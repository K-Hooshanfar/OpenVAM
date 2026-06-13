#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for OpenVAM with PRETRAINED WEIGHTS from no_vit.py.

This script:
1. Creates a OpenVAM model (checkpoint compatible)
2. Loads pretrained weights from a trained no_vit.py checkpoint
3. Fine-tunes the model with text conditioning

The compatible model uses the same DPT scratch architecture as no_vit.py,
so weights can be loaded directly without shape mismatches.

Usage:
    # With stage-1 (no_vit) weights:
    python scripts/train_stage2_visual.py \
        --pretrained_checkpoint path/to/novit_checkpoint.pth \
        --train_jsonl datasets/salience_train.jsonl \
        --val_jsonl datasets/salience_test.jsonl \
        --saliency_dir salicon_256/saliency \
        --fixation_dir salicon_256/fixations

    Without stage-1 weights: add --no_pretrained and omit --pretrained_checkpoint.

Merged layout (train/val with stimuli, saliency, fixations per split):
    If your data has structure:
        merged/train/{stimuli,saliency,fixations}  and  merged/val/{stimuli,saliency,fixations}
    and JSONL lines have "id" (e.g. {"id": "CAT2000_256_Action_001", ...}), pass the merged
    base directory as both --saliency_dir and --fixation_dir:
        --train_jsonl .../merged/merged_train.jsonl \
        --val_jsonl .../merged/merged_val.jsonl \
        --saliency_dir /path/to/merged \
        --fixation_dir /path/to/merged
"""

import os
import sys
import argparse
import json
import math
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from PIL import Image
from tqdm import tqdm

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the OpenVAM model and dataset-type helpers for prompts
from net.openvam import (
    OpenVAM,
    get_dataset_from_id,
    get_dataset_type_from_name,
    get_user_prompt_for_dataset_type,
)
from utils.losses import loss_KLdiv, loss_CC, loss_similarity, loss_NSS


# =============================================================================
# Weight Loading for Compatible Model
# =============================================================================

def load_novit_weights_into_compatible_model(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
    verbose: bool = True,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load weights from a trained no_vit.py checkpoint into OpenVAM model.
    
    The compatible model uses the same structure as no_vit.py, so most weights
    can be loaded directly:
    - proj_to_l1, proj_to_l2, proj_to_l3, proj_to_l4 -> same names
    - scratch.layerX_rn -> same names
    - scratch.refinenetX -> same names
    - scratch.output_conv -> same names
    - dino.* -> same names
    
    Args:
        model: OpenVAM model instance
        checkpoint_path: Path to the no_vit.py checkpoint (.pth file)
        strict: If True, raise error on missing keys
        verbose: If True, print detailed loading info
        
    Returns:
        Tuple of (loaded_keys, skipped_keys, missing_keys)
    """
    if verbose:
        print("="*70)
        print("Loading Pretrained Weights from no_vit.py (Compatible Model)")
        print("="*70)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    if verbose:
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Keys in checkpoint: {len(state_dict)}")
    
    # Helper function to safely get parameter shape
    def safe_get_shape(param):
        """Safely get parameter shape, handling uninitialized LazyConv2d parameters."""
        try:
            return param.shape
        except RuntimeError:
            # Parameter is uninitialized (LazyConv2d)
            return None
    
    # Initialize LazyConv2d modules by running a dummy forward pass
    if verbose:
        print("\nInitializing LazyConv2d modules...")
    try:
        # Get device and dtype from model
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        
        # Use model's dtype if available, otherwise use float32
        if hasattr(model, 'dtype'):
            model_dtype = model.dtype
        else:
            # Try to get dtype from parameters, but prefer float32 for initialization
            model_dtype = torch.float32
        
        # Use float32 for dummy input to avoid dtype mismatches during initialization
        # The model will handle dtype conversion internally
        dummy_img = torch.randn(1, 3, 256, 256, device=device, dtype=torch.float32)
        
        # Set model to eval mode and use autocast if needed
        was_training = model.training
        model.eval()
        
        with torch.no_grad():
            # Use autocast if model uses bfloat16
            if model_dtype == torch.bfloat16 and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _ = model(dummy_img, text_prompt="dummy")
            else:
                _ = model(dummy_img, text_prompt="dummy")
        
        # Restore training mode
        if was_training:
            model.train()
        
        if verbose:
            print("  ✓ LazyConv2d modules initialized")
    except Exception as e:
        if verbose:
            print(f"  ⚠️  Could not initialize LazyConv2d: {e}")
            print("  Will handle uninitialized parameters during loading")
    
    # Get model's current state dict after initialization attempt
    model_state_dict = model.state_dict()
    
    # Map weights (compatible model uses same names as no_vit.py)
    new_state_dict = OrderedDict()
    loaded_keys = []
    skipped_keys = []
    
    # Direct mappings (same names)
    direct_transfer_keys = [
        "dino.",
        "proj_to_l1.",
        "proj_to_l2.",
        "proj_to_l3.",
        "proj_to_l4.",
        "scratch.",
    ]
    
    for old_name, weight in state_dict.items():
        # Check if this key should be transferred
        should_transfer = any(old_name.startswith(prefix) for prefix in direct_transfer_keys)
        
        if should_transfer:
            # Use same name (compatible model structure matches no_vit.py)
            new_name = old_name
            if new_name in model_state_dict:
                model_param = model_state_dict[new_name]
                model_shape = safe_get_shape(model_param)
                
                if model_shape is None:
                    # Parameter is uninitialized - load the weight anyway (it will initialize it)
                    new_state_dict[new_name] = weight
                    loaded_keys.append(f"{old_name} -> {new_name} (uninitialized, will initialize)")
                elif model_shape == weight.shape:
                    new_state_dict[new_name] = weight
                    loaded_keys.append(f"{old_name} -> {new_name}")
                else:
                    skipped_keys.append(f"{old_name} (shape mismatch: checkpoint {weight.shape} vs model {model_shape})")
            else:
                skipped_keys.append(f"{old_name} (not in model)")
        else:
            skipped_keys.append(old_name)
    
    if verbose:
        print(f"\nMapped weights: {len(loaded_keys)}")
        print(f"Skipped weights: {len(skipped_keys)}")
    
    # Find missing keys (in model but not in loaded weights)
    missing_keys = [k for k in model_state_dict.keys() if k not in new_state_dict]
    
    # Find extra keys (in loaded weights but not in model)
    extra_keys = [k for k in new_state_dict.keys() if k not in model_state_dict]
    
    if verbose:
        print(f"\nMissing keys (will keep random init): {len(missing_keys)}")
        if missing_keys:
            missing_components = {}
            for key in missing_keys:
                component = key.split(".")[0]
                missing_components[component] = missing_components.get(component, 0) + 1
            for component, count in sorted(missing_components.items()):
                print(f"  {component}: {count} weights")
        
        print(f"Extra keys (will be ignored): {len(extra_keys)}")
    
    # Load weights
    model.load_state_dict(new_state_dict, strict=False)
    
    if verbose:
        print(f"\n✓ Successfully loaded {len(new_state_dict)} weights")
        
        # Print summary of what was loaded
        print("\n" + "-"*70)
        print("LOADED COMPONENTS:")
        print("-"*70)
        
        component_counts = {}
        for key in new_state_dict:
            component = key.split(".")[0]
            component_counts[component] = component_counts.get(component, 0) + 1
        
        for component, count in sorted(component_counts.items()):
            print(f"  {component}: {count} weights")
        
        print("="*70)
    
    if strict and missing_keys:
        raise RuntimeError(f"Missing keys in checkpoint: {missing_keys[:10]}...")
    
    return loaded_keys, skipped_keys, missing_keys


# =============================================================================
# Dataset (same as original)
# =============================================================================

class SalienceDataset(Dataset):
    """Dataset for saliency prediction with text prompts."""
    
    def __init__(
        self,
        jsonl_path: str,
        saliency_dir: str,
        fixation_dir: Optional[str] = None,
        image_size: int = 256,
        augment: bool = False,
    ):
        self.saliency_dir = Path(saliency_dir)
        self.fixation_dir = Path(fixation_dir) if fixation_dir else None
        self.image_size = image_size
        self.augment = augment
        self.missing_saliency_count = 0
        
        # Detect if saliency_dir has train/val subdirectories
        self.has_subdirs = False
        train_subdir = self.saliency_dir / "train"
        val_subdir = self.saliency_dir / "val"
        if train_subdir.exists() or val_subdir.exists():
            self.has_subdirs = True
            print(f"  Detected train/val subdirectories in saliency_dir")
        
        # Determine which subdirectory to use based on JSONL path
        self.saliency_subdir = None
        if self.has_subdirs:
            jsonl_name = Path(jsonl_path).stem.lower()
            if "train" in jsonl_name:
                self.saliency_subdir = train_subdir if train_subdir.exists() else self.saliency_dir
            elif "val" in jsonl_name or "test" in jsonl_name:
                self.saliency_subdir = val_subdir if val_subdir.exists() else self.saliency_dir
            else:
                if train_subdir.exists() and len(list(train_subdir.glob("*"))) > 0:
                    self.saliency_subdir = train_subdir
                elif val_subdir.exists() and len(list(val_subdir.glob("*"))) > 0:
                    self.saliency_subdir = val_subdir
                else:
                    self.saliency_subdir = self.saliency_dir
            
            if self.saliency_subdir != self.saliency_dir:
                # Merged layout: train/saliency/ and val/saliency/ contain the maps
                nested_sal = self.saliency_subdir / "saliency"
                if nested_sal.is_dir():
                    self.saliency_subdir = nested_sal
                print(f"  Using saliency subdirectory: {self.saliency_subdir}")
        
        # Same for fixation_dir - check if it has subdirectories independently
        self.fixation_subdir = None
        if self.fixation_dir:
            train_fix_subdir = self.fixation_dir / "train"
            val_fix_subdir = self.fixation_dir / "val"
            train_edit_subdir = self.fixation_dir / "train_edit"
            val_edit_subdir = self.fixation_dir / "val_edit"
            
            train_exists = train_fix_subdir.exists() or train_edit_subdir.exists()
            val_exists = val_fix_subdir.exists() or val_edit_subdir.exists()
            fixation_has_subdirs = train_exists or val_exists
            
            if fixation_has_subdirs:
                print(f"  Detected train/val subdirectories in fixation_dir")
                
                if "train" in jsonl_name:
                    if train_edit_subdir.exists():
                        self.fixation_subdir = train_edit_subdir
                    elif train_fix_subdir.exists():
                        self.fixation_subdir = train_fix_subdir
                    else:
                        self.fixation_subdir = self.fixation_dir
                elif "val" in jsonl_name or "test" in jsonl_name:
                    if val_edit_subdir.exists():
                        self.fixation_subdir = val_edit_subdir
                    elif val_fix_subdir.exists():
                        self.fixation_subdir = val_fix_subdir
                    else:
                        self.fixation_subdir = self.fixation_dir
                else:
                    if train_edit_subdir.exists() and len(list(train_edit_subdir.glob("*"))) > 0:
                        self.fixation_subdir = train_edit_subdir
                    elif train_fix_subdir.exists() and len(list(train_fix_subdir.glob("*"))) > 0:
                        self.fixation_subdir = train_fix_subdir
                    elif val_edit_subdir.exists() and len(list(val_edit_subdir.glob("*"))) > 0:
                        self.fixation_subdir = val_edit_subdir
                    elif val_fix_subdir.exists() and len(list(val_fix_subdir.glob("*"))) > 0:
                        self.fixation_subdir = val_fix_subdir
                    else:
                        self.fixation_subdir = self.fixation_dir
                
                if self.fixation_subdir != self.fixation_dir:
                    # Merged layout: train/fixations/ and val/fixations/ contain the maps
                    nested_fix = self.fixation_subdir / "fixations"
                    if nested_fix.is_dir():
                        self.fixation_subdir = nested_fix
                    elif (self.fixation_subdir / "fixation").is_dir():
                        self.fixation_subdir = self.fixation_subdir / "fixation"
                    print(f"  Using fixation subdirectory: {self.fixation_subdir}")
            else:
                self.fixation_subdir = self.fixation_dir
        
        # Load JSONL
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        
        # Detect merged layout: base_dir/train|val/{stimuli,saliency,fixations}, JSONL has "id"
        self.merged_layout = False
        self.merged_base = None
        self.merged_subdir = None
        train_sal = self.saliency_dir / "train" / "saliency"
        val_sal = self.saliency_dir / "val" / "saliency"
        if (train_sal.exists() or val_sal.exists()) and len(self.samples) > 0 and self.samples[0].get("id") is not None:
            self.merged_layout = True
            self.merged_base = self.saliency_dir
            jsonl_name = Path(jsonl_path).stem.lower()
            if "train" in jsonl_name:
                self.merged_subdir = "train" if (self.merged_base / "train" / "stimuli").exists() else "val"
            else:
                self.merged_subdir = "val" if (self.merged_base / "val" / "stimuli").exists() else "train"
            self.saliency_subdir = self.merged_base / self.merged_subdir / "saliency"
            if self.fixation_dir:
                self.fixation_subdir = self.merged_base / self.merged_subdir / "fixations"
            print(f"  Using merged layout: base={self.merged_base}, subdir={self.merged_subdir}")
        
        # Quick check: count how many saliency files exist
        def _sample_image_name(sample):
            if self.merged_layout:
                return sample.get("id", "")
            image_path = sample.get("image", sample.get("images", [""])[0])
            if isinstance(image_path, list):
                image_path = image_path[0]
            return Path(image_path).stem if image_path else ""
        
        search_dir = self.saliency_subdir if self.saliency_subdir else self.saliency_dir
        found_count = 0
        for sample in self.samples[:min(100, len(self.samples))]:
            image_name = _sample_image_name(sample)
            if not image_name:
                continue
            for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                if (search_dir / f"{image_name}{ext}").exists():
                    found_count += 1
                    break
        
        # Quick check: count how many fixation files exist
        fix_found_count = 0
        if self.fixation_dir:
            fix_search_dir = self.fixation_subdir if self.fixation_subdir else self.fixation_dir
            for sample in self.samples[:min(100, len(self.samples))]:
                image_name = _sample_image_name(sample)
                if not image_name:
                    continue
                for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                    if (fix_search_dir / f"{image_name}{ext}").exists():
                        fix_found_count += 1
                        break
        
        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")
        if len(self.samples) > 0:
            sample_rate = found_count / min(100, len(self.samples)) * 100
            print(f"  Saliency file check (first 100 samples): {found_count}/{min(100, len(self.samples))} found ({sample_rate:.1f}%)")
            if sample_rate < 50:
                print(f"  WARNING: Many saliency files may be missing! Check directory: {search_dir}")
            
            if self.fixation_dir:
                fix_rate = fix_found_count / min(100, len(self.samples)) * 100
                fix_search_dir = self.fixation_subdir if self.fixation_subdir else self.fixation_dir
                print(f"  Fixation file check (first 100 samples): {fix_found_count}/{min(100, len(self.samples))} found ({fix_rate:.1f}%)")
                if fix_rate < 50:
                    print(f"  WARNING: Many fixation files may be missing! Check directory: {fix_search_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def _load_image(self, path: str) -> torch.Tensor:
        """Load and preprocess image."""
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img = torch.from_numpy(np.array(img)).float() / 255.0
        img = img.permute(2, 0, 1)  # [3, H, W]
        return img
    
    def _load_saliency(self, image_name: str, image_filename: str = None) -> torch.Tensor:
        """Load saliency map."""
        smap_path = None
        search_dir = self.saliency_subdir if self.saliency_subdir else self.saliency_dir
        
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
            candidate = search_dir / f"{image_name}{ext}"
            if candidate.exists():
                smap_path = candidate
                break
        
        if smap_path is None and image_filename:
            candidate = search_dir / image_filename
            if candidate.exists():
                smap_path = candidate
        
        if smap_path is None or not smap_path.exists():
            self.missing_saliency_count += 1
            smap = torch.zeros(1, self.image_size, self.image_size)
            smap = smap / (smap.sum() + 1e-8)
            return smap
        
        smap = Image.open(smap_path).convert("L")
        smap = smap.resize((self.image_size, self.image_size), Image.BILINEAR)
        smap = torch.from_numpy(np.array(smap)).float() / 255.0
        smap = smap.unsqueeze(0)
        smap = smap / (smap.sum() + 1e-8)
        return smap
    
    def _load_fixation(self, image_name: str, image_filename: str = None) -> Optional[torch.Tensor]:
        """Load fixation map."""
        if self.fixation_dir is None:
            return None
        
        fix_path = None
        search_dir = self.fixation_subdir if self.fixation_subdir else self.fixation_dir
        
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".mat"]:
            candidate = search_dir / f"{image_name}{ext}"
            if candidate.exists():
                fix_path = candidate
                break
        
        if fix_path is None and image_filename:
            candidate = search_dir / image_filename
            if candidate.exists():
                fix_path = candidate
        
        if fix_path is None or not fix_path.exists():
            return None
        
        if fix_path.suffix.lower() == ".mat":
            return None
        
        fix = Image.open(fix_path).convert("L")
        fix = fix.resize((self.image_size, self.image_size), Image.NEAREST)
        fix = torch.from_numpy(np.array(fix)).float() / 255.0
        fix = fix.unsqueeze(0)
        fix = (fix > 0.5).float()  # Binarize
        return fix
    
    # Fallback when dataset type cannot be resolved (must match model natural_scene user prompt)
    TEXT_PROMPT_CANONICAL = (
        "What would draw a typical viewer's attention in this image? "
        "List 2–6 salient objects/regions in descending importance. "
        "For each, give a location phrase and 1–2 short justifications grounded only in what is visible. "
        "Do not guess hidden details or invent objects; use \"Uncertain:\" if needed."
    )

    def _extract_text_prompt(self, sample: dict) -> str:
        """Extract text prompt; use dataset-type-specific prompt when sample has id (merged layout)."""
        sample_id = sample.get("id", "")
        if sample_id:
            ds_name = get_dataset_from_id(sample_id)
            ds_type = get_dataset_type_from_name(ds_name)
            return get_user_prompt_for_dataset_type(ds_type)
        return self.TEXT_PROMPT_CANONICAL
    
    def _resolve_stimuli_path(self, sample: dict) -> str:
        """Resolve image path; support merged layout (id -> base/subdir/stimuli/id.jpg)."""
        # Explicit path from JSONL (stimuli / image / images)
        path = sample.get("stimuli") or sample.get("image") or (sample.get("images") or [""])[0]
        if isinstance(path, list):
            path = path[0] if path else ""
        if path and isinstance(path, str) and Path(path).exists():
            return path
        if self.merged_layout and self.merged_base is not None:
            sid = sample.get("id", "")
            if sid:
                for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                    p = self.merged_base / self.merged_subdir / "stimuli" / f"{sid}{ext}"
                    if p.exists():
                        return str(p)
                return str(self.merged_base / self.merged_subdir / "stimuli" / f"{sid}.jpg")
        image_path = sample.get("image", sample.get("images", [""])[0])
        if isinstance(image_path, list):
            image_path = image_path[0]
        return image_path or ""
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = self._resolve_stimuli_path(sample)
        img = self._load_image(image_path)
        if self.merged_layout:
            image_name = sample.get("id", Path(image_path).stem)
            image_filename = None
        else:
            image_path_obj = Path(image_path)
            image_name = image_path_obj.stem
            image_filename = image_path_obj.name
        
        smap = self._load_saliency(image_name, image_filename)
        fmap = self._load_fixation(image_name, image_filename)
        if fmap is None:
            fmap = torch.zeros_like(smap)
        
        text_prompt = self._extract_text_prompt(sample)
        
        # if self.augment and random.random() > 0.5:
        #     img = torch.flip(img, [-1])
        #     smap = torch.flip(smap, [-1])
        #     fmap = torch.flip(fmap, [-1])
        
        return {
            "image": img,
            "saliency": smap,
            "fixation": fmap,
            "text": text_prompt,
            "image_path": image_path,
            "id": sample.get("id", ""),
        }


def collate_fn(batch):
    """Custom collate function to handle text prompts and dataset_type."""
    images = torch.stack([b["image"] for b in batch])
    saliency = torch.stack([b["saliency"] for b in batch])
    fixation = torch.stack([b["fixation"] for b in batch])
    texts = [b["text"] for b in batch]
    paths = [b["image_path"] for b in batch]
    ids = [b.get("id", "") for b in batch]
    dataset_types = [get_dataset_type_from_name(get_dataset_from_id(i)) for i in ids]
    
    return {
        "image": images,
        "saliency": saliency,
        "fixation": fixation,
        "text": texts,
        "image_path": paths,
        "id": ids,
        "dataset_type": dataset_types,
    }


# =============================================================================
# Loss Functions
# =============================================================================

class CombinedSaliencyLoss(nn.Module):
    """Combined loss for saliency prediction."""
    
    def __init__(
        self,
        kld_weight: float = 1.0,
        cc_weight: float = 1.0,
        sim_weight: float = 1.0,
        nss_weight: float = 1.0,
        mse_weight: float = 1.0,
    ):
        super().__init__()
        self.kld_weight = kld_weight
        self.cc_weight = cc_weight
        self.sim_weight = sim_weight
        self.nss_weight = nss_weight
        self.mse_weight = mse_weight
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        smap: torch.Tensor,
        fmap: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss."""
        if pred.shape[-2:] != smap.shape[-2:]:
            pred = F.interpolate(pred, size=smap.shape[-2:], mode="bilinear", align_corners=False)
        
        B = pred.shape[0]
        mse_val = self.mse(pred, smap)
        
        pred_sq = pred.squeeze(1)
        smap_sq = smap.squeeze(1)
        fmap_sq = fmap.squeeze(1) if fmap is not None else None
        
        kld_losses = []
        cc_losses = []
        sim_losses = []
        nss_losses = []
        
        for i in range(B):
            if self.kld_weight > 0 and loss_KLdiv is not None:
                kld = loss_KLdiv(pred_sq[i], smap_sq[i])
                if not torch.isnan(kld):
                    kld_losses.append(kld)
            
            if self.cc_weight > 0 and loss_CC is not None:
                cc = loss_CC(pred_sq[i], smap_sq[i])
                if not torch.isnan(cc):
                    cc_losses.append(cc)
            
            if self.sim_weight > 0 and loss_similarity is not None:
                sim = loss_similarity(pred_sq[i], smap_sq[i])
                if not torch.isnan(sim):
                    sim_losses.append(sim)
            
            if self.nss_weight > 0 and loss_NSS is not None and fmap_sq is not None:
                fix_sum = fmap_sq[i].sum()
                if fix_sum > 0:
                    nss = loss_NSS(pred_sq[i], fmap_sq[i])
                    if not torch.isnan(nss) and not torch.isinf(nss):
                        nss_losses.append(nss)
        
        kld_val = torch.stack(kld_losses).mean() if kld_losses else torch.tensor(0.0, device=pred.device)
        cc_val = torch.stack(cc_losses).mean() if cc_losses else torch.tensor(0.0, device=pred.device)
        sim_val = torch.stack(sim_losses).mean() if sim_losses else torch.tensor(0.0, device=pred.device)
        nss_val = torch.stack(nss_losses).mean() if nss_losses else torch.tensor(0.0, device=pred.device)
        
        total_loss = (
            self.mse_weight * mse_val +
            self.kld_weight * kld_val +
            self.cc_weight * (1.0 - cc_val) +
            self.sim_weight * (1.0 - sim_val) +
            self.nss_weight * (-nss_val)
        )
        
        losses = {
            "total": total_loss.item(),
            "mse": mse_val.item(),
            "kld": kld_val.item(),
            "cc": cc_val.item(),
            "sim": sim_val.item(),
            "nss": nss_val.item(),
        }
        
        return total_loss, losses


# # =============================================================================
# # Chat Template Verification
# # =============================================================================

# def verify_chat_template(model: nn.Module, sample_text: str, device: torch.device = None):
#     """
#     Verify that chat template is correctly applied in the model.
    
#     This checks:
#     1. Chat template method exists and works
#     2. Formatted text contains expected special tokens
#     3. Tokenization produces correct format
#     4. encode_text() uses the template correctly
    
#     Args:
#         model: The model to verify
#         sample_text: Sample prompt text
#         device: Device to use for verification
#     """
#     print("\n" + "="*70)
#     print("Verifying Chat Template Application")
#     print("="*70)
    
#     try:
#         # Check if _apply_chat_template method exists
#         if not hasattr(model, '_apply_chat_template'):
#             print("  ❌ ERROR: Model does not have _apply_chat_template method!")
#             print("     This means chat template is NOT being applied.")
#             print("     Training and inference will have format mismatches!")
#             return False
        
#         # Test chat template application
#         print(f"\n  Testing chat template:")
#         print(f"    Dataset prompt (IGNORED): '{sample_text[:50]}...'")
#         print(f"    Note: Model uses fixed prompt, ignoring dataset text.")
        
#         # Test inference format (user only)
#         formatted_inference = model._apply_chat_template(sample_text)
#         print(f"\n  ✓ Inference format (user only):")
#         print(f"    Length: {len(formatted_inference)} chars")
#         print(f"    Preview: {formatted_inference[:100]}...")
        
#         # Check for expected special tokens
#         has_im_start = "<|im_start|>" in formatted_inference or "im_start" in formatted_inference.lower()
#         has_im_end = "<|im_end|>" in formatted_inference or "im_end" in formatted_inference.lower()
#         has_user = "user" in formatted_inference.lower()
#         has_assistant = "assistant" in formatted_inference.lower()
        
#         print(f"\n  Special token check:")
#         print(f"    Contains user role: {has_user} {'✓' if has_user else '❌'}")
#         print(f"    Contains assistant role: {has_assistant} {'✓' if has_assistant else '❌'}")
#         print(f"    Contains im_start/im_end tokens: {has_im_start or has_im_end} {'✓' if (has_im_start or has_im_end) else '⚠️'}")
        
#         # Verify tokenization produces expected format
#         print(f"\n  Verifying tokenization...")
#         tokens_inference = model.tokenizer(
#             formatted_inference,
#             return_tensors="pt",
#             add_special_tokens=False,  # Template already has special tokens
#         )
#         print(f"    Inference tokens: {tokens_inference['input_ids'].shape[1]} tokens")
        
#         # Compare with raw tokenization (should be different if template is applied)
#         tokens_raw = model.tokenizer(
#             sample_text,
#             return_tensors="pt",
#             add_special_tokens=False,
#         )
#         raw_length = tokens_raw['input_ids'].shape[1]
#         template_length = tokens_inference['input_ids'].shape[1]
        
#         if template_length > raw_length:
#             print(f"    ✓ Template adds {template_length - raw_length} tokens (expected)")
#         elif template_length == raw_length:
#             print(f"    ⚠️  WARNING: Template length equals raw length!")
#             print(f"       This suggests template might not be applied correctly.")
#         else:
#             print(f"    ⚠️  WARNING: Template length is shorter than raw (unexpected)")
        
#         # Verify encode_text uses template
#         print(f"\n  Verifying encode_text() uses template...")
#         verify_device = device if device is not None else torch.device("cpu")
#         text_embeds, text_mask, text_len = model.encode_text(sample_text, batch_size=1, device=verify_device)
#         print(f"    encode_text() output length: {text_len} tokens")
        
#         if text_len == template_length:
#             print(f"    ✓ encode_text() uses chat template correctly")
#         else:
#             print(f"    ⚠️  WARNING: encode_text() length ({text_len}) != template length ({template_length})")
#             print(f"       This suggests encode_text() might not be using the template!")
        
#         print(f"\n  {'='*70}")
#         print(f"  ✓ Chat template verification PASSED")
#         print(f"  {'='*70}")
#         return True
        
#     except Exception as e:
#         print(f"\n  ❌ ERROR during chat template verification: {e}")
#         import traceback
#         traceback.print_exc()
#         return False


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    accumulation_steps: int = 1,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    
    total_losses = {}
    num_batches = 0
    
    # Track truncation statistics
    truncation_count = 0
    total_texts = 0
    max_text_len = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device, dtype=raw_model.dtype)
        saliency = batch["saliency"].to(device, dtype=raw_model.dtype)
        fixation = batch["fixation"].to(device, dtype=raw_model.dtype)
        texts = batch["text"]
        
        # Check for truncation (only on first batch of first epoch)
        if epoch == 1 and batch_idx == 0:
            # # Log complete prompt template once at start of training
            # model_for_prompt = model.module if isinstance(model, nn.DataParallel) else model
            # if hasattr(model_for_prompt, "_apply_chat_template"):
            #     sample_prompt = texts[0] if texts else "(none)"
            #     formatted_template = model_for_prompt._apply_chat_template(sample_prompt)
            #     print("\n" + "=" * 70)
            #     print("Complete prompt template that goes to Qwen (first batch, epoch 1)")
            #     print("=" * 70)
            #     print("Raw prompt (before template):")
            #     print(sample_prompt)
            #     print("\n--- Full template string sent to tokenizer (inference / prompt-only) ---")
            #     print(formatted_template)
            #     print("--- End template ---")
            #     print("=" * 70 + "\n")
            for text in texts:
                total_texts += 1
                tokens = raw_model.tokenizer(text, return_tensors="pt", add_special_tokens=False)
                text_len = tokens['input_ids'].shape[1]
                max_text_len = max(max_text_len, text_len)
                if text_len > raw_model.max_text_length:
                    truncation_count += 1
        
        dataset_type = batch.get("dataset_type", [None])
        dt = dataset_type[0] if dataset_type else None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(images, text_prompt=texts[0], dataset_type=dt)
            loss, loss_dict = loss_fn(pred, saliency, fixation)
            loss = loss / accumulation_steps
        
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
        
        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v
        num_batches += 1
        
        pbar.set_postfix({
            "loss": f"{loss_dict['total']:.4f}",
            "mse": f"{loss_dict.get('mse', 0):.4f}",
            "kld": f"{loss_dict.get('kld', 0):.4f}",
            "cc": f"{loss_dict.get('cc', 0):.4f}",
        })
    
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    # Print truncation stats on first epoch
    if epoch == 1 and total_texts > 0:
        truncation_pct = 100 * truncation_count / total_texts
        print(f"\n  Text Length Stats (first batch): max={max_text_len}, truncations={truncation_count}/{total_texts} ({truncation_pct:.1f}%)")
        if truncation_count > 0:
            print(f"  ⚠️  WARNING: {truncation_count} texts truncated (max_text_length={raw_model.max_text_length})")
    
    return avg_losses


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    
    total_losses = {}
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    for batch in pbar:
        images = batch["image"].to(device, dtype=raw_model.dtype)
        saliency = batch["saliency"].to(device, dtype=raw_model.dtype)
        fixation = batch["fixation"].to(device, dtype=raw_model.dtype)
        texts = batch["text"]
        dataset_type = batch.get("dataset_type", [None])
        dt = dataset_type[0] if dataset_type else None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(images, text_prompt=texts[0], dataset_type=dt)
            _, loss_dict = loss_fn(pred, saliency, fixation)
        
        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v
        num_batches += 1
        
        pbar.set_postfix({
            "loss": f"{loss_dict['total']:.4f}",
            "mse": f"{loss_dict.get('mse', 0):.4f}",
            "cc": f"{loss_dict.get('cc', 0):.4f}",
        })
    
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return avg_losses


@torch.no_grad()
def validate_per_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Run validation per sample and aggregate MSE, KLD, CC, SIM, NSS by dataset (from sample id)."""
    model.eval()
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    per_dataset = {}
    for batch in tqdm(dataloader, desc="Val (per-dataset)", leave=False):
        images = batch["image"].to(device, dtype=raw_model.dtype)
        saliency = batch["saliency"].to(device, dtype=raw_model.dtype)
        fixation = batch["fixation"].to(device, dtype=raw_model.dtype)
        texts = batch["text"]
        ids = batch.get("id", [""] * len(texts))
        for i in range(len(texts)):
            ds = get_dataset_from_id(ids[i] if i < len(ids) else "")
            if ds not in per_dataset:
                per_dataset[ds] = {"mse": [], "kld": [], "cc": [], "sim": [], "nss": []}
            text_prompt = texts[i] if texts[i] else None
            dataset_type = get_dataset_type_from_name(ds)
            image_i = images[i : i + 1]
            saliency_i = saliency[i : i + 1]
            fixation_i = fixation[i : i + 1]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(image_i, text_prompt=text_prompt, dataset_type=dataset_type)
                _, loss_dict = loss_fn(pred, saliency_i, fixation_i)
            pred = pred.float()
            saliency_i = saliency_i.float()
            fixation_i = fixation_i.float()
            per_dataset[ds]["mse"].append(loss_dict.get("mse", 0))
            per_dataset[ds]["kld"].append(loss_dict.get("kld", 0))
            per_dataset[ds]["cc"].append(loss_dict.get("cc", 0))
            per_dataset[ds]["sim"].append(loss_dict.get("sim", 0))
            per_dataset[ds]["nss"].append(loss_dict.get("nss", 0))
    results = {}
    for ds, vals in per_dataset.items():
        n = len(vals["mse"])
        if n == 0:
            continue
        results[ds] = {
            "n_samples": n,
            "mse": sum(vals["mse"]) / n,
            "kld": sum(vals["kld"]) / n,
            "cc": sum(vals["cc"]) / n,
            "sim": sum(vals["sim"]) / n,
            "nss": sum(vals["nss"]) / n,
        }
    return results


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_loss: float,
    save_path: str,
):
    """Save a checkpoint (unwraps DataParallel so keys have no 'module.' prefix)."""
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({
        "epoch": epoch,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_loss": best_loss,
    }, save_path)
    print(f"Saved checkpoint to {save_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    # When stdout is piped (e.g. to tee for logging), disable tqdm so logs stay minimal
    if not sys.stdout.isatty():
        os.environ["TQDM_DISABLE"] = "1"
    parser = argparse.ArgumentParser(description="Train OpenVAM with pretrained weights")
    
    # Pretrained weights (stage-1 no_vit checkpoint). Omit with --no_pretrained.
    parser.add_argument("--pretrained_checkpoint", type=str, default=None,
                        help="Path to no_vit.py pretrained checkpoint (ignored if --no_pretrained)")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Do not load stage-1 weights; keep DINO+DPT random init (Qwen still loads from HF)")
    parser.add_argument("--verify_weights", action="store_true",
                        help="Verify weight transfer from checkpoint")
    parser.add_argument("--verify_prompts", action="store_true",
                        help="Print dataset-type prompts (system/user/formatted) that go to the model, then continue training")
    
    # Data paths
    parser.add_argument("--train_jsonl", type=str, default="datasets/salience_train.jsonl")
    parser.add_argument("--val_jsonl", type=str, default="datasets/salience_test.jsonl")
    parser.add_argument("--saliency_dir", type=str, default="salicon_256/saliency",
                        help="Saliency maps directory for training")
    parser.add_argument("--fixation_dir", type=str, default="salicon_256/fixations",
                        help="Fixation maps directory for training")
    parser.add_argument("--val_saliency_dir", type=str, default=None,
                        help="Saliency maps directory for validation (default: use saliency_dir)")
    parser.add_argument("--val_fixation_dir", type=str, default=None,
                        help="Fixation maps directory for validation (default: use fixation_dir)")
    
    # Model config
    parser.add_argument("--dino_model", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--backbone", type=str, default="vitb_rn50_384",
                        help="DPT backbone name (used to size scratch layers)")
    parser.add_argument("--features", type=int, default=256,
                        help="DPT features (used by DPT base class)")
    parser.add_argument("--image_size", type=int, default=256)
    
    # Freeze options
    parser.add_argument("--freeze_dino", action="store_true", help="Freeze DINO backbone")
    parser.add_argument("--freeze_dpt", action="store_true", help="Freeze DPT components")
    parser.add_argument("--freeze_projector", action="store_true", help="Freeze projector")
    
    # Training config
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--num_epochs", type=int, default=None, help="Alias for --epochs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--learning_rate", type=float, default=None, help="Alias for --lr")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    
    # Loss weights
    parser.add_argument("--mse_weight", type=float, default=0)
    parser.add_argument("--kld_weight", type=float, default=1.0)
    parser.add_argument("--cc_weight", type=float, default=1.0)
    parser.add_argument("--sim_weight", type=float, default=0.5)
    parser.add_argument("--nss_weight", type=float, default=0.5)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints_text_pretrained_compatible")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Stop training after this many epochs without validation improvement (0 = disabled)")
    parser.add_argument("--best_metric", type=str, default="val_loss",
                        choices=("val_loss", "avg_cc", "min_kld"),
                        help="Which metric to use for saving best_model.pth: val_loss (default), avg_cc (mean CC over datasets), min_kld (lowest mean KLD across datasets)")
    parser.add_argument("--log_per_dataset", action="store_true",
                        help="Print per-dataset validation metrics after each epoch (default: off)")
    
    args = parser.parse_args()
    if not args.no_pretrained and not args.pretrained_checkpoint:
        parser.error("Either pass --pretrained_checkpoint PATH or use --no_pretrained")
    
    # Handle aliases
    if args.num_epochs is not None:
        args.epochs = args.num_epochs
    if args.learning_rate is not None:
        args.lr = args.learning_rate
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save args
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ==========================================================================
    # Create Model
    # ==========================================================================
    print("\n" + "="*70)
    print("Creating OpenVAM Model")
    print("="*70)
    
    model = OpenVAM(
        dino_model_name=args.dino_model,
        qwen_model_name=args.qwen_model,
        backbone=args.backbone,
        features=args.features,
        freeze_dino=False,
        freeze_qwen_lm=True,
        freeze_projector=False,
    )
    
    # ==========================================================================
    # Load Pretrained Weights (optional)
    # ==========================================================================
    if args.no_pretrained:
        print("\n" + "="*70)
        print("Skipping stage-1 checkpoint (--no_pretrained)")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("Loading Pretrained Weights from no_vit.py")
        print("="*70)
        load_novit_weights_into_compatible_model(
            model,
            args.pretrained_checkpoint,
            strict=False,
            verbose=True,
        )
    
    # ==========================================================================
    # Apply Freezing
    # ==========================================================================
    if args.freeze_dino:
        print("\nFreezing DINO backbone...")
        for p in model.dino.parameters():
            p.requires_grad = False
    
    if args.freeze_dpt:
        print("\nFreezing DPT components...")
        dpt_modules = [
            model.proj_to_l1, model.proj_to_l2, model.proj_to_l3, model.proj_to_l4,
            model.scratch.layer1_rn, model.scratch.layer2_rn, 
            model.scratch.layer3_rn, model.scratch.layer4_rn,
            model.scratch.refinenet1, model.scratch.refinenet2, 
            model.scratch.refinenet3, model.scratch.refinenet4,
            model.scratch.output_conv,
        ]
        for module in dpt_modules:
            for p in module.parameters():
                p.requires_grad = False
    
    if args.freeze_projector:
        print("\nFreezing projector...")
        for p in model.dino_adapter.parameters():
            p.requires_grad = False
        for p in model.patch_merger.parameters():
            p.requires_grad = False
    
    model = model.to(device)
    
    # Enable gradient checkpointing to reduce activation memory
    print("\nEnabling gradient checkpointing...")
    model.enable_gradient_checkpointing()
    
    # Multi-GPU model parallelism: split DINO+DPT on GPU 0, Qwen on GPU 1
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"\nFound {num_gpus} GPUs — enabling model parallelism (pipeline split)")
        model.enable_model_parallel(device0='cuda:0', device1='cuda:1')
    
    # raw_model is always the model itself (no DataParallel wrapper)
    raw_model = model
    
    # Print params (will initialize LazyConv2d if needed)
    try:
        raw_model.print_params()
    except Exception as e:
        print(f"Warning: Could not print params: {e}")
        print("  This is okay - LazyConv2d will be initialized during first forward pass")
    
    # Optional: verify that the correct dataset-type prompts go to the model
    if args.verify_prompts:
        raw_model.verify_dataset_prompts(max_system_user_chars=600, max_formatted_chars=1000)
    
    # ==========================================================================
    # Datasets
    # ==========================================================================
    print("\n" + "="*70)
    print("Loading Datasets")
    print("="*70)
    
    train_dataset = SalienceDataset(
        jsonl_path=args.train_jsonl,
        saliency_dir=args.saliency_dir,
        fixation_dir=args.fixation_dir,
        image_size=args.image_size,
        augment=False,
    )
    
    val_saliency_dir = args.val_saliency_dir if args.val_saliency_dir else args.saliency_dir
    val_fixation_dir = args.val_fixation_dir if args.val_fixation_dir else args.fixation_dir
    
    val_dataset = SalienceDataset(
        jsonl_path=args.val_jsonl,
        saliency_dir=val_saliency_dir,
        fixation_dir=val_fixation_dir,
        image_size=args.image_size,
        augment=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # # ==========================================================================
    # # Verify Chat Template (CRITICAL SAFETY CHECK)
    # # ==========================================================================
    # if len(train_dataset) > 0:
    #     # Get a sample from the dataset
    #     sample = train_dataset[0]
    #     sample_text = sample.get("text", "Describe what draws attention in this image.")
        
    #     # Print complete prompt template that goes to Qwen (inference / prompt-only)
    #     print("\n" + "="*70)
    #     print("Training Data Sample Preview")
    #     print("="*70)
    #     print(f"  Processed Input Prompt: {sample_text}")
    #     if hasattr(model, "_apply_chat_template"):
    #         template_inference = model._apply_chat_template(sample_text)
    #         print("\n  --- Complete prompt template sent to Qwen (inference / prompt-only) ---")
    #         print(template_inference)
    #         print("  --- End prompt template ---")
    #     print("="*70)
        
    #     # Verify chat template is correctly applied
    #     template_ok = verify_chat_template(model, sample_text, device=device)
        
    #     if not template_ok:
    #         print("\n" + "="*70)
    #         print("⚠️  WARNING: Chat template verification FAILED!")
    #         print("="*70)
    #         print("Training may proceed, but format mismatches between training")
    #         print("and inference are likely. This can cause poor model performance.")
    #         print("="*70)
    #         response = input("\nContinue training anyway? (y/n): ")
    #         if response.lower() != 'y':
    #             print("Training aborted by user.")
    #             return
    #     else:
    #         print("\n✓ Chat template verification passed - training format is correct!")
    # else:
    #     print("\n⚠️  WARNING: No training samples available - skipping chat template verification")
    
    # ==========================================================================
    # Optimizer & Scheduler
    # ==========================================================================
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=args.warmup_epochs * len(train_loader),
    )
    
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(args.epochs - args.warmup_epochs) * len(train_loader),
        eta_min=args.lr * 0.01,
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[args.warmup_epochs * len(train_loader)],
    )
    
    # ==========================================================================
    # Loss Function
    # ==========================================================================
    loss_fn = CombinedSaliencyLoss(
        kld_weight=args.kld_weight,
        cc_weight=args.cc_weight,
        sim_weight=args.sim_weight,
        nss_weight=args.nss_weight,
        mse_weight=args.mse_weight,
    )
    
    # ==========================================================================
    # Training Loop
    # ==========================================================================
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    best_val_loss = float("inf")
    best_avg_cc = -1.0
    best_avg_kld = float("inf")
    epochs_without_improvement = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*70}")
        
        train_losses = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch,
            accumulation_steps=args.accumulation_steps,
            scheduler=scheduler,
        )
        
        val_losses = validate(model, val_loader, loss_fn, device, epoch)
        
        print(f"\nTrain Loss: {train_losses['total']:.4f} "
              f"(MSE: {train_losses.get('mse', 0):.4f}, KLD: {train_losses.get('kld', 0):.4f}, "
              f"CC: {train_losses.get('cc', 0):.4f}, SIM: {train_losses.get('sim', 0):.4f}, "
              f"NSS: {train_losses.get('nss', 0):.4f})")
        print(f"Val Loss: {val_losses['total']:.4f} "
              f"(MSE: {val_losses.get('mse', 0):.4f}, KLD: {val_losses.get('kld', 0):.4f}, "
              f"CC: {val_losses.get('cc', 0):.4f}, SIM: {val_losses.get('sim', 0):.4f}, "
              f"NSS: {val_losses.get('nss', 0):.4f})")
        
        # Per-dataset validation metrics — only run when needed for logging or best-metric selection
        needs_per_ds = args.log_per_dataset or args.best_metric in ("avg_cc", "min_kld")
        per_ds = validate_per_dataset(model, val_loader, loss_fn, device) if needs_per_ds else {}
        if per_ds and args.log_per_dataset:
            print("\n  Val metrics per dataset:")
            for name in sorted(per_ds.keys()):
                m = per_ds[name]
                print(f"    {name}: MSE={m['mse']:.4f}  KLD={m['kld']:.4f}  CC={m['cc']:.4f}  SIM={m['sim']:.4f}  NSS={m['nss']:.4f}  (n={m['n_samples']})")
        
        # Decide whether this epoch is "best" according to --best_metric
        is_best = False
        if args.best_metric == "val_loss":
            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                is_best = True
        elif args.best_metric == "avg_cc" and per_ds:
            avg_cc = sum(per_ds[d]["cc"] for d in per_ds) / len(per_ds)
            if avg_cc > best_avg_cc:
                best_avg_cc = avg_cc
                is_best = True
        elif args.best_metric == "min_kld" and per_ds:
            avg_kld = sum(per_ds[d]["kld"] for d in per_ds) / len(per_ds)
            if avg_kld < best_avg_kld:
                best_avg_kld = avg_kld
                is_best = True
        elif args.best_metric != "val_loss" and not per_ds:
            # No per-dataset metrics; fall back to val_loss
            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                is_best = True
        
        if is_best:
            epochs_without_improvement = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_losses["total"],
                os.path.join(args.output_dir, "best_model.pth"),
            )
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(f"\nEarly stopping: no improvement for {args.early_stopping_patience} epochs.")
                break
        
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_losses["total"],
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth"),
            )
    
    save_checkpoint(
        model, optimizer, scheduler, epoch, val_losses["total"],
        os.path.join(args.output_dir, "final_model.pth"),
    )
    
    print("\n" + "="*70)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

