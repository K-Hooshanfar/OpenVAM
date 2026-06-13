#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for OpenVAM with LoRA (Low-Rank Adaptation).

This script uses LoRA to efficiently fine-tune the Qwen LLM components,
significantly reducing memory usage and training time while maintaining performance.

Usage:
    Single GPU:
        python scripts/train_stage3_lora.py \
            --pretrained_checkpoint path/to/checkpoint.pth \
            --train_jsonl datasets/salience_train.jsonl ...

    Multi-GPU (DistributedDataParallel):
        torchrun --nproc_per_node=N scripts/train_stage3_lora.py \
            --pretrained_checkpoint path/to/checkpoint.pth \
            --train_jsonl datasets/salience_train.jsonl \
            --val_jsonl datasets/salience_test.jsonl \
            --saliency_dir salicon_256/saliency \
            --fixation_dir salicon_256/fixations \
            --lora_r 16 --lora_alpha 32 --lora_dropout 0.1
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
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from PIL import Image
from tqdm import tqdm

# LoRA imports
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel,
        prepare_model_for_kbit_training,
    )
    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False
    print("Warning: peft library not found. Install with: pip install peft")
    print("LoRA training requires peft library.")

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model classes (selected via --model_type at runtime)
from net.openvam import (
    OpenVAM,
    get_dataset_from_id,
    get_dataset_type_from_name,
    get_user_prompt_for_dataset_type,
)
try:
    from net.openvam_qwen_vit import QwenViTDPTWithText
    _QWEN_VIT_AVAILABLE = True
except ImportError:
    QwenViTDPTWithText = None
    _QWEN_VIT_AVAILABLE = False

from utils.losses import loss_KLdiv, loss_CC, loss_similarity, loss_NSS


# =============================================================================
# LoRA Setup Functions
# =============================================================================

def setup_lora_for_qwen_layers(
    model: nn.Module,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    lora_lm_head: bool = False,
    lora_lm_head_r: int = 8,
    lora_lm_head_alpha: int = 16,
    lora_lm_head_dropout: float = 0.0,
) -> nn.Module:
    """
    Apply LoRA to Qwen LLM layers.
    
    Args:
        model: OpenVAM model
        r: LoRA rank for main LLM modules
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: LoRA dropout rate
        target_modules: List of module names to apply LoRA to (default: q_proj, k_proj, v_proj, o_proj, ...)
        lora_lm_head: If True, apply LoRA to lm_head with its own rank/alpha via rank_pattern/alpha_pattern
        lora_lm_head_r: LoRA rank for lm_head only (used when lora_lm_head=True)
        lora_lm_head_alpha: LoRA alpha for lm_head only
        lora_lm_head_dropout: LoRA dropout for lm_head only (not supported with rank_pattern; uses lora_dropout)
    
    Returns:
        Model with LoRA adapters applied
    """
    if not _HAS_PEFT:
        raise ImportError("peft library is required for LoRA training. Install with: pip install peft")
    
    # Default target modules for Qwen architecture
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # When lora_lm_head=True, use rank_pattern/alpha_pattern in a single adapter.
    # Ensure lm_head is included in target_modules so LoRA is applied to it.
    main_target_modules = list(target_modules)
    if lora_lm_head and "lm_head" not in main_target_modules:
        main_target_modules.append("lm_head")
    
    # Get the Qwen language model layers
    if hasattr(model, 'qwen_layers'):
        qwen_layers = model.qwen_layers
    else:
        raise AttributeError("Model does not have qwen_layers attribute. Cannot apply LoRA.")
    
    print(f"\n{'='*70}")
    print("Setting up LoRA for Qwen LLM Layers")
    print(f"{'='*70}")
    print(f"LoRA Config:")
    print(f"  r (rank): {r}")
    print(f"  alpha: {lora_alpha}")
    print(f"  dropout: {lora_dropout}")
    print(f"  target_modules: {main_target_modules}")
    if lora_lm_head:
        print(f"  lm_head: rank_pattern/alpha_pattern (r={lora_lm_head_r}, alpha={lora_lm_head_alpha})")
        if lora_lm_head_dropout != lora_dropout:
            print(f"  note: lm_head dropout override not supported with rank_pattern; using lora_dropout={lora_dropout}")
    print(f"  Number of LLM layers: {len(qwen_layers)}")
    
    # Create LoRA config for main modules
    # IMPORTANT: Use CAUSAL_LM for text generation tasks
    rank_pattern = None
    alpha_pattern = None
    if lora_lm_head:
        # Use exact module name match; PEFT adds end-of-string anchor automatically.
        rank_pattern = {"lm_head": lora_lm_head_r}
        alpha_pattern = {"lm_head": lora_lm_head_alpha}

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Changed from FEATURE_EXTRACTION for proper generation
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=main_target_modules,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        bias="none",
    )
    
    # Apply LoRA to the Qwen layers module
    # We need to wrap the qwen_layers (which is a nn.ModuleList) with LoRA
    # Since peft works on the model level, we'll apply it to the full model
    # but configure it to only target the qwen_layers
    
    # Create a wrapper model that contains only the qwen_layers for LoRA
    # Actually, we need to apply LoRA to the full model but target specific modules
    # The issue is that qwen_layers is a ModuleList, so we need to target the individual layers
    
    # Better approach: Apply LoRA to the full Qwen model (qwen_full_model)
    # But we need to be careful - we only want to train the LLM layers, not the vision encoder
    
    # Since we're using the model's qwen_layers directly, we can't easily use PEFT's
    # automatic module detection. Instead, we'll manually apply LoRA to each layer.
    
    # Actually, the best approach is to use PEFT's get_peft_model on the qwen_full_model
    # but configure it to only target the language_model layers
    
    # Apply LoRA to the full Qwen model's language model
    if not hasattr(model, 'qwen_full_model'):
        raise AttributeError("Model does not have qwen_full_model attribute. Cannot apply LoRA.")
    
    qwen_model = model.qwen_full_model
    
    # Navigate to the language model
    if not (hasattr(qwen_model, 'model') and hasattr(qwen_model.model, 'language_model')):
        raise AttributeError("Cannot find language_model in qwen_full_model structure")
    
    lm_model = qwen_model.model.language_model
    print(f"  Found language_model: {type(lm_model).__name__}")
    
    # Get the inner model if it exists
    if hasattr(lm_model, 'model'):
        inner_model = lm_model.model
        print(f"  Found inner model: {type(inner_model).__name__}")
        num_layers = len(inner_model.layers) if hasattr(inner_model, 'layers') else 0
    else:
        inner_model = lm_model
        num_layers = len(inner_model.layers) if hasattr(inner_model, 'layers') else 0
    
    print(f"  Number of LLM layers: {num_layers}")
    
    # Apply LoRA to the language model
    print(f"\n  Applying LoRA to language model...")
    
    # PEFT's CAUSAL_LM task type expects prepare_inputs_for_generation method
    # Use a pass-through version to ensure inputs_embeds/position_ids are preserved
    if not hasattr(lm_model, 'prepare_inputs_for_generation'):
        def prepare_inputs_for_generation(self, input_ids=None, **kwargs):
            """Pass-through method for PEFT/Generation compatibility."""
            model_inputs = {}
            if input_ids is not None:
                model_inputs["input_ids"] = input_ids
            # Pass through everything generation needs
            for k in ["inputs_embeds", "attention_mask", "position_ids", "past_key_values", "use_cache"]:
                if k in kwargs and kwargs[k] is not None:
                    model_inputs[k] = kwargs[k]
            return model_inputs
        
        import types
        lm_model.prepare_inputs_for_generation = types.MethodType(
            prepare_inputs_for_generation, lm_model
        )
    
    peft_model = get_peft_model(lm_model, lora_config)
    
    # NOTE: We intentionally avoid multiple adapters + set_adapter(list), which is not
    # supported in some PEFT versions for PeftModel and can raise "unhashable type: 'list'".

    # Replace the language model in qwen_full_model
    qwen_model.model.language_model = peft_model
    
    # CRITICAL FIX: Update references to point to the LoRA-wrapped model, NOT the base model
    # With PEFT, we must use peft_model.model.layers (the wrapped path) so that
    # forward() calls go through the LoRA adapters. Using get_base_model() would
    # bypass the LoRA wrapper and training wouldn't update LoRA weights.
    
    # Access layers through the PeftModel wrapper (this ensures LoRA adapters are active)
    if hasattr(peft_model, 'model'):
        # PeftModel wraps the base model in .model
        if hasattr(peft_model.model, 'layers'):
            model.qwen_layers = peft_model.model.layers  # Use wrapped path
            print(f"  ✓ Updated model.qwen_layers to point to LoRA-wrapped layers")
        if hasattr(peft_model.model, 'norm'):
            model.qwen_norm = peft_model.model.norm
        if hasattr(peft_model.model, 'embed_tokens'):
            model.embed_tokens = peft_model.model.embed_tokens
    elif hasattr(peft_model, 'layers'):
        # Fallback: direct access (shouldn't happen with standard PEFT)
        model.qwen_layers = peft_model.layers
        print(f"  ⚠️  Warning: Using direct layers access (unexpected PEFT structure)")
    
    # Also update qwen_backbone to point to the wrapped model for generation
    # This ensures generate_text() uses LoRA weights
    model.qwen_backbone = peft_model
    print(f"  ✓ Updated model.qwen_backbone to point to LoRA-wrapped model")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    
    print(f"\n✓ LoRA applied successfully")
    print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total parameters: {total_params:,}")
    
    # --------------------------------------------------------------------------
    # CRITICAL CHECK: Verify LoRA path consistency for training vs generation
    # --------------------------------------------------------------------------
    print("\n" + "-"*50)
    print("LoRA Integration Verification (Training vs Generation)")
    print("-"*50)
    
    # Check the reference used by the main model (for training forward)
    has_lora_training = False
    training_layer_id = None
    try:
        if hasattr(model, 'qwen_layers') and len(model.qwen_layers) > 0:
            training_layer = model.qwen_layers[0].self_attn.q_proj
            has_lora_training = hasattr(training_layer, "lora_A")
            training_layer_id = id(training_layer)  # Get object identity
            print(f"  LoRA in model.qwen_layers (Training path):   {has_lora_training}")
            print(f"    Layer object ID: {training_layer_id}")
    except Exception as e:
        print(f"  ⚠️ Error checking training path: {e}")
    
    # Check the path used by .generate() - should be the SAME object
    gen_lm = model.qwen_full_model.model.language_model
    has_lora_gen = False
    gen_layer_id = None
    sample_layer = None
    
    try:
        # CRITICAL: Check if qwen_backbone points to the same model as language_model
        if hasattr(model, 'qwen_backbone'):
            backbone_id = id(model.qwen_backbone)
            gen_lm_id = id(gen_lm)
            print(f"  qwen_backbone ID: {backbone_id}")
            print(f"  language_model ID: {gen_lm_id}")
            if backbone_id == gen_lm_id:
                print(f"  ✓ qwen_backbone and language_model are the SAME object (correct!)")
            else:
                print(f"  ❌ WARNING: qwen_backbone and language_model are DIFFERENT objects!")
                print(f"     This means generation may not use LoRA weights!")
        
        # Find a sample layer through the generation path
        if hasattr(gen_lm, "model") and hasattr(gen_lm.model, "layers"):
            # This is the correct path for PeftModel
            if len(gen_lm.model.layers) > 0:
                sample_layer = gen_lm.model.layers[0].self_attn.q_proj
                gen_layer_id = id(sample_layer)
        elif hasattr(gen_lm, "layers"):
            if len(gen_lm.layers) > 0:
                sample_layer = gen_lm.layers[0].self_attn.q_proj
                gen_layer_id = id(sample_layer)
    except (AttributeError, IndexError) as e:
        print(f"  ⚠️ Error accessing generation path: {e}")
        sample_layer = None
    
    if sample_layer:
        has_lora_gen = hasattr(sample_layer, "lora_A")
        print(f"  LoRA in qwen_full_model (Generation path): {has_lora_gen}")
        print(f"    Layer object ID: {gen_layer_id}")
        
        # CRITICAL CHECK: Are they the SAME object?
        if training_layer_id is not None and gen_layer_id is not None:
            if training_layer_id == gen_layer_id:
                print(f"  ✅ Training and generation use the SAME layer object (perfect!)")
            else:
                print(f"  ❌ CRITICAL: Training and generation use DIFFERENT layer objects!")
                print(f"     Training updates one set of LoRA weights, generation uses another!")
    else:
        print("  ⚠️ Could not find layers in qwen_full_model to verify LoRA!")
        has_lora_gen = False
        
    if not (has_lora_training and has_lora_gen):
        print("\n  ❌ WARNING: LoRA might be bypassed in one of the paths!")
        print("     Training may not affect generation output!")
    elif training_layer_id is not None and gen_layer_id is not None and training_layer_id != gen_layer_id:
        print("\n  ❌ CRITICAL: Training and generation use different layer objects!")
        print("     LoRA weights updated during training won't be used during generation!")
    else:
        print("\n  ✅ LoRA is correctly active in both training and generation paths.")
        print("     Both paths use the same LoRA-wrapped layers.")
    print("-"*50 + "\n")
    
    return model


def load_checkpoint_weights(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
    verbose: bool = True,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load weights from a checkpoint into the model.
    Handles both regular checkpoints and LoRA checkpoints.
    """
    if verbose:
        print("="*70)
        print("Loading Checkpoint Weights")
        print("="*70)
    
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
    
    # Check if this is a LoRA checkpoint
    is_lora_checkpoint = any("lora" in k.lower() for k in state_dict.keys())
    
    if is_lora_checkpoint:
        if verbose:
            print("Detected LoRA checkpoint - loading LoRA weights...")
        # For LoRA, we need to load the base model first, then the LoRA adapters
        # Extract base model weights (non-LoRA)
        base_state_dict = {k: v for k, v in state_dict.items() if "lora" not in k.lower()}
        lora_state_dict = {k: v for k, v in state_dict.items() if "lora" in k.lower()}
        
        # Load base weights
        if base_state_dict:
            model.load_state_dict(base_state_dict, strict=False)
        
        # Load LoRA weights if we have a PEFT model
        if lora_state_dict and hasattr(model, 'qwen_full_model'):
            try:
                lm = model.qwen_full_model.model.language_model
                if isinstance(lm, PeftModel):
                    # Load LoRA adapters
                    lm.load_state_dict(lora_state_dict, strict=False)
                    if verbose:
                        print(f"Loaded {len(lora_state_dict)} LoRA adapter weights")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not load LoRA weights: {e}")
    else:
        # Regular checkpoint loading with shape mismatch handling
        # Filter out incompatible weights (shape mismatches)
        model_state_dict = model.state_dict()
        compatible_state_dict = {}
        skipped_keys = []
        shape_mismatches = []
        
        for key, value in state_dict.items():
            if key in model_state_dict:
                try:
                    # Try to get shape - may fail for uninitialized LazyConv2d
                    model_shape = model_state_dict[key].shape
                    checkpoint_shape = value.shape
                    
                    if model_shape == checkpoint_shape:
                        compatible_state_dict[key] = value
                    else:
                        # Skip projection layers with shape mismatches (architecture changed)
                        if "proj_to_l" in key:
                            shape_mismatches.append(f"{key}: checkpoint {checkpoint_shape} vs model {model_shape}")
                        else:
                            skipped_keys.append(f"{key}: shape mismatch {checkpoint_shape} vs {model_shape}")
                except RuntimeError as e:
                    # Handle uninitialized LazyConv2d parameters
                    if "uninitialized" in str(e).lower():
                        # For uninitialized LazyConv2d, we can't check shape, so skip it
                        # It will be initialized during first forward pass with random weights
                        if "proj_to_l" in key:
                            shape_mismatches.append(f"{key}: uninitialized (will be randomly initialized)")
                        else:
                            skipped_keys.append(f"{key}: uninitialized parameter")
                    else:
                        raise
            else:
                skipped_keys.append(f"{key}: not in model")
        
        # Load compatible weights
        model.load_state_dict(compatible_state_dict, strict=False)
        
        if verbose:
            print(f"Loaded {len(compatible_state_dict)} compatible weights")
            if shape_mismatches:
                print(f"\n⚠️  Skipped {len(shape_mismatches)} projection layers with shape mismatches:")
                for mismatch in shape_mismatches[:5]:
                    print(f"    - {mismatch}")
                if len(shape_mismatches) > 5:
                    print(f"    ... and {len(shape_mismatches) - 5} more")
                print("    (These will be randomly initialized - this is expected if architecture changed)")
            if skipped_keys and verbose:
                print(f"\nSkipped {len(skipped_keys)} incompatible keys:")
                for key in skipped_keys[:10]:
                    print(f"    - {key}")
                if len(skipped_keys) > 10:
                    print(f"    ... and {len(skipped_keys) - 10} more")
    
    return [], [], []


# =============================================================================
# Dataset (same as original)
# =============================================================================

class SalienceTextDataset(Dataset):
    """Dataset for saliency prediction with text prompts and target text."""
    
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
        
        # Detect if saliency_dir has train/val subdirectories (flat: train/, val/ or nested: train/saliency/, val/saliency/)
        self.has_subdirs = False
        train_subdir = self.saliency_dir / "train"
        val_subdir = self.saliency_dir / "val"
        train_saliency_nested = self.saliency_dir / "train" / "saliency"
        val_saliency_nested = self.saliency_dir / "val" / "saliency"
        if train_subdir.exists() or val_subdir.exists():
            self.has_subdirs = True
            print(f"  Detected train/val subdirectories in saliency_dir")
        # Prefer nested layout train/saliency and val/saliency when present
        if train_saliency_nested.exists() or val_saliency_nested.exists():
            train_subdir = train_saliency_nested if train_saliency_nested.exists() else train_subdir
            val_subdir = val_saliency_nested if val_saliency_nested.exists() else val_subdir
            if self.has_subdirs:
                print(f"  Using nested train/saliency and val/saliency")
        
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
                print(f"  Using saliency subdirectory: {self.saliency_subdir}")
        
        # Same for fixation_dir - check if it has subdirectories (flat or nested train/fixations, val/fixations)
        self.fixation_subdir = None
        if self.fixation_dir:
            train_fix_subdir = self.fixation_dir / "train"
            val_fix_subdir = self.fixation_dir / "val"
            train_fix_nested = self.fixation_dir / "train" / "fixations"
            val_fix_nested = self.fixation_dir / "val" / "fixations"
            train_edit_subdir = self.fixation_dir / "train_edit"
            val_edit_subdir = self.fixation_dir / "val_edit"
            # Prefer nested train/fixations and val/fixations when present
            if train_fix_nested.exists():
                train_fix_subdir = train_fix_nested
            if val_fix_nested.exists():
                val_fix_subdir = val_fix_nested
            train_exists = train_fix_subdir.exists() or train_edit_subdir.exists()
            val_exists = val_fix_subdir.exists() or val_edit_subdir.exists()
            fixation_has_subdirs = train_exists or val_exists
            
            if fixation_has_subdirs:
                print(f"  Detected train/val subdirectories in fixation_dir")
                
                jsonl_name = Path(jsonl_path).stem.lower()
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
                    print(f"  Using fixation subdirectory: {self.fixation_subdir}")
            else:
                self.fixation_subdir = self.fixation_dir
        
        # Load JSONL
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        
        # Quick check: count how many saliency files exist
        search_dir = self.saliency_subdir if self.saliency_subdir else self.saliency_dir
        found_count = 0
        for sample in self.samples[:min(100, len(self.samples))]:
            image_path = sample.get("image", sample.get("images", [""])[0])
            if isinstance(image_path, list):
                image_path = image_path[0]
            image_name = Path(image_path).stem
            for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                if (search_dir / f"{image_name}{ext}").exists():
                    found_count += 1
                    break
        
        # Quick check: count how many fixation files exist
        fix_found_count = 0
        fix_with_data_count = 0
        if self.fixation_dir:
            fix_search_dir = self.fixation_subdir if self.fixation_subdir else self.fixation_dir
            for sample in self.samples[:min(100, len(self.samples))]:
                image_path = sample.get("image", sample.get("images", [""])[0])
                if isinstance(image_path, list):
                    image_path = image_path[0]
                image_name = Path(image_path).stem
                for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                    if (fix_search_dir / f"{image_name}{ext}").exists():
                        fix_found_count += 1
                        # Check if it has data
                        try:
                            fix = Image.open(fix_search_dir / f"{image_name}{ext}").convert("L")
                            fix_array = np.array(fix)
                            if fix_array.sum() > 0:
                                fix_with_data_count += 1
                        except:
                            pass
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
                if fix_with_data_count == 0 and fix_found_count > 0:
                    print(f"  ⚠️ WARNING: Fixation files found but all are empty! NSS will be 0.0")
                elif fix_with_data_count == 0:
                    print(f"  ⚠️ WARNING: No fixations with data found! NSS will be 0.0")
    
    def __len__(self):
        return len(self.samples)
    
    def _load_image(self, path: str) -> torch.Tensor:
        """Load and preprocess image."""
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img = torch.from_numpy(np.array(img)).float() / 255.0
        img = img.permute(2, 0, 1)  # [3, H, W]
        return img
    
    def _load_saliency(self, image_name: str) -> torch.Tensor:
        """Load saliency map."""
        search_dir = self.saliency_subdir if self.saliency_subdir else self.saliency_dir
        
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
            candidate = search_dir / f"{image_name}{ext}"
            if candidate.exists():
                smap = Image.open(candidate).convert("L")
                smap = smap.resize((self.image_size, self.image_size), Image.BILINEAR)
                smap = torch.from_numpy(np.array(smap)).float() / 255.0
                smap = smap.unsqueeze(0)
                smap = smap / (smap.sum() + 1e-8)
                return smap
        
        self.missing_saliency_count += 1
        smap = torch.zeros(1, self.image_size, self.image_size)
        smap = smap / (smap.sum() + 1e-8)
        return smap
    
    def _load_fixation(self, image_name: str) -> Optional[torch.Tensor]:
        """Load fixation map."""
        if self.fixation_dir is None:
            return None
        
        search_dir = self.fixation_subdir if self.fixation_subdir else self.fixation_dir
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
            candidate = search_dir / f"{image_name}{ext}"
            if candidate.exists():
                fix = Image.open(candidate).convert("L")
                fix = fix.resize((self.image_size, self.image_size), Image.NEAREST)
                fix = torch.from_numpy(np.array(fix)).float() / 255.0
                fix = fix.unsqueeze(0)
                fix = (fix > 0.5).float()
                return fix
        return None
    
    def _extract_dataset_type(self, sample: dict) -> str:
        """Infer dataset type from sample id or image path (for merged UI/e-commerce/natural_scene)."""
        sample_id = sample.get("id", "")
        if sample_id:
            ds_name = get_dataset_from_id(sample_id)
            return get_dataset_type_from_name(ds_name)
        # Fallback: try stem of image path (e.g. datasets_UI_256_xxx -> datasets_UI_256)
        image_path = sample.get("image", sample.get("images", [""])[0])
        if isinstance(image_path, list):
            image_path = image_path[0] or ""
        if image_path:
            stem = Path(image_path).stem
            for name in ("datasets_UI_256", "SalEC", "CAT2000_256", "MIT1003_256", "OSIE_256", "salicon_256"):
                if stem.startswith(name + "_") or stem == name:
                    return get_dataset_type_from_name(name)
        return "natural_scene"

    def _extract_text_prompt(self, sample: dict) -> str:
        """Return dataset-type-specific user prompt (natural_scene / webpage / e_commerce) for mixed datasets."""
        dataset_type = self._extract_dataset_type(sample)
        return get_user_prompt_for_dataset_type(dataset_type)
    
    def _extract_target_text(self, sample: dict) -> Optional[str]:
        """Extract target text (expected output) from JSONL sample."""
        if "conversations" in sample:
            for conv in sample["conversations"]:
                if conv.get("from") == "gpt" or conv.get("role") == "assistant":
                    content = conv.get("value", "") or conv.get("content", "")
                    return content.strip()
        # Fallback to salience_explanation if available
        if "salience_explanation" in sample:
            return sample["salience_explanation"].strip()
        return None
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample.get("image", sample.get("images", [""])[0])
        if isinstance(image_path, list):
            image_path = image_path[0]
        
        img = self._load_image(image_path)
        image_path_obj = Path(image_path)
        image_name = image_path_obj.stem
        
        smap = self._load_saliency(image_name)
        fmap = self._load_fixation(image_name)
        if fmap is None:
            fmap = torch.zeros_like(smap)
        
        text_prompt = self._extract_text_prompt(sample)
        target_text = self._extract_target_text(sample)
        dataset_type = self._extract_dataset_type(sample)
        
        # if self.augment and random.random() > 0.5:
        #     img = torch.flip(img, [-1])
        #     smap = torch.flip(smap, [-1])
        #     fmap = torch.flip(fmap, [-1])
        
        return {
            "image": img,
            "saliency": smap,
            "fixation": fmap,
            "text": text_prompt,
            "target_text": target_text,
            "dataset_type": dataset_type,
            "image_path": image_path,
            "id": sample.get("id", ""),
        }


def collate_fn(batch):
    """Custom collate function to handle text prompts and dataset_type."""
    images = torch.stack([b["image"] for b in batch])
    saliency = torch.stack([b["saliency"] for b in batch])
    fixation = torch.stack([b["fixation"] for b in batch])
    texts = [b["text"] for b in batch]
    target_texts = [b["target_text"] for b in batch]
    dataset_types = [b.get("dataset_type", "natural_scene") for b in batch]
    paths = [b["image_path"] for b in batch]
    ids = [b["id"] for b in batch]
    
    return {
        "image": images,
        "saliency": saliency,
        "fixation": fixation,
        "text": texts,
        "target_text": target_texts,
        "dataset_type": dataset_types,
        "image_path": paths,
        "id": ids,
    }


def _has_valid_target_text(t) -> bool:
    """Return True iff target_text is present and non-empty (used for LoRA text loss)."""
    return t is not None and isinstance(t, str) and len(t.strip()) > 0


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
        
        # Track how many samples had valid fixations for NSS
        nss_valid_count = len(nss_losses)
        nss_total_count = B if (self.nss_weight > 0 and loss_NSS is not None and fmap_sq is not None) else 0
        
        losses = {
            "total": total_loss.item(),
            "mse": mse_val.item(),
            "kld": kld_val.item(),
            "cc": cc_val.item(),
            "sim": sim_val.item(),
            "nss": nss_val.item(),
            "nss_valid": nss_valid_count,
            "nss_total": nss_total_count,
        }
        
        return total_loss, losses


class LanguageModelingLoss(nn.Module):
    """Loss for text generation (next-token prediction)."""
    
    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)
    
    def forward(
        self,
        logits: torch.Tensor,  # [B, seq_len, vocab_size]
        labels: torch.Tensor,  # [B, seq_len] with -100 for tokens to ignore
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute language modeling loss.
        NOTE: This version expects ALREADY SHIFTED logits and labels
        to match standard causal LM training loops.
        """
        # Flatten for CrossEntropyLoss
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        
        # Compute loss
        loss = self.ce_loss(flat_logits, flat_labels)
        
        # Compute perplexity (only on non-ignored tokens)
        with torch.no_grad():
            mask = (flat_labels != self.ignore_index)
            if mask.sum() > 0:
                valid_logits = flat_logits[mask]
                valid_labels = flat_labels[mask]
                valid_loss = self.ce_loss(valid_logits, valid_labels)
                perplexity = torch.exp(valid_loss.clamp(max=10.0)).item()
            else:
                perplexity = 1.0
        
        return loss, {"lm_loss": loss.item(), "perplexity": perplexity}


# =============================================================================
# Chat Template Verification
# =============================================================================

def verify_chat_template(model: nn.Module, sample_text: str, sample_target: str = None, device: torch.device = None):
    """
    Verify that chat template is correctly applied in the model.
    
    This checks:
    1. Chat template method exists and works
    2. Formatted text contains expected special tokens
    3. Tokenization produces correct format
    4. Training and inference formats match
    
    Args:
        model: The model to verify
        sample_text: Sample prompt text
        sample_target: Optional sample target text
    """
    print("\n" + "="*70)
    print("Verifying Chat Template Application")
    print("="*70)
    
    # Get base model (handles DataParallel)
    base_model = _get_base_model(model)
    
    try:
        # Check if _apply_chat_template method exists
        if not hasattr(base_model, '_apply_chat_template'):
            print("  ❌ ERROR: Model does not have _apply_chat_template method!")
            print("     This means chat template is NOT being applied.")
            print("     Training and inference will have format mismatches!")
            return False
        
        # Test chat template application
        print(f"\n  Testing chat template:")
        print(f"    Dataset-type prompt (per sample): '{sample_text[:50]}...'")
        print(f"    Note: Prompt is dataset-type-specific (natural_scene / webpage / e_commerce) from sample id.")
        
        # Test inference format (no target)
        formatted_inference = base_model._apply_chat_template(sample_text)
        print(f"\n  ✓ Inference format (user only):")
        print(f"    Length: {len(formatted_inference)} chars")
        print(f"    Preview: {formatted_inference[:100]}...")
        
        # Check for expected special tokens
        has_im_start = "<|im_start|>" in formatted_inference or "im_start" in formatted_inference.lower()
        has_im_end = "<|im_end|>" in formatted_inference or "im_end" in formatted_inference.lower()
        has_user = "user" in formatted_inference.lower()
        has_assistant = "assistant" in formatted_inference.lower()
        
        print(f"\n  Special token check:")
        print(f"    Contains user role: {has_user} {'✓' if has_user else '❌'}")
        print(f"    Contains assistant role: {has_assistant} {'✓' if has_assistant else '❌'}")
        print(f"    Contains im_start/im_end tokens: {has_im_start or has_im_end} {'✓' if (has_im_start or has_im_end) else '⚠️'}")
        
        # Test training format (with target)
        if sample_target:
            formatted_training = base_model._apply_chat_template(sample_text, assistant_message=sample_target)
            print(f"\n  ✓ Training format (user + assistant):")
            print(f"    Length: {len(formatted_training)} chars")
            print(f"    Preview: {formatted_training[:150]}...")
            
            # Verify training format has both user and assistant
            has_user_training = "user" in formatted_training.lower()
            has_assistant_training = "assistant" in formatted_training.lower()
            print(f"\n  Training format check:")
            print(f"    Contains user role: {has_user_training} {'✓' if has_user_training else '❌'}")
            print(f"    Contains assistant role: {has_assistant_training} {'✓' if has_assistant_training else '❌'}")
        
        # Verify tokenization produces expected format
        print(f"\n  Verifying tokenization...")
        tokens_inference = base_model.tokenizer(
            formatted_inference,
            return_tensors="pt",
            add_special_tokens=False,  # Template already has special tokens
        )
        print(f"    Inference tokens: {tokens_inference['input_ids'].shape[1]} tokens")
        
        # Compare with raw tokenization (should be different if template is applied)
        tokens_raw = base_model.tokenizer(
            sample_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        raw_length = tokens_raw['input_ids'].shape[1]
        template_length = tokens_inference['input_ids'].shape[1]
        
        if template_length > raw_length:
            print(f"    ✓ Template adds {template_length - raw_length} tokens (expected)")
        elif template_length == raw_length:
            print(f"    ⚠️  WARNING: Template length equals raw length!")
            print(f"       This suggests template might not be applied correctly.")
        else:
            print(f"    ⚠️  WARNING: Template length is shorter than raw (unexpected)")
        
        # Verify encode_text uses template
        print(f"\n  Verifying encode_text() uses template...")
        verify_device = device if device is not None else torch.device("cpu")
        text_embeds, text_mask, text_len = base_model.encode_text(sample_text, batch_size=1, device=verify_device)
        print(f"    encode_text() output length: {text_len} tokens")
        
        if text_len == template_length:
            print(f"    ✓ encode_text() uses chat template correctly")
        else:
            print(f"    ⚠️  WARNING: encode_text() length ({text_len}) != template length ({template_length})")
            print(f"       This suggests encode_text() might not be using the template!")
        
        # Final check: verify get_text_generation_logits uses template
        if sample_target:
            print(f"\n  Verifying get_text_generation_logits() uses template...")
            try:
                # This will tokenize internally, we just want to check it doesn't error
                # and that the format is consistent
                verify_device = device if device is not None else torch.device("cpu")
                model_dtype = getattr(base_model, "dtype", torch.float32)
                dummy_image = torch.randn(1, 3, 256, 256, device=verify_device, dtype=model_dtype)
                logits, labels, prompt_len = base_model.get_text_generation_logits(
                    dummy_image,
                    sample_text,
                    target_text=sample_target
                )
                print(f"    ✓ get_text_generation_logits() works with template")
                print(f"    Prompt length: {prompt_len}, Logits shape: {logits.shape}")
            except Exception as e:
                print(f"    ❌ ERROR: get_text_generation_logits() failed: {e}")
                return False
        
        print(f"\n  {'='*70}")
        print(f"  ✓ Chat template verification PASSED")
        print(f"  {'='*70}")
        return True
        
    except Exception as e:
        print(f"\n  ❌ ERROR during chat template verification: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Training Functions
# =============================================================================

def print_lora_stats(model: nn.Module):
    """Print statistics for LoRA parameters to see if they are learning."""
    # Get base model to handle DataParallel
    model_to_check = _get_base_model(model)
    
    # Check if model has qwen_layers
    if not hasattr(model_to_check, 'qwen_layers'):
        return

    print("\n" + "-"*30)
    print("LoRA Parameter Stats (Diagnostics)")
    print("-"*30)
    
    found_lora = False
    num_layers = len(model_to_check.qwen_layers)
    for name, param in model_to_check.named_parameters():
        if 'lora' in name and param.requires_grad:
            # Only print for a few layers (start and end) to avoid too much output
            if 'layers.0.' in name or f'layers.{num_layers-1}.' in name:
                found_lora = True
                print(f"  {name:50s} | mean={param.data.mean():.8f} | std={param.data.std():.8f}")
    
    if not found_lora:
        print("  No trainable LoRA parameters found!")
    print("-"*30 + "\n")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    saliency_loss_fn: nn.Module,
    lm_loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    accumulation_steps: int = 1,
    saliency_weight: float = 1.0,
    text_weight: float = 1.0,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_losses = {}
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    optimizer.zero_grad()
    
    # Get base model for attribute access (handles DataParallel and DDP)
    base_model = _get_base_model(model)
    skipped_target_text_warned_this_epoch = False  # warn once per epoch when target_text missing/empty

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device, dtype=base_model.dtype)
        saliency = batch["saliency"].to(device, dtype=base_model.dtype)
        fixation = batch["fixation"].to(device, dtype=base_model.dtype)
        texts = batch["text"]
        target_texts = batch["target_text"]
        dataset_types = batch.get("dataset_type", ["natural_scene"] * len(texts))
        B = len(texts)
        
        # Log complete prompt template once at start of training (first batch, first epoch)
        if epoch == 1 and batch_idx == 0 and hasattr(base_model, "_apply_chat_template"):
            sample_prompt = texts[0] if texts else "(none)"
            sample_target_first = target_texts[0] if target_texts else None
            sample_dtype = dataset_types[0] if dataset_types else "natural_scene"
            try:
                formatted_template = base_model._apply_chat_template(sample_prompt, dataset_type=sample_dtype)
            except TypeError:
                formatted_template = base_model._apply_chat_template(sample_prompt)
            print("\n" + "=" * 70)
            print("Complete prompt template that goes to Qwen (first batch, epoch 1)")
            print("=" * 70)
            print("Raw prompt (before template):")
            print(sample_prompt)
            print("\n--- Full template string sent to tokenizer (inference / prompt-only) ---")
            print(formatted_template)
            print("--- End template ---")
            if sample_target_first and _has_valid_target_text(sample_target_first):
                try:
                    formatted_with_target = base_model._apply_chat_template(sample_prompt, assistant_message=sample_target_first, dataset_type=sample_dtype)
                except TypeError:
                    formatted_with_target = base_model._apply_chat_template(sample_prompt, assistant_message=sample_target_first)
                print("\n--- Full template with target (training / teacher-forcing) ---")
                print(formatted_with_target)
                print("--- End template with target ---")
            print("=" * 70 + "\n")
        
        batch_saliency_loss = 0.0
        batch_text_loss = 0.0
        batch_mse = 0.0
        batch_kld = 0.0
        batch_cc = 0.0
        batch_sim = 0.0
        batch_nss = 0.0
        batch_nss_valid = 0
        batch_nss_total = 0
        batch_count = 0

        for i in range(B):
            text_prompt = texts[i]
            target_text = target_texts[i]
            dataset_type_i = dataset_types[i] if i < len(dataset_types) else "natural_scene"
            image_i = images[i:i+1]
            saliency_i = saliency[i:i+1]
            fixation_i = fixation[i:i+1]

            if not _has_valid_target_text(target_text) and not skipped_target_text_warned_this_epoch:
                skipped_target_text_warned_this_epoch = True
                print(f"  [LoRA] Skipping text loss for sample (target_text missing or empty). "
                      f"Confirm dataset has 'conversations'/'assistant' or 'salience_explanation'.")
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Saliency prediction (dataset_type selects natural_scene / webpage / e_commerce prompts)
                try:
                    pred_saliency = base_model(image_i, text_prompt=text_prompt, dataset_type=dataset_type_i)
                except TypeError:
                    pred_saliency = base_model(image_i, text_prompt=text_prompt)
                saliency_loss, saliency_loss_dict = saliency_loss_fn(pred_saliency, saliency_i, fixation_i)
                
                # Text generation (proper teacher forcing); only when target_text is present and non-empty
                text_loss = torch.tensor(0.0, device=device)
                if _has_valid_target_text(target_text):
                    try:
                        # Get logits for prompt+target sequence with proper masking
                        logits, labels, prompt_len = base_model.get_text_generation_logits(
                            image_i, text_prompt, target_text=target_text, dataset_type=dataset_type_i
                        )
                        
                        # Debug: Print masking info for first batch of first epoch
                        if batch_idx == 0 and i == 0 and epoch == 1:
                            num_masked = (labels == -100).sum().item()
                            num_total = labels.numel()
                            num_padding = (labels == base_model.tokenizer.pad_token_id).sum().item()
                            num_valid = ((labels != -100) & (labels != base_model.tokenizer.pad_token_id)).sum().item()
                            print(f"\n[DEBUG] Text generation masking check (teacher forcing):")
                            print(f"  Prompt length: {prompt_len}")
                            print(f"  Total labels: {num_total}")
                            print(f"  Masked (-100): {num_masked} ({100*num_masked/num_total:.1f}%)")
                            print(f"  Padding tokens (should be 0): {num_padding}")
                            print(f"  Valid target tokens: {num_valid}")
                        
                        # labels already has prompt tokens masked with -100 (if new method)
                        # This is proper next-token prediction: predict target given prompt
                        
                        # SHIFT: logits[i] predicts labels[i+1]
                        # logits: [B, L, V], labels: [B, L]
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = labels[:, 1:].contiguous()
                        
                        # Sanity check: count non-masked labels once per epoch
                        if batch_idx == 0 and i == 0:
                            num_active_labels = (shift_labels != -100).sum().item()
                            print(f"\n[DEBUG] Epoch {epoch} active labels check:")
                            print(f"  Total shifted tokens: {shift_labels.numel()}")
                            print(f"  Active labels (not -100): {num_active_labels}")
                            if num_active_labels == 0:
                                print("  ⚠️ WARNING: 0 active labels! Model will learn nothing for text.")
                        
                        text_loss, _ = lm_loss_fn(shift_logits, shift_labels)
                    except Exception as e:
                        print(f"Warning: Text generation loss failed: {e}")
                        import traceback
                        traceback.print_exc()
                        text_loss = torch.tensor(0.0, device=device, requires_grad=True)
                
                total_loss = saliency_weight * saliency_loss + text_weight * text_loss
                total_loss = total_loss / (B * accumulation_steps)  # mean per sample, then scale for grad accum
                total_loss.backward()
                
                batch_saliency_loss += saliency_loss.item()
                batch_text_loss += text_loss.item()
                batch_mse += saliency_loss_dict.get("mse", 0)
                batch_kld += saliency_loss_dict.get("kld", 0)
                batch_cc += saliency_loss_dict.get("cc", 0)
                batch_sim += saliency_loss_dict.get("sim", 0)
                batch_nss += saliency_loss_dict.get("nss", 0)
                batch_nss_valid += saliency_loss_dict.get("nss_valid", 0)
                batch_nss_total += saliency_loss_dict.get("nss_total", 0)
                batch_count += 1
        
        if batch_count > 0:
            batch_saliency_loss = batch_saliency_loss / batch_count
            batch_text_loss = batch_text_loss / batch_count
            batch_mse = batch_mse / batch_count
            batch_kld = batch_kld / batch_count
            batch_cc = batch_cc / batch_count
            batch_sim = batch_sim / batch_count
            batch_nss = batch_nss / batch_count
        
        if (batch_idx + 1) % accumulation_steps == 0:
            # Get base model for gradient clipping (handles DataParallel and DDP)
            base_model_for_grad = _get_base_model(model)
            torch.nn.utils.clip_grad_norm_(base_model_for_grad.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
        
        total_losses["saliency"] = total_losses.get("saliency", 0) + batch_saliency_loss
        total_losses["text_lm"] = total_losses.get("text_lm", 0) + batch_text_loss
        total_losses["total"] = total_losses.get("total", 0) + (saliency_weight * batch_saliency_loss + text_weight * batch_text_loss)
        total_losses["mse"] = total_losses.get("mse", 0) + batch_mse
        total_losses["kld"] = total_losses.get("kld", 0) + batch_kld
        total_losses["cc"] = total_losses.get("cc", 0) + batch_cc
        total_losses["sim"] = total_losses.get("sim", 0) + batch_sim
        total_losses["nss"] = total_losses.get("nss", 0) + batch_nss
        total_losses["nss_valid"] = total_losses.get("nss_valid", 0) + batch_nss_valid
        total_losses["nss_total"] = total_losses.get("nss_total", 0) + batch_nss_total
        num_batches += 1
        
        pbar.set_postfix({
            "sal": f"{batch_saliency_loss:.4f}",
            "text": f"{batch_text_loss:.4f}",
            "CC": f"{batch_cc:.3f}",
            "NSS": f"{batch_nss:.3f}",
        })
    
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return avg_losses


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    saliency_loss_fn: nn.Module,
    lm_loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    saliency_weight: float = 1.0,
    text_weight: float = 1.0,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    
    total_losses = {}
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    base_model = _get_base_model(model)
    
    for batch in pbar:
        images = batch["image"].to(device, dtype=base_model.dtype)
        saliency = batch["saliency"].to(device, dtype=base_model.dtype)
        fixation = batch["fixation"].to(device, dtype=base_model.dtype)
        texts = batch["text"]
        target_texts = batch["target_text"]
        dataset_types = batch.get("dataset_type", ["natural_scene"] * len(texts))
        
        batch_saliency_loss = 0.0
        batch_text_loss = 0.0
        batch_mse = 0.0
        batch_kld = 0.0
        batch_cc = 0.0
        batch_sim = 0.0
        batch_nss = 0.0
        batch_nss_valid = 0
        batch_nss_total = 0
        batch_count = 0
        
        for i in range(len(texts)):
            text_prompt = texts[i]
            target_text = target_texts[i]
            dataset_type_i = dataset_types[i] if i < len(dataset_types) else "natural_scene"
            image_i = images[i:i+1]
            saliency_i = saliency[i:i+1]
            fixation_i = fixation[i:i+1]
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                try:
                    pred_saliency = base_model(image_i, text_prompt=text_prompt, dataset_type=dataset_type_i)
                except TypeError:
                    pred_saliency = base_model(image_i, text_prompt=text_prompt)
                saliency_loss, saliency_loss_dict = saliency_loss_fn(pred_saliency, saliency_i, fixation_i)
                
                text_loss = torch.tensor(0.0, device=device)
                if _has_valid_target_text(target_text):
                    try:
                        # Check if method accepts target_text parameter
                        import inspect
                        sig = inspect.signature(base_model.get_text_generation_logits)
                        has_target_param = 'target_text' in sig.parameters
                        
                        if has_target_param:
                            # Get logits for prompt+target sequence with proper masking
                            logits, labels, prompt_len = base_model.get_text_generation_logits(
                                image_i, text_prompt, target_text=target_text, dataset_type=dataset_type_i
                            )
                        else:
                            # Fallback: old method signature
                            logits = base_model.get_text_generation_logits(image_i, text_prompt)
                            target_tokens = base_model.tokenizer(
                                target_text,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=True,
                                max_length=base_model.max_text_length,
                            )
                            labels = target_tokens["input_ids"].to(device)
                            target_mask = target_tokens["attention_mask"].to(device)
                            
                            if labels.shape[1] > logits.shape[1]:
                                labels = labels[:, :logits.shape[1]]
                                target_mask = target_mask[:, :logits.shape[1]]
                            elif labels.shape[1] < logits.shape[1]:
                                padding = torch.full(
                                    (1, logits.shape[1] - labels.shape[1]),
                                    base_model.tokenizer.pad_token_id,
                                    device=device
                                )
                                labels = torch.cat([labels, padding], dim=1)
                                mask_padding = torch.zeros(
                                    (1, logits.shape[1] - target_mask.shape[1]),
                                    device=device,
                                    dtype=target_mask.dtype
                                )
                                target_mask = torch.cat([target_mask, mask_padding], dim=1)
                            
                            # Mask padding tokens (important: prevents learning to predict padding)
                            labels[target_mask == 0] = -100
                        
                        # SHIFT for validation too
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = labels[:, 1:].contiguous()
                        
                        # Sanity check: count non-masked labels once per validation run
                        if num_batches == 0 and i == 0:
                            num_active_labels = (shift_labels != -100).sum().item()
                            print(f"\n[DEBUG] Validation active labels check:")
                            print(f"  Total shifted tokens: {shift_labels.numel()}")
                            print(f"  Active labels (not -100): {num_active_labels}")
                        
                        # labels already has prompt tokens masked with -100 (if new method)
                        text_loss, _ = lm_loss_fn(shift_logits, shift_labels)
                    except Exception as e:
                        # Print error during validation for debugging
                        if epoch <= 3:  # Only print in first few epochs to avoid spam
                            print(f"Warning: Text generation loss failed: {e}")
                            import traceback
                            traceback.print_exc()
                
                batch_saliency_loss += saliency_loss.item()
                batch_text_loss += text_loss.item()
                batch_mse += saliency_loss_dict.get("mse", 0)
                batch_kld += saliency_loss_dict.get("kld", 0)
                batch_cc += saliency_loss_dict.get("cc", 0)
                batch_sim += saliency_loss_dict.get("sim", 0)
                batch_nss += saliency_loss_dict.get("nss", 0)
                batch_nss_valid += saliency_loss_dict.get("nss_valid", 0)
                batch_nss_total += saliency_loss_dict.get("nss_total", 0)
                batch_count += 1
        
        if batch_count > 0:
            batch_saliency_loss = batch_saliency_loss / batch_count
            batch_text_loss = batch_text_loss / batch_count
            batch_mse = batch_mse / batch_count
            batch_kld = batch_kld / batch_count
            batch_cc = batch_cc / batch_count
            batch_sim = batch_sim / batch_count
            batch_nss = batch_nss / batch_count
        
        total_losses["saliency"] = total_losses.get("saliency", 0) + batch_saliency_loss
        total_losses["text_lm"] = total_losses.get("text_lm", 0) + batch_text_loss
        total_losses["total"] = total_losses.get("total", 0) + (saliency_weight * batch_saliency_loss + text_weight * batch_text_loss)
        total_losses["mse"] = total_losses.get("mse", 0) + batch_mse
        total_losses["kld"] = total_losses.get("kld", 0) + batch_kld
        total_losses["cc"] = total_losses.get("cc", 0) + batch_cc
        total_losses["sim"] = total_losses.get("sim", 0) + batch_sim
        total_losses["nss"] = total_losses.get("nss", 0) + batch_nss
        total_losses["nss_valid"] = total_losses.get("nss_valid", 0) + batch_nss_valid
        total_losses["nss_total"] = total_losses.get("nss_total", 0) + batch_nss_total
        num_batches += 1
        
        pbar.set_postfix({
            "sal": f"{batch_saliency_loss:.4f}",
            "text": f"{batch_text_loss:.4f}",
            "CC": f"{batch_cc:.3f}",
            "NSS": f"{batch_nss:.3f}",
        })
    
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return avg_losses


def _get_base_model(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel or DistributedDataParallel."""
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_loss: float,
    save_path: str,
    is_main_process: bool = True,
):
    """Save a checkpoint (including LoRA adapters if present). Only saves when is_main_process is True (e.g. rank 0)."""
    if not is_main_process:
        return
    model_to_save = _get_base_model(model)
    
    # Check if model has LoRA adapters
    has_lora = False
    if hasattr(model_to_save, 'qwen_full_model'):
        qwen_model = model_to_save.qwen_full_model
        if hasattr(qwen_model, 'model') and hasattr(qwen_model.model, 'language_model'):
            lm = qwen_model.model.language_model
            if isinstance(lm, PeftModel):
                has_lora = True
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_loss": best_loss,
        "has_lora": has_lora,
    }
    
    torch.save(checkpoint, save_path)
    if is_main_process:
        print(f"Saved checkpoint to {save_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    # When stdout is piped (e.g. to tee for logging), disable tqdm so logs stay minimal
    if not sys.stdout.isatty():
        os.environ["TQDM_DISABLE"] = "1"
    parser = argparse.ArgumentParser(description="Train OpenVAM with LoRA")
    
    # Pretrained weights
    parser.add_argument("--pretrained_checkpoint", type=str, required=True,
                        help="Path to pretrained checkpoint (.pth file)")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to a mid-training checkpoint to resume from "
                             "(e.g. checkpoints_text_focused_joint/best_model.pth). "
                             "Restores model weights, optimizer, scheduler, epoch, and best_loss.")
    
    # Data paths
    parser.add_argument("--train_jsonl", type=str, default="datasets/salience_train.jsonl")
    parser.add_argument("--val_jsonl", type=str, default="datasets/salience_test.jsonl")
    parser.add_argument("--saliency_dir", type=str, default="salicon_256/saliency")
    parser.add_argument("--fixation_dir", type=str, default="salicon_256/fixations")
    
    # Model config
    parser.add_argument("--dino_model", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--backbone", type=str, default="vitb_rn50_384")
    parser.add_argument("--features", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=256)
    
    # LoRA config
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank (lower = fewer parameters)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout rate")
    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        help="Target modules for LoRA")
    parser.add_argument("--lora_lm_head", action="store_true",
                        help="Apply LoRA to lm_head instead of fully unfreezing it (uses separate adapter with --lora_lm_head_* flags)")
    parser.add_argument("--lora_lm_head_r", type=int, default=8,
                        help="LoRA rank for lm_head only (used when --lora_lm_head)")
    parser.add_argument("--lora_lm_head_alpha", type=int, default=16,
                        help="LoRA alpha for lm_head only (used when --lora_lm_head)")
    parser.add_argument("--lora_lm_head_dropout", type=float, default=0.0,
                        help="LoRA dropout for lm_head only (used when --lora_lm_head)")
    parser.add_argument("--train_qwen_norm", action="store_true",
                        help="Unfreeze qwen_norm for training (task-adapted hidden-state scaling before lm_head)")
    
    # Freeze options
    parser.add_argument("--freeze_dino", action="store_true", help="Freeze DINO backbone")
    parser.add_argument("--freeze_dpt", action="store_true", help="Freeze DPT components")
    parser.add_argument("--freeze_projector", action="store_true", help="Freeze projector")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing on the Qwen backbone to reduce activation memory "
                             "at the cost of ~20%% extra compute. Strongly recommended for 7B+ models on 24 GB GPUs.")
    
    # Training config
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    
    # Loss weights
    parser.add_argument("--saliency_weight", type=float, default=1.0)
    parser.add_argument("--text_weight", type=float, default=1.0)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--kld_weight", type=float, default=1.0)
    parser.add_argument("--cc_weight", type=float, default=1.0)
    parser.add_argument("--sim_weight", type=float, default=0.5)
    parser.add_argument("--nss_weight", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing for text CE loss (e.g. 0.05–0.1 to reduce overconfidence/hallucinations)")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints_salience_text_lora")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--early_stop_patience", type=int, default=5,
                        help="Stop training if val loss does not improve for this many consecutive epochs.")
    parser.add_argument("--model_type", type=str, default="qwen_dino",
                        choices=["qwen_dino", "qwen_vit"],
                        help="qwen_dino: Qwen LLM + DINOv3 + DPT (default). "
                             "qwen_vit: Qwen VL ViT + DPT, no separate DINOv3.")

    # Multi-GPU (DistributedDataParallel)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (set by torchrun)")
    
    args = parser.parse_args()
    
    # Resolve local_rank from environment (torchrun sets this)
    if args.local_rank == -1 and "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    
    use_ddp = args.local_rank >= 0 and torch.cuda.is_available()
    if use_ddp:
        dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size() if use_ddp else 1
    rank = dist.get_rank() if use_ddp else 0
    is_main_process = rank == 0
    
    if not _HAS_PEFT:
        raise ImportError("peft library is required for LoRA training. Install with: pip install peft")
    
    # Device (per-process for DDP)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if use_ddp:
            device = torch.device(f"cuda:{args.local_rank}")
            if is_main_process:
                print(f"Distributed training: world_size={world_size}, rank={rank}, local_rank={args.local_rank}")
        else:
            device = torch.device("cuda:0")
        if is_main_process:
            print(f"Found {num_gpus} GPU(s)")
            for i in range(num_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        num_gpus = 0
        device = torch.device("cpu")
    
    if is_main_process:
        print(f"Using device: {device}")
    
    # Create output directory and save args (rank 0 only)
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    if use_ddp:
        dist.barrier()
    
    # ==========================================================================
    # Create Model
    # ==========================================================================
    print("\n" + "="*70)
    if args.model_type == "qwen_vit":
        if not _QWEN_VIT_AVAILABLE:
            raise ImportError("QwenViTDPTWithText could not be imported. Check qwen_vit_dpt_text.py.")
        print("Creating QwenViTDPTWithText Model (Qwen VL ViT + DPT, no DINOv3)")
        print("="*70)
        model = QwenViTDPTWithText(
            qwen_model_name=args.qwen_model,
            backbone=args.backbone,
            features=args.features,
            freeze_qwen_vit=False,
            freeze_qwen_lm=True,
            freeze_projector=False,
        )
    else:
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
    # Move Model to Device BEFORE Initialization
    # ==========================================================================
    model = model.to(device)
    
    # ==========================================================================
    # Initialize LazyConv2d BEFORE Loading Weights (CRITICAL!)
    # ==========================================================================
    print("\n" + "="*70)
    print("Initializing LazyConv2d Layers (Before Weight Loading)")
    print("="*70)
    print("  Running dummy forward pass to initialize LazyConv2d...")
    
    # CRITICAL: LazyConv2d layers don't initialize until first forward pass
    # If we load weights before initialization, they'll be silently skipped!
    model.eval()
    
    with torch.no_grad():
        try:
            # Use the model's native dtype for the dummy input to avoid the float32
            # conversion that would double VRAM usage (fatal on 24 GB GPUs with 7B models).
            # The scratch layers are already cast to model.dtype in OpenVAM.__init__,
            # so no dtype mismatch occurs.
            model_dtype = getattr(model, 'dtype', torch.bfloat16)
            dummy_img = torch.randn(1, 3, args.image_size, args.image_size, device=device, dtype=model_dtype)
            dummy_text = "dummy prompt for initialization"
            
            # Run forward pass to initialize all LazyConv2d layers
            _ = model(dummy_img, text_prompt=dummy_text)
            
            print("  ✓ LazyConv2d layers initialized successfully")
        except Exception as e:
            print(f"  ⚠️  Warning: Could not initialize LazyConv2d: {e}")
            print("  Will attempt to load weights anyway (may fail silently)")
            import traceback
            traceback.print_exc()
    
    # ==========================================================================
    # Load Pretrained Weights
    # ==========================================================================
    print("\n" + "="*70)
    print("Loading Pretrained Weights")
    print("="*70)
    
    load_checkpoint_weights(model, args.pretrained_checkpoint, strict=False, verbose=True)
    
    # ==========================================================================
    # Apply LoRA
    # ==========================================================================
    lora_target_modules = list(args.lora_target_modules)
    lora_lm_head = getattr(args, "lora_lm_head", False)
    if lora_lm_head:
        if "lm_head" not in lora_target_modules:
            lora_target_modules.append("lm_head")
        print("\nLoRA lm_head: enabled (separate adapter with r=%d, alpha=%d, dropout=%s)"
              % (args.lora_lm_head_r, args.lora_lm_head_alpha, args.lora_lm_head_dropout))
    model = setup_lora_for_qwen_layers(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_target_modules,
        lora_lm_head=lora_lm_head,
        lora_lm_head_r=args.lora_lm_head_r,
        lora_lm_head_alpha=args.lora_lm_head_alpha,
        lora_lm_head_dropout=args.lora_lm_head_dropout,
    )
    
    # CRITICAL: Rebind references after LoRA is applied
    # After LoRA wraps the language_model, self.qwen_backbone still points to the base model.
    # We must rebind it to use the LoRA-wrapped module for generation.
    if hasattr(model, 'rebind_lora_references'):
        print("\nRebinding references to LoRA-wrapped language model...")
        model.rebind_lora_references()
    
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

    # Unfreeze lm_head for task-adapted text generation (output projection)
    # When --lora_lm_head: lm_head is adapted via LoRA only (base frozen); skip full unfreeze.
    base_for_lm_head = _get_base_model(model)
    lm_head_module = None
    if not getattr(args, "lora_lm_head", False):
        if hasattr(base_for_lm_head, 'qwen_full_model'):
            qwen = base_for_lm_head.qwen_full_model
            if hasattr(qwen, 'lm_head'):
                lm_head_module = qwen.lm_head
            elif hasattr(qwen, 'model') and hasattr(qwen.model, 'language_model') and hasattr(qwen.model.language_model, 'lm_head'):
                lm_head_module = qwen.model.language_model.lm_head
        if lm_head_module is not None:
            print("\nUnfreezing lm_head for training (task-adapted text generation)...")
            for p in lm_head_module.parameters():
                p.requires_grad = True
            n_lm_head = sum(p.numel() for p in lm_head_module.parameters())
            print(f"  lm_head parameters: {n_lm_head:,} (trainable)")
        else:
            print("\n⚠️  Could not find lm_head on model; skipping lm_head unfreeze.")
    else:
        print("\nlm_head: using LoRA adaptation only (base frozen, --lora_lm_head).")

    # Unfreeze qwen_norm only when --train_qwen_norm (optional)
    if getattr(args, "train_qwen_norm", False):
        if hasattr(base_for_lm_head, 'qwen_norm'):
            print("\nUnfreezing qwen_norm for training...")
            for p in base_for_lm_head.qwen_norm.parameters():
                p.requires_grad = True
            n_norm = sum(p.numel() for p in base_for_lm_head.qwen_norm.parameters())
            print(f"  qwen_norm parameters: {n_norm:,} (trainable)")
        else:
            print("\n⚠️  Could not find qwen_norm on model; skipping qwen_norm unfreeze.")
    else:
        print("\nqwen_norm: frozen (use --train_qwen_norm to make trainable).")
    
    # Model is already on device (moved before initialization)

    # Enable gradient checkpointing BEFORE DDP wrapping (required order)
    if getattr(args, "gradient_checkpointing", False):
        base_for_gc = _get_base_model(model)
        if hasattr(base_for_gc, "enable_gradient_checkpointing"):
            base_for_gc.enable_gradient_checkpointing()
            if is_main_process:
                print("\nGradient checkpointing: ENABLED (saves activation memory, ~20% slower)")
        else:
            if is_main_process:
                print("\n⚠️  Model does not expose enable_gradient_checkpointing(); skipping.")

    # Wrap with DistributedDataParallel for multi-GPU
    if use_ddp:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,
                    find_unused_parameters=False)
        if is_main_process:
            print(f"\nMulti-GPU: model wrapped with DistributedDataParallel (world_size={world_size})")
    elif num_gpus > 1 and is_main_process:
        print(f"\nNote: {num_gpus} GPUs available but using single GPU only (cuda:0). Use torchrun for multi-GPU.")
    
    # Print trainable parameters
    model_for_params = model
    trainable_params = sum(p.numel() for p in model_for_params.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_for_params.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Total parameters: {total_params:,}")
    
    # ==========================================================================
    # LoRA and Trainable Parameter Verification
    # ==========================================================================
    print("\n" + "="*70)
    print("Verifying Trainable Parameters (LoRA Check)")
    print("="*70)
    
    trainable_names = [name for name, param in model_for_params.named_parameters() if param.requires_grad]
    
    # 1. Check for LoRA parameters
    lora_params = [n for n in trainable_names if "lora_" in n]
    print(f"  Found {len(lora_params)} trainable LoRA parameters")
    if lora_params:
        print("  Sample LoRA parameters:")
        for n in lora_params[:5]:
            print(f"    - {n}")
        if len(lora_params) > 5:
            print(f"    ... and {len(lora_params) - 5} more")
    else:
        print("  ⚠️ WARNING: No LoRA parameters found among trainable parameters!")
        
    # 2. Check for non-LoRA trainable parameters
    non_lora_trainable = [n for n in trainable_names if "lora_" not in n]
    if non_lora_trainable:
        print(f"\n  Found {len(non_lora_trainable)} non-LoRA trainable parameters:")
        # Group by component for readability
        components = {}
        for n in non_lora_trainable:
            parts = n.split('.')
            comp = parts[0]
            if comp == 'scratch' or comp == 'proj_to_l1' or comp == 'proj_to_l2' or comp == 'proj_to_l3' or comp == 'proj_to_l4':
                comp = 'DPT/Scratch'
            elif comp == 'dino_adapter' or comp == 'patch_merger':
                comp = 'Projector'
            elif comp == 'dino':
                comp = 'DINO'
            elif 'lm_head' in n:
                comp = 'lm_head (output, intentionally trainable)'
            elif 'qwen_norm' in n:
                comp = 'qwen_norm (intentionally trainable)'
            elif comp == 'qwen_layers' or comp == 'embed_tokens':
                comp = 'Qwen (Non-LoRA!)'
            
            components[comp] = components.get(comp, 0) + 1
            
        for comp, count in components.items():
            print(f"    - {comp}: {count} parameters")
            if 'Non-LoRA!' in comp:
                print(f"      ⚠️ WARNING: {comp} parameters should typically be frozen when using LoRA!")
    else:
        print("\n  ✓ No non-LoRA parameters are trainable (pure LoRA fine-tuning)")
    print("="*70)
    
    # ==========================================================================
    # Datasets
    # ==========================================================================
    print("\n" + "="*70)
    print("Loading Datasets")
    print("="*70)
    
    train_dataset = SalienceTextDataset(
        jsonl_path=args.train_jsonl,
        saliency_dir=args.saliency_dir,
        fixation_dir=args.fixation_dir,
        image_size=args.image_size,
        augment=False,
    )
    
    val_dataset = SalienceTextDataset(
        jsonl_path=args.val_jsonl,
        saliency_dir=args.saliency_dir,
        fixation_dir=args.fixation_dir,
        image_size=args.image_size,
        augment=False,
    )
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=world_size, rank=rank) if use_ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
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
    
    if is_main_process:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
    
    # ==========================================================================
    # Verify Chat Template (CRITICAL SAFETY CHECK) — rank 0 only; broadcast abort to all
    # ==========================================================================
    should_abort = 0  # 0 = continue, 1 = abort
    if len(train_dataset) > 0:
        # Get a sample from the dataset
        sample = train_dataset[0]
        sample_text = sample.get("text", "Describe what draws attention in this image.")
        sample_target = sample.get("target_text", None)
        
        if is_main_process:
            raw_sample = train_dataset.samples[0]
            print("\n" + "="*70)
            print("Training Data Sample Preview")
            print("="*70)
            print(f"  Raw Sample Key Structure: {list(raw_sample.keys())}")
            if "conversations" in raw_sample:
                print(f"  Raw Conversations: {json.dumps(raw_sample['conversations'], indent=2)}")
            print(f"\n  Processed Input Prompt: {sample_text}")
            print(f"  Processed Target Text:  {sample_target}")
            # Complete prompt template that goes to Qwen (inference: prompt only; training: prompt + assistant)
            base_for_template = _get_base_model(model)
            if hasattr(base_for_template, "_apply_chat_template"):
                template_inference = base_for_template._apply_chat_template(sample_text)
                print("\n  --- Complete prompt template sent to Qwen (inference / prompt-only) ---")
                print(template_inference)
                print("  --- End prompt template (inference) ---")
                if sample_target:
                    template_training = base_for_template._apply_chat_template(sample_text, assistant_message=sample_target)
                    print("\n  --- Complete prompt+target template sent to Qwen (training / teacher-forcing) ---")
                    print(template_training)
                    print("  --- End prompt+target template (training) ---")
            print("="*70)
        
        # Verify chat template (rank 0 only for prompt; all ranks need model for verify)
        template_ok = verify_chat_template(model, sample_text, sample_target, device=device)
        
        if is_main_process:
            if not template_ok:
                print("\n" + "="*70)
                print("⚠️  WARNING: Chat template verification FAILED!")
                print("="*70)
                print("Training may proceed, but format mismatches between training")
                print("and inference are likely. This can cause poor model performance.")
                print("="*70)
                response = input("\nContinue training anyway? (y/n): ")
                if response.lower() != 'y':
                    print("Training aborted by user.")
                    should_abort = 1
            else:
                print("\n✓ Chat template verification passed - training format is correct!")
        
        if use_ddp:
            should_abort_t = torch.tensor([should_abort], device=device, dtype=torch.long)
            dist.broadcast(should_abort_t, src=0)
            should_abort = should_abort_t.item()
        if should_abort:
            if use_ddp:
                dist.destroy_process_group()
            sys.exit(0)
    else:
        if is_main_process:
            print("\n⚠️  WARNING: No training samples available - skipping chat template verification")
    
    # ==========================================================================
    # Optimizer & Scheduler
    # ==========================================================================
    # Get base model for optimizer (handles DataParallel)
    base_model_for_optimizer = _get_base_model(model)
    optimizer = AdamW(
        [p for p in base_model_for_optimizer.parameters() if p.requires_grad],
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
    # Loss Functions
    # ==========================================================================
    saliency_loss_fn = CombinedSaliencyLoss(
        kld_weight=args.kld_weight,
        cc_weight=args.cc_weight,
        sim_weight=args.sim_weight,
        nss_weight=args.nss_weight,
        mse_weight=args.mse_weight,
    )
    
    lm_loss_fn = LanguageModelingLoss(label_smoothing=getattr(args, "label_smoothing", 0.0))
    
    # ==========================================================================
    # Resume from mid-training checkpoint (optional)
    # ==========================================================================
    start_epoch = 1
    best_val_loss = float("inf")
    epochs_no_improve = 0

    if args.resume_checkpoint is not None:
        if is_main_process:
            print("\n" + "="*70)
            print(f"Resuming from checkpoint: {args.resume_checkpoint}")
            print("="*70)
        resume_ckpt = torch.load(args.resume_checkpoint, map_location=device)
        if isinstance(resume_ckpt, dict):
            # Restore model weights
            state_dict = resume_ckpt.get("model_state_dict", resume_ckpt.get("state_dict", resume_ckpt))
            missing, unexpected = _get_base_model(model).load_state_dict(state_dict, strict=False)
            if is_main_process:
                print(f"  Loaded model weights: {len(state_dict) - len(unexpected)} matched, "
                      f"{len(missing)} missing, {len(unexpected)} unexpected")
            # Restore optimizer
            if "optimizer_state_dict" in resume_ckpt and resume_ckpt["optimizer_state_dict"] is not None:
                try:
                    optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
                    if is_main_process:
                        print("  Restored optimizer state")
                except Exception as e:
                    if is_main_process:
                        print(f"  ⚠️  Could not restore optimizer state: {e} — starting fresh optimizer")
            # Restore scheduler
            if "scheduler_state_dict" in resume_ckpt and resume_ckpt["scheduler_state_dict"] is not None:
                try:
                    scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
                    if is_main_process:
                        print("  Restored scheduler state")
                except Exception as e:
                    if is_main_process:
                        print(f"  ⚠️  Could not restore scheduler state: {e} — starting fresh scheduler")
            # Restore epoch and best loss
            saved_epoch = resume_ckpt.get("epoch", 0)
            start_epoch = saved_epoch + 1
            best_val_loss = resume_ckpt.get("best_loss", float("inf"))
            if is_main_process:
                print(f"  Resuming from epoch {start_epoch}/{args.epochs}")
                print(f"  Best val loss so far: {best_val_loss:.4f}")
        else:
            if is_main_process:
                print("  ⚠️  Checkpoint format not recognised — starting from epoch 1")

    # ==========================================================================
    # Training Loop
    # ==========================================================================
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)

    early_stop_patience = args.early_stop_patience

    for epoch in range(start_epoch, args.epochs + 1):
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_main_process:
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"{'='*70}")
        
        train_losses = train_epoch(
            model, train_loader, optimizer, saliency_loss_fn, lm_loss_fn, device, epoch,
            accumulation_steps=args.accumulation_steps,
            saliency_weight=args.saliency_weight,
            text_weight=args.text_weight,
            scheduler=scheduler,
        )
        
        val_losses = validate(
            model, val_loader, saliency_loss_fn, lm_loss_fn, device, epoch,
            saliency_weight=args.saliency_weight,
            text_weight=args.text_weight,
        )
        
        if is_main_process:
            print(f"\nTrain Loss: {train_losses['total']:.4f} "
                  f"(Saliency: {train_losses.get('saliency', 0):.4f}, "
                  f"Text: {train_losses.get('text_lm', 0):.4f})")
            nss_train_valid = train_losses.get('nss_valid', 0)
            nss_train_total = train_losses.get('nss_total', 0)
            nss_train_str = f"{train_losses.get('nss', 0):.4f}" if nss_train_valid > 0 else f"0.0000 (no fixations: {nss_train_valid}/{nss_train_total})"
            print(f"  Saliency Metrics - MSE: {train_losses.get('mse', 0):.4f}, "
                  f"KLD: {train_losses.get('kld', 0):.4f}, "
                  f"CC: {train_losses.get('cc', 0):.4f}, "
                  f"SIM: {train_losses.get('sim', 0):.4f}, "
                  f"NSS: {nss_train_str}")
            print(f"Val Loss: {val_losses['total']:.4f} "
                  f"(Saliency: {val_losses.get('saliency', 0):.4f}, "
                  f"Text: {val_losses.get('text_lm', 0):.4f})")
            nss_val_valid = val_losses.get('nss_valid', 0)
            nss_val_total = val_losses.get('nss_total', 0)
            nss_val_str = f"{val_losses.get('nss', 0):.4f}" if nss_val_valid > 0 else f"0.0000 (no fixations: {nss_val_valid}/{nss_val_total})"
            print(f"  Saliency Metrics - MSE: {val_losses.get('mse', 0):.4f}, "
                  f"KLD: {val_losses.get('kld', 0):.4f}, "
                  f"CC: {val_losses.get('cc', 0):.4f}, "
                  f"SIM: {val_losses.get('sim', 0):.4f}, "
                  f"NSS: {nss_val_str}")
        
        # Diagnostic: Print LoRA weight stats (rank 0 only)
        if is_main_process:
            print_lora_stats(model)
        
        # ======================================================================
        # Qualitative Check: Generate Sample Text (Fixed 10 Samples) — rank 0 only
        # ======================================================================
        if args.text_weight > 0 and is_main_process:
            print("\n" + "-"*30)
            print(f"Qualitative Check (Epoch {epoch}) - Tracking 10 fixed samples")
            print("-"*30)
            model.eval()
            base_model_eval = _get_base_model(model)
            
            # Pick 10 fixed indices from validation set for consistent tracking
            num_val = len(val_dataset)
            check_indices = [i * (num_val // 10) for i in range(10)]
            
            for idx in check_indices:
                sample = val_dataset[idx]
                img = sample['image'].unsqueeze(0).to(device, dtype=base_model_eval.dtype)
                prompt = sample['text']
                target = sample['target_text']
                dataset_type_check = sample.get('dataset_type', 'natural_scene')
                image_path = sample.get('image_path', 'N/A')  # Get image filename for verification
                
                try:
                    with torch.no_grad():
                        # Use greedy search for consistent monitoring
                        # Enable diagnostics on first sample of first epoch only (to avoid spam)
                        enable_diag = (epoch == 1 and idx == check_indices[0])
                        try:
                            gen = base_model_eval.generate_text(
                                img, prompt,
                                max_new_tokens=100,
                                do_sample=False,
                                enable_attention_diagnostics=enable_diag,
                                dataset_type=dataset_type_check,
                            )
                        except TypeError:
                            gen = base_model_eval.generate_text(
                                img, prompt,
                                max_new_tokens=100,
                                do_sample=False,
                                enable_attention_diagnostics=enable_diag,
                            )
                    
                    print(f"\n[Sample {idx}] Image: {image_path}")
                    print(f"  TARGET: {target[:80]}...")
                    print(f"  GEN   : {gen}")
                except Exception as e:
                    print(f"\n[Sample {idx}] Generation failed: {e}")
            
            print("-"*30 + "\n")
            model.train()
        
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            epochs_no_improve = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_loss,
                os.path.join(args.output_dir, "best_model.pth"),
                is_main_process=is_main_process,
            )
        else:
            epochs_no_improve += 1
            if is_main_process:
                print(f"  No improvement for {epochs_no_improve}/{early_stop_patience} epochs.")
            if epochs_no_improve >= early_stop_patience:
                if is_main_process:
                    print(f"\nEarly stopping triggered: no improvement for {early_stop_patience} consecutive epochs.")
                break

        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_losses["total"],
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth"),
                is_main_process=is_main_process,
            )
    
    save_checkpoint(
        model, optimizer, scheduler, epoch, val_losses["total"],
        os.path.join(args.output_dir, "final_model.pth"),
        is_main_process=is_main_process,
    )
    
    if use_ddp:
        dist.destroy_process_group()
    
    if is_main_process:
        print("\n" + "="*70)
        print("Training Complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {args.output_dir}")
        print("="*70)


if __name__ == "__main__":
    main()
