#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL + DINOv3 + DPT Decoder with TEXT CONDITIONING (Checkpoint Compatible).

This version is compatible with checkpoints from no_vit.py:
- Uses DPT base class to get scratch layers with varying channel sizes (256, 512, 768)
- Uses LazyConv2d projections to adapt to scratch layer input channels
- Adds text conditioning through Qwen LLM layers

Architecture:
    Image → DINOv3 ─┬─→ Intermediate layers [L3, L6, L9] ─→ DPT Decoder ─→ Dense Output
                    │                                            ↑
                    └─→ PatchMerger ─┐                           │
                                     ├─→ Qwen LLM ─→ Visual Tokens (text-conditioned)
    Text → Tokenize → Embed ─────────┘
"""

import os
import math
import contextlib
from typing import Optional, List, Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# DPT imports (external module)
from DPT.dpt.models import DPT

# HF imports: support both Qwen2.5-VL and Qwen3-VL (choice by --qwen_model name)
try:
    from transformers import (
        AutoModel,
        AutoTokenizer,
        Qwen2_5_VLForConditionalGeneration,
    )
    _QWEN25_VL_AVAILABLE = True
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None
    _QWEN25_VL_AVAILABLE = False
try:
    from transformers import Qwen3VLForConditionalGeneration
    _QWEN3_VL_AVAILABLE = True
except ImportError:
    Qwen3VLForConditionalGeneration = None
    _QWEN3_VL_AVAILABLE = False
_HAS_TRANSFORMERS = _QWEN25_VL_AVAILABLE or _QWEN3_VL_AVAILABLE


def _get_qwen_vl_class(model_name: str):
    """Return the correct Qwen-VL class for the given HuggingFace model name."""
    name_lower = model_name.lower()
    if "2.5" in name_lower or "qwen2_5" in name_lower or "qwen2.5" in name_lower:
        if _QWEN25_VL_AVAILABLE:
            return Qwen2_5_VLForConditionalGeneration
        raise ImportError(
            f"Model name looks like Qwen2.5-VL ({model_name}) but Qwen2_5_VLForConditionalGeneration "
            "is not available. Install transformers with Qwen2.5-VL support."
        )
    if _QWEN3_VL_AVAILABLE:
        return Qwen3VLForConditionalGeneration
    raise ImportError(
        f"Qwen3VLForConditionalGeneration is not available. Install transformers with Qwen3-VL support, "
        f"or use a Qwen2.5-VL model (e.g. Qwen/Qwen2.5-VL-3B-Instruct)."
    )


# Dataset-type-specific prompts: natural_scene, webpage, e_commerce
# Used when dataset_type is passed to forward/encode_text (e.g. from sample id prefix).
DATASET_TYPE_PROMPTS = {
    "natural_scene": {
        "system": (
            "You are a visual attention explainer for saliency prediction.\n"
            "The input is a natural-scene photograph (real-world photo).\n"
            "List the 2-6 most visually salient objects or regions in descending order of attention.\n"
            "One item per line: <Object or region> (<location phrase>): <1-2 short evidence-based sentences>\n"
            "Rules: Use concrete location phrases (e.g., center, top-left, foreground, background, near the edge, behind <object>).\n"
            "Justify with visible cues only (e.g., faces/gaze, readable text, strong contrast/color, sharpness, size, implied motion/pose, interaction/pointing, uniqueness)\n"
            "Do NOT invent objects; if uncertain, write \"Uncertain: <object>\". No headers, bullets, or numbering."
        ),
        "user": (
            "This is a natural-scene photograph (real-world photo).\n"
            "Describe the 2-6 most visually salient objects/regions in this image, in descending order of attention.\n"
            "One per line: <Object or region> (<location>): <1-2 short evidence-based sentences>.\n"
            "Use concrete locations (e.g., center, top-left, foreground). Justify with visible cues only (faces/gaze, readable text, strong contrast/color, sharpness, size, implied motion/pose, interaction/pointing, uniqueness).\n"
            "Do NOT invent objects; use \"Uncertain: <object>\" if needed."
        ),
    },
    "webpage": {
        "system": (
            "You are a visual attention explainer for webpage/UI saliency.\n"
            "List the 2-6 most visually salient objects or regions in descending order of attention. Prioritize UI attention targets: headline/title text, buttons, search bar, navigation, "
            "modals, forms, product cards, key images, icons (cart/profile), prices/discounts, notification banners.\n"
            "One per line: <UI element or region> (<location>): <1-2 evidence-based sentences>\n"
            "Rules: Use concrete location phrases such as center, top-left, upper-right, left edge, lower-right, foreground, background, "
            "near the edge, or below/above <element>.\n"
            "- If a salient region is text, name it by function (e.g., \"headline text\", \"button label\", \"price text\", \"notification banner\") "
            "and do NOT invent unreadable words; if unreadable, say \"unreadable text\".\n"
            "Justify with visible cues only. No inventing; use \"Uncertain: <element>\". No headers, bullets, or numbering."
        ),
        "user": (
            "This image is a webpage/app UI screenshot (not a natural photo).\n"
            "List the 2-6 most visually salient objects or regions in this image.\n"
            "Prioritize UI elements: headline, buttons, search bar, navigation, modals, forms, product cards, icons, prices, banners. "
            "One per line: <element> (<location>): <1-2 short evidence-based sentences>. Use concrete locations. No inventing."
        ),
    },
    "e_commerce": {
        "system": (
            "You are a visual attention explainer for e-commerce/shopping images.\n"
            "List the 2-6 most salient objects or regions. Focus on: main product, brand/logo, price or discount text, "
            "promotional badges (SALE/NEW), key text, call-to-action, faces/models, standout accessories.\n"
            "One per line: <Object or region> (<location>): <1-2 evidence-based sentences>\n"
            "Rules: Use concrete locations (e.g., center, top-left, foreground).\n"
            "For text, DO NOT transcribe the exact words. Describe it by function only, e.g., "
            "\"price text\", \"discount text\", \"product title text\", \"brand/logo\", \"promo badge\", \"call-to-action button\", "
            "\"rating/review text\", \"shipping/offer text\".\n"
            "Justify with visible cues only (contrast, size, placement, faces). No inventing; if text unreadable say \"unreadable promotional text\". No headers, bullets, or numbering."
        ),
        "user": (
            "This image is an e-commerce/shopping image (product page/listing/promo).\n"
            "List the 2-6 most visually salient objects or regions in descending order of attention.\n"
            "Focus on: main product, brand/logo, price/discount area, promo badges, key text areas, call-to-action area, faces/models. "
            "One per line: <object/region> (<location>): <1-2 short evidence-based sentences>. "
            "Use concrete locations. For text, do NOT quote or transcribe, describe by function only."
        ),
    },
}

# Map merged dataset name (from sample id prefix) to prompt type
DATASET_NAME_TO_TYPE = {
    "CAT2000_256": "natural_scene",
    "MIT1003_256": "natural_scene",
    "OSIE_256": "natural_scene",
    "salicon_256": "natural_scene",
    "datasets_UI_256": "webpage",
    "SalEC": "e_commerce",
}

# Order by length desc so "datasets_UI_256" matches before "datasets"
MERGED_DATASET_NAMES = sorted(DATASET_NAME_TO_TYPE.keys(), key=len, reverse=True)


def get_dataset_from_id(sample_id: str) -> str:
    """Extract dataset name from merged sample id (e.g. CAT2000_256_Action_001 -> CAT2000_256)."""
    if not sample_id:
        return "unknown"
    for name in MERGED_DATASET_NAMES:
        if sample_id.startswith(name + "_"):
            return name
    return "other"


def get_dataset_type_from_name(dataset_name: str) -> str:
    """Return prompt type for a merged dataset name. Defaults to natural_scene."""
    return DATASET_NAME_TO_TYPE.get(dataset_name, "natural_scene")


def get_user_prompt_for_dataset_type(dataset_type: str) -> str:
    """Return the user prompt string for a dataset type (for use in datasets)."""
    return DATASET_TYPE_PROMPTS.get(dataset_type, DATASET_TYPE_PROMPTS["natural_scene"])["user"]


try:
    from transformers import AutoImageProcessor
    _HAS_IMAGE_PROCESSOR = True
except ImportError:
    _HAS_IMAGE_PROCESSOR = False


# ============================================================================
# Qwen-VL Projector Components (compatible with Qwen2.5-VL / Qwen3-VL)
# ============================================================================

class Qwen2RMSNorm(nn.Module):
    """RMSNorm as used in Qwen-VL (Qwen2.5-VL / Qwen3-VL)."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


class Qwen2_5_VLPatchMerger(nn.Module):
    """
    Qwen-VL style PatchMerger (Qwen2.5-VL / Qwen3-VL compatible).
    """
    
    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.context_dim = context_dim
        self.dim = dim
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        
        self.ln_q = Qwen2RMSNorm(context_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )
    
    def forward(self, x: torch.Tensor, h: int, w: int) -> Tuple[torch.Tensor, int, int]:
        B, N, D = x.shape
        s = self.spatial_merge_size
        
        x = self.ln_q(x)
        x = x.view(B, h, w, D)
        
        pad_h = (s - h % s) % s
        pad_w = (s - w % s) % s
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            h, w = h + pad_h, w + pad_w
        
        new_h, new_w = h // s, w // s
        x = x.view(B, new_h, s, new_w, s, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, new_h * new_w, s * s * D)
        x = self.mlp(x)
        
        return x, new_h, new_w


class VisualAdapter(nn.Module):
    """Adapts DINO hidden size to Qwen's vision encoder hidden size."""
    
    def __init__(self, dino_dim: int, qwen_vision_dim: int):
        super().__init__()
        self.needs_adapt = (dino_dim != qwen_vision_dim)
        self.adapter = nn.Linear(dino_dim, qwen_vision_dim) if self.needs_adapt else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)


# ============================================================================
# Main Model with Text Conditioning (Checkpoint Compatible)
# ============================================================================

class OpenVAM(DPT):
    """
    Multimodal Dense Prediction with Qwen3-VL + DINOv3 + DPT + TEXT CONDITIONING.
    
    This version is compatible with checkpoints from no_vit.py:
    - Inherits from DPT to get scratch layers with varying channel sizes
    - Uses LazyConv2d projections to adapt to scratch layer input channels
    - Adds text conditioning through Qwen LLM layers
    """
    
    def __init__(
        self,
        # Model names
        dino_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        qwen_model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        
        # HuggingFace settings
        hf_token: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        
        # DPT decoder settings (inherited from DPT base class)
        backbone: str = "vitb_rn50_384",  # Used by DPT to size scratch layers
        features: int = 256,
        readout: str = "project",
        channels_last: bool = False,
        use_bn: bool = False,
        enable_attention_hooks: bool = False,
        upsample_output_to_input_res: bool = True,
        out_channels: int = 1,
        use_sigmoid: bool = True,
        
        # Text conditioning settings
        max_text_length: int = 1024,
        
        # Freeze settings
        freeze_dino: bool = False,
        freeze_qwen_lm: bool = True,
        freeze_projector: bool = False,
        
        # Multi-GPU: spread Qwen LLM across GPUs (e.g. "auto" for 7B on 2 GPUs)
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        
        # Which DINO layers to tap
        dino_hooks: Optional[List[int]] = None,
    ):
        # Build DPT head
        head_layers = [
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.ReLU(True),
        ]
        if upsample_output_to_input_res:
            head_layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
        head_layers += [
            nn.Conv2d(features // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, out_channels, kernel_size=1),
        ]
        if use_sigmoid:
            head_layers.append(nn.Sigmoid())
        head = nn.Sequential(*head_layers)
        
        # Initialize DPT base class (we'll use its scratch layers)
        super().__init__(
            head=head,
            features=features,
            backbone=backbone,
            readout=readout,
            channels_last=channels_last,
            use_bn=use_bn,
            enable_attention_hooks=enable_attention_hooks,
        )
        
        # Remove unused timm backbone (like no_vit.py)
        if hasattr(self, "pretrained"):
            del self.pretrained
        
        assert _HAS_TRANSFORMERS, "Install transformers>=4.45.0"
        
        self.token = hf_token or os.getenv("HF_TOKEN")
        self.dtype = dtype or torch.bfloat16
        self.max_text_length = max_text_length
        self._device_map = device_map
        
        # ===== 1. Load DINOv3 =====
        print(f"[1/4] Loading DINOv3: {dino_model_name}")
        self.dino = AutoModel.from_pretrained(
            dino_model_name,
            torch_dtype=self.dtype,
            token=self.token,
        )
        
        self.patch_size = int(getattr(self.dino.config, "patch_size", 16))
        self.dino_dim = int(getattr(self.dino.config, "hidden_size", 768))
        self.dino_num_layers = int(getattr(self.dino.config, "num_hidden_layers", 12))
        self.num_registers = int(getattr(self.dino.config, "num_register_tokens", 4))
        
        # Set up DINO hooks (3 intermediate layers)
        if dino_hooks is not None:
            self.dino_hooks = dino_hooks
        else:
            n = self.dino_num_layers
            if n == 12:
                self.dino_hooks = [3, 6, 9]
            elif n == 24:
                self.dino_hooks = [6, 12, 18]
            else:
                step = n // 4
                self.dino_hooks = [step, 2*step, 3*step]
        
        self._setup_dino_norm(dino_model_name)
        
        if freeze_dino:
            self.dino.eval()
            for p in self.dino.parameters():
                p.requires_grad = False
        
        print(f"      dim={self.dino_dim}, layers={self.dino_num_layers}, hooks={self.dino_hooks}")
        
        # ===== 2. Load Qwen-VL (2.5 or 3, from model name) =====
        QwenVLClass = _get_qwen_vl_class(qwen_model_name)
        print(f"[2/4] Loading Qwen-VL ({QwenVLClass.__name__}): {qwen_model_name}")
        if self._device_map:
            print(f"      device_map={self._device_map} (multi-GPU)")

        load_kw = dict(
            torch_dtype=self.dtype,
            token=self.token,
            trust_remote_code=True,
        )
        if self._device_map is not None:
            load_kw["device_map"] = self._device_map
        qwen_full = QwenVLClass.from_pretrained(qwen_model_name, **load_kw)
        
        # Qwen-VL: hidden_size/vocab_size live on text_config, not top-level config
        self.qwen_dim = getattr(
            qwen_full.config, "hidden_size", None
        ) or getattr(qwen_full.config.text_config, "hidden_size", 2048)
        self.vocab_size = getattr(
            qwen_full.config, "vocab_size", None
        ) or getattr(qwen_full.config.text_config, "vocab_size", 151936)

        if hasattr(qwen_full.config, 'vision_config'):
            vc = qwen_full.config.vision_config
            self.spatial_merge = getattr(vc, 'spatial_merge_size', getattr(vc, 'merge_size', 2))
            self.qwen_vision_dim = getattr(vc, 'hidden_size', getattr(vc, 'embed_dim', 1280))
        else:
            self.spatial_merge = 2
            self.qwen_vision_dim = 1280
        
        # Keep full Qwen model for generation (Option A)
        vl_model = qwen_full.model
        
        if hasattr(vl_model, 'language_model'):
            lm = vl_model.language_model
            print(f"      Found language_model: {type(lm).__name__}")
            
            if hasattr(lm, 'model'):
                lm_inner = lm.model
                self.qwen_backbone = lm_inner # Backbone that returns last_hidden_state
                self.embed_tokens = lm_inner.embed_tokens
                self.qwen_layers = lm_inner.layers
                self.qwen_norm = lm_inner.norm
            else:
                self.qwen_backbone = lm # Backbone that returns last_hidden_state
                self.embed_tokens = lm.embed_tokens
                self.qwen_layers = lm.layers
                self.qwen_norm = lm.norm
            
            print(f"      ✓ qwen_backbone initialized: {type(self.qwen_backbone).__name__}")
            self._has_printed_backbone_info = False
        else:
            raise AttributeError("Cannot find language_model in Qwen-VL model")
        
        # Keep full Qwen model for text generation
        self.qwen_full_model = qwen_full
        if hasattr(vl_model, 'visual'):
            del vl_model.visual
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if freeze_qwen_lm:
            for p in self.embed_tokens.parameters():
                p.requires_grad = False
            for p in self.qwen_layers.parameters():
                p.requires_grad = False
            for p in self.qwen_norm.parameters():
                p.requires_grad = False
        
        print(f"      dim={self.qwen_dim}, vision_dim={self.qwen_vision_dim}")
        print(f"      Using ALL {len(self.qwen_layers)} LLM layers for cross-modal fusion (full conditioning)")
        
        # ===== 3. Build Projector =====
        print(f"[3/4] Building Qwen-style PatchMerger")
        
        self.dino_adapter = VisualAdapter(self.dino_dim, self.qwen_vision_dim)
        if self.dino_adapter.needs_adapt:
            print(f"      DINO Adapter: {self.dino_dim} -> {self.qwen_vision_dim}")
        
        self.patch_merger = Qwen2_5_VLPatchMerger(
            dim=self.qwen_dim,
            context_dim=self.qwen_vision_dim,
            spatial_merge_size=self.spatial_merge,
        )
        
        if freeze_projector:
            for p in self.dino_adapter.parameters():
                p.requires_grad = False
            for p in self.patch_merger.parameters():
                p.requires_grad = False
        
        # ===== 4. Build DPT Projections (Compatible with checkpoint) =====
        print(f"[4/4] Building DPT projections (compatible with checkpoint)")
        
        # Get scratch layer input channels (varying: 256, 512, 768)
        in_ch_l1 = self.scratch.layer1_rn.in_channels
        in_ch_l2 = self.scratch.layer2_rn.in_channels
        in_ch_l3 = self.scratch.layer3_rn.in_channels
        in_ch_l4 = self.scratch.layer4_rn.in_channels
        
        print(f"      Scratch layer input channels: L1={in_ch_l1}, L2={in_ch_l2}, L3={in_ch_l3}, L4={in_ch_l4}")
        
        # Use LazyConv2d to adapt DINO features to scratch layer input channels
        # (like no_vit.py does)
        self.proj_to_l1 = nn.LazyConv2d(in_ch_l1, kernel_size=1)
        self.proj_to_l2 = nn.LazyConv2d(in_ch_l2, kernel_size=1)
        self.proj_to_l3 = nn.LazyConv2d(in_ch_l3, kernel_size=1)
        # For layer4, we'll project Qwen features (qwen_dim -> in_ch_l4)
        # We need to know qwen_dim, but it's already set above
        # Use a regular Conv2d since we know the input dimension
        self.proj_to_l4 = nn.Conv2d(self.qwen_dim, in_ch_l4, kernel_size=1)
        
        # Convert all new modules and DPT scratch to model dtype (so decoder matches DINO/Qwen)
        self.dino_adapter = self.dino_adapter.to(self.dtype)
        self.patch_merger = self.patch_merger.to(self.dtype)
        self.proj_to_l1 = self.proj_to_l1.to(self.dtype)
        self.proj_to_l2 = self.proj_to_l2.to(self.dtype)
        self.proj_to_l3 = self.proj_to_l3.to(self.dtype)
        self.proj_to_l4 = self.proj_to_l4.to(self.dtype)
        self.scratch = self.scratch.to(self.dtype)

        # Tokenizer for text processing (prefer cache to avoid Hub timeouts)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                qwen_model_name, token=self.token, local_files_only=True
            )
        except (OSError, ValueError):
            # Cache miss or missing file: allow network with longer timeout
            _prev = os.environ.get("HF_HUB_ETAG_TIMEOUT")
            os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    qwen_model_name, token=self.token
                )
            finally:
                if _prev is not None:
                    os.environ["HF_HUB_ETAG_TIMEOUT"] = _prev
                else:
                    os.environ.pop("HF_HUB_ETAG_TIMEOUT", None)
        self.tokenizer.padding_side = "left" 
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Default text prompt when none provided (natural_scene; must match _apply_chat_template)
        self.default_prompt = DATASET_TYPE_PROMPTS["natural_scene"]["user"]
        
        print(f"\nModel ready with TEXT CONDITIONING (Checkpoint Compatible)!")
        print(f"  DINO hooks: {self.dino_hooks}")
        print(f"  Scratch layer channels: L1={in_ch_l1}, L2={in_ch_l2}, L3={in_ch_l3}, L4={in_ch_l4}")
        print(f"  LLM layers for fusion: ALL {len(self.qwen_layers)} layers (full conditioning)")
        print(f"  Max text length: {self.max_text_length}")
        print(f"  Full Qwen model kept for text generation (Option A)")
    
    def rebind_lora_references(self):
        """
        Rebind qwen_backbone and related references after LoRA is applied.
        
        CRITICAL: If LoRA is applied to qwen_full_model.model.language_model after initialization,
        self.qwen_backbone will still point to the frozen base model. This method rebinds
        the references to point to the LoRA-wrapped module so generation uses LoRA weights.
        
        IMPORTANT: With PEFT, we must use the PeftModel wrapper (peft_model) for qwen_backbone,
        and access layers through peft_model.model.layers (not get_base_model().layers) to
        ensure LoRA adapters are active during both training and generation.
        
        Call this method AFTER LoRA is applied (e.g., after setup_lora_for_qwen_layers).
        
        Example:
            model = OpenVAM(...)
            model = setup_lora_for_qwen_layers(model, ...)  # LoRA wraps language_model
            model.rebind_lora_references()  # Rebind to use LoRA-wrapped module
        """
        if not hasattr(self, 'qwen_full_model'):
            print("  ⚠️  Warning: qwen_full_model not found, cannot rebind LoRA references")
            return
        
        vl_model = self.qwen_full_model.model
        if not hasattr(vl_model, 'language_model'):
            print("  ⚠️  Warning: language_model not found, cannot rebind LoRA references")
            return
        
        lm = vl_model.language_model
        print(f"  Rebinding references to language_model: {type(lm).__name__}")
        
        # Check if lm is a PeftModel (LoRA-wrapped)
        from peft import PeftModel
        is_peft = isinstance(lm, PeftModel)
        
        if isinstance(lm, PeftModel):
            # CRITICAL: Use the PeftModel wrapper itself for qwen_backbone
            # This ensures forward() calls go through LoRA adapters
            self.qwen_backbone = lm
            
            # Access layers through the wrapper (peft_model.model.layers)
            # NOT through get_base_model() which would bypass LoRA
            if hasattr(lm, 'model'):
                lm_inner = lm.model
                self.embed_tokens = lm_inner.embed_tokens
                self.qwen_layers = lm_inner.layers  # Wrapped path - LoRA adapters active
                self.qwen_norm = lm_inner.norm
            else:
                # Fallback (shouldn't happen with standard PEFT)
                self.embed_tokens = lm.embed_tokens
                self.qwen_layers = lm.layers
                self.qwen_norm = lm.norm
            print(f"  ✓ Using PeftModel wrapper for qwen_backbone (LoRA adapters active)")
        else:
            # Not LoRA-wrapped, use standard rebinding
            if hasattr(lm, 'model'):
                lm_inner = lm.model
                self.qwen_backbone = lm_inner
                self.embed_tokens = lm_inner.embed_tokens
                self.qwen_layers = lm_inner.layers
                self.qwen_norm = lm_inner.norm
            else:
                self.qwen_backbone = lm
                self.embed_tokens = lm.embed_tokens
                self.qwen_layers = lm.layers
                self.qwen_norm = lm.norm
            print(f"  ✓ Language model is not LoRA-wrapped - using standard references")
        
        print(f"  ✓ References rebound: qwen_backbone={type(self.qwen_backbone).__name__}")
        print(f"    embed_tokens={type(self.embed_tokens).__name__}")
        print(f"    qwen_layers={type(self.qwen_layers).__name__}")
        print(f"    qwen_norm={type(self.qwen_norm).__name__}")
        
        # Verify LoRA is accessible through the rebound references
        try:
            if hasattr(self.qwen_layers, '__len__') and len(self.qwen_layers) > 0:
                sample_layer = self.qwen_layers[0]
                if hasattr(sample_layer, 'self_attn') and hasattr(sample_layer.self_attn, 'q_proj'):
                    has_lora = hasattr(sample_layer.self_attn.q_proj, 'lora_A')
                    print(f"  ✓ LoRA verification: {'LoRA adapters found' if has_lora else 'No LoRA adapters (expected if not wrapped)'}")
        except Exception as e:
            print(f"  ⚠️  Could not verify LoRA: {e}")
    
    def _setup_dino_norm(self, model_name: str):
        """Set up normalization for DINO."""
        if _HAS_IMAGE_PROCESSOR:
            try:
                proc = AutoImageProcessor.from_pretrained(model_name, token=self.token)
                self.register_buffer("img_mean", torch.tensor(proc.image_mean).view(1,3,1,1))
                self.register_buffer("img_std", torch.tensor(proc.image_std).view(1,3,1,1))
                return
            except:
                pass
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
    
    @staticmethod
    def _safe_div(H: int, W: int, div: int) -> Tuple[int, int]:
        return (max(1, H // div), max(1, W // div))
    
    def _tokens_to_map(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Convert [B, N, D] tokens to [B, D, h, w] feature map."""
        B, N, D = tokens.shape
        return tokens.permute(0, 2, 1).contiguous().view(B, D, h, w)
    
    def _dino_tokens_to_map(self, tokens: torch.Tensor, H_pad: int, W_pad: int) -> torch.Tensor:
        """
        tokens: [B, T, D] -> map: [B, D, h, w]
        Note: tokens should already have CLS + register tokens removed.
        """
        B, T, D = tokens.shape
        
        # Calculate expected grid dimensions from padded image size
        h = max(1, H_pad // self.patch_size)
        w = max(1, W_pad // self.patch_size)
        expected_N = h * w
        
        # If token count doesn't match expected, use actual token count to infer dimensions
        if T != expected_N:
            # Infer dimensions from actual token count
            h = int(round(math.sqrt(T)))
            w = max(1, T // h)
            # Ensure h * w == T
            if h * w != T:
                # Try to find factors that work
                for h_candidate in range(int(math.sqrt(T)), 0, -1):
                    if T % h_candidate == 0:
                        h = h_candidate
                        w = T // h_candidate
                        break
                # If still doesn't work, use floor division
                if h * w != T:
                    h = int(math.sqrt(T))
                    w = (T + h - 1) // h  # Ceiling division
        
        return tokens.permute(0, 2, 1).contiguous().view(B, D, h, w)
    
    def _apply_chat_template(
        self,
        user_message: str,
        assistant_message: Optional[str] = None,
        dataset_type: Optional[str] = None,
    ) -> str:
        """
        Apply chat template to format messages for Qwen-VL-Instruct.
        
        This matches the standard Qwen-VL fine-tuning format:
        - System message: Defines the model's role as a vision assistant
        - Training: [system, user message, assistant response] → full conversation
        - Inference: [system, user message] → with generation prompt
        
        If dataset_type is set (natural_scene, webpage, e_commerce), uses that type's system and user prompts.
        
        Args:
            user_message: The user's prompt/question
            assistant_message: Optional assistant response (for training)
            dataset_type: Optional. One of natural_scene, webpage, e_commerce.
            
        Returns:
            Formatted text string with proper special tokens (Qwen-VL format)
        """
        if dataset_type and dataset_type in DATASET_TYPE_PROMPTS:
            prompts = DATASET_TYPE_PROMPTS[dataset_type]
            system_prompt = prompts["system"]
            full_user_message = user_message if user_message else prompts["user"]
        else:
            full_user_message = user_message if user_message else self.default_prompt
            system_prompt = DATASET_TYPE_PROMPTS["natural_scene"]["system"]
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user_message},
        ]
        if assistant_message is not None:
            messages.append({"role": "assistant", "content": assistant_message})
        
        # Check if tokenizer has chat template (Qwen-VL should have one)
        # Use getattr to safely check for chat_template (handles empty string, None, etc.)
        has_chat_template = (
            hasattr(self.tokenizer, "apply_chat_template") and 
            getattr(self.tokenizer, "chat_template", None)
        )
        
        if not has_chat_template:
            # Fallback: use tokenizer's actual special tokens if available
            # Get special tokens from tokenizer (more robust than hardcoding)
            im_start = getattr(self.tokenizer, "im_start_token", "<|im_start|>")
            im_end = getattr(self.tokenizer, "im_end_token", "<|im_end|>")
            
            # Try to get from tokenizer's special tokens dict if available
            if hasattr(self.tokenizer, "special_tokens_map"):
                im_start = self.tokenizer.special_tokens_map.get("im_start", im_start)
                im_end = self.tokenizer.special_tokens_map.get("im_end", im_end)
            
            # Fallback format using tokenizer's special tokens
            if assistant_message is not None:
                return f"{im_start}system\n{system_prompt}{im_end}\n{im_start}user\n{full_user_message}{im_end}\n{im_start}assistant\n{assistant_message}{im_end}"
            else:
                return f"{im_start}system\n{system_prompt}{im_end}\n{im_start}user\n{full_user_message}{im_end}\n{im_start}assistant\n"
        
        # Apply chat template (standard Qwen-VL format)
        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=(assistant_message is None),
        )
        
        return formatted_text
    
    def verify_dataset_prompts(self, max_system_user_chars: int = 600, max_formatted_chars: int = 1000):
        """
        Print system/user and the actual formatted prompt for each dataset type so you can verify
        the correct prompts go to the model. Call this before training (e.g. with --verify_prompts).
        """
        print("\n" + "=" * 70)
        print("Dataset-type prompts verification (exactly what goes to the model)")
        print("=" * 70)
        for dtype in ("natural_scene", "webpage", "e_commerce"):
            prompts = DATASET_TYPE_PROMPTS.get(dtype)
            if not prompts:
                continue
            sys_str = prompts["system"]
            user_str = prompts["user"]
            print(f"\n--- {dtype} ---")
            print("[System prompt]")
            print(sys_str[:max_system_user_chars] + ("..." if len(sys_str) > max_system_user_chars else ""))
            print("\n[User prompt]")
            print(user_str[:max_system_user_chars] + ("..." if len(user_str) > max_system_user_chars else ""))
            formatted = self._apply_chat_template(user_str, dataset_type=dtype)
            print("\n[Formatted string sent to tokenizer (first {} chars)]".format(max_formatted_chars))
            print(formatted[:max_formatted_chars] + ("..." if len(formatted) > max_formatted_chars else ""))
        print("\n" + "=" * 70 + "\n")
    
    def encode_text(
        self,
        text_prompt: str,
        batch_size: int,
        device: torch.device,
        use_chat_template: bool = True,
        dataset_type: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Tokenize and embed text prompt.
        
        Args:
            text_prompt: The text prompt (when dataset_type is set, type's user prompt is used if text_prompt is empty)
            batch_size: Batch size to replicate for
            device: Device to put tensors on
            use_chat_template: If True, apply chat template (recommended for Qwen-Instruct)
            dataset_type: Optional. One of natural_scene, webpage, e_commerce (uses that type's system+user prompts).
            
        Returns:
            text_embeds: [B, text_len, qwen_dim] text embeddings
            attention_mask: [B, text_len] attention mask
            text_len: Number of text tokens
        """
        # Apply chat template if requested
        if use_chat_template:
            if dataset_type and dataset_type in DATASET_TYPE_PROMPTS:
                user_msg = text_prompt if text_prompt else DATASET_TYPE_PROMPTS[dataset_type]["user"]
                formatted_text = self._apply_chat_template(user_msg, dataset_type=dataset_type)
            else:
                formatted_text = self._apply_chat_template(text_prompt)
        else:
            formatted_text = text_prompt
        
        # Tokenize WITHOUT padding - use actual sequence length
        # add_special_tokens=False: chat template string already has special tokens; avoid double BOS/EOS
        tokens = self.tokenizer(
            formatted_text,
            return_tensors="pt",
            padding=False,  # NO PADDING - use real length
            truncation=True,
            max_length=self.max_text_length,
            add_special_tokens=False,
        )
        
        input_ids = tokens["input_ids"].to(device)  # [1, L] where L is actual length
        attention_mask = tokens["attention_mask"].to(device)  # [1, L]
        
        # Get actual text length from shape (more reliable than sum when no padding)
        text_len = input_ids.shape[1]
        
        # Embed text tokens
        text_embeds = self.embed_tokens(input_ids)  # [1, L, qwen_dim]
        
        # Replicate for batch
        if batch_size > 1:
            text_embeds = text_embeds.expand(batch_size, -1, -1)
            attention_mask = attention_mask.expand(batch_size, -1)
        
        return text_embeds, attention_mask, text_len
    
    def encode_dino(
        self,
        pixel_values: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, int, int]:
        """
        Encode image with DINOv3, returning intermediate features and final tokens.
        """
        B, _, H, W = pixel_values.shape
        device = pixel_values.device
        
        # Normalize in model dtype (DINO is loaded with torch_dtype=self.dtype; fp32 input can crash/upcast)
        x = pixel_values.to(dtype=self.dtype)
        mean = self.img_mean.to(device=device, dtype=self.dtype)
        std = self.img_std.to(device=device, dtype=self.dtype)
        x = (x - mean) / std
        
        # Pad to patch size
        H_pad = math.ceil(H / self.patch_size) * self.patch_size
        W_pad = math.ceil(W / self.patch_size) * self.patch_size
        if H != H_pad or W != W_pad:
            x = F.pad(x, (0, W_pad - W, 0, H_pad - H), mode="reflect")
        
        grid_h = H_pad // self.patch_size
        grid_w = W_pad // self.patch_size
        
        # Forward with hidden states (autocast on CUDA for bf16/fp16 to match loaded dtype)
        use_amp = device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16)
        ctx = torch.autocast("cuda", dtype=self.dtype) if use_amp else contextlib.nullcontext()
        with ctx:
            with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.dino.parameters())):
                outputs = self.dino(pixel_values=x, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states
        n_drop = 1 + self.num_registers
        
        # Extract hooked layers
        dino_features = []
        for hook_idx in self.dino_hooks:
            idx = min(max(hook_idx, 1), len(hidden_states) - 1)
            tokens = hidden_states[idx][:, n_drop:, :]
            feat_map = self._dino_tokens_to_map(tokens, H_pad, W_pad)
            dino_features.append(feat_map)
        
        # Final patch tokens
        patch_tokens = outputs.last_hidden_state[:, n_drop:, :]

        # Ensure vision features match projector/merger dtype (avoid Float vs BFloat16 matmul)
        vision_dtype = None
        try:
            vision_dtype = next(self.dino_adapter.parameters()).dtype
        except Exception:
            vision_dtype = None
        if vision_dtype is None:
            try:
                vision_dtype = next(self.patch_merger.parameters()).dtype
            except Exception:
                vision_dtype = None
        if vision_dtype is None:
            vision_dtype = self.dtype

        if patch_tokens.dtype != vision_dtype:
            patch_tokens = patch_tokens.to(dtype=vision_dtype)
            dino_features = [f.to(dtype=vision_dtype) for f in dino_features]
        
        return dino_features, patch_tokens, grid_h, grid_w
    
    def _build_mrope_position_ids(
        self,
        num_visual_tokens: int,
        num_text_tokens: int,
        grid_h: int,
        grid_w: int,
        device: torch.device,
        vision_first: bool = True,
    ) -> torch.Tensor:
        """
        Build proper M-RoPE position IDs for Qwen-VL multimodal inputs.
        
        Follows Qwen-VL's "segment-by-segment offset" convention where
        each modality segment starts at (previous_max_position + 1).
        
        Qwen2.5-VL / Qwen3-VL use 3-axis M-RoPE:
        - Axis 0 (t/temporal): 0 for static images (+ offset)
        - Axis 1 (h/height): Row index in grid (+ offset)
        - Axis 2 (w/width): Column index in grid (+ offset)
        
        For vision-first [vision, text]:
        - Vision: positions (t,h,w) starting at 0 (spatial grid positions)
        - Text: sequential positions starting at offset = max(grid_h, grid_w) so text
          starts "after" the vision grid and avoids position collisions with vision.
        
        For text-first [text, vision]:
        - Text: positions 0..N_text-1 (all 3 axes same)
        - Vision: positions (t,h,w) + N_text offset
        
        Args:
            num_visual_tokens: Number of visual tokens (should equal grid_h * grid_w)
            num_text_tokens: Number of text tokens
            grid_h: Height of the visual token grid
            grid_w: Width of the visual token grid
            device: Target device
            vision_first: If True, vision tokens come before text tokens
            
        Returns:
            position_ids: [3, 1, N_total] tensor for M-RoPE
        """
        num_vis = grid_h * grid_w
        
        # Strict check: num_visual_tokens MUST match grid dimensions
        # If this fails, there's an upstream bug in PatchMerger or caller
        assert num_visual_tokens == num_vis, (
            f"M-RoPE position_ids error: num_visual_tokens ({num_visual_tokens}) != "
            f"grid_h*grid_w ({grid_h}*{grid_w}={num_vis}). "
            f"Check PatchMerger output or caller logic."
        )
        
        # Base vision grid (single image => t=0)
        vis_t = torch.zeros(num_vis, device=device, dtype=torch.long)
        vis_h = torch.arange(grid_h, device=device, dtype=torch.long).repeat_interleave(grid_w)
        vis_w = torch.arange(grid_w, device=device, dtype=torch.long).repeat(grid_h)
        
        if vision_first:
            # Vision starts at 0 (spatial grid positions)
            vis_pos = torch.stack([vis_t, vis_h, vis_w], dim=0)  # [3, N_vis]
            # Text starts after vision grid to avoid position collisions (safer for VL M-RoPE)
            offset = max(grid_h, grid_w)
            txt = torch.arange(num_text_tokens, device=device, dtype=torch.long) + offset
            txt_pos = torch.stack([txt, txt, txt], dim=0)  # [3, N_text]
            pos = torch.cat([vis_pos, txt_pos], dim=1)  # [3, N_total]
        else:
            # Text starts at 0
            txt = torch.arange(num_text_tokens, device=device, dtype=torch.long)
            txt_pos = torch.stack([txt, txt, txt], dim=0)  # [3, N_text]
            
            # Vision starts after max(text)+1 = num_text
            vis_offset = txt_pos.max().item() + 1  # == num_text
            vis_pos = torch.stack([vis_t, vis_h, vis_w], dim=0) + vis_offset  # [3, N_vis]
            
            pos = torch.cat([txt_pos, vis_pos], dim=1)  # [3, N_total]
        
        return pos.unsqueeze(1)  # [3, 1, N_total]
    
    def _check_visual_text_token_alignment(
        self,
        visual_tokens: torch.Tensor,
        text_embeds: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> None:
        """
        Verify that visual_tokens and text_embeds shapes match expectations, and that
        num_visual_tokens is aligned with grid dimensions (DINOv3 + PatchMerger).
        """
        # Expected: visual_tokens [B, N_vis, dim], text_embeds [B, N_text, dim]
        assert visual_tokens.ndim == 3, (
            f"visual_tokens must be 3D [B, N_vis, dim], got ndim={visual_tokens.ndim}"
        )
        assert text_embeds.ndim == 3, (
            f"text_embeds must be 3D [B, N_text, dim], got ndim={text_embeds.ndim}"
        )
        B_vis, N_vis, dim_vis = visual_tokens.shape
        B_txt, N_text, dim_txt = text_embeds.shape
        assert B_vis >= 1 and N_vis >= 1, (
            f"visual_tokens shape invalid: expected B>=1, N_vis>=1, got [{B_vis}, {N_vis}, {dim_vis}]"
        )
        assert B_txt >= 1 and N_text >= 1, (
            f"text_embeds shape invalid: expected B>=1, N_text>=1, got [{B_txt}, {N_text}, {dim_txt}]"
        )
        assert B_vis == B_txt, (
            f"Visual/text batch size mismatch: visual {B_vis} vs text {B_txt}"
        )
        assert dim_vis == dim_txt, (
            f"Visual/text embedding dim mismatch: visual {dim_vis} vs text {dim_txt}. "
            "DINO adapter + PatchMerger must output same dim as Qwen embed_tokens."
        )
        # Confirm num_visual_tokens is aligned with grid dimensions
        expected_vis = grid_h * grid_w
        assert N_vis == expected_vis, (
            f"num_visual_tokens not aligned with grid: got {N_vis} visual tokens, "
            f"expected grid_h*grid_w = {grid_h}*{grid_w} = {expected_vis}. Check PatchMerger output."
        )
        assert grid_h >= 1 and grid_w >= 1, (
            f"Grid dimensions must be positive: grid_h={grid_h}, grid_w={grid_w}"
        )

    def _verify_position_ids_mixed_vision_text(
        self,
        position_ids: torch.Tensor,
        num_visual_tokens: int,
        num_text_tokens: int,
        batch_size: int,
        vision_first: bool,
    ) -> None:
        """
        Ensure position_ids are built correctly for mixed vision-text input.
        Expects position_ids [3, B, N_total] with N_total = num_visual_tokens + num_text_tokens.
        """
        assert position_ids.ndim == 3, (
            f"position_ids must be 3D [3, B, N_total] for M-RoPE, got ndim={position_ids.ndim}"
        )
        n_axes, B, N_total = position_ids.shape
        assert n_axes == 3, (
            f"position_ids must have 3 axes for M-RoPE (t,h,w), got {n_axes}"
        )
        assert B == batch_size, (
            f"position_ids batch size {B} != expected {batch_size}"
        )
        expected_total = num_visual_tokens + num_text_tokens
        assert N_total == expected_total, (
            f"position_ids sequence length {N_total} != num_visual_tokens + num_text_tokens "
            f"({num_visual_tokens} + {num_text_tokens} = {expected_total})"
        )

    def _validate_qwen_backbone_inputs(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        """
        Ensure qwen_backbone receives consistent inputs: merged visual+text embeddings
        with position_ids and attention_mask that match in batch size and sequence length.
        Call before passing inputs to qwen_backbone (full forward, no past_key_values).
        """
        B_emb = inputs_embeds.shape[0]
        seq_emb = inputs_embeds.shape[1]
        assert inputs_embeds.ndim == 3, (
            f"inputs_embeds must be 3D [B, seq_len, dim], got ndim={inputs_embeds.ndim}"
        )
        assert attention_mask.ndim == 2, (
            f"attention_mask must be 2D [B, seq_len], got ndim={attention_mask.ndim}"
        )
        assert position_ids.ndim == 3, (
            f"position_ids must be 3D [3, B, seq_len] for M-RoPE, got ndim={position_ids.ndim}"
        )
        B_mask, seq_mask = attention_mask.shape
        n_axes, B_pos, seq_pos = position_ids.shape
        assert n_axes == 3, (
            f"position_ids must have 3 axes (t,h,w), got {n_axes}"
        )
        assert B_emb == B_mask == B_pos, (
            f"Batch size mismatch: inputs_embeds {B_emb}, attention_mask {B_mask}, position_ids {B_pos}"
        )
        assert seq_emb == seq_mask == seq_pos, (
            f"Sequence length mismatch: inputs_embeds {seq_emb}, attention_mask {seq_mask}, "
            f"position_ids {seq_pos}. All must match the merged sequence length."
        )

    def _apply_lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply Qwen norm then LM head to get logits. Use this for all generation/training
        logits so the path is consistent. No masking or other ops should be applied to
        hidden_states before calling this; masking is applied only to labels (e.g. -100
        on prompt positions), not to hidden state.
        hidden_states: [B, N, dim] (sequence) or [B, dim] (single token).
        Returns: logits [B, N, vocab_size] or [B, vocab_size].
        """
        # Ensure dtype matches norm/lm_head (e.g. without autocast, hidden_states can be float32)
        head_dtype = next(self.qwen_norm.parameters()).dtype
        hidden_states = hidden_states.to(dtype=head_dtype)
        normalized = self.qwen_norm(hidden_states)
        if hasattr(self.qwen_full_model, 'lm_head'):
            return self.qwen_full_model.lm_head(normalized)
        if hasattr(self.qwen_full_model.model.language_model, 'lm_head'):
            return self.qwen_full_model.model.language_model.lm_head(normalized)
        raise AttributeError("Could not find lm_head in Qwen model.")

    def fuse_text_and_vision(
        self,
        visual_tokens: torch.Tensor,
        text_embeds: torch.Tensor,
        text_attention_mask: torch.Tensor,
        grid_h: int,
        grid_w: int,
        return_full_sequence: bool = False,
        order: str = "text_first",
        debug_shapes: bool = False,
    ) -> torch.Tensor:
        """
        Fuse text and visual tokens through Qwen LLM layers.
        qwen_backbone receives merged inputs_embeds (visual + text in the chosen order),
        with position_ids and attention_mask that match in batch size and sequence length.

        Causal attention: token i cannot attend to tokens after it. So:
        - order="vision_first" [vision, text]: text tokens can attend to vision → vision-conditioned
          text. Use for saliency when you want vision-first sequencing.
        - order="text_first" [text, vision]: vision tokens can attend to text → text-conditioned
          visual features. Use if you want prompt-guided visual features.

        Args:
            visual_tokens: [B, N_vis, dim] visual token embeddings
            text_embeds: [B, N_text, dim] text token embeddings
            text_attention_mask: [B, N_text] attention mask for text
            grid_h: Height of visual token grid (for M-RoPE)
            grid_w: Width of visual token grid (for M-RoPE)
            return_full_sequence: If True, return full sequence; else just visual tokens
            order: "vision_first" for saliency (vision then text), "text_first" for text-conditioned visuals
            debug_shapes: If True, print shapes of combined, position_ids, attention_mask (for testing)
        """
        B = visual_tokens.shape[0]
        device = visual_tokens.device
        N_vis = visual_tokens.shape[1]
        N_text = text_embeds.shape[1]

        # Check that custom visual tokens (DINOv3 + PatchMerger) are aligned with text tokens
        self._check_visual_text_token_alignment(visual_tokens, text_embeds, grid_h, grid_w)

        if order == "text_first":
            combined = torch.cat([text_embeds, visual_tokens], dim=1)  # [B, N_text + N_vis, dim]
            attn_mask = torch.cat(
                [text_attention_mask, torch.ones(B, N_vis, device=device, dtype=torch.long)],
                dim=1,
            )
            position_ids = self._build_mrope_position_ids(
                num_visual_tokens=N_vis,
                num_text_tokens=N_text,
                grid_h=grid_h,
                grid_w=grid_w,
                device=device,
                vision_first=False,
            )
            position_ids = position_ids.expand(3, B, -1)
            assert combined.shape[1] == position_ids.shape[2], (
                f"Combined length vs position_ids: {combined.shape[1]} != {position_ids.shape[2]} (text+vision alignment)"
            )
            self._verify_position_ids_mixed_vision_text(
                position_ids, N_vis, N_text, B, vision_first=False
            )
            # Ensure qwen_backbone receives consistent inputs (merged embeds, mask, position_ids)
            self._validate_qwen_backbone_inputs(combined, attn_mask, position_ids)
            if debug_shapes:
                self._debug_log_sequence_shapes(
                    visual_tokens=visual_tokens,
                    text_embeds=text_embeds,
                    combined=combined,
                    position_ids=position_ids,
                    attention_mask=attn_mask,
                    order="text_first",
                    num_visual_tokens=N_vis,
                    num_text_tokens=N_text,
                    label="fuse_text_and_vision (text_first)",
                )

            qwen_outputs = self.qwen_backbone(
                inputs_embeds=combined,
                attention_mask=attn_mask,
                position_ids=position_ids,
                return_dict=True,
                use_cache=False,
            )
            hs = qwen_outputs.last_hidden_state
            if return_full_sequence:
                return hs
            return hs[:, N_text:, :]  # visual part (text-conditioned)

        elif order == "vision_first":
            combined = torch.cat([visual_tokens, text_embeds], dim=1)  # [B, N_vis + N_text, dim]
            attn_mask = torch.cat(
                [torch.ones(B, N_vis, device=device, dtype=torch.long), text_attention_mask],
                dim=1,
            )
            position_ids = self._build_mrope_position_ids(
                num_visual_tokens=N_vis,
                num_text_tokens=N_text,
                grid_h=grid_h,
                grid_w=grid_w,
                device=device,
                vision_first=True,
            )
            position_ids = position_ids.expand(3, B, -1)
            assert combined.shape[1] == position_ids.shape[2], (
                f"Combined length vs position_ids: {combined.shape[1]} != {position_ids.shape[2]} (text+vision alignment)"
            )
            self._verify_position_ids_mixed_vision_text(
                position_ids, N_vis, N_text, B, vision_first=True
            )
            # Ensure qwen_backbone receives consistent inputs (merged embeds, mask, position_ids)
            self._validate_qwen_backbone_inputs(combined, attn_mask, position_ids)
            if debug_shapes:
                self._debug_log_sequence_shapes(
                    visual_tokens=visual_tokens,
                    text_embeds=text_embeds,
                    combined=combined,
                    position_ids=position_ids,
                    attention_mask=attn_mask,
                    order="vision_first",
                    num_visual_tokens=N_vis,
                    num_text_tokens=N_text,
                    label="fuse_text_and_vision (vision_first)",
                )

            qwen_outputs = self.qwen_backbone(
                inputs_embeds=combined,
                attention_mask=attn_mask,
                position_ids=position_ids,
                return_dict=True,
                use_cache=False,
            )
            hs = qwen_outputs.last_hidden_state
            if return_full_sequence:
                return hs
            return hs[:, :N_vis, :]  # visual part (NOT text-conditioned)

        else:
            raise ValueError(f"Unknown order={order!r}")
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        text_prompt: Optional[str] = None,
        dataset_type: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Forward pass with text conditioning.

        Supports both single-GPU and model-parallel (two-GPU) configurations.
        Device placement is detected automatically from where DINO and Qwen
        embed_tokens actually live, so no code changes are needed when switching
        between single-GPU and model-parallel modes.
        
        Args:
            pixel_values: [B, 3, H, W] input images (in [0, 1] range)
            text_prompt: Text prompt to guide the prediction
            dataset_type: Optional. One of natural_scene, webpage, e_commerce.
            
        Returns:
            output: [B, out_channels, H, W] dense prediction
        """
        B, _, H_img, W_img = pixel_values.shape

        # Auto-detect component devices so this forward works for both single-GPU
        # and model-parallel (e.g. DINO+DPT on cuda:0, Qwen on cuda:1).
        dino_device = next(self.dino.parameters()).device
        qwen_device = next(self.embed_tokens.parameters()).device

        # ===== 1. Encode text (on qwen_device, where embed_tokens lives) =====
        text_embeds, text_mask, text_len = self.encode_text(
            text_prompt or self.default_prompt, B, qwen_device, dataset_type=dataset_type
        )
        
        # ===== 2. Encode with DINO (input moved to dino_device) =====
        dino_features, patch_tokens, grid_h, grid_w = self.encode_dino(pixel_values.to(dino_device))
        
        # ===== 3. Project visual tokens (adapter + merger on qwen_device) =====
        adapted_tokens = self.dino_adapter(patch_tokens.to(qwen_device))
        visual_tokens, merged_h, merged_w = self.patch_merger(adapted_tokens, grid_h, grid_w)
        num_visual_tokens = visual_tokens.shape[1]
        
        # Safety check: verify visual tokens match grid dimensions
        assert num_visual_tokens == merged_h * merged_w, \
            f"Visual token count mismatch: {num_visual_tokens} != {merged_h}*{merged_w}={merged_h*merged_w}"
        
        # ===== 4. Fuse text and vision through LLM (on qwen_device) =====
        fused_visual = self.fuse_text_and_vision(
            visual_tokens, text_embeds, text_mask,
            grid_h=merged_h, grid_w=merged_w,
            order="vision_first",
        )
        # fused_visual: [B, N_vis, qwen_dim] on qwen_device
        
        # ===== 5. Convert to feature map for DPT (still on qwen_device) =====
        qwen_map = self._tokens_to_map(fused_visual, merged_h, merged_w)
        
        # ===== 6. Build multi-scale pyramid =====
        target_sizes = [
            self._safe_div(H_img, W_img, 4),
            self._safe_div(H_img, W_img, 8),
            self._safe_div(H_img, W_img, 16),
            self._safe_div(H_img, W_img, 32),
        ]
        
        # proj_to_l1/2/3 live on dino_device; dino_features already on dino_device
        f1 = F.interpolate(self.proj_to_l1(dino_features[0]), size=target_sizes[0], mode="bilinear", align_corners=False)
        f2 = F.interpolate(self.proj_to_l2(dino_features[1]), size=target_sizes[1], mode="bilinear", align_corners=False)
        f3 = F.interpolate(self.proj_to_l3(dino_features[2]), size=target_sizes[2], mode="bilinear", align_corners=False)
        # proj_to_l4 lives on dino_device; move qwen_map there first
        f4 = F.interpolate(self.proj_to_l4(qwen_map.to(dino_device)), size=target_sizes[3], mode="bilinear", align_corners=False)

        # Cast pyramid features to scratch layer dtype
        _scratch_param = next(self.scratch.parameters(), None)
        scratch_dtype = _scratch_param.dtype if _scratch_param is not None else self.dtype
        f1 = f1.to(dtype=scratch_dtype)
        f2 = f2.to(dtype=scratch_dtype)
        f3 = f3.to(dtype=scratch_dtype)
        f4 = f4.to(dtype=scratch_dtype)

        # ===== 7. DPT decoder on dino_device =====
        l1_rn = self.scratch.layer1_rn(f1)
        l2_rn = self.scratch.layer2_rn(f2)
        l3_rn = self.scratch.layer3_rn(f3)
        l4_rn = self.scratch.layer4_rn(f4)
        
        # Refinement (coarse to fine)
        p4 = self.scratch.refinenet4(l4_rn)
        p3 = self.scratch.refinenet3(p4, l3_rn)
        p2 = self.scratch.refinenet2(p3, l2_rn)
        p1 = self.scratch.refinenet1(p2, l1_rn)
        
        # Output
        out = self.scratch.output_conv(p1)
        
        return out
    
    def forward_vision_and_text(
        self,
        pixel_values: torch.Tensor,
        text_prompt: str,
        return_salience: bool = False,
        dataset_type: Optional[str] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass using BOTH vision and text tokens for SALIENCY CONDITIONING.
        
        Uses [vision, text] order (order="vision_first") so the sequence is vision then text.
        Generation uses [vision, text] in generate_text / get_text_generation_logits so text attends to vision.
        
        Args:
            pixel_values: [B, 3, H, W] input images (in [0, 1] range)
            text_prompt: Text prompt for conditioning
            return_salience: If True, also return salience map
            dataset_type: Optional. One of natural_scene, webpage, e_commerce.
            
        Returns:
            salience_map: [B, out_channels, H, W] dense prediction (if return_salience=True)
            combined_features: [B, N_vis + N_text, qwen_dim] (vision-first); vision at 0:N_vis
        """
        B, _, H_img, W_img = pixel_values.shape
        device = pixel_values.device
        
        # ===== 1. Encode text =====
        text_embeds, text_mask, text_len = self.encode_text(
            text_prompt or self.default_prompt, B, device, dataset_type=dataset_type
        )
        
        # ===== 2. Encode with DINO =====
        dino_features, patch_tokens, grid_h, grid_w = self.encode_dino(pixel_values)
        
        # ===== 3. Project visual tokens =====
        adapted_tokens = self.dino_adapter(patch_tokens)
        visual_tokens, merged_h, merged_w = self.patch_merger(adapted_tokens, grid_h, grid_w)
        num_visual_tokens = visual_tokens.shape[1]
        
        # Safety check: verify visual tokens match grid dimensions
        assert num_visual_tokens == merged_h * merged_w, \
            f"Visual token count mismatch: {num_visual_tokens} != {merged_h}*{merged_w}={merged_h*merged_w}"
        
        # ===== 4. Fuse text and vision through LLM (vision_first for [vision, text] sequencing) =====
        combined_features = self.fuse_text_and_vision(
            visual_tokens, text_embeds, text_mask,
            grid_h=merged_h, grid_w=merged_w,
            return_full_sequence=True,
            order="vision_first",
        )  # [B, N_vis + N_text, dim]; vision at 0:N_vis, text at N_vis:
        # Optionally return salience map
        salience_map = None
        if return_salience:
            # Extract visual tokens (before text) for salience map (vision-first)
            fused_visual = combined_features[:, :num_visual_tokens, :]  # [B, N_vis, dim]
            # Convert fused visual tokens to feature map
            qwen_map = self._tokens_to_map(fused_visual, merged_h, merged_w)
            
            target_sizes = [
                self._safe_div(H_img, W_img, 4),
                self._safe_div(H_img, W_img, 8),
                self._safe_div(H_img, W_img, 16),
                self._safe_div(H_img, W_img, 32),
            ]
            
            f1 = F.interpolate(self.proj_to_l1(dino_features[0]), size=target_sizes[0], mode="bilinear", align_corners=False)
            f2 = F.interpolate(self.proj_to_l2(dino_features[1]), size=target_sizes[1], mode="bilinear", align_corners=False)
            f3 = F.interpolate(self.proj_to_l3(dino_features[2]), size=target_sizes[2], mode="bilinear", align_corners=False)
            f4 = F.interpolate(self.proj_to_l4(qwen_map), size=target_sizes[3], mode="bilinear", align_corners=False)
            _scratch_param = next(self.scratch.parameters(), None)
            scratch_dtype = _scratch_param.dtype if _scratch_param is not None else self.dtype
            f1 = f1.to(dtype=scratch_dtype)
            f2 = f2.to(dtype=scratch_dtype)
            f3 = f3.to(dtype=scratch_dtype)
            f4 = f4.to(dtype=scratch_dtype)

            l1_rn = self.scratch.layer1_rn(f1)
            l2_rn = self.scratch.layer2_rn(f2)
            l3_rn = self.scratch.layer3_rn(f3)
            l4_rn = self.scratch.layer4_rn(f4)
            
            p4 = self.scratch.refinenet4(l4_rn)
            p3 = self.scratch.refinenet3(p4, l3_rn)
            p2 = self.scratch.refinenet2(p3, l2_rn)
            p1 = self.scratch.refinenet1(p2, l1_rn)
            
            salience_map = self.scratch.output_conv(p1)
        
        return salience_map, combined_features

    def _debug_log_sequence_shapes(
        self,
        *,
        visual_tokens: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
        combined: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        order: str = "vision_first",
        num_visual_tokens: Optional[int] = None,
        num_text_tokens: Optional[int] = None,
        label: str = "sequence",
    ) -> None:
        """
        Debug log: print shapes of visual_tokens, text_embeds, combined, position_ids,
        attention_mask. Confirm order (vision first, text second) and that visual
        tokens are not discarded. Use during testing to ensure dimensions match expectations.
        """
        print("\n" + "-"*60)
        print(f"[DEBUG] SHAPE CHECK: {label}")
        print("-"*60)
        if visual_tokens is not None:
            print(f"  visual_tokens.shape: {visual_tokens.shape}")
            nv = visual_tokens.shape[1]
            if num_visual_tokens is not None and nv != num_visual_tokens:
                print(f"    ⚠️  expected num_visual_tokens={num_visual_tokens}, got {nv}")
        if text_embeds is not None:
            print(f"  text_embeds.shape:   {text_embeds.shape}")
            nt = text_embeds.shape[1]
            if num_text_tokens is not None and nt != num_text_tokens:
                print(f"    ⚠️  expected num_text_tokens={num_text_tokens}, got {nt}")
        if combined is not None:
            print(f"  combined.shape:      {combined.shape}")
            seq_len = combined.shape[1]
            if num_visual_tokens is not None and num_text_tokens is not None:
                expected = num_visual_tokens + num_text_tokens
                if seq_len != expected:
                    print(f"    ⚠️  expected seq_len={expected} (N_vis+N_text), got {seq_len}")
                else:
                    print(f"    ✓ seq_len = N_vis + N_text = {num_visual_tokens} + {num_text_tokens} = {seq_len}")
        if position_ids is not None:
            print(f"  position_ids.shape: {position_ids.shape}")
            if combined is not None and position_ids.shape[2] != combined.shape[1]:
                print(f"    ⚠️  position_ids seq_len ({position_ids.shape[2]}) != combined ({combined.shape[1]})")
        if attention_mask is not None:
            print(f"  attention_mask.shape: {attention_mask.shape}")
            if combined is not None and attention_mask.shape[1] != combined.shape[1]:
                print(f"    ⚠️  attention_mask seq_len ({attention_mask.shape[1]}) != combined ({combined.shape[1]})")
        order_msg = "vision first, text second" if order == "vision_first" else "text first, vision second"
        print(f"  Order: {order_msg}")
        if num_visual_tokens is not None and combined is not None:
            print(f"  Visual tokens are NOT discarded: positions 0:{num_visual_tokens} in combined are vision.")
            print(f"  Text tokens: positions {num_visual_tokens}:{combined.shape[1]} in combined.")
        print("-"*60 + "\n")

    def _diagnose_visual_tokens(
        self,
        visual_tokens: torch.Tensor,
        num_visual_tokens: int,
        grid_h: int,
        grid_w: int,
        device: torch.device,
        text_embeds: Optional[torch.Tensor] = None,
    ):
        """Diagnostic: Verify visual tokens are valid and compare norms with text embeddings."""
        print("\n" + "="*70)
        print("VISUAL TOKEN DIAGNOSTICS")
        print("="*70)
        
        # Check token statistics
        vis_mean = visual_tokens.mean().item()
        vis_std = visual_tokens.std().item()
        vis_min = visual_tokens.min().item()
        vis_max = visual_tokens.max().item()
        vis_norm = visual_tokens.norm(dim=-1).mean().item()
        
        print(f"Visual Token Statistics:")
        print(f"  Shape: {visual_tokens.shape}")
        print(f"  Grid: {grid_h}x{grid_w} = {num_visual_tokens} tokens")
        print(f"  Mean: {vis_mean:.6f}")
        print(f"  Std: {vis_std:.6f}")
        print(f"  Min: {vis_min:.6f}")
        print(f"  Max: {vis_max:.6f}")
        print(f"  Avg Norm: {vis_norm:.6f}")
        
        # Compare with text embeddings if provided
        if text_embeds is not None:
            text_norm = text_embeds.norm(dim=-1).mean().item()
            norm_ratio = text_norm / vis_norm if vis_norm > 0 else float('inf')
            
            print(f"\nText Embedding Statistics:")
            print(f"  Shape: {text_embeds.shape}")
            print(f"  Avg Norm: {text_norm:.6f}")
            print(f"  Norm Ratio (text/vision): {norm_ratio:.2f}x")
            
            if norm_ratio > 3.0:
                print(f"  ⚠️  WARNING: Text embeddings are {norm_ratio:.1f}x larger than visual tokens!")
                print(f"     This may cause attention to ignore visual tokens.")
                print(f"     Consider normalizing visual tokens to match text embedding scale.")
            elif norm_ratio < 0.5:
                print(f"  ⚠️  WARNING: Visual tokens are {1/norm_ratio:.1f}x larger than text embeddings!")
                print(f"     This may cause attention to ignore text tokens.")
            else:
                print(f"  ✓ Norms are reasonably matched (ratio: {norm_ratio:.2f}x)")
        
        # Check for zero or near-zero tokens (bad sign)
        zero_threshold = 1e-6
        near_zero_mask = (visual_tokens.norm(dim=-1) < zero_threshold)
        num_near_zero = near_zero_mask.sum().item()
        zero_pct = 100 * num_near_zero / num_visual_tokens
        
        if num_near_zero > 0:
            print(f"  ⚠️  WARNING: {num_near_zero}/{num_visual_tokens} tokens ({zero_pct:.1f}%) are near-zero!")
            print(f"     This suggests visual encoding may be broken.")
        else:
            print(f"  ✓ All {num_visual_tokens} visual tokens have non-zero embeddings")
        
        # Check for NaN or Inf
        has_nan = torch.isnan(visual_tokens).any().item()
        has_inf = torch.isinf(visual_tokens).any().item()
        
        if has_nan:
            print(f"  ❌ ERROR: Visual tokens contain NaN values!")
        if has_inf:
            print(f"  ❌ ERROR: Visual tokens contain Inf values!")
        if not has_nan and not has_inf:
            print(f"  ✓ No NaN or Inf values in visual tokens")
        
        print("="*70 + "\n")
    
    def _setup_attention_hooks(self):
        """Set up hooks to capture attention weights during generation."""
        hooks = []
        attention_data = {
            'layer_attentions': [],
            'layer_names': [],
        }
        
        def make_attention_hook(layer_idx, layer_name):
            def attention_hook(module, input, output):
                attn_weights = None
                
                if isinstance(output, tuple):
                    for item_idx, item in enumerate(output):
                        if isinstance(item, torch.Tensor):
                            if item.dim() == 4:
                                if item.shape[-1] == item.shape[-2] or item.shape[-1] <= item.shape[-2]:
                                    attn_weights = item
                                    break
                
                if attn_weights is None and hasattr(module, '_attn_weights'):
                    attn_weights = module._attn_weights
                
                if attn_weights is None and hasattr(module, 'attn_weights'):
                    attn_weights = module.attn_weights
                
                if attn_weights is not None:
                    if attn_weights.dim() == 4 and attn_weights.shape[-1] > 0:
                        attention_data['layer_attentions'].append(attn_weights.detach().cpu())
                        attention_data['layer_names'].append(f"{layer_name}_layer_{layer_idx}")
            
            return attention_hook
        

        if hasattr(self, 'qwen_layers'):
            for idx, layer in enumerate(self.qwen_layers):
                if hasattr(layer, 'self_attn'):
                    hook = layer.self_attn.register_forward_hook(
                        make_attention_hook(idx, "qwen")
                    )
                    hooks.append(hook)
        
        return hooks, attention_data
    
    def _analyze_attention_patterns(
        self,
        attention_data: dict,
        num_visual_tokens: int,
        num_prompt_tokens: int,
        generated_text: str,
    ):
        """
        Analyze attention patterns to verify how well the model attends to visual tokens.
        Attention hooks and this analysis give useful insight: prompt→vision attention
        should be non-negligible if the model is using visual information.
        """
        print("\n" + "="*70)
        print("ATTENTION PATTERN ANALYSIS (how well the model attends to visual tokens)")
        print("="*70)
        print("  Use these logs to check whether prompt tokens attend to vision tokens.")
        print("  Higher prompt→vision attention suggests the model is using visual info.")
        print("-"*70)

        if not attention_data.get('layer_attentions'):
            print("  ⚠️  No attention weights captured")
            print("  This is EXPECTED if Qwen uses FlashAttention/SDPA (fused kernels)")
            print("  that don't materialize full attention matrices to save memory.")
            print("  Even with output_attentions=True, FlashAttention may not return weights.")
            print("  The vision impact test below confirms vision is being used.")
            print("="*70 + "\n")
            return

        print(f"Generated text length: {len(generated_text)} chars")
        print(f"Generated text preview: {generated_text[:100]}...")
        print(f"\nVision tokens: {num_visual_tokens}, Prompt tokens: {num_prompt_tokens}")
        print(f"Total prefix length: {num_visual_tokens + num_prompt_tokens}")
        print(f"  (Order: positions 0:{num_visual_tokens}=vision, {num_visual_tokens}:{num_visual_tokens+num_prompt_tokens}=prompt)")

        # Analyze attention from prompt tokens to vision tokens
        all_mean_attn = []
        for layer_idx, (attn_weights, layer_name) in enumerate(
            zip(attention_data['layer_attentions'], attention_data['layer_names'])
        ):
            # attn_weights: [B, num_heads, seq_len, seq_len]
            if attn_weights.dim() != 4:
                continue
            
            B, num_heads, seq_len, _ = attn_weights.shape
            
            # Focus on first batch item and average over heads
            attn = attn_weights[0].mean(dim=0)  # [seq_len, seq_len]
            
            # Extract attention from prompt tokens (after vision) to vision tokens
            prompt_start = num_visual_tokens
            prompt_end = num_visual_tokens + num_prompt_tokens
            
            if prompt_end <= seq_len:
                # Attention from prompt tokens to vision tokens
                prompt_to_vision = attn[prompt_start:prompt_end, :num_visual_tokens]  # [N_prompt, N_vis]
                
                # Average attention per prompt token to vision tokens
                avg_attention_to_vision = prompt_to_vision.mean(dim=1)  # [N_prompt]
                max_attention_to_vision = prompt_to_vision.max(dim=1)[0]  # [N_prompt]
                
                # Overall statistics
                mean_attn = avg_attention_to_vision.mean().item()
                max_attn = max_attention_to_vision.max().item()
                min_attn = avg_attention_to_vision.min().item()
                
                print(f"\n{layer_name}:")
                print(f"  Mean attention (prompt→vision): {mean_attn:.4f}")
                print(f"  Max attention (prompt→vision): {max_attn:.4f}")
                print(f"  Min attention (prompt→vision): {min_attn:.4f}")
                
                # Check if attention is significant
                if mean_attn > 0.01:  # 1% average attention
                    print(f"  ✓ Prompt tokens ARE attending to vision tokens (mean={mean_attn:.4f})")
                else:
                    print(f"  ⚠️  WARNING: Low attention to vision tokens (mean={mean_attn:.4f})")
                    print(f"     Model may be ignoring visual information!")

                all_mean_attn.append(mean_attn)

                # Check attention distribution
                high_attn_count = (avg_attention_to_vision > 0.05).sum().item()
                high_attn_pct = 100 * high_attn_count / num_prompt_tokens
                print(f"  Prompt tokens with >5% vision attention: {high_attn_count}/{num_prompt_tokens} ({high_attn_pct:.1f}%)")

        if all_mean_attn:
            overall_mean = sum(all_mean_attn) / len(all_mean_attn)
            overall_pct = 100 * overall_mean
            print("\n  --- Visual attention summary ---")
            print(f"  Mean prompt→vision attention across layers: {overall_mean:.4f} ({overall_pct:.2f}%)")
            if overall_mean > 0.01:
                print(f"  ✓ Attention hooks indicate the model IS attending to visual tokens.")
            else:
                print(f"  ⚠️  Low overall attention to vision; check that visual tokens are not discarded and order is correct.")

        print("="*70 + "\n")
    
    def _compare_with_zeroed_vision(
        self,
        pixel_values: torch.Tensor,
        text_prompt: str,
        original_text: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        generation_kwargs: dict,
    ):
        """Compare generation with and without visual tokens to verify vision is being used."""
        print("\n" + "="*70)
        print("VISION TOKEN IMPACT TEST")
        print("="*70)
        print("Testing if model output changes when visual tokens are zeroed...")
        
        try:
            # Generate with zeroed visual tokens (by zeroing the image)
            zeroed_image = torch.zeros_like(pixel_values)
            
            # Temporarily disable diagnostics to avoid recursion
            text_without_vision = self.generate_text(
                zeroed_image,
                text_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                enable_attention_diagnostics=False,  # Disable to avoid recursion
                **generation_kwargs,
            )
            
            # Compare outputs
            print(f"\nOriginal (with vision):")
            print(f"  {original_text[:200]}...")
            print(f"\nWithout vision (zeroed image):")
            print(f"  {text_without_vision[:200]}...")
            
            # Simple similarity check
            if original_text == text_without_vision:
                print(f"\n  ❌ CRITICAL: Outputs are IDENTICAL!")
                print(f"     Model is NOT using visual information - it's ignoring vision tokens!")
            else:
                # Calculate simple word overlap
                orig_words = set(original_text.lower().split())
                zeroed_words = set(text_without_vision.lower().split())
                overlap = len(orig_words & zeroed_words)
                total_unique = len(orig_words | zeroed_words)
                similarity = overlap / total_unique if total_unique > 0 else 0.0
                
                print(f"\n  Word overlap: {overlap}/{total_unique} ({similarity*100:.1f}%)")
                
                if similarity > 0.8:
                    print(f"  ⚠️  WARNING: High similarity ({similarity*100:.1f}%)")
                    print(f"     Model may not be using visual information effectively")
                else:
                    print(f"  ✓ Outputs differ significantly (similarity={similarity*100:.1f}%)")
                    print(f"     Model appears to be using visual information")
        
        except Exception as e:
            print(f"  ⚠️  Could not perform comparison test: {e}")
        
        print("="*70 + "\n")
    
    def generate_text_only(
        self,
        text_prompt: Optional[str] = None,
        max_new_tokens: int = 128,
        do_sample: bool = False,
    ) -> str:
        """
        Text-only generation (no visual prefix). Sanity check for decoding stack:
        backbone + norm + lm_head. Uses same path as generate_text() but with no image.
        If this produces normal text → tokenizer/LM head are fine; visual prefix is the cause.
        If this degenerates → head/norm mismatch in manual decoding path.
        """
        device = next(self.parameters()).device
        prompt = text_prompt if text_prompt is not None else self.default_prompt
        formatted_prompt = self._apply_chat_template(prompt)
        tok = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        prompt_embeds = self.embed_tokens(tok["input_ids"])  # [1, N, D]
        B, N, _ = prompt_embeds.shape
        attn_mask = tok["attention_mask"]  # [1, N]
        # M-RoPE position_ids for text-only: sequential 0..N-1 on all 3 axes
        pos_1d = torch.arange(N, device=device, dtype=torch.long)
        position_ids = torch.stack([pos_1d, pos_1d, pos_1d], dim=0).unsqueeze(1)  # [3, 1, N]
        
        def hidden_to_logits(h):
            return self._apply_lm_head(h)
        
        was_training = self.training
        self.eval()
        with torch.no_grad():
            out = self.qwen_backbone(
                inputs_embeds=prompt_embeds,
                attention_mask=attn_mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
            past = getattr(out, "past_key_values", None)
            if past is None:
                raise RuntimeError(
                    "qwen_backbone did not return past_key_values despite use_cache=True. "
                    "Autoregressive generation requires the cache to be maintained and passed at each step."
                )
            hidden_last = out.last_hidden_state[:, -1, :]
            pos_val = N
            eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.pad_token_id
            generated_ids = []
            for _ in range(max_new_tokens):
                logits = hidden_to_logits(hidden_last)
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
                generated_ids.append(next_id)
                if next_id.item() == eos_token_id:
                    break
                next_embed = self.embed_tokens(next_id)
                attn_mask = torch.cat([attn_mask, torch.ones(B, 1, device=device, dtype=attn_mask.dtype)], dim=1)
                next_pos = torch.full((3, B, 1), pos_val, device=device, dtype=torch.long)
                pos_val += 1
                out = self.qwen_backbone(
                    inputs_embeds=next_embed,
                    attention_mask=attn_mask,
                    position_ids=next_pos,
                    past_key_values=past,  # cache for all previous tokens
                    use_cache=True,
                    return_dict=True,
                )
                past = getattr(out, "past_key_values", None)
                if past is None:
                    raise RuntimeError(
                        "qwen_backbone did not return past_key_values in autoregressive step. "
                        "Cache must be maintained and passed at each step."
                    )
                hidden_last = out.last_hidden_state[:, -1, :]
        if was_training:
            self.train()
        if not generated_ids:
            return ""
        gen = torch.cat(generated_ids, dim=1)[0].tolist()
        return self.tokenizer.decode(gen, skip_special_tokens=True)
    
    def generate_text(
        self,
        pixel_values: torch.Tensor,
        text_prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.5,
        do_sample: bool = True,
        enable_attention_diagnostics: bool = False,
        dataset_type: Optional[str] = None,
        **generation_kwargs,
    ) -> str:
        """
        Generate text explanation conditioned on vision.
        
        Defaults are tuned for structured grounded descriptions (low temperature,
        small top_p). Use do_sample=False for greedy decoding.
        
        ⚠️  IMPORTANT: This model was NOT trained for text generation!
        
        The LLM is frozen during training and only used for visual-text fusion.
        Text generation is EXPERIMENTAL and may produce gibberish or repetitive output.
        
        For meaningful text generation, you would need to:
        1. Unfreeze the LLM during training (freeze_qwen_lm=False)
        2. Add a text generation loss (cross-entropy on generated text)
        3. Train with both saliency and text generation objectives
        
        FIXED VERSION: Uses proper token embeddings and correct token order.
        
        For causal generation, tokens must be ordered as [visual, prompt] so that
        the prompt can attend to visual tokens. We use pre-transformer token embeddings
        (not post-transformer hidden states) as inputs_embeds.
        
        Args:
            pixel_values: [B, 3, H, W] input images (in [0, 1] range)
            text_prompt: Text prompt (should match training prompt style)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (default 0.2 for structured output; use 0.1-0.3)
            top_p: Nucleus sampling (default 0.5; use small value for grounded descriptions)
            do_sample: If False, use greedy decoding; if True, use temperature/top_p
            **generation_kwargs: Additional generation arguments
            
        Returns:
            Generated text string (may be gibberish if model wasn't trained for generation)
        """
        B = pixel_values.shape[0]
        device = pixel_values.device
        
        # ===== 1. Get visual tokens (pre-transformer embeddings) =====
        dino_features, patch_tokens, grid_h, grid_w = self.encode_dino(pixel_values)
        adapted_tokens = self.dino_adapter(patch_tokens)
        visual_tokens, merged_h, merged_w = self.patch_merger(adapted_tokens, grid_h, grid_w)
        num_visual_tokens = visual_tokens.shape[1]  # [B, N_vis, qwen_dim]
        
        # Safety check: verify visual tokens match grid dimensions
        assert num_visual_tokens == merged_h * merged_w, \
            f"Visual token count mismatch: {num_visual_tokens} != {merged_h}*{merged_w}={merged_h*merged_w}"
        
        # ===== 2. Get prompt token embeddings (pre-transformer) =====
        # Apply chat template for consistent formatting with training (dataset_type for system+user)
        formatted_prompt = self._apply_chat_template(text_prompt, dataset_type=dataset_type)
        
        prompt_tokens = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=False,  # Don't pad - we'll handle sequence length
            truncation=True,
            max_length=self.max_text_length,
            add_special_tokens=False,  # template already has special tokens
        )
        prompt_input_ids = prompt_tokens["input_ids"].to(device)  # [1, prompt_len]
        prompt_attention_mask = prompt_tokens["attention_mask"].to(device)
        N_prompt = prompt_input_ids.shape[1]  # Use shape, not sum (no padding)
        
        # Embed prompt tokens (pre-transformer embeddings)
        prompt_embeds = self.embed_tokens(prompt_input_ids)  # [1, N_prompt, qwen_dim]
        
        # Expand to batch size for batch-safe concatenation
        prompt_embeds = prompt_embeds.expand(B, -1, -1)  # [B, N_prompt, qwen_dim]
        prompt_attention_mask = prompt_attention_mask.expand(B, -1)  # [B, N_prompt]
        # Alias: text token embeddings for the prefix (prompt); used so order is explicit below
        text_embeds = prompt_embeds

        # Verify visual and text token alignment before building prefix
        self._check_visual_text_token_alignment(visual_tokens, text_embeds, merged_h, merged_w)

        # ===== DIAGNOSTICS: Verify visual tokens and compare with text embeddings =====
        if enable_attention_diagnostics:
            self._diagnose_visual_tokens(
                visual_tokens, num_visual_tokens, merged_h, merged_w, device,
                text_embeds=prompt_embeds
            )

        # ===== 4. Concatenate: [Vision, Text] — visual tokens FIRST =====
        # Ensure visual tokens (DINOv3) come first so text tokens can attend to them.
        # Order: [visual_tokens, text_embeds] → [B, N_vis + N_prompt, qwen_dim]
        # When using device_map, adapter/merger may be on CPU; align to text (embed) device for cat.
        target_device = text_embeds.device
        visual_tokens = visual_tokens.to(device=target_device, dtype=text_embeds.dtype)
        generation_prefix = torch.cat([visual_tokens, text_embeds], dim=1)
        total_len = generation_prefix.shape[1]
        assert total_len == num_visual_tokens + N_prompt, (
            f"Prefix length {total_len} != num_visual_tokens + N_prompt ({num_visual_tokens} + {N_prompt})"
        )

        # ===== 5. Build M-RoPE position_ids for [vision, text] sequence =====
        # position_ids must match the concatenated order: first num_visual_tokens positions
        # are vision (spatial grid), next N_prompt positions are text (sequential).
        # vision_first=True builds IDs for [vision, text] so they correspond correctly.
        prefix_position_ids = self._build_mrope_position_ids(
            num_visual_tokens=num_visual_tokens,
            num_text_tokens=N_prompt,
            grid_h=merged_h,
            grid_w=merged_w,
            device=device,
            vision_first=True,  # matches [visual_tokens, text_embeds] order
        )
        prefix_position_ids = prefix_position_ids.expand(3, B, -1)  # [3, B, total_len]
        self._verify_position_ids_mixed_vision_text(
            prefix_position_ids, num_visual_tokens, N_prompt, B, vision_first=True
        )
        # Verify position_ids correspond to concatenated sequence length
        assert prefix_position_ids.shape[2] == total_len, (
            f"position_ids seq_len ({prefix_position_ids.shape[2]}) must equal "
            f"len(visual_tokens)+len(text_embeds) = {total_len}"
        )

        # ===== 6. Create attention_mask for concatenated [vision, text] sequence =====
        # One position per token; all visible for prefix. Must match generation_prefix length.
        prefix_attention_mask = torch.ones(B, total_len, device=device, dtype=torch.long)
        assert prefix_attention_mask.shape[1] == total_len, (
            f"attention_mask seq_len ({prefix_attention_mask.shape[1]}) must equal prefix length {total_len}"
        )

        # ===== 7. Dummy Input IDs (Must match total_len exactly) =====
        dummy_input_ids = torch.zeros((B, total_len), dtype=torch.long, device=device)
        
        # ===== DIAGNOSTICS: Set up attention hooks if enabled =====
        attention_hooks = []
        attention_data = {}
        if enable_attention_diagnostics:
            attention_hooks, attention_data = self._setup_attention_hooks()
        
        try:
            # ===== 8. Generate using inputs_embeds with CORRECT alignment =====
            # Debug logs: shapes and order (vision first, text second); confirm visual tokens not discarded
            if enable_attention_diagnostics:
                self._debug_log_sequence_shapes(
                    visual_tokens=visual_tokens,
                    text_embeds=text_embeds,
                    combined=generation_prefix,
                    position_ids=prefix_position_ids,
                    attention_mask=prefix_attention_mask,
                    order="vision_first",
                    num_visual_tokens=num_visual_tokens,
                    num_text_tokens=N_prompt,
                    label="generate_text prefix (vision first, text second)",
                )
            if enable_attention_diagnostics:
                print(f"\n[DEBUG] Generation input shapes (verifying alignment):")
                print(f"  generation_prefix shape: {generation_prefix.shape}")
                print(f"  dummy_input_ids shape: {dummy_input_ids.shape}")
                print(f"  prefix_attention_mask shape: {prefix_attention_mask.shape}")
                print(f"  prefix_position_ids shape: {prefix_position_ids.shape}")
                print(f"  total_len (expected sequence length): {total_len}")
                print(f"  Breakdown: num_visual_tokens={num_visual_tokens}, N_prompt={N_prompt}")
                print(f"  Visual grid: {merged_h}x{merged_w} = {merged_h*merged_w} tokens")
            
            # CRITICAL VALIDATION: generation_prefix = [visual_tokens, text_embeds]; position_ids
            # and attention_mask must correspond to this concatenated sequence (length = N_vis + N_prompt).
            assert generation_prefix.shape[1] == total_len, f"generation_prefix seq_len ({generation_prefix.shape[1]}) != total_len ({total_len})"
            assert dummy_input_ids.shape[1] == total_len, f"dummy_input_ids seq_len ({dummy_input_ids.shape[1]}) != total_len ({total_len})"
            assert prefix_attention_mask.shape[1] == total_len, f"prefix_attention_mask seq_len ({prefix_attention_mask.shape[1]}) != total_len ({total_len})"
            assert prefix_position_ids.shape[2] == total_len, f"prefix_position_ids seq_len ({prefix_position_ids.shape[2]}) != total_len ({total_len})"
            assert prefix_position_ids.shape[0] == 3, f"prefix_position_ids must have 3 axes for M-RoPE, got {prefix_position_ids.shape[0]}"
            if enable_attention_diagnostics:
                print(f"  ✓ All tensor shapes are aligned to {total_len} tokens (visual first, then text)")
                print(f"  ✓ Visual tokens are NOT discarded: positions 0:{num_visual_tokens} = vision; {num_visual_tokens}:{total_len} = text.")
            
            
            # Helper function for top-p sampling
            def _sample_top_p(logits, top_p=0.9):
                # logits: [B, V]
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cum = torch.cumsum(sorted_probs, dim=-1)
                # Keep at least 1 token
                cutoff = cum > top_p
                cutoff[..., 0] = False
                sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_in_sorted = torch.multinomial(sorted_probs, num_samples=1)  # [B,1]
                next_token = sorted_idx.gather(-1, next_in_sorted)  # [B,1]
                return next_token
            
            # Get logits from last hidden state: norm then lm_head (same as _apply_lm_head)
            # No masking on hidden state; repetition penalty is applied to logits only
            def hidden_to_logits(h):
                return self._apply_lm_head(h)  # [B, D] -> [B, V]

            # Set model to eval mode for generation
            was_training = self.qwen_full_model.training
            self.qwen_full_model.eval()
            
            # 1) Prime the cache with the full prefix (your total_len tokens)
            # Ensure qwen_backbone receives merged [visual, text] with matching mask and position_ids
            self._validate_qwen_backbone_inputs(
                generation_prefix, prefix_attention_mask, prefix_position_ids
            )
            # Pass output_attentions=True to enable attention weight capture by hooks
            with torch.no_grad():
                out = self.qwen_backbone(
                    inputs_embeds=generation_prefix,  # [B, total_len, D]
                    attention_mask=prefix_attention_mask,  # [B, total_len]
                    position_ids=prefix_position_ids,  # [3, B, total_len]
                    use_cache=True,
                    output_attentions=enable_attention_diagnostics,  # Enable attention weights for diagnostics
                    return_dict=True,
                )
                past = getattr(out, "past_key_values", None)
                # Cache must be returned for autoregressive decoding; without it each step would recompute the full prefix
                if past is None:
                    raise RuntimeError(
                        "qwen_backbone did not return past_key_values despite use_cache=True. "
                        "Autoregressive generation requires the cache to be maintained and passed at each step."
                    )
                hidden_last = out.last_hidden_state[:, -1, :]  # [B, D] - last token's hidden state

                # If output_attentions=True, attention weights are in out.attentions
                # Store them in attention_data for analysis
                if enable_attention_diagnostics and hasattr(out, 'attentions') and out.attentions is not None:
                    # out.attentions is a tuple of attention weights from each layer
                    # Each element is [B, num_heads, seq_len, seq_len]
                    for layer_idx, layer_attn in enumerate(out.attentions):
                        if layer_attn is not None:
                            attention_data['layer_attentions'].append(layer_attn.detach().cpu())
                            attention_data['layer_names'].append(f"qwen_layer_{layer_idx}")
            
            # Current position value for next token: increment after last prefix position
            # (text positions are sequential and identical across the 3 axes for text)
            pos_val = int(prefix_position_ids[0, 0, -1].item()) + 1
            
            generated_ids = []
            min_tokens_generated = 16  # Minimum tokens to generate
            eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.pad_token_id
            
            # Track recent tokens for repetition detection
            recent_token_window = 10  # Check last N tokens for repetition

            # 2) Autoregressive generation loop
            # Invariant: past_key_values caches K/V for positions [0 .. total_len+step-1].
            # Each step: pass the new token's inputs_embeds and position_ids; backbone returns
            # updated past (including the new token) and last_hidden_state for the new token.
            # We maintain and pass the cache correctly so the model does not recompute the prefix.
            for step in range(max_new_tokens):
                # Get logits for next token
                logits = hidden_to_logits(hidden_last)  # [B, vocab_size]
                
                # Check for degenerate logits (all zeros or NaN)
                if torch.isnan(logits).any() or (logits == 0).all():
                    print(f"  ⚠️  Stopping generation: Degenerate logits detected at step {step}")
                    break
                
                # Only apply entropy early-stop in sampling mode (greedy decoding is often low-entropy and fine)
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                    if entropy.item() < 0.01:  # Very low entropy suggests degenerate output when sampling
                        next_id_debug = logits.argmax(dim=-1).item()
                        try:
                            tok_str = self.tokenizer.decode([next_id_debug])
                        except Exception:
                            tok_str = "<decode failed>"
                        print(f"  ⚠️  Stopping generation: Very low entropy ({entropy.item():.4f}) at step {step} (argmax id={next_id_debug}, token={repr(tok_str)})")
                        break
                
                # Apply repetition penalty to recently generated tokens.
                # Applied in BOTH greedy and sampling modes — greedy decoding
                # is especially prone to repetition loops without this penalty.
                if len(generated_ids) > 0:
                    # Use a larger window (50 tokens) for better repetition coverage
                    recent_tokens = [g.item() for g in generated_ids[-50:]]
                    token_counts = {}
                    for token_id in recent_tokens:
                        token_counts[token_id] = token_counts.get(token_id, 0) + 1
                    for token_id, count in token_counts.items():
                        if token_id < logits.shape[-1]:
                            penalty = 0.7 if count == 1 else (0.4 if count == 2 else 0.1)
                            logits[0, token_id] = logits[0, token_id] * penalty
                
                # Sample next token
                if not do_sample or temperature is None or temperature <= 0:
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)  # [B,1]
                else:
                    logits = logits / float(temperature)
                    next_id = _sample_top_p(logits, top_p=top_p)  # [B,1]
                
                generated_ids.append(next_id)
                min_tokens_generated -= 1
                
                # Stop on EOS after minimum tokens generated
                if next_id.shape[0] == 1:
                    token_id = next_id.item()
                    if token_id == eos_token_id and min_tokens_generated <= 0:
                        break
                    
                    # Enhanced repetition detection: check for multiple patterns
                    if len(generated_ids) >= 3:
                        # Pattern 1: Same token repeated 2+ times consecutively (stricter)
                        recent_ids = [g.item() for g in generated_ids[-3:]]
                        if len(recent_ids) >= 2 and len(set(recent_ids)) == 1:
                            # Decode to see what token this is
                            try:
                                token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                                print(f"  ⚠️  Stopping generation: Same token repeated 2+ times (token ID {token_id}, text: '{token_text}')")
                            except:
                                print(f"  ⚠️  Stopping generation: Same token repeated 2+ times (token ID {token_id})")
                            break
                        
                        # Pattern 2: phrase-level repetition over the FULL generated history
                        if len(generated_ids) >= 20:
                            try:
                                full_gen = torch.cat(generated_ids, dim=1)
                                full_text = self.tokenizer.decode(full_gen[0], skip_special_tokens=True)
                                words = full_text.split()
                                stop_loop = False
                                # Check if any 4-word phrase appears 3+ times
                                if len(words) >= 12:
                                    phrase_counts: dict = {}
                                    for i in range(len(words) - 3):
                                        phrase = " ".join(words[i:i+4])
                                        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
                                        if phrase_counts[phrase] >= 3 and phrase.strip():
                                            print(f"  ⚠️  Stopping: phrase repeated 3x: '{phrase}'")
                                            stop_loop = True
                                            break
                                if stop_loop:
                                    break
                            except Exception:
                                pass
                
                # Embed the next token
                next_embed = self.embed_tokens(next_id)  # [B,1,D]
                
                # Update attention mask length (+1)
                prefix_attention_mask = torch.cat(
                    [prefix_attention_mask, torch.ones((B, 1), device=device, dtype=torch.long)],
                    dim=1
                )  # [B, total_len+step+1]
                
                # Position ids for THIS new token only: [3,B,1]
                # For text tokens, all 3 axes use the same sequential position
                next_pos = torch.full((3, B, 1), pos_val, device=device, dtype=torch.long)
                pos_val += 1
                
                # One-step decode: pass only the new token; cache carries the rest
                out = self.qwen_backbone(
                    inputs_embeds=next_embed,  # [B,1,D]
                    attention_mask=prefix_attention_mask,  # [B, total_len+step+1] full context length
                    position_ids=next_pos,  # [3,B,1] position of this new token
                    past_key_values=past,  # cache for positions 0..total_len+step-1
                    use_cache=True,
                    output_attentions=enable_attention_diagnostics,
                    return_dict=True,
                )
                past = getattr(out, "past_key_values", None)
                if past is None:
                    raise RuntimeError(
                        f"qwen_backbone did not return past_key_values at step {step}. "
                        "Cache must be maintained and passed at each autoregressive step."
                    )
                hidden_last = out.last_hidden_state[:, -1, :]  # [B,D] hidden state for the new token

            # Restore training mode
            if was_training:
                self.qwen_full_model.train()
            
            # Decode only the newly generated ids
            if len(generated_ids) == 0:
                return "[Empty decode - no tokens generated]"
            
            gen = torch.cat(generated_ids, dim=1)  # [B, T]
            
            # Filter out pad tokens and other special tokens before decoding
            # Convert to list and filter
            gen_list = gen[0].tolist()
            # Remove pad tokens and other unwanted tokens
            filtered_gen = [tid for tid in gen_list if tid != self.tokenizer.pad_token_id]
            
            if len(filtered_gen) == 0:
                return "[Empty decode - no valid tokens generated]"
            
            # Decode with skip_special_tokens to avoid special token artifacts
            generated_text = self.tokenizer.decode(filtered_gen, skip_special_tokens=True)
            
            # Enhanced cleanup: remove excessive repetition patterns
            words = generated_text.split()
            if len(words) > 0:
                cleaned_words = [words[0]]
                for i, word in enumerate(words[1:], 1):
                    # Pattern 1: Don't add if it's the same as the last 2 words (prevent triple repetition)
                    if len(cleaned_words) >= 2 and word == cleaned_words[-1] == cleaned_words[-2]:
                        continue
                    
                    # Pattern 2: Check for phrase-level repetition (same 3-word phrase)
                    if len(cleaned_words) >= 6:
                        recent_phrase = " ".join(cleaned_words[-3:])
                        # Check if current word + next 2 words would repeat the phrase
                        if i + 1 < len(words):
                            potential_phrase = " ".join([word] + words[i+1:i+3] if i+2 < len(words) else [word])
                            if potential_phrase == recent_phrase:
                                # Skip this word to break the repetition
                                continue
                    
                    cleaned_words.append(word)
                generated_text = " ".join(cleaned_words)
                
                # Additional cleanup: Remove "Unanswerable:" prefix if it's just a prefix
                if generated_text.startswith("Unanswerable:") and len(generated_text.split()) <= 5:
                    # If the whole text is just "Unanswerable: ..." with few words, it might be a mistake
                    generated_text = generated_text.replace("Unanswerable:", "").strip()
            
            # ===== DIAGNOSTICS: Analyze attention patterns =====
            if enable_attention_diagnostics:
                self._analyze_attention_patterns(
                    attention_data, num_visual_tokens, N_prompt, generated_text
                )
                # Remove hooks after analysis
                for hook in attention_hooks:
                    hook.remove()
                
                # Additional diagnostic: Compare with zeroed visual tokens
                self._compare_with_zeroed_vision(
                    pixel_values, text_prompt, generated_text,
                    max_new_tokens, temperature, top_p, do_sample,
                    generation_kwargs
                )
            
            # Return decoded text or fallback message if empty
            return generated_text.strip() if generated_text.strip() else "[Empty decode]"
            
        except Exception as e:
            print(f"  [DEBUG] Vision-conditioned generation failed: {e}")
            import traceback
            # Only print first few lines of traceback to avoid spam
            tb_lines = traceback.format_exc().splitlines()
            for line in tb_lines[:10]:
                print(f"    {line}")
            
            # Fallback: Try with input_ids only (no vision conditioning)
            try:
                print("  [DEBUG] Trying fallback method (NO VISION)...")
                generation_outputs = self.qwen_full_model.generate(
                    input_ids=prompt_input_ids,
                    attention_mask=prompt_attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=16,  # Enforce minimum generation
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **generation_kwargs,
                )
                
                # Robust decoding for fallback too
                out_ids = generation_outputs[0]
                if len(out_ids) <= N_prompt:
                    new_ids = out_ids
                else:
                    new_ids = out_ids[N_prompt:]
                
                if len(new_ids) == 0:
                    return "[Empty decode - no tokens generated]"
                
                generated_text = self.tokenizer.decode(
                    new_ids,
                    skip_special_tokens=True
                )
                return generated_text.strip() if generated_text.strip() else "[Empty decode]"
            except Exception as e2:
                print(f"  Fallback also failed: {e2}")
                return "[Text generation failed - model may not be trained for text generation]"
    
    def get_text_generation_logits(
        self,
        pixel_values: torch.Tensor,
        text_prompt: str,
        target_text: Optional[str] = None,
        debug_shapes: bool = False,
        dataset_type: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get logits for text generation (used during training).
        
        This method supports two modes:
        1. Prompt-only: Returns logits for prompt tokens (for backward compatibility)
        2. Prompt+Target: Returns logits for full sequence (prompt + target) for proper teacher forcing
        
        Args:
            pixel_values: [B, 3, H, W] input images (in [0, 1] range)
            text_prompt: Text prompt for conditioning
            target_text: Optional target text to append (for teacher forcing)
            debug_shapes: If True, print shapes of combined, position_ids, mask (for testing)
            dataset_type: Optional. One of natural_scene, webpage, e_commerce (uses type-specific system+user prompts).
            
        Returns:
            logits: [B, seq_len, vocab_size] logits for tokens
            labels: [B, seq_len] token IDs for loss computation (with -100 for prompt tokens if target provided)
            prompt_length: Length of prompt tokens (for masking)
        """
        B = pixel_values.shape[0]
        device = pixel_values.device
        # Ensure model dtype so DINO/adapter/backbone path is consistent (e.g. verification without autocast)
        pixel_values = pixel_values.to(device=device, dtype=self.dtype)

        if target_text is None:
            # Prompt-only logits (VISION-CONDITIONED): build [vision, text] so text can attend to vision.
            # This aligns prompt-only logits with the teacher-forcing/generation ordering.
            formatted_prompt = self._apply_chat_template(text_prompt, dataset_type=dataset_type)
            prompt_tokens = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=False,  # NO PADDING
                truncation=True,
                max_length=self.max_text_length,
                add_special_tokens=False,  # template already has special tokens
            )
            prompt_input_ids = prompt_tokens["input_ids"].to(device)  # [1, L]
            prompt_attention_mask = prompt_tokens["attention_mask"].to(device)  # [1, L]
            if B > 1:
                prompt_input_ids = prompt_input_ids.expand(B, -1)
                prompt_attention_mask = prompt_attention_mask.expand(B, -1)
            N_prompt = prompt_input_ids.shape[1]

            # Get visual tokens
            dino_features, patch_tokens, grid_h, grid_w = self.encode_dino(pixel_values)
            adapted_tokens = self.dino_adapter(patch_tokens)
            visual_tokens, merged_h, merged_w = self.patch_merger(adapted_tokens, grid_h, grid_w)
            num_visual_tokens = visual_tokens.shape[1]
            assert num_visual_tokens == merged_h * merged_w, \
                f"Visual token count mismatch: {num_visual_tokens} != {merged_h}*{merged_w}={merged_h*merged_w}"

            # Embed prompt tokens
            prompt_embeds = self.embed_tokens(prompt_input_ids)  # [B, N_prompt, qwen_dim]

            # Cast to backbone dtype
            backbone_dtype = next(self.qwen_backbone.parameters()).dtype
            visual_tokens = visual_tokens.to(dtype=backbone_dtype)
            prompt_embeds = prompt_embeds.to(dtype=backbone_dtype)

            # Concatenate: [visual, text]
            combined = torch.cat([visual_tokens, prompt_embeds], dim=1)  # [B, N_vis + N_prompt, dim]
            N_total = num_visual_tokens + N_prompt

            # Position ids for [vision, text]
            position_ids = self._build_mrope_position_ids(
                num_visual_tokens=num_visual_tokens,
                num_text_tokens=N_prompt,
                grid_h=merged_h,
                grid_w=merged_w,
                device=device,
                vision_first=True,
            )
            position_ids = position_ids.expand(3, B, -1)
            self._verify_position_ids_mixed_vision_text(
                position_ids, num_visual_tokens, N_prompt, B, vision_first=True
            )

            # Attention mask
            mask_long = torch.ones(B, N_total, device=device, dtype=torch.long)
            mask_long[:, num_visual_tokens:] = prompt_attention_mask

            # Forward backbone
            self._validate_qwen_backbone_inputs(combined, mask_long, position_ids)
            out = self.qwen_backbone(
                inputs_embeds=combined,
                attention_mask=mask_long,
                position_ids=position_ids,
                return_dict=True,
                use_cache=False,
            )
            hs = out.last_hidden_state  # [B, N_vis + N_prompt, dim]
            hs_text = hs[:, num_visual_tokens:, :]  # [B, N_prompt, dim]
            assert hs_text.shape[1] == N_prompt, (
                f"Text slice length {hs_text.shape[1]} != N_prompt ({N_prompt}); "
                "hs_text must be the exact text subsequence for lm_head."
            )
            logits = self._apply_lm_head(hs_text)  # [B, N_prompt, vocab_size]

            labels = prompt_input_ids  # [B, N_prompt]
            return logits, labels, N_prompt
        
        else:
            # NEW: Proper teacher forcing with prompt + target
            # Use chat template for consistent formatting with inference
            formatted_prompt = self._apply_chat_template(text_prompt, dataset_type=dataset_type)
            formatted_full = self._apply_chat_template(text_prompt, assistant_message=target_text, dataset_type=dataset_type)
            
            # Tokenize WITHOUT padding - use actual sequence lengths
            # Since we're duplicating the same string B times, tokenize once and expand
            
            # Tokenize full sequence (prompt + target with chat template)
            full_tokens = self.tokenizer(
                formatted_full,
                return_tensors="pt",
                padding=False,  # NO PADDING
                truncation=True,
                max_length=self.max_text_length,
                add_special_tokens=False,  # template already has special tokens
            )
            full_input_ids = full_tokens["input_ids"].to(device)  # [1, L_full]
            full_attention_mask = full_tokens["attention_mask"].to(device)  # [1, L_full]
            
            # Tokenize prompt separately (using chat template)
            prompt_tokens = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=False,  # NO PADDING
                truncation=True,
                max_length=self.max_text_length,
                add_special_tokens=False,  # template already has special tokens
            )
            # Mask boundary: use longest common prefix of prompt_ids and full_ids so we don't
            # train on role/header tokens or accidentally mask assistant content (tokenizer
            # context can make tokenize(prompt) not a strict prefix of tokenize(full)).
            prompt_ids = prompt_tokens["input_ids"][0]
            full_ids = full_tokens["input_ids"][0]
            k = 0
            while k < min(len(prompt_ids), len(full_ids)) and prompt_ids[k].item() == full_ids[k].item():
                k += 1
            N_prompt = k  # number of positions to mask (template-robust)
            
            # Expand for batch size
            if B > 1:
                full_input_ids = full_input_ids.expand(B, -1)
                full_attention_mask = full_attention_mask.expand(B, -1)
            
            # Get visual tokens
            dino_features, patch_tokens, grid_h, grid_w = self.encode_dino(pixel_values)
            adapted_tokens = self.dino_adapter(patch_tokens)
            visual_tokens, merged_h, merged_w = self.patch_merger(adapted_tokens, grid_h, grid_w)
            num_visual_tokens = visual_tokens.shape[1]
            
            # Safety check: verify visual tokens match grid dimensions
            assert num_visual_tokens == merged_h * merged_w, \
                f"Visual token count mismatch: {num_visual_tokens} != {merged_h}*{merged_w}={merged_h*merged_w}"
            
            # Embed the FULL text sequence (prompt + target)
            full_text_embeds = self.embed_tokens(full_input_ids)  # [B, full_len, qwen_dim]
            full_text_mask = full_attention_mask

            # Cast both to backbone dtype before concat (avoids Float vs BFloat16 in backbone/lm_head)
            backbone_dtype = next(self.qwen_backbone.parameters()).dtype
            visual_tokens = visual_tokens.to(dtype=backbone_dtype)
            full_text_embeds = full_text_embeds.to(dtype=backbone_dtype)

            # Concatenate: [visual_tokens, full_text_tokens] - CORRECT ORDER for causal LM
            # Visual tokens must come FIRST so text tokens can attend to them
            combined = torch.cat([visual_tokens, full_text_embeds], dim=1)  # [B, N_vis + N_full, dim]

            N_full = full_text_embeds.shape[1]
            N_total = num_visual_tokens + N_full
            
            # Create proper M-RoPE position_ids with spatial structure for vision tokens
            # Order is [visual, text], so vision_first=True
            position_ids = self._build_mrope_position_ids(
                num_visual_tokens=num_visual_tokens,
                num_text_tokens=N_full,
                grid_h=merged_h,
                grid_w=merged_w,
                device=device,
                vision_first=True,  # vision comes first in this function
            )
            # Expand for batch: [3, 1, N_total] -> [3, B, N_total]
            position_ids = position_ids.expand(3, B, -1)
            self._verify_position_ids_mixed_vision_text(
                position_ids, num_visual_tokens, N_full, B, vision_first=True
            )
            
            # Create standard 2D attention mask (padding mask)
            # The model's internal forward() will handle causal masking
            mask_long = torch.ones(B, N_total, device=device, dtype=torch.long)
            mask_long[:, num_visual_tokens:] = full_text_mask  # text part comes after visual tokens

            # Ensure qwen_backbone receives merged [visual, text] with matching dimensions and order
            self._validate_qwen_backbone_inputs(combined, mask_long, position_ids)
            if debug_shapes:
                self._debug_log_sequence_shapes(
                    visual_tokens=visual_tokens,
                    text_embeds=full_text_embeds,
                    combined=combined,
                    position_ids=position_ids,
                    attention_mask=mask_long,
                    order="vision_first",
                    num_visual_tokens=num_visual_tokens,
                    num_text_tokens=N_full,
                    label="get_text_generation_logits (vision first, text second)",
                )

            # 3. Forward through backbone only (avoid VL wrapper modality bookkeeping / "666 tokens")
            # Same pattern as generate_text: backbone -> last_hidden_state -> norm -> lm_head
            # Defensive: ensure combined and backbone path in same dtype (e.g. CPU or no autocast)
            combined = combined.to(dtype=backbone_dtype)
            out = self.qwen_backbone(
                inputs_embeds=combined,
                attention_mask=mask_long,
                position_ids=position_ids,
                return_dict=True,
                use_cache=False,
            )
            hs = out.last_hidden_state  # [B, N_vis + N_full, dim]
            # LM head is applied only to TEXT positions (after visual); no masking on hidden state
            hs_text = hs[:, num_visual_tokens:, :]  # [B, N_full, dim]
            assert hs_text.shape[1] == N_full, (
                f"Text slice length {hs_text.shape[1]} != N_full ({N_full}); "
                "hs_text must be the exact text subsequence for lm_head."
            )
            # Cast so lm_head/norm see correct dtype (backbone can return float32 on CPU)
            head_dtype = next(self.qwen_norm.parameters()).dtype
            hs_text = hs_text.to(dtype=head_dtype)
            logits = self._apply_lm_head(hs_text)  # [B, N_full, vocab_size]
            # Masking is on labels only (prompt positions = -100), not on hidden state
            # 5. Create labels: mask prompt prefix (common prefix length k) with -100
            labels = full_input_ids.clone()
            labels[:, :N_prompt] = -100  # N_prompt = k from longest common prefix
            
            return logits, labels, N_prompt
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on the Qwen backbone to reduce activation memory.

        With PEFT/LoRA, enable_input_require_grads() must be called first so that
        gradients can flow back through the checkpointed (recomputed) activations
        even when the base-model inputs have requires_grad=False.
        """
        if hasattr(self.qwen_backbone, 'enable_input_require_grads'):
            self.qwen_backbone.enable_input_require_grads()
        if hasattr(self.qwen_backbone, 'gradient_checkpointing_enable'):
            # use_reentrant=False avoids issues with PEFT adapters and is
            # the recommended default for newer PyTorch / Transformers.
            try:
                self.qwen_backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.qwen_backbone.gradient_checkpointing_enable()
            print("  Gradient checkpointing enabled via qwen_backbone.gradient_checkpointing_enable()")
        else:
            for layer in self.qwen_layers:
                if hasattr(layer, 'gradient_checkpointing'):
                    layer.gradient_checkpointing = True
            print("  Gradient checkpointing enabled on individual Qwen layers")

    def enable_model_parallel(self, device0: str = 'cuda:0', device1: str = 'cuda:1'):
        """
        Split the model across two GPUs for pipeline parallelism.

        GPU 0 (device0): DINO encoder, DPT projections, DPT scratch decoder
        GPU 1 (device1): Qwen LLM backbone, DINO adapter, patch merger

        The forward() method auto-detects device placement via parameter queries,
        so no flag is needed; just call this once after model.to(device0).
        """
        # DINO + DPT stay on device0
        self.dino.to(device0)
        self.proj_to_l1.to(device0)
        self.proj_to_l2.to(device0)
        self.proj_to_l3.to(device0)
        self.proj_to_l4.to(device0)
        self.scratch.to(device0)
        if hasattr(self, 'head'):
            self.head.to(device0)

        # Qwen LLM + vision-language adapters move to device1
        self.dino_adapter.to(device1)
        self.patch_merger.to(device1)
        # qwen_backbone contains embed_tokens, qwen_layers, qwen_norm as children
        self.qwen_backbone.to(device1)
        # qwen_full_model holds lm_head and other Qwen components
        if hasattr(self, 'qwen_full_model'):
            self.qwen_full_model.to(device1)

        print(f"  Model parallel: DINO+DPT on {device0}, Qwen+adapters on {device1}")

    def print_params(self):
        """Print parameter statistics."""
        def count(m):
            try:
                return sum(p.numel() for p in m.parameters())
            except ValueError:
                # Handle uninitialized LazyConv2d parameters
                return 0
        
        def trainable(m):
            try:
                return sum(p.numel() for p in m.parameters() if p.requires_grad)
            except ValueError:
                # Handle uninitialized LazyConv2d parameters
                return 0
        
        print("\n" + "="*70)
        print("Parameter Statistics (with Text Conditioning - Checkpoint Compatible)")
        print("="*70)
        
        # Initialize LazyConv2d modules if needed (by running a dummy forward pass)
        try:
            device = next(self.parameters()).device
            # Use float32 for dummy input to avoid dtype mismatches during initialization
            dummy_img = torch.randn(1, 3, 256, 256, device=device, dtype=torch.float32)
            
            was_training = self.training
            self.eval()
            
            with torch.no_grad():
                # Use autocast if model uses bfloat16
                if self.dtype == torch.bfloat16 and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        _ = self(dummy_img, text_prompt="dummy")
                else:
                    _ = self(dummy_img, text_prompt="dummy")
            
            # Restore training mode
            if was_training:
                self.train()
        except Exception as e:
            print(f"  Warning: Could not initialize LazyConv2d modules: {e}")
        
        components = [
            ("DINOv3 Encoder", self.dino),
            ("DINO Adapter", self.dino_adapter),
            ("PatchMerger", self.patch_merger),
            ("Qwen Embeddings", self.embed_tokens),
            ("Qwen Layers", self.qwen_layers),
            ("Qwen Norm", self.qwen_norm),
            ("DPT Projections (4)", nn.ModuleList([self.proj_to_l1, self.proj_to_l2, self.proj_to_l3, self.proj_to_l4])),
            ("DPT Scratch", self.scratch),
        ]
        
        total = sum(count(m) for _, m in components)
        train_total = sum(trainable(m) for _, m in components)
        
        for name, module in components:
            t = count(module)
            tr = trainable(module)
            pct = 100 * t / total if total > 0 else 0
            print(f"  {name:25s}: {t:>12,} ({tr:>12,} trainable) [{pct:5.1f}%]")
        
        print("-"*70)
        print(f"  {'TOTAL':25s}: {total:>12,} ({train_total:>12,} trainable)")
        if total > 0:
            print(f"  Trainable: {100*train_total/total:.2f}%")
        print("="*70)
