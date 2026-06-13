#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for OpenVAM model.

Loads a trained checkpoint and runs inference on an image to generate:
1. Saliency map (dense prediction) - MAIN OUTPUT
2. Text explanation (optional) - If model was trained for text generation

Supports both regular checkpoints and LoRA checkpoints.

Usage:
    # Basic inference (saliency map only):
    python scripts/inference.py \
        --checkpoint /path/to/best_model.pth \
        --image /path/to/image.jpg \
        --text_prompt "Describe the salient regions in this image." \
        --output_dir ./outputs

    # By category (model uses the correct prompt for that image type):
    python scripts/inference.py \
        --checkpoint /path/to/best_model.pth \
        --image /path/to/image.jpg \
        --category natural_scene \
        --output_dir ./outputs
    # Categories: natural_scene, webpage, e_commerce (or dataset names: CAT2000_256, datasets_UI_256, SalEC, etc.)
    
    # With text generation (if model was trained for it):
    python scripts/inference.py \
        --checkpoint /path/to/best_model.pth \
        --image /path/to/image.jpg \
        --generate_text \
        --output_dir ./outputs
    
    # LoRA checkpoint (automatically detected):
    python scripts/inference.py \
        --checkpoint /path/to/checkpoints_salience_text_lora/best_model.pth \
        --image /path/to/image.jpg \
        --output_dir ./outputs
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, visualization will be limited")

try:
    from scipy.ndimage import zoom
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, using basic resizing for visualization")

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the OpenVAM model and dataset-type prompt helpers
from net.openvam import (
    OpenVAM,
    get_dataset_type_from_name,
    get_user_prompt_for_dataset_type,
    DATASET_TYPE_PROMPTS,
)

# LoRA imports
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel,
    )
    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False


def load_training_args(checkpoint_path):
    """Try to load training args from checkpoint directory."""
    checkpoint_dir = Path(checkpoint_path).parent
    args_json_path = checkpoint_dir / "args.json"
    
    if args_json_path.exists():
        print(f"  Found args.json in checkpoint directory")
        with open(args_json_path, "r") as f:
            training_args = json.load(f)
        return training_args
    return None


def setup_lora_for_inference(
    model: torch.nn.Module,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
) -> torch.nn.Module:
    """Apply LoRA to model for inference (same as training setup)."""
    if not _HAS_PEFT:
        raise ImportError("peft library is required for LoRA inference. Install with: pip install peft")
    
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    if not hasattr(model, 'qwen_full_model'):
        raise AttributeError("Model does not have qwen_full_model attribute. Cannot apply LoRA.")
    
    qwen_model = model.qwen_full_model
    
    if not (hasattr(qwen_model, 'model') and hasattr(qwen_model.model, 'language_model')):
        raise AttributeError("Cannot find language_model in qwen_full_model structure")
    
    lm_model = qwen_model.model.language_model
    
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
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Use CAUSAL_LM for text generation
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    
    print(f"  Applying LoRA (r={lora_r}, alpha={lora_alpha})...")
    peft_model = get_peft_model(lm_model, lora_config)
    
    qwen_model.model.language_model = peft_model
    
    # CRITICAL: Rebind qwen_backbone to the PeftModel so state_dict keys match the
    # training checkpoint (qwen_backbone.base_model.model.* and qwen_backbone.lora_*).
    # Without this, model.qwen_backbone still points to the base model, giving
    # 74 missing / 74 unexpected keys when loading the checkpoint.
    if hasattr(model, 'rebind_lora_references'):
        model.rebind_lora_references()
    else:
        # Fallback: update refs manually so at least qwen_backbone matches checkpoint
        model.qwen_backbone = peft_model
        if hasattr(peft_model, 'get_base_model'):
            base_model = peft_model.get_base_model()
            if hasattr(base_model, 'model'):
                model.qwen_layers = base_model.model.layers
                model.qwen_norm = base_model.model.norm
            else:
                model.qwen_layers = base_model.layers
                model.qwen_norm = base_model.norm

    return model


def check_if_lora_checkpoint(checkpoint_path):
    """Check if checkpoint contains LoRA weights."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if isinstance(checkpoint, dict):
        # Check for LoRA flag
        if checkpoint.get("has_lora", False):
            return True
        # Check state dict for LoRA keys
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Check for LoRA structure (base_layer, lora_A, lora_B)
        if any("base_layer" in k or "lora_A" in k or "lora_B" in k for k in state_dict.keys()):
            return True
    
    return False


def load_checkpoint(model, checkpoint_path, device):
    """Load checkpoint into model. Supports both regular and LoRA checkpoints."""
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print(f"  Found 'model_state_dict' in checkpoint")
            if "epoch" in checkpoint:
                print(f"  Checkpoint epoch: {checkpoint['epoch']}")
            if "best_loss" in checkpoint:
                print(f"  Checkpoint best_loss: {checkpoint['best_loss']:.4f}")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print(f"  Found 'state_dict' in checkpoint")
        else:
            state_dict = checkpoint
            print(f"  Using checkpoint as state_dict directly")
    else:
        state_dict = checkpoint
    
    # Check if this is a LoRA checkpoint
    is_lora_checkpoint = False
    if isinstance(checkpoint, dict):
        # Check for LoRA flag
        if checkpoint.get("has_lora", False):
            is_lora_checkpoint = True
        # Or check for LoRA keys in state dict
        elif any("lora" in k.lower() for k in state_dict.keys()):
            is_lora_checkpoint = True
    
    if is_lora_checkpoint:
        print(f"  ✓ Detected LoRA checkpoint")
        print(f"  Loading base model weights and LoRA adapters...")
        # For LoRA checkpoints, the model should already have LoRA applied
        # Check if model has LoRA structure
        has_lora_structure = False
        if hasattr(model, 'qwen_full_model'):
            qwen_model = model.qwen_full_model
            if hasattr(qwen_model, 'model') and hasattr(qwen_model.model, 'language_model'):
                lm = qwen_model.model.language_model
                if isinstance(lm, PeftModel):
                    has_lora_structure = True
        
        if not has_lora_structure:
            print(f"  ⚠️  WARNING: LoRA checkpoint detected but model doesn't have LoRA structure!")
            print(f"      LoRA should be applied BEFORE loading checkpoint.")
            print(f"      Some weights may not load correctly.")
    else:
        print(f"  ✓ Regular checkpoint (no LoRA)")
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    num_loaded = len(state_dict) - len(unexpected_keys)

    # For LoRA checkpoints, missing/unexpected keys are expected if structure matches
    if is_lora_checkpoint:
        # Filter keys to see what's actually missing/unexpected
        lora_missing = [k for k in missing_keys if any(x in k.lower() for x in ["lora", "base_layer"])]
        non_lora_missing = [k for k in missing_keys if not any(x in k.lower() for x in ["lora", "base_layer"])]
        
        lora_unexpected = [k for k in unexpected_keys if any(x in k.lower() for x in ["lora", "base_layer"])]
        non_lora_unexpected = [k for k in unexpected_keys if not any(x in k.lower() for x in ["lora", "base_layer"])]
        
        if non_lora_missing:
            print(f"  ⚠️  Missing non-LoRA keys: {len(non_lora_missing)}")
            if len(non_lora_missing) <= 5:
                for key in non_lora_missing:
                    print(f"    - {key}")
            else:
                for key in non_lora_missing[:3]:
                    print(f"    - {key}")
                print(f"    ... and {len(non_lora_missing) - 3} more")
        
        if non_lora_unexpected:
            print(f"  ⚠️  Unexpected non-LoRA keys: {len(non_lora_unexpected)}")
            if len(non_lora_unexpected) <= 5:
                for key in non_lora_unexpected:
                    print(f"    - {key}")
            else:
                for key in non_lora_unexpected[:3]:
                    print(f"    - {key}")
                print(f"    ... and {len(non_lora_unexpected) - 3} more")
        
        if lora_missing or lora_unexpected:
            print(f"  Note: {len(lora_missing)} LoRA keys missing, {len(lora_unexpected)} LoRA keys unexpected")
            print(f"        This is normal if LoRA structure matches checkpoint structure.")
    else:
        # Regular checkpoint: missing keys often = Qwen LLM not in checkpoint (use HF weights)
        qwen_backbone_prefix = "qwen_backbone."
        missing_qwen = [k for k in missing_keys if k.startswith(qwen_backbone_prefix)]
        missing_other = [k for k in missing_keys if not k.startswith(qwen_backbone_prefix)]

        if missing_keys:
            if len(missing_qwen) == len(missing_keys) or len(missing_qwen) > len(missing_keys) // 2:
                # Most or all missing keys are Qwen LLM → expected when checkpoint has no LLM weights
                print(f"  ℹ️  Checkpoint does not contain Qwen LLM weights ({len(missing_qwen)} keys).")
                print(f"      Using Hugging Face pretrained weights for qwen_backbone (normal for DINO/DPT-only checkpoints).")
            else:
                print(f"  ⚠️  Missing keys: {len(missing_keys)}")
                if len(missing_other) <= 10:
                    for key in missing_other:
                        print(f"    - {key}")
                else:
                    for key in missing_other[:5]:
                        print(f"    - {key}")
                    print(f"    ... and {len(missing_other) - 5} more")
            if missing_other and len(missing_qwen) > 0:
                print(f"      (+ {len(missing_qwen)} qwen_backbone keys not in checkpoint, using HF weights)")
        if unexpected_keys:
            print(f"  ⚠️  Unexpected keys: {len(unexpected_keys)}")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    print(f"    - {key}")
            else:
                for key in unexpected_keys[:5]:
                    print(f"    - {key}")
                print(f"    ... and {len(unexpected_keys) - 5} more")
        print(f"  Loaded {num_loaded} weights from checkpoint.")
    print(f"  ✓ Checkpoint loaded successfully")
    if is_lora_checkpoint:
        print(f"  ✓ LoRA adapters loaded")
    return model


def preprocess_image(image_path, image_size=256):
    """Load and preprocess image."""
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (W, H)
    
    # Resize to model input size
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    pixel_values = transform(image).unsqueeze(0)  # [1, 3, H, W]
    
    return pixel_values, image, original_size


def visualize_saliency(image, saliency_map, output_path, alpha=0.6):
    """Visualize saliency map overlaid on image."""
    # Convert saliency to numpy (convert bfloat16 to float32 first)
    if isinstance(saliency_map, torch.Tensor):
        saliency_np = saliency_map.squeeze().cpu().float().numpy()
    else:
        saliency_np = saliency_map
    
    # Normalize to [0, 1]
    saliency_np = (saliency_np - saliency_np.min()) / (saliency_np.max() - saliency_np.min() + 1e-8)
    
    # Resize saliency to match image size
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image
    
    if saliency_np.shape != img_np.shape[:2]:
        if HAS_SCIPY:
            zoom_factors = (img_np.shape[0] / saliency_np.shape[0], 
                           img_np.shape[1] / saliency_np.shape[1])
            saliency_np = zoom(saliency_np, zoom_factors, order=1)
        else:
            # Fallback: use PIL resize
            saliency_img = Image.fromarray((saliency_np * 255).astype(np.uint8))
            saliency_img = saliency_img.resize((img_np.shape[1], img_np.shape[0]), Image.BILINEAR)
            saliency_np = np.array(saliency_img).astype(np.float32) / 255.0
    
    # Create colormap
    if HAS_MATPLOTLIB:
        import matplotlib.cm as cm
        saliency_colored = cm.jet(saliency_np)[:, :, :3]  # [H, W, 3]
    else:
        # Fallback: simple colormap
        saliency_colored = np.stack([saliency_np, saliency_np * 0.5, 1 - saliency_np], axis=2)
        saliency_colored = np.clip(saliency_colored, 0, 1)
    
    # Overlay
    overlay = alpha * saliency_colored + (1 - alpha) * (img_np / 255.0)
    overlay = np.clip(overlay, 0, 1)
    
    if not HAS_MATPLOTLIB:
        # Fallback: save overlay as simple image
        overlay_uint8 = (overlay * 255).astype(np.uint8)
        overlay_img = Image.fromarray(overlay_uint8)
        overlay_img.save(output_path)
        print(f"  ✓ Saved overlay to: {output_path}")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Saliency map
    axes[1].imshow(saliency_np, cmap="jet")
    axes[1].set_title("Saliency Map")
    axes[1].axis("off")
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ Saved visualization to: {output_path}")


def save_saliency_map(saliency_map, output_path):
    """Save saliency map as grayscale image."""
    # Convert saliency to numpy (convert bfloat16 to float32 first)
    if isinstance(saliency_map, torch.Tensor):
        saliency_np = saliency_map.squeeze().cpu().float().numpy()
    else:
        saliency_np = saliency_map
    
    # Normalize to [0, 1]
    saliency_np = (saliency_np - saliency_np.min()) / (saliency_np.max() - saliency_np.min() + 1e-8)
    
    # Convert to uint8
    saliency_uint8 = (saliency_np * 255).astype(np.uint8)
    
    # Save as image
    saliency_img = Image.fromarray(saliency_uint8, mode="L")
    saliency_img.save(output_path)
    
    print(f"  ✓ Saved saliency map to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with OpenVAM")
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (.pth file)")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    
    # Optional arguments
    parser.add_argument("--category", type=str, default=None,
                        help="Image category for prompt selection. Use one of: natural_scene, webpage, e_commerce; "
                        "or a dataset name (e.g. CAT2000_256, MIT1003_256, salicon_256, datasets_UI_256, SalEC) "
                        "which is mapped to the correct prompt type. When set, overrides --text_prompt.")
    parser.add_argument("--text_prompt", type=str, default=None,
                        help="Text prompt for conditioning (ignored if --category is set; default: model default)")
    parser.add_argument("--output_dir", type=str, default="./inference_outputs",
                        help="Output directory for results")
    
    # Model config (should match training config)
    parser.add_argument("--dino_model", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m",
                        help="DINO model name")
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                        help="Qwen model name (e.g. Qwen3-VL-2B-Instruct or Qwen3-VL-4B-Instruct)")
    parser.add_argument("--backbone", type=str, default="vitb_rn50_384",
                        help="DPT backbone name")
    parser.add_argument("--features", type=int, default=256,
                        help="DPT features")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Input image size")
    
    # Generation options
    parser.add_argument("--generate_text", action="store_true",
                        help="Generate text explanation (requires model trained with text_weight > 0)")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Max tokens for text generation")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for text generation (lower = more deterministic, recommended: 0.1-0.3)")
    
    # LoRA config (for LoRA checkpoints)
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank (only needed if args.json not found)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (only needed if args.json not found)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout (only needed if args.json not found)")
    
    # HuggingFace token
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    
    args = parser.parse_args()
    
    # Check if this is a LoRA checkpoint BEFORE creating model
    is_lora_checkpoint = check_if_lora_checkpoint(args.checkpoint)
    
    # Try to load training args from checkpoint directory
    training_args = load_training_args(args.checkpoint)
    if training_args:
        print("\n" + "="*70)
        print("Loading Training Configuration from args.json")
        print("="*70)
        # Override args with training config (user can still override via command line)
        args.dino_model = training_args.get("dino_model", args.dino_model)
        args.qwen_model = training_args.get("qwen_model", args.qwen_model)
        args.backbone = training_args.get("backbone", args.backbone)
        args.features = training_args.get("features", args.features)
        args.image_size = training_args.get("image_size", args.image_size)
        
        # Get LoRA config from training args if available
        if is_lora_checkpoint:
            args.lora_r = training_args.get("lora_r", 16)
            args.lora_alpha = training_args.get("lora_alpha", 32)
            args.lora_dropout = training_args.get("lora_dropout", 0.1)
            args.lora_target_modules = training_args.get("lora_target_modules", 
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        
        print(f"  Using config: dino={args.dino_model}, qwen={args.qwen_model}")
        print(f"  backbone={args.backbone}, features={args.features}")
        if is_lora_checkpoint:
            print(f"  LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ==========================================================================
    # Load Model
    # ==========================================================================
    print("\n" + "="*70)
    print("Loading Model")
    print("="*70)
    
    model = OpenVAM(
        dino_model_name=args.dino_model,
        qwen_model_name=args.qwen_model,
        backbone=args.backbone,
        features=args.features,
        freeze_dino=False,
        freeze_qwen_lm=True,
        freeze_projector=False,
        dtype=torch.bfloat16,
        hf_token=args.hf_token or os.getenv("HF_TOKEN"),
    )
    
    model = model.to(device)
    model.eval()

    # # ------------------------------------------------------------------
    # # Initialize LazyConv2d layers BEFORE loading checkpoint (CRITICAL!)
    # # LazyConv2d layers (DPT scratch/backbone) do not materialize their
    # # parameters until the first forward pass. If we call load_state_dict
    # # before this, those weights are silently skipped (strict=False hides
    # # the miss), causing randomly-initialized DPT layers at inference time.
    # # ------------------------------------------------------------------
    # print("\nInitializing LazyConv2d layers via dummy forward pass...")
    # with torch.no_grad():
    #     try:
    #         was_dtype = getattr(model, 'dtype', None)
    #         if was_dtype == torch.bfloat16:
    #             model = model.to(torch.float32)
    #         dummy_img = torch.randn(
    #             1, 3, args.image_size, args.image_size,
    #             device=device, dtype=torch.float32,
    #         )
    #         _ = model(dummy_img, text_prompt="dummy")
    #         if was_dtype is not None:
    #             model = model.to(was_dtype)
    #         print("  ✓ LazyConv2d layers initialized")
    #     except Exception as e:
    #         print(f"  ⚠  Warning: dummy forward pass failed ({e}); "
    #               "checkpoint weights for uninitialized layers may be skipped.")
    #         if was_dtype is not None:
    #             try:
    #                 model = model.to(was_dtype)
    #             except Exception:
    #                 pass

    # Apply LoRA BEFORE loading checkpoint if this is a LoRA checkpoint
    if is_lora_checkpoint:
        if not _HAS_PEFT:
            raise ImportError("LoRA checkpoint detected but peft library not found. Install with: pip install peft")
        
        print("\n" + "="*70)
        print("Applying LoRA (Required for LoRA Checkpoint)")
        print("="*70)
        model = setup_lora_for_inference(
            model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules if hasattr(args, 'lora_target_modules') else None,
        )
        print("✓ LoRA applied - model structure now matches checkpoint")
    
    # Load checkpoint
    model = load_checkpoint(model, args.checkpoint, device)
    
    print("✓ Model ready for inference")
    
    # ==========================================================================
    # Load Image
    # ==========================================================================
    print("\n" + "="*70)
    print("Loading Image")
    print("="*70)
    
    pixel_values, original_image, original_size = preprocess_image(
        args.image, args.image_size
    )
    pixel_values = pixel_values.to(device, dtype=model.dtype)
    
    print(f"✓ Image loaded: {pixel_values.shape}")
    print(f"  Original size: {original_size}")
    
    # ==========================================================================
    # Run Inference
    # ==========================================================================
    print("\n" + "="*70)
    print("Running Inference")
    print("="*70)
    
    # Resolve category -> dataset_type and prompt
    dataset_type = None
    if args.category:
        # Allow either dataset type (natural_scene, webpage, e_commerce) or dataset name (e.g. CAT2000_256)
        category = args.category.strip()
        if category in DATASET_TYPE_PROMPTS:
            dataset_type = category
        else:
            dataset_type = get_dataset_type_from_name(category)
        text_prompt = get_user_prompt_for_dataset_type(dataset_type)
        print(f"Category: {args.category} -> dataset_type: {dataset_type}")
    else:
        text_prompt = args.text_prompt if args.text_prompt else model.default_prompt
    print(f"Text prompt: {text_prompt[:80]}..." if len(text_prompt) > 80 else f"Text prompt: {text_prompt}")
    
    with torch.no_grad():
        with torch.autocast(device_type="cuda:1", dtype=torch.bfloat16):
            # Generate saliency map (pass dataset_type so model uses correct system+user prompts)
            saliency_map = model(pixel_values, text_prompt=text_prompt, dataset_type=dataset_type)
    
    print(f"✓ Saliency map generated: {saliency_map.shape}")
    
    # ==========================================================================
    # Save Results
    # ==========================================================================
    print("\n" + "="*70)
    print("Saving Results")
    print("="*70)
    
    # Save saliency map
    saliency_path = os.path.join(args.output_dir, "saliency_map.png")
    save_saliency_map(saliency_map[0, 0], saliency_path)
    
    # Save visualization
    vis_path = os.path.join(args.output_dir, "saliency_visualization.png")
    visualize_saliency(original_image, saliency_map[0, 0], vis_path)
    
    # Save text prompt
    prompt_path = os.path.join(args.output_dir, "text_prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(f"Text Prompt:\n{text_prompt}\n")
    print(f"  ✓ Saved text prompt to: {prompt_path}")
    
    # ==========================================================================
    # Generate Text (Optional - EXPERIMENTAL)
    # ==========================================================================
    if args.generate_text:
        print("\n" + "="*70)
        print("Generating Text Explanation")
        print("="*70)
        print("⚠️  IMPORTANT: Prompt Format Mismatch")
        print("="*70)
        print("Your model was trained to EXPLAIN why listed objects are salient,")
        print("NOT to DISCOVER which objects are salient.")
        print()
        print("Training format:")
        print("  Input: 'Salient-object sentence: [list of objects]. Explain why each is salient.'")
        print("  Output: Explanations for each object")
        print()
        print("Your current prompt:")
        print(f"  '{text_prompt}'")
        print()
        print("This asks the model to DISCOVER objects, which it wasn't trained for.")
        print("Result: Repetitive or generic outputs.")
        print()
        print("RECOMMENDED: Use a prompt that matches training format:")
        print("  Example: 'Salient-object sentence: A person's hand holding a camera.")
        print("            Explain why each object is salient.'")
        print("="*70)
        
        try:
            with torch.no_grad():
                with torch.autocast(device_type="cuda:1", dtype=torch.bfloat16):
                    # Use more conservative generation settings (dataset_type for correct template)
                    generated_text = model.generate_text(
                        pixel_values,
                        text_prompt,
                        max_new_tokens=args.max_new_tokens,
                        # temperature=0.3,  # Lower temperature for more deterministic output
                        top_p=0.9,
                        do_sample=False,
                        dataset_type=dataset_type,
                    )
            
            print(f"✓ Generated text: {generated_text}")
            
            # Save generated text
            text_path = os.path.join(args.output_dir, "generated_text.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(f"Input Prompt:\n{text_prompt}\n\n")
                f.write(f"Generated Explanation (EXPERIMENTAL - may be gibberish):\n{generated_text}\n")
            print(f"  ✓ Saved generated text to: {text_path}")
            print("\n  ⚠️  Note: If the output is gibberish, this is expected.")
            print("     This model was not trained for text generation.")
            
        except Exception as e:
            print(f"  ⚠️  Text generation failed: {e}")
            print(f"  This is okay - saliency map generation succeeded")
            import traceback
            traceback.print_exc()
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "="*70)
    print("Inference Complete!")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - saliency_map.png: Grayscale saliency map")
    print(f"  - saliency_visualization.png: Overlay visualization")
    print(f"  - text_prompt.txt: Input text prompt")
    if args.generate_text:
        print(f"  - generated_text.txt: Generated text explanation (EXPERIMENTAL)")
    print("\nNote: This model is designed for SALIENCY PREDICTION.")
    print("      Text generation is experimental and not recommended.")
    print("="*70)


if __name__ == "__main__":
    main()
