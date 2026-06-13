#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch text-generation inference on a merged val JSONL.

Uses the exact same model-loading and generation logic as scripts/inference.py,
wrapped in a loop over a JSONL file.

For every sample in the JSONL the script:
  1. Resolves the image path from the merged layout
  2. Determines the dataset type from the sample id
  3. Runs model.generate_text() to produce a salient-region description
  4. Appends one JSON line to the output file:

     {"id": "CAT2000_256_Action_005",
      "image": "/abs/path/to/stimuli/CAT2000_256_Action_005.jpg",
      "source_dataset": "CAT2000_256",
      "prompt": "<user prompt for that dataset type>",
      "prediction": "<generated text>"}

The output file is written incrementally; use --resume to skip already-processed ids.

Usage:
    python scripts/inference_batch.py \\
        --checkpoint /path/to/best_model.pth \\
        --val_jsonl  /path/to/merged/merged_val.jsonl \\
        --merged_base /path/to/merged \\
        --output_jsonl ./val_text_predictions.jsonl \\
        --qwen_model Qwen/Qwen2.5-VL-3B-Instruct \\
        --max_new_tokens 1024 \\
        --resume

  For 7B (or large) models to avoid OOM, spread the Qwen LLM across 2 GPUs:
    python scripts/inference_batch.py ... --device_map auto ...
  Optional: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation.
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
from tqdm import tqdm

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from net.openvam import (
    OpenVAM,
    get_dataset_from_id,
    get_dataset_type_from_name,
    get_user_prompt_for_dataset_type,
    DATASET_TYPE_PROMPTS,
)

# LoRA imports — same as scripts/inference.py
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


# =============================================================================
# All helpers copied verbatim from scripts/inference.py
# =============================================================================

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

    if not hasattr(lm_model, 'prepare_inputs_for_generation'):
        def prepare_inputs_for_generation(self, input_ids=None, **kwargs):
            model_inputs = {}
            if input_ids is not None:
                model_inputs["input_ids"] = input_ids
            for k in ["inputs_embeds", "attention_mask", "position_ids", "past_key_values", "use_cache"]:
                if k in kwargs and kwargs[k] is not None:
                    model_inputs[k] = kwargs[k]
            return model_inputs

        import types
        lm_model.prepare_inputs_for_generation = types.MethodType(
            prepare_inputs_for_generation, lm_model
        )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    print(f"  Applying LoRA (r={lora_r}, alpha={lora_alpha})...")
    peft_model = get_peft_model(lm_model, lora_config)
    qwen_model.model.language_model = peft_model

    if hasattr(model, 'rebind_lora_references'):
        model.rebind_lora_references()
    else:
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
        if checkpoint.get("has_lora", False):
            return True
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        if any("base_layer" in k or "lora_A" in k or "lora_B" in k for k in state_dict.keys()):
            return True

    return False


def load_checkpoint(model, checkpoint_path, device, use_cpu_map=False):
    """Load checkpoint into model. Supports both regular and LoRA checkpoints.
    use_cpu_map: if True (multi-GPU device_map), load to CPU first so state_dict
    is applied without pushing the full checkpoint onto one GPU.
    """
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    map_location = "cpu" if use_cpu_map else device
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

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

    is_lora_checkpoint = False
    if isinstance(checkpoint, dict):
        if checkpoint.get("has_lora", False):
            is_lora_checkpoint = True
        elif any("lora" in k.lower() for k in state_dict.keys()):
            is_lora_checkpoint = True

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    num_loaded = len(state_dict) - len(unexpected_keys)

    if not is_lora_checkpoint:
        qwen_backbone_prefix = "qwen_backbone."
        missing_qwen = [k for k in missing_keys if k.startswith(qwen_backbone_prefix)]
        missing_other = [k for k in missing_keys if not k.startswith(qwen_backbone_prefix)]

        if missing_keys:
            if len(missing_qwen) == len(missing_keys) or len(missing_qwen) > len(missing_keys) // 2:
                print(f"  Using HF pretrained weights for qwen_backbone ({len(missing_qwen)} keys).")
            else:
                print(f"  Missing keys: {len(missing_keys)}")
                for key in missing_other[:5]:
                    print(f"    - {key}")
                if len(missing_other) > 5:
                    print(f"    ... and {len(missing_other) - 5} more")
        if unexpected_keys:
            print(f"  Unexpected keys: {len(unexpected_keys)}")

    print(f"  Loaded {num_loaded} weights from checkpoint.")
    print(f"  ✓ Checkpoint loaded successfully")
    return model


def preprocess_image(image_path, image_size=256):
    """Load and preprocess image — identical to scripts/inference.py."""
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (W, H)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    pixel_values = transform(image).unsqueeze(0)  # [1, 3, H, W]
    return pixel_values, image, original_size


# =============================================================================
# Batch-specific helpers
# =============================================================================

_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def resolve_image_path(sample: dict, merged_base: Path, split: str = "val") -> Optional[str]:
    """Resolve absolute path of the stimulus image."""
    for field in ("stimuli", "image", "images"):
        val = sample.get(field)
        if isinstance(val, list):
            val = val[0] if val else None
        if val and Path(val).exists():
            return str(val)

    sid = sample.get("id", "")
    if sid and merged_base is not None:
        stim_dir = merged_base / split / "stimuli"
        for ext in _IMAGE_EXTENSIONS:
            p = stim_dir / f"{sid}{ext}"
            if p.exists():
                return str(p)

    return None


def _normalize_name(name: str) -> str:
    """Normalize a filename stem for matching."""
    return name.strip().lower()


def collect_sample_image_stems(sample_folder: Path) -> set:
    """Collect normalized image stems from a sample folder."""
    sample_stems = set()
    for p in sample_folder.iterdir():
        if p.is_file() and p.suffix in _IMAGE_EXTENSIONS:
            sample_stems.add(_normalize_name(p.stem))
    return sample_stems


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch text inference on merged val JSONL"
    )

    # Required
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained .pth checkpoint")
    parser.add_argument("--val_jsonl", required=True,
                        help="Path to merged_val.jsonl")
    parser.add_argument("--merged_base", required=True,
                        help="Root of the merged dataset (contains train/ and val/)")
    parser.add_argument("--output_jsonl", default="val_text_predictions.jsonl",
                        help="Output JSONL file (written incrementally)")

    # Model config (auto-read from args.json if present)
    parser.add_argument("--dino_model", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--backbone", type=str, default="vitb_rn50_384")
    parser.add_argument("--features", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=256)

    # LoRA config (for LoRA checkpoints)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # Generation — exact same defaults as scripts/inference.py
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true",
                        help="Use nucleus sampling; default is greedy decoding")

    # Which JSONL split sub-directory to look for stimuli
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--sample_folder",
        type=str,
        default=None,
        help="Optional folder containing sample images. "
             "If set, only JSONL rows whose resolved image name contains a sample image name "
             "(stem-based, normalized matching) are processed.",
    )

    # Device — default cuda:1 so inference does not conflict with training on cuda:0
    parser.add_argument("--device", default=None,
                        help="Device (e.g. cuda:0, cuda:1). "
                             "Default: cuda:1 if two GPUs present, else cuda:0.")

    # Multi-GPU: spread the Qwen LLM across GPUs (e.g. for 7B to avoid OOM on one GPU)
    parser.add_argument("--device_map", type=str, default=None,
                        help="Passed to Qwen model load (e.g. 'auto' or 'balanced'). "
                             "Use 'auto' to spread 7B across 2 GPUs. Ignored if not set.")

    # Resume / misc
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples already present in the output file")
    parser.add_argument("--hf_token", type=str, default=None)

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Check LoRA BEFORE model creation (same as scripts/inference.py)
    # ------------------------------------------------------------------
    is_lora_checkpoint = check_if_lora_checkpoint(args.checkpoint)

    # ------------------------------------------------------------------
    # Load training args from args.json (same as scripts/inference.py)
    # ------------------------------------------------------------------
    training_args = load_training_args(args.checkpoint)
    if training_args:
        print("\n" + "=" * 70)
        print("Loading Training Configuration from args.json")
        print("=" * 70)
        args.dino_model = training_args.get("dino_model", args.dino_model)
        args.qwen_model = training_args.get("qwen_model", args.qwen_model)
        args.backbone = training_args.get("backbone", args.backbone)
        args.features = training_args.get("features", args.features)
        args.image_size = training_args.get("image_size", args.image_size)
        if is_lora_checkpoint:
            args.lora_r = training_args.get("lora_r", args.lora_r)
            args.lora_alpha = training_args.get("lora_alpha", args.lora_alpha)
            args.lora_dropout = training_args.get("lora_dropout", args.lora_dropout)
            args.lora_target_modules = training_args.get(
                "lora_target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
        print(f"  qwen={args.qwen_model}")

    merged_base = Path(args.merged_base)

    # ------------------------------------------------------------------
    # Device (default cuda:1, matching scripts/inference.py behaviour)
    # When --device_map is set, device is set after model creation (to DINO device).
    # ------------------------------------------------------------------
    use_device_map = (args.device_map is not None and args.device_map.strip() != "")
    if not use_device_map:
        if args.device:
            device = torch.device(args.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")
    else:
        device = None  # set after model creation from DINO device
        print(f"Using device_map: {args.device_map} (multi-GPU); primary device will be set after model load.")

    # ------------------------------------------------------------------
    # Load JSONL samples
    # ------------------------------------------------------------------
    samples = []
    with open(args.val_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} samples from {args.val_jsonl}")

    sample_image_stems = None
    if args.sample_folder:
        sample_folder = Path(args.sample_folder)
        if not sample_folder.exists() or not sample_folder.is_dir():
            raise ValueError(f"--sample_folder is not a valid directory: {sample_folder}")
        sample_image_stems = collect_sample_image_stems(sample_folder)
        if not sample_image_stems:
            print(f"No images found in sample folder: {sample_folder}")
            return
        print(f"Sample folder filter enabled: {len(sample_image_stems)} images from {sample_folder}")

    # ------------------------------------------------------------------
    # Resume: collect already-processed ids
    # ------------------------------------------------------------------
    done_ids: set = set()
    output_path = Path(args.output_jsonl)
    if args.resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_ids.add(json.loads(line)["id"])
                    except Exception:
                        pass
        print(f"Resuming — {len(done_ids)} samples already done, skipping them")

    pending = [s for s in samples if s.get("id", "") not in done_ids]
    print(f"Samples to process: {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    if sample_image_stems is not None:
        filtered_pending = []
        skipped_not_in_sample_folder = 0
        used_sample_stems = set()
        for s in pending:
            img_path = resolve_image_path(s, merged_base, split=args.split)
            if img_path is None:
                continue
            merged_stem_norm = _normalize_name(Path(img_path).stem)
            matching_stems = [stem for stem in sample_image_stems if stem in merged_stem_norm]
            if not matching_stems:
                skipped_not_in_sample_folder += 1
                continue

            # Keep only one merged sample per sample-folder image.
            # If multiple sample stems match, prefer the longest (most specific) one.
            best_stem = max(matching_stems, key=len)
            if best_stem in used_sample_stems:
                skipped_not_in_sample_folder += 1
                continue

            used_sample_stems.add(best_stem)
            if len(used_sample_stems) <= len(sample_image_stems):
                filtered_pending.append(s)
            else:
                skipped_not_in_sample_folder += 1
        pending = filtered_pending
        print(
            f"After --sample_folder filter: {len(pending)} samples "
            f"({skipped_not_in_sample_folder} excluded)"
        )
        if not pending:
            print("Nothing to do after sample folder filter.")
            return

    # ------------------------------------------------------------------
    # Build model — identical to scripts/inference.py
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Loading Model")
    print("=" * 70)

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
        device_map=args.device_map if use_device_map else None,
    )

    if use_device_map:
        # Qwen is already spread across GPUs; DINO/DPT stay on first GPU. Use DINO device for inputs.
        device = next(model.dino.parameters()).device
        print(f"Primary device (DINO/inputs): {device}")
    else:
        model = model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Initialize LazyConv2d layers BEFORE loading checkpoint (CRITICAL!)
    # LazyConv2d layers (DPT scratch/backbone) do not materialize their
    # parameters until the first forward pass. If we call load_state_dict
    # before this, those weights are silently skipped (strict=False hides
    # the miss), and inference then runs with randomly-initialized DPT
    # layers — producing hallucinated descriptions unrelated to the image.
    # The training script does this same dummy-forward step; we must too.
    # ------------------------------------------------------------------
    print("\nInitializing LazyConv2d layers via dummy forward pass...")
    with torch.no_grad():
        try:
            was_dtype = getattr(model, 'dtype', None)
            # When using device_map, avoid moving whole model to float32 (would break multi-GPU placement)
            if use_device_map:
                dummy_img = torch.randn(
                    1, 3, args.image_size, args.image_size,
                    device=device, dtype=was_dtype or torch.bfloat16,
                )
                _ = model(dummy_img, text_prompt="dummy")
            else:
                if was_dtype == torch.bfloat16:
                    model = model.to(torch.float32)
                dummy_img = torch.randn(
                    1, 3, args.image_size, args.image_size,
                    device=device, dtype=torch.float32,
                )
                _ = model(dummy_img, text_prompt="dummy")
                if was_dtype is not None:
                    model = model.to(was_dtype)
            print("  ✓ LazyConv2d layers initialized")
        except Exception as e:
            print(f"  ⚠  Warning: dummy forward pass failed ({e}); "
                  "checkpoint weights for uninitialized layers may be skipped.")
            if not use_device_map and was_dtype is not None:
                try:
                    model = model.to(was_dtype)
                except Exception:
                    pass

    # Apply LoRA BEFORE loading checkpoint (same order as scripts/inference.py)
    if is_lora_checkpoint:
        if not _HAS_PEFT:
            raise ImportError("LoRA checkpoint detected but peft library not found.")
        print("\n" + "=" * 70)
        print("Applying LoRA (Required for LoRA Checkpoint)")
        print("=" * 70)
        model = setup_lora_for_inference(
            model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=getattr(args, 'lora_target_modules', None),
        )
        print("✓ LoRA applied - model structure now matches checkpoint")

    model = load_checkpoint(model, args.checkpoint, device, use_cpu_map=use_device_map)
    print("✓ Model ready for inference")

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    errors = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as out_f:
        for sample in tqdm(pending, desc="Generating", dynamic_ncols=True):
            sid = sample.get("id", "")

            img_path = resolve_image_path(sample, merged_base, split=args.split)
            if img_path is None:
                tqdm.write(f"  SKIP (image not found): {sid}")
                errors += 1
                continue

            # Resolve dataset type and prompt — same logic as scripts/inference.py
            # when called with --category <dataset_name>
            dataset_name = get_dataset_from_id(sid)
            dataset_type = get_dataset_type_from_name(dataset_name)
            text_prompt = get_user_prompt_for_dataset_type(dataset_type)

            try:
                pixel_values, _, _ = preprocess_image(img_path, args.image_size)
                pixel_values = pixel_values.to(device, dtype=model.dtype)

                # Exact same call as scripts/inference.py --generate_text
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        prediction = model.generate_text(
                            pixel_values,
                            text_prompt,
                            max_new_tokens=args.max_new_tokens,
                            top_p=args.top_p,
                            do_sample=args.do_sample,
                            dataset_type=dataset_type,
                        )

                record = {
                    "id": sid,
                    "image": img_path,
                    "source_dataset": dataset_name,
                    "prompt": text_prompt,
                    "prediction": prediction,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

            except Exception as exc:
                tqdm.write(f"  ERROR ({sid}): {exc}")
                errors += 1
                import traceback
                traceback.print_exc()
                continue

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_done = len(done_ids) + len(pending) - errors
    print(f"\nDone. {total_done} predictions written to {output_path}")
    if errors:
        print(f"  {errors} samples skipped due to errors")


if __name__ == "__main__":
    main()
