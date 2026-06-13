#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained OpenVAM checkpoint (Stage I OpenVAMSaliencyNet or a full OpenVAM
model) on the merged validation set and report saliency metrics (MSE, KLD, CC, SIM,
NSS) per dataset. Uses merged_val.jsonl only; sample ids have the form
{dataset_name}_{original_id}, so metrics are grouped by dataset name prefix.

Usage:
    python scripts/eval_per_dataset.py --checkpoint path/to/best_model.pth
    python scripts/eval_per_dataset.py --checkpoint path/to/best_model.pth --datasets_dir datasets --output_metrics results.json
"""

import math
import os
import sys
import json
import argparse
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model, dataset, and loss from the Stage-III LoRA training script (saliency-only here)
from scripts.train_stage3_lora import (
    OpenVAM,
    SalienceTextDataset,
    collate_fn,
    CombinedSaliencyLoss,
    _get_base_model,
    setup_lora_for_qwen_layers,
    load_checkpoint_weights,
)
from scripts.inference import load_training_args, check_if_lora_checkpoint
from net.openvam import get_dataset_type_from_name
from tqdm import tqdm

from utils.losses import AUC_Judd


def _is_no_vit_checkpoint(checkpoint_path: str) -> bool:
    """Return True if checkpoint is from no_vit.py (DPT-only). False for text-compatible (has dino_adapter/patch_merger)."""
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    keys = set(state_dict.keys())
    # state_dict keys are "dino_adapter.xxx", "patch_merger.xxx", not bare "dino_adapter"
    has_text_compatible = any(
        k.startswith("dino_adapter") or k.startswith("patch_merger") for k in keys
    )
    # Also treat proj_to_l4 with in_channels 2048 as text-compatible (no_vit has 768)
    if not has_text_compatible and "proj_to_l4.weight" in state_dict:
        w = state_dict["proj_to_l4.weight"]
        if w.shape[1] == 2048:
            has_text_compatible = True
    return not has_text_compatible

# Dataset names in merged JSONL ids (id format: {dataset_name}_{original_id}). Sort by length desc so "datasets_UI_256" matches before "datasets".
MERGED_DATASET_NAMES = [
    "datasets_UI_256",
    "CAT2000_256",
    "MIT1003_256",
    "OSIE_256",
    "salicon_256",
    "SalEC",
]


def get_dataset_from_id(sample_id: str) -> str:
    """Extract dataset name from merged sample id (e.g. CAT2000_256_Action_001 -> CAT2000_256)."""
    if not sample_id:
        return "unknown"
    for name in MERGED_DATASET_NAMES:
        if sample_id.startswith(name + "_"):
            return name
    return "other"


# ImageNet normalization for no_vit (OpenVAMSaliencyNet expects normalized input)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@torch.no_grad()
def validate_saliency_only_per_dataset(model, dataloader, saliency_loss_fn, device, use_no_vit=False):
    """Run validation on merged val; compute saliency metrics per dataset. use_no_vit: model is OpenVAMSaliencyNet (no text)."""
    model.eval()
    base_model = _get_base_model(model)
    base_dtype = getattr(base_model, "dtype", next(base_model.parameters(), torch.tensor(0.0)).dtype)
    if not torch.is_tensor(base_dtype):
        base_dtype = torch.float32
    per_dataset = {}
    for batch in tqdm(dataloader, desc="Val (saliency)"):
        images = batch["image"].to(device, dtype=base_dtype)
        saliency = batch["saliency"].to(device, dtype=base_dtype)
        fixation = batch["fixation"].to(device, dtype=base_dtype)
        texts = batch["text"]
        ids = batch.get("id", [""] * len(texts))
        if use_no_vit:
            # no_vit expects ImageNet-normalized input (images from dataset are in [0, 1])
            mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=images.dtype).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD, device=device, dtype=images.dtype).view(1, 3, 1, 1)
            images = (images - mean) / std
        for i in range(len(texts)):
            ds = get_dataset_from_id(ids[i] if i < len(ids) else "")
            if ds not in per_dataset:
                per_dataset[ds] = {"mse": [], "kld": [], "cc": [], "sim": [], "nss": [], "auc": []}
            image_i = images[i : i + 1]
            saliency_i = saliency[i : i + 1]
            fixation_i = fixation[i : i + 1]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if use_no_vit:
                    pred_saliency = base_model(image_i)
                else:
                    text_prompt = texts[i] if texts[i] else None
                    dataset_type = get_dataset_type_from_name(ds)
                    pred_saliency = base_model(image_i, text_prompt=text_prompt, dataset_type=dataset_type)
            # Run loss in float32 so AUC_Judd gets .numpy()-compatible tensors (bfloat16 raises)
            pred_saliency = pred_saliency.float()
            saliency_i = saliency_i.float()
            fixation_i = fixation_i.float()
            _, saliency_loss_dict = saliency_loss_fn(pred_saliency, saliency_i, fixation_i)
            per_dataset[ds]["mse"].append(saliency_loss_dict.get("mse", 0))
            per_dataset[ds]["kld"].append(saliency_loss_dict.get("kld", 0))
            per_dataset[ds]["cc"].append(saliency_loss_dict.get("cc", 0))
            per_dataset[ds]["sim"].append(saliency_loss_dict.get("sim", 0))
            per_dataset[ds]["nss"].append(saliency_loss_dict.get("nss", 0))
            # Compute AUC directly so we don't depend on loss returning "auc" (shapes: pred [1,1,H,W], fix [1,1,H,W])
            if AUC_Judd is not None and fixation_i[0, 0].sum() > 0:
                auc_v = float(AUC_Judd(pred_saliency[0, 0], fixation_i[0, 0]))
            else:
                auc_v = saliency_loss_dict.get("auc", float("nan"))
            per_dataset[ds]["auc"].append(auc_v)
    # Average per dataset (skip None/nan for AUC)
    results = {}
    for ds, vals in per_dataset.items():
        n = len(vals["mse"])
        if n == 0:
            continue
        auc_list = [x for x in vals["auc"] if x is not None and not (isinstance(x, float) and math.isnan(x))]
        results[ds] = {
            "n_samples": n,
            "mse": sum(vals["mse"]) / n,
            "kld": sum(vals["kld"]) / n,
            "cc": sum(vals["cc"]) / n,
            "sim": sum(vals["sim"]) / n,
            "nss": sum(vals["nss"]) / n,
            "auc": sum(auc_list) / len(auc_list) if auc_list else float("nan"),
            "total": (sum(vals["mse"]) + sum(vals["kld"])) / n,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on each dataset val and report metrics")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (Stage 1 best_model.pth or Stage 2 LoRA best_model.pth)")
    parser.add_argument("--datasets_dir", type=str, default="datasets",
                        help="Directory containing *_val.jsonl and optionally merged/merged_val.jsonl")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--output_metrics", type=str, default=None,
                        help="Path to save metrics JSON (default: <checkpoint_dir>/eval_metrics_per_dataset.json)")
    parser.add_argument("--device_map", type=str, default=None,
                        help="Device map for multi-GPU (e.g. 'auto' or 'balanced'). Use for 7B to avoid OOM.")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    checkpoint_dir = checkpoint_path.parent

    # Detect no_vit checkpoint (from no_vit.py - Stage 1 saliency-only, no text)
    is_no_vit = _is_no_vit_checkpoint(str(checkpoint_path))
    if is_no_vit:
        print("Detected no_vit (Stage 1 / no_vit.py) checkpoint. Using OpenVAMSaliencyNet and loading full weights.")

    # Load training args from checkpoint directory (optional for Stage 1 / no_vit checkpoints)
    training_args = load_training_args(str(checkpoint_path))
    if not training_args:
        print("WARNING: No args.json in checkpoint directory. Using default model config.")
        training_args = {}
    dino_model = training_args.get("dino_model", "facebook/dinov3-vitb16-pretrain-lvd1689m")
    backbone = training_args.get("backbone", "vitb_rn50_384")
    features = training_args.get("features", 256)
    image_size = training_args.get("image_size", 256)
    qwen_model = training_args.get("qwen_model", "Qwen/Qwen2.5-VL-3B-Instruct")
    lora_r = training_args.get("lora_r", 16)
    lora_alpha = training_args.get("lora_alpha", 32)
    lora_dropout = training_args.get("lora_dropout", 0.1)
    lora_target_modules = training_args.get("lora_target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    lora_lm_head = training_args.get("lora_lm_head", False)
    lora_lm_head_r = training_args.get("lora_lm_head_r", 8)
    lora_lm_head_alpha = training_args.get("lora_lm_head_alpha", 16)
    lora_lm_head_dropout = training_args.get("lora_lm_head_dropout", 0.0)
    saliency_weight = training_args.get("saliency_weight", 0.2)
    text_weight = training_args.get("text_weight", 0.5)
    mse_weight = training_args.get("mse_weight", 1.0)
    kld_weight = training_args.get("kld_weight", 1.0)
    cc_weight = training_args.get("cc_weight", 1.0)
    sim_weight = training_args.get("sim_weight", 0.5)
    nss_weight = training_args.get("nss_weight", 0.1)

    use_device_map = (args.device_map is not None and args.device_map.strip() != "")
    if use_device_map:
        device = None  # set after model creation from DINO device
        print(f"Using device_map: {args.device_map} (multi-GPU); primary device will be set after model load.")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    if is_no_vit:
        # Build no_vit model (OpenVAMSaliencyNet from no_vit.py) and load full checkpoint
        from net.saliency_net import OpenVAMSaliencyNet
        print("\nCreating OpenVAMSaliencyNet (no_vit)...")
        model = OpenVAMSaliencyNet(
            use_hf=True,
            hf_model_name=dino_model,
            backbone=backbone,
            features=features,
            readout="project",
            upsample_output_to_input_res=True,
            out_channels=1,
            freeze_dino=False,
            freeze_vit=False,
        )
        model = model.to(device)
        model.eval()
        # Initialize LazyConv2d with a dummy forward
        print("Initializing LazyConv2d (no_vit)...")
        with torch.no_grad():
            try:
                mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
                std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
                dummy = (torch.rand(1, 3, image_size, image_size, device=device) - mean) / std
                _ = model(dummy)
            except Exception as e:
                print(f"  Warning: dummy forward failed: {e}")
        # Load checkpoint (full state_dict for no_vit)
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded {len(state_dict)} weights into OpenVAMSaliencyNet")
    else:
        # Build text-compatible model (Qwen + DINO + DPT)
        print("\nCreating model...")
        model = OpenVAM(
            dino_model_name=dino_model,
            qwen_model_name=qwen_model,
            backbone=backbone,
            features=features,
            freeze_dino=False,
            freeze_qwen_lm=True,
            freeze_projector=False,
            dtype=torch.bfloat16 if use_device_map else None,
            device_map=args.device_map if use_device_map else None,
        )
        if use_device_map:
            device = next(model.dino.parameters()).device
            print(f"Primary device (DINO/inputs): {device}")
        else:
            model = model.to(device)
        model.eval()
        print("Initializing LazyConv2d...")
        with torch.no_grad():
            try:
                if use_device_map:
                    dummy_img = torch.randn(1, 3, image_size, image_size, device=device, dtype=torch.bfloat16)
                    _ = model(dummy_img, text_prompt="dummy")
                else:
                    dummy_img = torch.randn(1, 3, image_size, image_size, device=device, dtype=torch.float32)
                    _ = model(dummy_img, text_prompt="dummy")
            except Exception as e:
                print(f"  Warning: dummy forward failed: {e}")
        is_lora = check_if_lora_checkpoint(str(checkpoint_path))
        if is_lora:
            print("Applying LoRA...")
            model = setup_lora_for_qwen_layers(
                model,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                lora_lm_head=lora_lm_head,
                lora_lm_head_r=lora_lm_head_r,
                lora_lm_head_alpha=lora_lm_head_alpha,
                lora_lm_head_dropout=lora_lm_head_dropout,
            )
            if hasattr(model, "rebind_lora_references"):
                model.rebind_lora_references()
        else:
            print("Stage 1 (non-LoRA) checkpoint: skipping LoRA.")
        print(f"Loading checkpoint: {checkpoint_path}")
        load_checkpoint_weights(model, str(checkpoint_path), strict=False, verbose=True)

    # Loss for validation (saliency metrics only; text weight can be 0 for reporting)
    saliency_loss_fn = CombinedSaliencyLoss(
        kld_weight=kld_weight,
        cc_weight=cc_weight,
        sim_weight=sim_weight,
        nss_weight=nss_weight,
        mse_weight=mse_weight,
    ).to(device)

    # Use only merged val; metrics will be grouped by dataset name from sample id prefix
    datasets_dir = Path(args.datasets_dir)
    merged_val = datasets_dir / "merged" / "merged_val.jsonl"
    merged_dir = datasets_dir / "merged"
    if not merged_val.exists():
        print(f"ERROR: Merged val not found: {merged_val}")
        print("Run merge_datasets_unified.py first so merged/merged_val.jsonl and merged/val/ exist.")
        sys.exit(1)

    print(f"\nLoading merged val: {merged_val}")
    dataset = SalienceTextDataset(
        jsonl_path=str(merged_val),
        saliency_dir=str(merged_dir),
        fixation_dir=str(merged_dir),
        image_size=image_size,
        augment=False,
    )
    if len(dataset) == 0:
        print("ERROR: Merged val dataset is empty.")
        sys.exit(1)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    print(f"Evaluating on {len(dataset)} samples; metrics will be grouped by dataset (from id prefix).")
    per_ds = validate_saliency_only_per_dataset(model, loader, saliency_loss_fn, device, use_no_vit=is_no_vit)
    results = {}
    for name, m in per_ds.items():
        auc_val = m.get("auc", float("nan"))
        results[name] = {
            "n_samples": m["n_samples"],
            "MSE": round(m["mse"], 4),
            "KLD": round(m["kld"], 4),
            "CC": round(m["cc"], 4),
            "SIM": round(m["sim"], 4),
            "NSS": round(m["nss"], 4),
            "AUC": round(auc_val, 4) if not math.isnan(auc_val) else None,
            "val_loss": round(m["total"], 4),
        }
        auc_s = f"  AUC={results[name]['AUC']}" if results[name].get("AUC") is not None else "  AUC=—"
        print(f"  {name}: n={results[name]['n_samples']}  MSE={results[name]['MSE']}  KLD={results[name]['KLD']}  CC={results[name]['CC']}  SIM={results[name]['SIM']}  NSS={results[name]['NSS']}{auc_s}")

    # Summary table
    print("\n" + "=" * 70)
    print("Metrics per dataset (val)")
    print("=" * 70)
    for name, m in results.items():
        auc_s = f"  AUC={m['AUC']}" if m.get("AUC") is not None else "  AUC=—"
        print(f"  {name}: MSE={m['MSE']}  KLD={m['KLD']}  CC={m['CC']}  SIM={m['SIM']}  NSS={m['NSS']}{auc_s}  (n={m['n_samples']})")
    print("=" * 70)

    out_path = args.output_metrics
    if out_path is None:
        out_path = checkpoint_dir / "eval_metrics_per_dataset.json"
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": str(checkpoint_path), "datasets": results}, f, indent=2)
    print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    main()
