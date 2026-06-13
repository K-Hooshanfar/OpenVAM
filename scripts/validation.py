
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run your trained saliency model on a directory of images and
evaluate predictions against ground-truth density + fixation maps.

Usage (example):
    python scripts/validation.py \
        --checkpoint /path/to/best_model.pth \
        "/path/to/dataset/images" \
        "/path/to/dataset/saliency_maps" \
        "/path/to/dataset/outputs" \
        my_dataset

Positional arguments:
    1. images_dir     : directory with original stimulus images
    2. saliency_dir   : directory with ground-truth saliency / density maps
    3. fixation_dir   : directory with fixation maps (from eye tracking)
    4. dataset_tag    : free-form string printed in the summary (e.g. my_dataset)

Required flag:
    --checkpoint      : path to the trained model checkpoint (.pth)

The script:
    - loads your trained model (Stage 1 or Stage 2 LoRA, auto-detected)
    - runs saliency prediction on each image
    - compares the prediction to the ground-truth saliency + fixation maps
    - prints average metrics (MSE, KLD, CC, SIM, NSS, AUC) at the end.
"""

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_stage3_lora import (
    OpenVAM,
    CombinedSaliencyLoss,
    _get_base_model,
    setup_lora_for_qwen_layers,
    load_checkpoint_weights,
)
from scripts.inference import load_training_args, check_if_lora_checkpoint
from utils.losses import AUC_Judd


VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _find_map_path(dir_path: Path, stem: str) -> Optional[Path]:
    """Find a map file with given stem in dir_path, trying common image extensions."""
    for ext in VALID_EXTS:
        for e in (ext, ext.upper()):
            candidate = dir_path / f"{stem}{e}"
            if candidate.exists():
                return candidate
    return None


def _load_gray_tensor(path: Path, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Load a single-channel map as float tensor in [0, 1], shape [1, H, W]."""
    img = Image.open(path).convert("L")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class SimpleValDataset(Dataset):
    """
    Validation dataset built from three folders:
      - images_dir   : RGB stimuli
      - saliency_dir : ground-truth density / saliency maps
      - fixation_dir : fixation maps

    Files are matched by stem (filename without extension).
    """

    def __init__(self, images_dir: Path, saliency_dir: Path, fixation_dir: Path, image_size: int = 256):
        self.images_dir = images_dir
        self.saliency_dir = saliency_dir
        self.fixation_dir = fixation_dir
        self.image_size = image_size

        self.samples: List[Tuple[str, Path, Path, Path]] = []
        stems: Dict[str, Dict[str, Path]] = {}

        def _add_files(root: Path, key: str) -> None:
            for p in root.iterdir():
                if p.is_file() and p.suffix.lower() in VALID_EXTS:
                    stem = p.stem
                    if stem not in stems:
                        stems[stem] = {}
                    stems[stem][key] = p

        _add_files(self.images_dir, "image")
        _add_files(self.saliency_dir, "saliency")
        _add_files(self.fixation_dir, "fixation")

        for stem, d in stems.items():
            if {"image", "saliency", "fixation"}.issubset(d.keys()):
                self.samples.append((stem, d["image"], d["saliency"], d["fixation"]))

        self.samples.sort(key=lambda x: x[0])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        stem, img_path, sal_path, fix_path = self.samples[idx]

        # Load and resize image to model input size
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_arr = np.asarray(img, dtype=np.float32) / 255.0
        img_arr = np.transpose(img_arr, (2, 0, 1))  # HWC -> CHW
        image = torch.from_numpy(img_arr)

        # Load GT saliency and fixation; keep their own resolution
        saliency = _load_gray_tensor(sal_path)
        fixation = _load_gray_tensor(fix_path)

        return {
            "image": image,
            "saliency": saliency,
            "fixation": fixation,
            "id": stem,
        }


def _is_no_vit_checkpoint(checkpoint_path: str) -> bool:
    """
    Detect if checkpoint comes from no_vit.py (DPT-only) instead of text-compatible model.
    Copied from eval_val_per_dataset.py.
    """
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
    has_text_compatible = any(
        k.startswith("dino_adapter") or k.startswith("patch_merger") for k in keys
    )
    if not has_text_compatible and "proj_to_l4.weight" in state_dict:
        w = state_dict["proj_to_l4.weight"]
        if w.shape[1] == 2048:
            has_text_compatible = True
    return not has_text_compatible


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run trained saliency model on a folder and report metrics."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (.pth) from saliency+text training.",
    )
    parser.add_argument("images_dir", type=str, help="Directory with original images")
    parser.add_argument(
        "saliency_dir",
        type=str,
        help="Directory with ground-truth saliency / density maps",
    )
    parser.add_argument(
        "fixation_dir",
        type=str,
        help="Directory with fixation maps",
    )
    parser.add_argument(
        "dataset_tag",
        type=str,
        help="Dataset tag/name for printing only (e.g. 'my_dataset')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader num_workers",
    )
    parser.add_argument(
        "--print_per_image",
        action="store_true",
        help="Print metrics for each individual image as well as the summary.",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    images_dir = Path(args.images_dir)
    saliency_dir = Path(args.saliency_dir)
    fixation_dir = Path(args.fixation_dir)

    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")
    if not saliency_dir.is_dir():
        raise SystemExit(f"Saliency directory not found: {saliency_dir}")
    if not fixation_dir.is_dir():
        raise SystemExit(f"Fixation directory not found: {fixation_dir}")

    # Load training config (image_size, LoRA config, etc.)
    training_args = load_training_args(str(checkpoint_path)) or {}
    image_size = training_args.get("image_size", 256)
    dino_model = training_args.get("dino_model", "facebook/dinov3-vitb16-pretrain-lvd1689m")
    backbone = training_args.get("backbone", "vitb_rn50_384")
    features = training_args.get("features", 256)
    qwen_model = training_args.get("qwen_model", "Qwen/Qwen2.5-VL-3B-Instruct")
    lora_r = training_args.get("lora_r", 16)
    lora_alpha = training_args.get("lora_alpha", 32)
    lora_dropout = training_args.get("lora_dropout", 0.1)
    lora_target_modules = training_args.get(
        "lora_target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    lora_lm_head = training_args.get("lora_lm_head", False)
    lora_lm_head_r = training_args.get("lora_lm_head_r", 8)
    lora_lm_head_alpha = training_args.get("lora_lm_head_alpha", 16)
    lora_lm_head_dropout = training_args.get("lora_lm_head_dropout", 0.0)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Detect if this is a no_vit (DPT-only) checkpoint
    is_no_vit = _is_no_vit_checkpoint(str(checkpoint_path))

    if is_no_vit:
        from net.saliency_net import OpenVAMSaliencyNet

        print("\nDetected no_vit checkpoint — using OpenVAMSaliencyNet.")
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
        ).to(device)

        # Initialize LazyConv2d via dummy forward
        with torch.no_grad():
            dummy = torch.rand(1, 3, image_size, image_size, device=device)
            _ = model(dummy)

        # Load checkpoint weights
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state_dict, strict=False)
        base_model = model
        use_no_vit = True
    else:
        print("\nCreating OpenVAM model...")
        model = OpenVAM(
            dino_model_name=dino_model,
            qwen_model_name=qwen_model,
            backbone=backbone,
            features=features,
            freeze_dino=False,
            freeze_qwen_lm=True,
            freeze_projector=False,
            dtype=torch.bfloat16,
            device_map=None,
        )
        model = model.to(device)

        # Initialize LazyConv2d via dummy forward before loading weights
        with torch.no_grad():
            try:
                dummy_img = torch.randn(1, 3, image_size, image_size, device=device, dtype=torch.float32)
                _ = model(dummy_img, text_prompt="dummy")
            except Exception as e:
                print(f"Warning: dummy forward failed: {e}")

        # Apply LoRA if checkpoint has LoRA weights
        is_lora = check_if_lora_checkpoint(str(checkpoint_path))
        if is_lora:
            print("Applying LoRA adapters for inference...")
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
            print("Non-LoRA (Stage 1) checkpoint detected.")

        print(f"Loading checkpoint weights from {checkpoint_path}")
        load_checkpoint_weights(model, str(checkpoint_path), strict=False, verbose=True)

        base_model = _get_base_model(model)
        use_no_vit = False

    model.eval()

    # Loss/metric function (reuses training CombinedSaliencyLoss)
    saliency_loss_fn = CombinedSaliencyLoss(
        kld_weight=1.0,
        cc_weight=1.0,
        sim_weight=1.0,
        nss_weight=1.0,
        mse_weight=1.0,
    ).to(device)

    # Dataset + loader
    dataset = SimpleValDataset(images_dir, saliency_dir, fixation_dir, image_size=image_size)
    if len(dataset) == 0:
        raise SystemExit("No matching (image, saliency, fixation) triplets found in the given folders.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Evaluating on {len(dataset)} images from '{args.dataset_tag}'")

    # Metric accumulators
    n_samples = 0
    mse_vals: List[float] = []
    kld_vals: List[float] = []
    cc_vals: List[float] = []
    sim_vals: List[float] = []
    nss_vals: List[float] = []
    auc_vals: List[float] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            saliency = batch["saliency"].to(device=device, dtype=torch.float32)
            fixation = batch["fixation"].to(device=device, dtype=torch.float32)
            ids = batch["id"]

            if use_no_vit:
                preds = base_model(images)
            else:
                # For external datasets we use no text prompt and a default dataset_type
                preds = base_model(images, text_prompt=None, dataset_type="natural_scene")

            preds = preds.float()
            saliency = saliency.float()
            fixation = fixation.float()

            _, loss_dict = saliency_loss_fn(preds, saliency, fixation)

            bsz = images.shape[0]
            for i in range(bsz):
                n_samples += 1
                mse_vals.append(float(loss_dict.get("mse", 0.0)))
                kld_vals.append(float(loss_dict.get("kld", 0.0)))
                cc_vals.append(float(loss_dict.get("cc", 0.0)))
                sim_vals.append(float(loss_dict.get("sim", 0.0)))
                nss_vals.append(float(loss_dict.get("nss", 0.0)))

                # Compute AUC per-sample, matching training/eval_val_per_dataset style
                p_i = preds[i : i + 1].float()
                f_i = fixation[i : i + 1].float()
                if AUC_Judd is not None and f_i[0, 0].sum() > 0:
                    auc_v = float(AUC_Judd(p_i[0, 0], f_i[0, 0]))
                else:
                    auc_v = float("nan")
                auc_vals.append(auc_v)

                if args.print_per_image:
                    name = ids[i]
                    auc_s = f"{auc_v:.4f}" if not math.isnan(auc_v) else "—"
                    print(
                        f"{name}: "
                        f"MSE={mse_vals[-1]:.4f}  "
                        f"KLD={kld_vals[-1]:.4f}  "
                        f"CC={cc_vals[-1]:.4f}  "
                        f"SIM={sim_vals[-1]:.4f}  "
                        f"NSS={nss_vals[-1]:.4f}  "
                        f"AUC={auc_s}"
                    )

    if n_samples == 0:
        raise SystemExit("No samples were evaluated; nothing to report.")

    def _avg(xs: List[float]) -> float:
        xs_f = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
        return sum(xs_f) / len(xs_f) if xs_f else float("nan")

    avg_mse = _avg(mse_vals)
    avg_kld = _avg(kld_vals)
    avg_cc = _avg(cc_vals)
    avg_sim = _avg(sim_vals)
    avg_nss = _avg(nss_vals)
    avg_auc = _avg(auc_vals)

    print("\n" + "=" * 70)
    print(f"Validation metrics for dataset '{args.dataset_tag}'")
    print("=" * 70)
    print(f"Num samples: {n_samples}")
    print(
        f"MSE={avg_mse:.4f}  "
        f"KLD={avg_kld:.4f}  "
        f"CC={avg_cc:.4f}  "
        f"SIM={avg_sim:.4f}  "
        f"NSS={avg_nss:.4f}  "
        f"AUC={avg_auc:.4f}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()

