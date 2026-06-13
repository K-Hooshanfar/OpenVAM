#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Qwen2.5-VL-3B on all images in a sample folder and save outputs to CSV.

CSV columns:
  - image_name
  - output

Example:
    python scripts/sample_folder_explain.py \
        --sample_folder /path/to/sample_images \
        --output_csv ./sample_qwen_outputs.csv \
        --qwen_model Qwen/Qwen2.5-VL-3B-Instruct
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from tqdm import tqdm


PROMPT_IMAGE_ONLY_SHARED = (
    "You are given one original stimulus image.\n"
    "Analyze only visible visual evidence in this image.\n\n"
)


SAL_PROMPT_NATURAL_IMAGE_ONLY = (
    PROMPT_IMAGE_ONLY_SHARED
    + "You are a visual attention explainer for saliency prediction.\n\n"
    "Task:\n"
    "List the 2-6 most visually salient objects or regions in descending order of attention.\n"
    "For EACH object/region you name, give 1-2 short evidence-based sentences explaining why it draws attention "
    "(e.g., size, contrast, color, uniqueness, motion/pose, central placement, sharpness, readable text, interaction/gaze).\n\n"
    "Output format (must follow exactly):\n"
    "One item per line:\n"
    "<Object or region> (<location phrase>): <1-2 short evidence-based sentences>\n\n"
    "Rules:\n"
    "- Use concrete location phrases such as \"center,\" \"left edge,\" \"lower-right,\" \"top-left,\" \"foreground,\" \"background,\" "
    "\"near the edge,\" or \"behind <object>\".\n"
    "- Justify using only visible cues (faces/gaze, readable text, strong contrast/color, sharpness, size, implied motion/pose, "
    "interaction/pointing, uniqueness).\n"
    "- No extra headers, no bullets, no numbering, no blank lines."
)


def try_import_qwen_vl_utils():
    try:
        from qwen_vl_utils import process_vision_info  # type: ignore

        return process_vision_info
    except ImportError:
        return None


def _truthy_env(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def use_local_hf_only(cli_flag: bool) -> bool:
    return cli_flag or _truthy_env("HF_HUB_OFFLINE") or _truthy_env("TRANSFORMERS_OFFLINE")


def hf_hub_root() -> Path:
    return Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"


def resolve_hub_model_to_snapshot_dir(model_id: str, label: str = "") -> str:
    mid = (model_id or "").strip()
    if not mid:
        return mid
    p = Path(mid)
    if p.is_dir():
        rp = str(p.resolve())
        if label:
            print(f"  [{label}] using local directory: {rp}")
        return rp
    if "/" not in mid:
        return mid
    org, name = mid.split("/", 1)
    snap_root = hf_hub_root() / f"models--{org}--{name}" / "snapshots"
    if not snap_root.is_dir():
        return mid
    snaps = sorted(
        [d for d in snap_root.iterdir() if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if not snaps:
        return mid
    chosen = str(snaps[0].resolve())
    if label:
        print(f"  [{label}] using HF hub snapshot (offline-friendly): {chosen}")
    return chosen


def load_qwen_vl(
    model_name: str,
    device_map: str,
    hf_token: str | None,
    local_files_only: bool = False,
):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    load_id = resolve_hub_model_to_snapshot_dir(model_name, label="Qwen") if local_files_only else model_name

    tok_kw: Dict[str, Any] = {"local_files_only": bool(local_files_only)}
    if hf_token:
        tok_kw["token"] = hf_token

    processor = AutoProcessor.from_pretrained(load_id, **tok_kw)

    load_kw: Dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": device_map,
        "local_files_only": bool(local_files_only),
    }
    if hf_token:
        load_kw["token"] = hf_token
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(load_id, **load_kw)
    return processor, model


def generate_for_image(
    processor,
    model,
    pil_image: Image.Image,
    user_text: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    process_vision_info,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

    gen_kw: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kw["temperature"] = temperature
        gen_kw["top_p"] = top_p

    with torch.inference_mode():
        out_ids = model.generate(**inputs, **gen_kw)

    in_len = inputs["input_ids"].shape[1]
    new_tokens = out_ids[0, in_len:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def collect_images(sample_folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}
    images = [p for p in sample_folder.iterdir() if p.is_file() and p.suffix in exts]
    images.sort(key=lambda p: p.name.lower())
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen2.5-VL image-only inference from sample folder to CSV")
    parser.add_argument("--sample_folder", type=str, required=True, help="Folder containing input images")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV path")
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device_map", type=str, default="auto", help="Qwen device_map, e.g. auto or cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip image names already in output CSV")
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load model from local HF cache/directory only (no internet).",
    )
    args = parser.parse_args()

    local_only = use_local_hf_only(args.local_files_only)
    if local_only and not args.local_files_only:
        print("Offline mode inferred from HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE.")
    if local_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    process_vision_info = try_import_qwen_vl_utils()
    if process_vision_info is None:
        raise RuntimeError("qwen_vl_utils is required. Install with: pip install qwen-vl-utils")

    sample_folder = Path(args.sample_folder)
    if not sample_folder.exists() or not sample_folder.is_dir():
        raise ValueError(f"--sample_folder is not a valid directory: {sample_folder}")

    images = collect_images(sample_folder)
    if not images:
        print(f"No images found in folder: {sample_folder}")
        return
    print(f"Found {len(images)} images in {sample_folder}")

    print(f"Loading Qwen model: {args.qwen_model} (device_map={args.device_map}, local_only={local_only})")
    token = args.hf_token or os.getenv("HF_TOKEN")
    processor, model = load_qwen_vl(
        args.qwen_model,
        args.device_map,
        token,
        local_files_only=local_only,
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    done_names = set()
    csv_exists = output_csv.exists()
    if args.resume and csv_exists:
        with open(output_csv, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            _ = next(reader, None)
            for row in reader:
                if row and row[0].strip():
                    done_names.add(row[0].strip())
        print(f"Resume enabled: {len(done_names)} images already in CSV")

    mode = "a" if (args.resume and csv_exists) else "w"
    with open(output_csv, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(["image_name", "output"])

        for img_path in tqdm(images, desc="qwen-image-only"):
            image_name = img_path.name
            if args.resume and image_name in done_names:
                continue
            try:
                pil_image = Image.open(img_path).convert("RGB")
                output_text = generate_for_image(
                    processor=processor,
                    model=model,
                    pil_image=pil_image,
                    user_text=SAL_PROMPT_NATURAL_IMAGE_ONLY,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    process_vision_info=process_vision_info,
                )
            except Exception as e:
                output_text = f"[ERROR] {e}"
            writer.writerow([image_name, output_text])
            f.flush()

    print(f"Done. Wrote CSV to: {output_csv}")


if __name__ == "__main__":
    main()

