#!/usr/bin/env python3
"""
Merge multiple saliency datasets into one unified layout:

  out_dir/
    train/
      stimuli/
      saliency/
      fixations/
    val/
      stimuli/
      saliency/
      fixations/

- Reads *_train.jsonl and *_val.jsonl from datasets_dir.
- Each sample gets a unique id: {dataset_name}_{original_id} to avoid collisions.
- Copies (or symlinks) stimuli, saliency maps, and fixation maps into out_dir.
- Supports CAT2000/datasets_UI layout (train/train_saliency, train/train_fixation) and
  salicon layout (saliency/train, fixations/train). Saliency and fixations copied by default.
- Writes merged_train.jsonl and merged_val.jsonl with updated ids and paths.

Usage (run on the machine where image files live):
  python merge_datasets_unified.py --datasets_dir datasets --out_dir /path/to/merged

If salicon (or others) are missing: JSONL may point to ff/salicon_256/ while data is under
ff2/z/datasets/. When the image path does not exist, the script looks under datasets_base (if
given) or else out_dir.parent (e.g. .../datasets/merged -> .../datasets). So with
--out_dir .../datasets/merged no extra flag is needed if the datasets live in .../datasets/.

Only datasets that have both {name}_train.jsonl and {name}_val.jsonl in datasets_dir
are merged. Each sample id becomes {dataset_name}_{id} so IDs are unique across sets.
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from collections import Counter

# Default dataset names that have JSONL in this repo (used for ordering and filtering)
DEFAULT_DATASETS = [
    "CAT2000_256",
    "MIT1003_256",
    "OSIE_256",
    "SalEC",
    "datasets_UI_256",
    "fiwi_256",
    "salicon_256",
]

# Extensions we treat as images or maps
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAP_EXTS = {".png", ".jpg", ".jpeg"}


def _sanitize_id(s: str) -> str:
    """Make id safe for filenames (no path separators, minimal special chars)."""
    return re.sub(r'[<>:"/\\|?*]', "_", s).strip() or "unknown"


def _find_saliency_or_fixation_dir(
    image_path: Path, dataset_name: str, kind: str, kind_aliases: list[str] | None = None
) -> Path | None:
    """
    Given image path and dataset name, try to find saliency or fixation dir.
    kind: primary name (e.g. "saliency", "fixations").
    kind_aliases: optional extra names to try (e.g. ["fixation"] for train_fixation).
    Layouts supported:
      - CAT2000_256, datasets_UI_256: base/train/train_saliency, base/train/train_fixation
      - salicon_256: base/saliency/train, base/fixations/train
    """
    # image_path might be .../CAT2000_256/train/train_stimuli/Action_001.jpg or .../salicon_256/stimuli/train/COCO_xxx.jpg
    parts = image_path.parts
    try:
        idx = parts.index(dataset_name)
    except ValueError:
        return None
    base = Path(*parts[: idx + 1])
    train_or_val = "train" if "train" in parts else "val"
    kinds_to_try = [kind] + (kind_aliases or [])
    # fiwi_256: fiwi_train/stimuli, fiwi_train/saliency, fiwi_val/...
    fiwi_split = "fiwi_val" if "fiwi_val" in parts else "fiwi_train"
    for k in kinds_to_try:
        candidates = [
            base / train_or_val / f"{train_or_val}_{k}",  # base/train/train_saliency (CAT2000, MIT, OSIE, SalEC, datasets_UI)
            base / train_or_val / k,                       # base/train/saliency
            base / fiwi_split / k,                         # base/fiwi_train/saliency (fiwi_256)
            base / k / train_or_val,                      # base/saliency/train (salicon saliency)
            base / k / ("output_train" if train_or_val == "train" else "output_val"),  # salicon fixations
            base / k / ("train_edit" if train_or_val == "train" else "val_edit"),
            base / f"{train_or_val}_{k}",                 # base/train_saliency (flat)
            base / k,
        ]
        for c in candidates:
            if c.exists() and c.is_dir():
                return c
    return None


def _resolve_path(path_str: str, path_rewrite: list[tuple[str, str]] | None) -> Path:
    """Apply path rewrites so the file can be found (e.g. different server path)."""
    p = Path(path_str)
    if path_rewrite:
        s = p.as_posix()
        for old_prefix, new_prefix in path_rewrite:
            if s.startswith(old_prefix):
                s = new_prefix + s[len(old_prefix) :]
                return Path(s)
    return p


def _find_image_under_base(datasets_base: Path, dataset_name: str, split: str, stem: str) -> Path | None:
    """
    When the path in JSONL does not exist, try to find the image under datasets_base/dataset_name/
    using known layout: train/train_stimuli, stimuli/train (salicon), train_images (datasets_UI), fiwi_train/stimuli.
    """
    base = datasets_base / dataset_name
    if not base.exists():
        return None
    train_val = "train" if split == "train" else "val"
    # Order: CAT2000-style, salicon, datasets_UI, fiwi
    candidates = [
        base / train_val / f"{train_val}_stimuli",
        base / train_val / f"{train_val}_images",
        base / "stimuli" / train_val,
        base / ("fiwi_train" if split == "train" else "fiwi_val") / "stimuli",
    ]
    for folder in candidates:
        if folder.exists():
            for ext in IMAGE_EXTS:
                for e in (ext, ext.upper()):
                    p = folder / f"{stem}{e}"
                    if p.exists():
                        return p
    return None


# SalEC and similar: fixation = {id}_fixPts.png, saliency = {id}_fixMap.jpg
MAP_STEM_SUFFIXES_SALIENCY = ("", "_fixMap", "_saliency")
MAP_STEM_SUFFIXES_FIXATION = ("", "_fixPts", "_fixMap")


def _find_map_file(
    search_dir: Path, stem: str, suffixes: tuple[str, ...] = ("",)
) -> Path | None:
    """Find a file with given stem (and optional suffixes like _fixPts, _fixMap) in search_dir."""
    for suffix in suffixes:
        for ext in MAP_EXTS:
            for e in (ext, ext.upper()):
                candidate = search_dir / f"{stem}{suffix}{e}"
                if candidate.exists():
                    return candidate
    return None


def main():
    ap = argparse.ArgumentParser(description="Merge saliency datasets into one train/val layout.")
    ap.add_argument("--datasets_dir", type=Path, default=Path(__file__).resolve().parent,
                    help="Directory containing *_train.jsonl and *_val.jsonl")
    ap.add_argument("--out_dir", type=Path,
                    default=Path("dataset_merged"),
                    help="Output root for merged dataset (train/stimuli, val/stimuli, etc.)")
    ap.add_argument("--path_rewrite", type=str, default=None,
                    help="Rewrite path prefix to find files, e.g. '/old/prefix:/new/prefix'")
    ap.add_argument("--datasets_base", type=Path, default=None,
                    help="If an image path from a JSONL does not exist, look under this dir instead. Useful when manifests use absolute paths from another machine.")
    ap.add_argument("--move", action="store_true", help="Move files instead of copy (rename to merged ids, frees source space)")
    ap.add_argument("--symlink", action="store_true", help="Symlink instead of copy (ignored if --move)")
    ap.add_argument("--no_saliency", action="store_true", help="Do not copy/move saliency maps (default: copy both)")
    ap.add_argument("--no_fixations", action="store_true", help="Do not copy fixation maps (default: copy both)")
    ap.add_argument("--datasets", type=str, nargs="*", default=None,
                    help="Dataset names to include (default: all *_train.jsonl stems)")
    args = ap.parse_args()
    copy_saliency = not args.no_saliency
    copy_fixations = not args.no_fixations

    path_rewrite: list[tuple[str, str]] = []
    if args.path_rewrite:
        for part in args.path_rewrite.split(";"):
            if ":" in part:
                a, b = part.split(":", 1)
                path_rewrite.append((a.strip(), b.strip()))

    datasets_dir = args.datasets_dir.resolve()
    out_dir = args.out_dir.resolve()
    # Layout: train/stimuli, train/saliency, train/fixations; val/ same
    out_train_stimuli = out_dir / "train" / "stimuli"
    out_val_stimuli = out_dir / "val" / "stimuli"
    out_train_saliency = out_dir / "train" / "saliency"
    out_val_saliency = out_dir / "val" / "saliency"
    out_train_fixations = out_dir / "train" / "fixations"
    out_val_fixations = out_dir / "val" / "fixations"

    for d in [out_train_stimuli, out_val_stimuli]:
        d.mkdir(parents=True, exist_ok=True)
    # Always create saliency/fixations dirs so merged layout is complete
    out_train_saliency.mkdir(parents=True, exist_ok=True)
    out_val_saliency.mkdir(parents=True, exist_ok=True)
    out_train_fixations.mkdir(parents=True, exist_ok=True)
    out_val_fixations.mkdir(parents=True, exist_ok=True)

    # Discover dataset names from JSONL files (exclude merged output so we don't use merged paths as sources)
    train_glob = [f for f in datasets_dir.glob("*_train.jsonl") if f.name not in ("merged_train.jsonl",)]
    if args.datasets:
        names = [n.strip() for n in args.datasets]
    else:
        names = sorted({f.stem.replace("_train", "") for f in train_glob})

    merged_train = []
    merged_val = []
    all_train_ids = []
    all_val_ids = []
    stats = {"train": 0, "val": 0, "missing_image": 0, "saliency_copied": 0, "fixation_copied": 0}
    per_dataset = {n: {"train": 0, "val": 0} for n in names}

    for dataset_name in names:
        train_file = datasets_dir / f"{dataset_name}_train.jsonl"
        val_file = datasets_dir / f"{dataset_name}_val.jsonl"
        for split, out_list, out_stimuli, out_sal, out_fix, id_list in [
            ("train", merged_train, out_train_stimuli, out_train_saliency, out_train_fixations, all_train_ids),
            ("val", merged_val, out_val_stimuli, out_val_saliency, out_val_fixations, all_val_ids),
        ]:
            src_file = train_file if split == "train" else val_file
            if not src_file.exists():
                print(f"  Skip {dataset_name} {split}: no {src_file.name}")
                continue
            with open(src_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    old_id = obj.get("id", "")
                    new_id = _sanitize_id(f"{dataset_name}_{old_id}")
                    image_path_str = obj.get("image") or (obj.get("images") or [""])[0]
                    if isinstance(image_path_str, list):
                        image_path_str = image_path_str[0]
                    src_image = _resolve_path(image_path_str, path_rewrite)
                    stem = Path(image_path_str).stem
                    ext = Path(image_path_str).suffix or ".jpg"
                    if ext.lower() not in {e.lower() for e in IMAGE_EXTS}:
                        ext = ".jpg"
                    dest_image = out_stimuli / f"{new_id}{ext}"

                    # If path from JSONL does not exist (e.g. salicon under ff/ vs data under ff2/z/datasets), try datasets_base or out_dir's parent
                    if not src_image.exists():
                        base_to_try = (args.datasets_base.resolve() if args.datasets_base else out_dir.parent)
                        fallback = _find_image_under_base(base_to_try, dataset_name, split, stem)
                        if fallback is not None:
                            src_image = fallback

                    if src_image.exists():
                        if args.move:
                            shutil.move(str(src_image), str(dest_image))
                        elif args.symlink:
                            if not dest_image.exists():
                                dest_image.symlink_to(src_image.resolve())
                        else:
                            shutil.copy2(src_image, dest_image)
                    else:
                        stats["missing_image"] += 1
                        if stats["missing_image"] <= 5:
                            print(f"  Missing image: {src_image}")

                    # Saliency / fixations by stem in source dir (support CAT2000 train_saliency/train_fixation and salicon saliency/train, fixations/train)
                    sal_dir = _find_saliency_or_fixation_dir(src_image, dataset_name, "saliency")
                    fix_dir = _find_saliency_or_fixation_dir(src_image, dataset_name, "fixations", kind_aliases=["fixation"])
                    if copy_saliency and sal_dir:
                        src_sal = _find_map_file(sal_dir, stem, MAP_STEM_SUFFIXES_SALIENCY)
                        if src_sal:
                            dest_sal = out_sal / f"{new_id}{src_sal.suffix}"
                            if args.move:
                                shutil.move(str(src_sal), str(dest_sal))
                            else:
                                shutil.copy2(src_sal, dest_sal)
                            stats["saliency_copied"] += 1
                    if copy_fixations and fix_dir:
                        src_fix = _find_map_file(fix_dir, stem, MAP_STEM_SUFFIXES_FIXATION)
                        if src_fix:
                            dest_fix = out_fix / f"{new_id}{src_fix.suffix}"
                            if args.move:
                                shutil.move(str(src_fix), str(dest_fix))
                            else:
                                shutil.copy2(src_fix, dest_fix)
                            stats["fixation_copied"] += 1

                    new_obj = {**obj, "id": new_id, "image": str(dest_image)}
                    if "metadata" not in new_obj:
                        new_obj["metadata"] = {}
                    new_obj["metadata"]["source_dataset"] = dataset_name
                    out_list.append(new_obj)
                    id_list.append(new_id)
                    stats[split] += 1
                    per_dataset[dataset_name][split] += 1

    # Write merged JSONL
    merged_train_path = out_dir / "merged_train.jsonl"
    merged_val_path = out_dir / "merged_val.jsonl"
    with open(merged_train_path, "w", encoding="utf-8") as f:
        for obj in merged_train:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    with open(merged_val_path, "w", encoding="utf-8") as f:
        for obj in merged_val:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Duplicate check
    train_dupes = [k for k, c in Counter(all_train_ids).items() if c > 1]
    val_dupes = [k for k, c in Counter(all_val_ids).items() if c > 1]

    print("Merged dataset written to:", out_dir)
    print(f"  train samples: {stats['train']} -> {merged_train_path}")
    print(f"  val   samples: {stats['val']}   -> {merged_val_path}")
    print("  Per-dataset breakdown (from JSONL lines):")
    for n in names:
        tr, va = per_dataset[n]["train"], per_dataset[n]["val"]
        if tr or va:
            print(f"    {n}: train={tr}, val={va}")
    if train_dupes:
        print(f"  WARNING: duplicate train ids: {len(train_dupes)} e.g. {train_dupes[:5]}")
    else:
        print("  No duplicate train ids.")
    if val_dupes:
        print(f"  WARNING: duplicate val ids: {len(val_dupes)} e.g. {val_dupes[:5]}")
    else:
        print("  No duplicate val ids.")
    if stats["missing_image"]:
        print(f"  Missing images: {stats['missing_image']}")
    verb = "moved" if args.move else "copied"
    print(f"  Saliency maps {verb}: {stats['saliency_copied']}")
    print(f"  Fixation maps {verb}: {stats['fixation_copied']}")
    print("\nTo train with merged data:")
    print(f"  --train_jsonl {merged_train_path}")
    print(f"  --val_jsonl {merged_val_path}")
    print(f"  --saliency_dir {out_dir}   # trainer will use train/saliency and val/saliency")
    print(f"  --fixation_dir {out_dir}  # trainer will use train/fixations and val/fixations")


if __name__ == "__main__":
    main()
