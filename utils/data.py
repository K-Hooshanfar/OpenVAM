import json
import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


def preprocess_img(img_dir, channels=3):

    if channels == 1:
        img = cv2.imread(img_dir, 0)
    elif channels == 3:
        img = cv2.imread(img_dir)

    image_org = img
    shape_r = 256
    shape_c = 256
    img_padded = np.ones((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)
    original_shape = img.shape
    rows_rate = original_shape[0] / shape_r
    cols_rate = original_shape[1] / shape_c
    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:,
        ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))

        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows),
        :] = img

    return img_padded , image_org


def postprocess_img(pred, org_dir):
    pred = np.array(pred)
    org = cv2.imread(org_dir, 0)
    shape_r = org.shape[0]
    shape_c = org.shape[1]
    predictions_shape = pred.shape

    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    return img


class TrainDataset(Dataset):
    def __init__(self, datasets_info, transform=None, num_classes=4):
        self.datasets = []
        self.num_classes = num_classes
        for dataset_info in datasets_info:
            ids = pd.read_csv(dataset_info['id_train'])
            self.datasets.append((ids, dataset_info, transform))

    def __len__(self):
        return sum(len(ids) for ids, _, _ in self.datasets)

    def __getitem__(self, idx):
        dataset_idx = 0
        while idx >= len(self.datasets[dataset_idx][0]):
            idx -= len(self.datasets[dataset_idx][0])
            dataset_idx += 1
        ids, dataset_info, transform = self.datasets[dataset_idx]

        # Load image
        im_path = dataset_info['stimuli_dir'] + ids.iloc[idx, 0]
        image = Image.open(im_path).convert('RGB')
        if transform:
            image = transform(image)

        # Load saliency map
        smap_path = dataset_info['saliency_dir'] + ids.iloc[idx, 1]
        saliency = Image.open(smap_path).convert('L')
        saliency = np.array(saliency, dtype=np.float32) / 255.
        saliency = torch.from_numpy(saliency).unsqueeze(0)

        # Load fixation map
        fmap_path = dataset_info['fixation_dir'] + ids.iloc[idx, 2]
        fixation = Image.open(fmap_path).convert('L')
        fixation = np.array(fixation, dtype=np.float32) / 255.
        fixation = torch.from_numpy(fixation).unsqueeze(0)

        # Convert label to one-hot vector
        label = torch.zeros(self.num_classes)
        label[dataset_info['label']] = 1

        sample = {'image': image, 'saliency': saliency, 'fixation': fixation, 'label': label}
        return sample

class ValDataset(Dataset):
    def __init__(self, ids_path, stimuli_dir, saliency_dir, fixation_dir, label, transform=None, num_classes=4):
        self.ids = pd.read_csv(ids_path)
        self.stimuli_dir = stimuli_dir
        self.saliency_dir = saliency_dir
        self.fixation_dir = fixation_dir
        self.label = label
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Load image
        im_path = self.stimuli_dir + self.ids.iloc[idx, 0]
        image = Image.open(im_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load saliency map
        smap_path = self.saliency_dir + self.ids.iloc[idx, 1]
        saliency = Image.open(smap_path).convert('L')
        saliency = np.array(saliency, dtype=np.float32) / 255.
        saliency = torch.from_numpy(saliency).unsqueeze(0)

        # Load fixation map
        fmap_path = self.fixation_dir + self.ids.iloc[idx, 2]
        fixation = Image.open(fmap_path).convert('L')
        fixation = np.array(fixation, dtype=np.float32) / 255.
        fixation = torch.from_numpy(fixation).unsqueeze(0)

        # Convert label to one-hot vector
        label = torch.zeros(self.num_classes)
        label[self.label] = 1

        sample = {'image': image, 'saliency': saliency, 'fixation': fixation, 'label': label}
        return sample


def _find_map_path(dir_path, stem, exts=(".png", ".jpg", ".jpeg")):
    """Find a file with given stem in dir_path, trying extensions. Returns path or None."""
    d = Path(dir_path)
    for ext in exts:
        for e in (ext, ext.upper()):
            p = d / f"{stem}{e}"
            if p.exists():
                return str(p)
    return None


# Target size for merged data (sources may have different resolutions)
MERGED_TARGET_SIZE = (256, 256)


def merged_collate_fn(batch):
    """Collate that stacks tensors and keeps 'id' as a list (for per-dataset val metrics)."""
    from torch.utils.data.dataloader import default_collate
    ids = [b["id"] for b in batch]
    stacked = default_collate([{k: v for k, v in b.items() if k != "id"} for b in batch])
    stacked["id"] = ids
    return stacked


def _resolve_image_path(merged_dir, subdir, obj):
    """Return the image path we would use in __getitem__ (for existence check)."""
    sid = obj.get("id", "")
    image_path = obj.get("image", "")
    if image_path and Path(image_path).exists():
        return image_path
    return str(Path(merged_dir) / subdir / "stimuli" / f"{sid}.jpg")


def _resize_if_needed(pil_img, target_size, sid, field_name, resized_seen, resized_report, mode=Image.BILINEAR):
    """Resize PIL image to target_size only if needed; log once per (sid, field). Returns PIL image."""
    w, h = pil_img.size
    if (w, h) == target_size:
        return pil_img
    key = (sid, field_name)
    if key not in resized_seen:
        resized_seen.add(key)
        resized_report.append((sid, field_name, (w, h)))
        print(f"[MergedDataset] resized {field_name} for id {sid!r} from {w}x{h} to {target_size[0]}x{target_size[1]}")
    return pil_img.resize(target_size, mode)


class MergedTrainDataset(Dataset):
    """Train dataset from merged_train.jsonl; uses merged_dir/train/{stimuli,saliency,fixations}."""

    def __init__(self, merged_dir, jsonl_name="merged_train.jsonl", transform=None, num_classes=4, skip_missing_images=True):
        self.merged_dir = Path(merged_dir)
        self.jsonl_path = self.merged_dir / jsonl_name
        loaded = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    loaded.append(json.loads(line))
        if skip_missing_images:
            self.samples = []
            for obj in loaded:
                p = _resolve_image_path(self.merged_dir, "train", obj)
                if Path(p).exists():
                    self.samples.append(obj)
                else:
                    pass  # skip missing
            if len(self.samples) < len(loaded):
                print(f"[MergedTrainDataset] Skipped {len(loaded) - len(self.samples)} samples with missing image (kept {len(self.samples)}).")
        else:
            self.samples = loaded
        self.transform = transform
        self.num_classes = num_classes
        self.train_sub = "train"
        self._resized_seen = set()
        self._resized_report = []

    def __len__(self):
        return len(self.samples)

    def get_resized_report(self):
        """Return list of (sample_id, field_name, original_size) for any sample that was resized."""
        return list(self._resized_report)

    def __getitem__(self, idx):
        obj = self.samples[idx]
        sid = obj.get("id", "")
        image_path = obj.get("image", "")
        if not image_path or not Path(image_path).exists():
            image_path = str(self.merged_dir / self.train_sub / "stimuli" / f"{sid}.jpg")
        image = Image.open(image_path).convert("RGB")
        image = _resize_if_needed(image, MERGED_TARGET_SIZE, sid, "image", self._resized_seen, self._resized_report, Image.BILINEAR)
        if self.transform:
            image = self.transform(image)
        sal_path = _find_map_path(self.merged_dir / self.train_sub / "saliency", sid)
        if sal_path:
            saliency = Image.open(sal_path).convert("L")
            saliency = _resize_if_needed(saliency, MERGED_TARGET_SIZE, sid, "saliency", self._resized_seen, self._resized_report, Image.BILINEAR)
        else:
            saliency = Image.fromarray(np.zeros(MERGED_TARGET_SIZE, dtype=np.uint8))
        saliency = np.array(saliency, dtype=np.float32) / 255.0
        saliency = torch.from_numpy(saliency).unsqueeze(0)
        fix_path = _find_map_path(self.merged_dir / self.train_sub / "fixations", sid)
        if fix_path:
            fixation = Image.open(fix_path).convert("L")
            fixation = _resize_if_needed(fixation, MERGED_TARGET_SIZE, sid, "fixation", self._resized_seen, self._resized_report, Image.NEAREST)
        else:
            fixation = Image.fromarray(np.zeros(MERGED_TARGET_SIZE, dtype=np.uint8))
        fixation = np.array(fixation, dtype=np.float32) / 255.0
        fixation = torch.from_numpy(fixation).unsqueeze(0)
        label = torch.zeros(self.num_classes)
        label[0] = 1
        return {"image": image, "saliency": saliency, "fixation": fixation, "label": label, "id": sid}

    def check_fixation_saliency_coverage(self, sample_limit=500):
        """Load up to sample_limit samples and return (n_with_fixation, n_with_saliency, n_total, ids_missing_fixation)."""
        n_fix, n_sal, n_total = 0, 0, 0
        ids_missing_fix = []
        for idx in range(min(sample_limit, len(self.samples))):
            out = self.__getitem__(idx)
            n_total += 1
            if out["fixation"].sum().item() > 0:
                n_fix += 1
            else:
                ids_missing_fix.append(out.get("id", ""))
            if out["saliency"].sum().item() > 0:
                n_sal += 1
        return n_fix, n_sal, n_total, ids_missing_fix


class MergedValDataset(Dataset):
    """Val dataset from merged_val.jsonl; uses merged_dir/val/{stimuli,saliency,fixations}."""

    def __init__(self, merged_dir, jsonl_name="merged_val.jsonl", transform=None, num_classes=4, skip_missing_images=True):
        self.merged_dir = Path(merged_dir)
        self.jsonl_path = self.merged_dir / jsonl_name
        loaded = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    loaded.append(json.loads(line))
        if skip_missing_images:
            self.samples = []
            for obj in loaded:
                p = _resolve_image_path(self.merged_dir, "val", obj)
                if Path(p).exists():
                    self.samples.append(obj)
                else:
                    pass  # skip missing
            if len(self.samples) < len(loaded):
                print(f"[MergedValDataset] Skipped {len(loaded) - len(self.samples)} samples with missing image (kept {len(self.samples)}).")
        else:
            self.samples = loaded
        self.transform = transform
        self.num_classes = num_classes
        self.val_sub = "val"
        self._resized_seen = set()
        self._resized_report = []

    def __len__(self):
        return len(self.samples)

    def get_resized_report(self):
        """Return list of (sample_id, field_name, original_size) for any sample that was resized."""
        return list(self._resized_report)

    def __getitem__(self, idx):
        obj = self.samples[idx]
        sid = obj.get("id", "")
        image_path = obj.get("image", "")
        if not image_path or not Path(image_path).exists():
            image_path = str(self.merged_dir / self.val_sub / "stimuli" / f"{sid}.jpg")
        image = Image.open(image_path).convert("RGB")
        image = _resize_if_needed(image, MERGED_TARGET_SIZE, sid, "image", self._resized_seen, self._resized_report, Image.BILINEAR)
        if self.transform:
            image = self.transform(image)
        sal_path = _find_map_path(self.merged_dir / self.val_sub / "saliency", sid)
        if sal_path:
            saliency = Image.open(sal_path).convert("L")
            saliency = _resize_if_needed(saliency, MERGED_TARGET_SIZE, sid, "saliency", self._resized_seen, self._resized_report, Image.BILINEAR)
        else:
            saliency = Image.fromarray(np.zeros(MERGED_TARGET_SIZE, dtype=np.uint8))
        saliency = np.array(saliency, dtype=np.float32) / 255.0
        saliency = torch.from_numpy(saliency).unsqueeze(0)
        fix_path = _find_map_path(self.merged_dir / self.val_sub / "fixations", sid)
        if fix_path:
            fixation = Image.open(fix_path).convert("L")
            fixation = _resize_if_needed(fixation, MERGED_TARGET_SIZE, sid, "fixation", self._resized_seen, self._resized_report, Image.NEAREST)
        else:
            fixation = Image.fromarray(np.zeros(MERGED_TARGET_SIZE, dtype=np.uint8))
        fixation = np.array(fixation, dtype=np.float32) / 255.0
        fixation = torch.from_numpy(fixation).unsqueeze(0)
        label = torch.zeros(self.num_classes)
        label[0] = 1
        return {"image": image, "saliency": saliency, "fixation": fixation, "label": label, "id": sid}

    def check_fixation_saliency_coverage(self, sample_limit=500):
        """Load up to sample_limit samples and return (n_with_fixation, n_with_saliency, n_total, ids_missing_fixation)."""
        n_fix, n_sal, n_total = 0, 0, 0
        ids_missing_fix = []
        for idx in range(min(sample_limit, len(self.samples))):
            out = self.__getitem__(idx)
            n_total += 1
            if out["fixation"].sum().item() > 0:
                n_fix += 1
            else:
                ids_missing_fix.append(out.get("id", ""))
            if out["saliency"].sum().item() > 0:
                n_sal += 1
        return n_fix, n_sal, n_total, ids_missing_fix
