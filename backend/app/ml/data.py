"""Dataset discovery, leakage-aware splitting, and preprocessing.

Preprocessing (identical for validation, test and inference):  RGB -> resize 224x224 -> tensor -> ImageNet normalisation.
Augmentation (training only): small rotation / translation / scale; horizontal flip is OFF by default because flipping
a chest radiograph swaps the side of the heart and is not anatomically neutral.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from app.ml.model import CLASS_NAMES, CLASS_TO_IDX

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class DatasetError(RuntimeError):
    pass


def eval_transform(size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def train_transform(size: int = 224, horizontal_flip: bool = False) -> T.Compose:
    aug = [
        T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomAffine(degrees=7, translate=(0.04, 0.04), scale=(0.95, 1.05)),
    ]
    if horizontal_flip:
        aug.append(T.RandomHorizontalFlip())
    return T.Compose([*aug, T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


@dataclass(frozen=True)
class Item:
    path: Path
    label: int
    group: str


def patient_group(path: Path, label: int) -> str:
    """Best-effort patient id from Kaggle file names (person12_bacteria_34.jpeg / IM-0115-0001.jpeg).
    Falls back to the file name, i.e. image-level grouping."""
    name = path.name
    m = re.search(r"person(\d+)_", name, flags=re.I)
    if m:
        return f"P{m.group(1)}"
    m = re.search(r"IM-(\d{4})", name)
    if m:
        return f"N{m.group(1)}"
    return f"{CLASS_NAMES[label]}:{path.stem}"


def _find_split_dir(root: Path, split: str) -> Path | None:
    for child in root.iterdir() if root.exists() else []:
        if child.is_dir() and child.name.lower() == split:
            return child
    return None


def scan_split(root: Path, split: str) -> list[Item]:
    d = _find_split_dir(root, split)
    if d is None:
        return []
    items: list[Item] = []
    for cls_dir in sorted(p for p in d.iterdir() if p.is_dir()):
        cname = cls_dir.name.upper()
        if cname not in CLASS_TO_IDX:
            continue
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in IMG_EXTS and not f.name.startswith("."):
                items.append(Item(f, CLASS_TO_IDX[cname], patient_group(f, CLASS_TO_IDX[cname])))
    return items


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def class_counts(items: list[Item]) -> dict[str, int]:
    out = {n: 0 for n in CLASS_NAMES}
    for it in items:
        out[CLASS_NAMES[it.label]] += 1
    return out


def subsample(items: list[Item], per_class: int | None, seed: int) -> list[Item]:
    if not per_class:
        return items
    rng = random.Random(seed)
    out: list[Item] = []
    for lbl in range(len(CLASS_NAMES)):
        pool = [i for i in items if i.label == lbl]
        rng.shuffle(pool)
        out += pool[:per_class]
    return out


def make_splits(root: Path, val_fraction: float, seed: int, max_images_per_class: int | None = None) -> dict:
    """Keep the official test split untouched. Build validation from `train` with a group-aware stratified split
    (the official `val` folder has only 16 images, far too few to select checkpoints/thresholds)."""
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split

    train_all, test = scan_split(root, "train"), scan_split(root, "test")
    if not train_all or not test:
        raise DatasetError("Expected train/ and test/ folders with NORMAL/ and PNEUMONIA/ sub-folders under the dataset root.")
    train_all = subsample(train_all, max_images_per_class, seed)
    test = subsample(test, max_images_per_class, seed)
    y = np.array([i.label for i in train_all])
    groups = np.array([i.group for i in train_all])
    k = max(2, round(1 / val_fraction))
    method = "stratified group split (patient id parsed from file names)"
    try:
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        tr_idx, va_idx = next(iter(sgkf.split(np.zeros(len(y)), y, groups)))
    except ValueError:
        tr_idx, va_idx = train_test_split(np.arange(len(y)), test_size=val_fraction, stratify=y, random_state=seed)
        method = "stratified random split (too few samples for group split)"
    train, val = [train_all[i] for i in tr_idx], [train_all[i] for i in va_idx]

    # Leakage guard: drop any training image whose bytes are identical to a validation/test image.
    held = {sha256_file(i.path) for i in val + test}
    kept = [i for i in train if sha256_file(i.path) not in held]
    removed = len(train) - len(kept)
    train = kept
    g_tr, g_va, g_te = ({i.group for i in s} for s in (train, val, test))
    return {
        "train": train,
        "val": val,
        "test": test,
        "summary": {
            "method": method,
            "seed": seed,
            "val_fraction": val_fraction,
            "counts": {"train": class_counts(train), "val": class_counts(val), "test": class_counts(test)},
            "train_images_removed_as_exact_duplicates_of_val_or_test": removed,
            "patient_group_overlap": {
                "train_val": len(g_tr & g_va),
                "train_test": len(g_tr & g_te),
                "val_test": len(g_va & g_te),
            },
            "note": (
                "Patient identifiers are parsed from Kaggle file names on a best-effort basis; true patient-level "
                "separation cannot be fully verified without metadata. The official test split is never used for training."
            ),
        },
    }


class CXRDataset(Dataset):
    def __init__(self, items: list[Item], transform):
        self.items, self.transform = items, transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        with Image.open(it.path) as im:
            img = im.convert("RGB")
        return self.transform(img), it.label


def make_loader(items: list[Item], transform, batch_size: int, shuffle: bool, num_workers: int = 0, seed: int = 0) -> DataLoader:
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        CXRDataset(items, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=g if shuffle else None,
        pin_memory=False,
    )
