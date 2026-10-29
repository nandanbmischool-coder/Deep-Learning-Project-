"""Dataset validation: structure, corrupt files, counts, imbalance, exact + near duplicates, cross-split leakage."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from app.ml.data import IMG_EXTS, patient_group, sha256_file
from app.ml.model import CLASS_NAMES

SPLITS = ("train", "val", "test")
MAX_LIST = 25


def quick_status(root: Path) -> dict:
    """Cheap availability + counts (no hashing)."""
    if not root.exists():
        return {"available": False, "counts": {}}
    counts: dict[str, dict[str, int]] = {}
    for d in (p for p in root.iterdir() if p.is_dir() and p.name.lower() in SPLITS):
        counts[d.name.lower()] = {
            c.name.upper(): sum(1 for f in c.iterdir() if f.suffix.lower() in IMG_EXTS) for c in d.iterdir() if c.is_dir()
        }
    return {"available": bool(counts), "counts": counts}


def _dhash(path: Path) -> int:
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def validate_dataset(root: Path, near_dup_distance: int = 2, progress=None) -> dict:
    report: dict = {"created_at": datetime.now(UTC).isoformat(), "root_exists": root.exists(), "issues": []}
    if not root.exists():
        report["issues"].append("Dataset folder not found. See the README for download instructions.")
        report["status"] = "missing"
        return report
    issues = report["issues"]
    found = {d.name.lower(): d for d in root.iterdir() if d.is_dir()}
    for s in ("train", "test"):
        if s not in found:
            issues.append(f"Missing required split folder '{s}'.")
    if "val" not in found:
        issues.append("No official 'val' folder (optional). A validation split is created from 'train' during training.")
    files: list[tuple[str, str, Path]] = []  # split, class, path
    unexpected, empty, corrupt = [], [], []
    for split, d in found.items():
        if split not in SPLITS:
            unexpected.append(d.name)
            continue
        classes = [c for c in d.iterdir() if c.is_dir()]
        for c in classes:
            if c.name.upper() not in CLASS_NAMES:
                unexpected.append(f"{split}/{c.name}")
                continue
            imgs = [f for f in c.iterdir() if f.suffix.lower() in IMG_EXTS and not f.name.startswith(".")]
            if not imgs:
                empty.append(f"{split}/{c.name}")
            files += [(split, c.name.upper(), f) for f in imgs]
        for need in CLASS_NAMES:
            if not any(c.name.upper() == need for c in classes):
                issues.append(f"Split '{split}' has no {need} folder.")
    counts = {s: {c: 0 for c in CLASS_NAMES} for s in SPLITS}
    sha: dict[str, list[tuple[str, str]]] = {}
    hashes: list[tuple[int, str, str]] = []
    for i, (split, cls, path) in enumerate(files):
        if progress and i % 200 == 0:
            progress(i, len(files))
        try:
            with Image.open(path) as im:
                im.verify()
            hashes.append((_dhash(path), split, f"{split}/{cls}/{path.name}"))
        except Exception:  # noqa: BLE001
            corrupt.append(f"{split}/{cls}/{path.name}")
            continue
        counts[split][cls] += 1
        sha.setdefault(sha256_file(path), []).append((split, f"{split}/{cls}/{path.name}"))
    exact = [v for v in sha.values() if len(v) > 1]
    exact_cross = [v for v in exact if len({s for s, _ in v}) > 1]
    # near-duplicates via 64-bit difference hash (Hamming distance), vectorised in chunks
    near_cross, near_total = [], 0
    if hashes:
        arr = np.array([h for h, _, _ in hashes], dtype=np.uint64)
        for i in range(len(arr)):
            x = arr[i + 1 :] ^ arr[i]
            dist = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
            for j in np.nonzero(dist <= near_dup_distance)[0]:
                near_total += 1
                a, b = hashes[i], hashes[i + 1 + j]
                if a[1] != b[1] and len(near_cross) < MAX_LIST:
                    near_cross.append([a[2], b[2], int(dist[j])])
    groups: dict[str, set[str]] = {s: set() for s in SPLITS}
    for split, cls, path in files:
        groups[split].add(patient_group(path, CLASS_NAMES.index(cls)))
    overlap = {f"{a}_{b}": len(groups[a] & groups[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    tot = {c: sum(counts[s][c] for s in SPLITS) for c in CLASS_NAMES}
    ratio = (max(tot.values()) / max(min(tot.values()), 1)) if any(tot.values()) else None
    if unexpected:
        issues.append(f"Unexpected folders ignored: {unexpected[:5]}")
    if empty:
        issues.append(f"Empty class folders: {empty}")
    if corrupt:
        issues.append(f"{len(corrupt)} unreadable/corrupt images were excluded from counts.")
    if exact_cross:
        issues.append(f"{len(exact_cross)} exact duplicate groups span multiple splits (data leakage risk).")
    if near_cross:
        issues.append("Near-duplicate images (difference-hash heuristic) found across splits - review before trusting metrics.")
    if ratio and ratio > 1.5:
        issues.append(f"Class imbalance: majority/minority ratio {ratio:.2f}. Training uses class-weighted loss.")
    report.update(
        {
            "status": "ok" if not (corrupt or exact_cross or empty) else "attention",
            "counts": counts,
            "totals": tot,
            "imbalance_ratio": ratio,
            "n_files": len(files),
            "corrupt_count": len(corrupt),
            "corrupt_examples": corrupt[:MAX_LIST],
            "empty_folders": empty,
            "unexpected_folders": unexpected[:MAX_LIST],
            "duplicates": {
                "exact_groups": len(exact),
                "exact_groups_across_splits": len(exact_cross),
                "exact_examples": [[p for _, p in g][:4] for g in exact_cross[:MAX_LIST]],
                "near_duplicate_pairs_total": near_total,
                "near_duplicate_pairs_across_splits_examples": near_cross,
                "near_duplicate_method": f"64-bit difference hash, Hamming distance <= {near_dup_distance} (heuristic)",
            },
            "patient_group_overlap": overlap,
            "leakage_note": (
                "Patient groups are parsed from Kaggle file names (best effort). Non-zero overlap means the "
                "official splits are not strictly patient-separated; training builds its own group-aware validation split."
            ),
        }
    )
    return report


if __name__ == "__main__":
    from app.core.config import get_settings

    out = validate_dataset(get_settings().dataset_dir)
    path = get_settings().artifacts_dir / "dataset_report.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out.get(k) for k in ("status", "totals", "issues")}, indent=2))
    os._exit(0)
