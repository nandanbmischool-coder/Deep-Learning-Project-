"""Download the Kaggle 'Chest X-Ray Images (Pneumonia)' dataset into data/raw/chest_xray.

Prerequisites (one time):
  1. Create a Kaggle account and an API token (Settings -> API -> Create New Token). Save kaggle.json to
       Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json      Linux/macOS: ~/.kaggle/kaggle.json
     (or export KAGGLE_USERNAME and KAGGLE_KEY).
  2. Open https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia while signed in and accept the dataset terms.
  3. pip install kaggle
Never paste your key into chat, source code or Git.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

SLUG = "paultimothymooney/chest-xray-pneumonia"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TARGET = RAW / "chest_xray"


def kaggle_cmd() -> list[str]:
    exe = shutil.which("kaggle")
    return [exe] if exe else [sys.executable, "-m", "kaggle"]


def main() -> int:
    if TARGET.exists() and any(TARGET.iterdir()):
        print(f"{TARGET} already exists - nothing to do. Delete it first to re-download.")
        return 0
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {SLUG} (about 1.2 GB) ...")
    res = subprocess.run([*kaggle_cmd(), "datasets", "download", "-d", SLUG, "-p", str(RAW), "--unzip"], check=False)
    if res.returncode != 0:
        print("\nDownload failed. Check: kaggle installed, kaggle.json in place, dataset terms accepted, internet access.", file=sys.stderr)
        return res.returncode
    shutil.rmtree(RAW / "__MACOSX", ignore_errors=True)
    nested = TARGET / "chest_xray"  # the Kaggle archive contains a duplicate nested copy
    if nested.exists():
        shutil.rmtree(nested)
        print("Removed the duplicate nested chest_xray/chest_xray folder.")
    missing = [p for p in ("train", "test") if not (TARGET / p).is_dir()]
    if missing:
        print(f"Unexpected layout: missing {missing} under {TARGET}.", file=sys.stderr)
        return 1
    print(f"Done. Dataset is at {TARGET}\nNext: cd backend && python -m app.ml.dataset_validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
