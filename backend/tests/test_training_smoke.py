"""End-to-end smoke test of the real training CLI on a tiny synthetic dataset (random init, 1 CPU: kept small)."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.ml import train as train_mod
from app.ml.data import make_splits, scan_split
from app.ml.dataset_validation import validate_dataset


def make_dataset(root: Path, n=14):
    rng = np.random.default_rng(1)
    for split in ("train", "test"):
        for cls, base in (("NORMAL", 90), ("PNEUMONIA", 150)):
            d = root / split / cls
            d.mkdir(parents=True)
            for i in range(n):
                arr = np.clip(rng.normal(base, 25, (96, 96)), 0, 255).astype(np.uint8)
                name = f"IM-{i:04d}-0001.jpeg" if cls == "NORMAL" else f"person{i}_bacteria_{i}.jpeg"
                Image.fromarray(arr).save(d / name)


def test_splits_validation_and_training(tmp_path):
    root = tmp_path / "chest_xray"
    make_dataset(root)
    splits = make_splits(root, 0.25, seed=3)
    s = splits["summary"]
    assert s["patient_group_overlap"]["train_val"] == 0 and s["patient_group_overlap"]["train_test"] >= 0
    assert len(scan_split(root, "test")) == 28 and sum(s["counts"]["test"].values()) == 28
    rep = validate_dataset(root)
    assert rep["totals"] == {"NORMAL": 28, "PNEUMONIA": 28} and rep["corrupt_count"] == 0

    run = tmp_path / "run"
    code = train_mod.main(
        [
            "--arch",
            "mobilenet_v3_large",
            "--no-pretrained",
            "--epochs",
            "2",
            "--head-epochs",
            "1",
            "--batch-size",
            "8",
            "--dataset-dir",
            str(root),
            "--run-dir",
            str(run),
            "--seed",
            "1",
        ]
    )
    assert code == 0
    for f in ("model.pt", "config.json", "history.json", "evaluation.json", "evaluation.html", "split_summary.json"):
        assert (run / f).exists(), f
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["pretrained"] is False and "random initialization" in cfg["pretrained_source"]
    ev = json.loads((run / "evaluation.json").read_text())
    assert ev["n_samples"] == 28 and set(ev["metrics"]["confusion"]) == {"tp", "tn", "fp", "fn"}
    assert json.loads((run / "status.json").read_text())["state"] == "completed"
    # refuses to overwrite an existing checkpoint
    assert (
        train_mod.main(["--arch", "mobilenet_v3_large", "--no-pretrained", "--dataset-dir", str(root), "--run-dir", str(run)])
        == 2
    )


def test_missing_dataset_is_reported_not_faked(tmp_path):
    code = train_mod.main(
        [
            "--arch",
            "mobilenet_v3_large",
            "--no-pretrained",
            "--dataset-dir",
            str(tmp_path / "nope"),
            "--run-dir",
            str(tmp_path / "r2"),
        ]
    )
    assert code == 1
    assert json.loads((tmp_path / "r2" / "status.json").read_text())["state"] == "failed"


@pytest.mark.parametrize("flag", ["--epochs", "--head-epochs"])
def test_cli_args_parse(flag):
    a = train_mod.parse_args(["--run-dir", "x", flag, "3"])
    assert a.arch == "densenet121"
