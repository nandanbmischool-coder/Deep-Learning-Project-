"""Optional model-comparison experiment: same seed => same splits, same preprocessing and evaluation protocol.
    python -m app.ml.compare --archs densenet121 resnet50 mobilenet_v3_large --epochs 10
Results are only written after each run has actually finished; nothing is estimated."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime

from app.core.config import get_settings


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--archs", nargs="+", default=["densenet121", "resnet50", "mobilenet_v3_large"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--head-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-images-per-class", type=int, default=None)
    a, extra = p.parse_known_args()
    root = get_settings().artifacts_dir
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    comp_dir = root / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for arch in a.archs:
        run_dir = root / "runs" / f"cmp_{stamp}_{arch}"
        cmd = [
            sys.executable,
            "-m",
            "app.ml.train",
            "--arch",
            arch,
            "--epochs",
            str(a.epochs),
            "--head-epochs",
            str(a.head_epochs),
            "--batch-size",
            str(a.batch_size),
            "--seed",
            str(a.seed),
            "--run-dir",
            str(run_dir),
            *extra,
        ]
        if a.max_images_per_class:
            cmd += ["--max-images-per-class", str(a.max_images_per_class)]
        print("Running:", " ".join(cmd), flush=True)
        if subprocess.run(cmd, check=False).returncode != 0:
            rows.append({"architecture": arch, "status": "failed"})
            continue
        ev = json.loads((run_dir / "evaluation.json").read_text())
        cfg = json.loads((run_dir / "config.json").read_text())
        rows.append(
            {
                "architecture": arch,
                "status": "completed",
                "run": run_dir.name,
                "parameters": ev["parameter_count"],
                "model_size_mb": ev["model_size_mb"],
                "median_latency_ms": ev["latency"]["median_ms"],
                "latency_device": ev["latency"]["device"],
                "metrics": ev["metrics"],
                "training_seconds": cfg["training_seconds"],
            }
        )
    (comp_dir / f"comparison_{stamp}.json").write_text(json.dumps({"created_at": stamp, "seed": a.seed, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
