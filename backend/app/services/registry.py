"""Registers finished training runs (model version + evaluation) in the database and syncs on start-up."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Evaluation, ModelVersion


def register_run(db: Session, run_dir: Path, make_active_if_first: bool = True) -> ModelVersion | None:
    """Create ModelVersion (+Evaluation) rows from a completed run directory. Idempotent."""
    cfg_path, ev_path = run_dir / "config.json", run_dir / "evaluation.json"
    if not (cfg_path.exists() and (run_dir / "model.pt").exists()):
        return None
    cfg = json.loads(cfg_path.read_text())
    rel = run_dir.relative_to(get_settings().artifacts_dir).as_posix()
    row = db.scalar(select(ModelVersion).where(ModelVersion.version == cfg["version"]))
    if row is None:
        has_active = db.scalar(select(ModelVersion).where(ModelVersion.is_active.is_(True))) is not None
        row = ModelVersion(
            version=cfg["version"],
            architecture=cfg["architecture"],
            pretrained=bool(cfg["pretrained"]),
            pretrained_source=cfg.get("pretrained_source", ""),
            run_dir=rel,
            parameter_count=cfg["parameter_count"],
            model_size_mb=cfg["model_size_mb"],
            threshold=cfg["threshold"],
            threshold_method=cfg.get("threshold_method", ""),
            is_active=make_active_if_first and not has_active,
            training_config={k: v for k, v in cfg.items() if k != "class_to_idx"},
        )
        db.add(row)
        db.flush()
    if ev_path.exists() and not row.evaluations:
        rep = json.loads(ev_path.read_text())
        db.add(
            Evaluation(
                model_version_id=row.id,
                split=rep.get("split", "test"),
                accuracy=rep["metrics"]["accuracy"],
                roc_auc=rep["metrics"]["roc_auc"],
                report=rep,
            )
        )
    db.commit()
    return row


def sync_runs(db: Session) -> int:
    runs = get_settings().artifacts_dir / "runs"
    n = 0
    if runs.exists():
        for d in sorted(p for p in runs.iterdir() if p.is_dir()):
            if register_run(db, d):
                n += 1
    return n
