from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import current_user
from app.api.predictions import to_out
from app.core.config import get_settings
from app.db.models import Analysis, Evaluation, ModelVersion, User
from app.db.session import get_db
from app.ml.dataset_validation import quick_status

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _active_eval(db: Session):
    mv = db.scalar(select(ModelVersion).where(ModelVersion.is_active.is_(True)))
    ev = (
        db.scalar(select(Evaluation).where(Evaluation.model_version_id == mv.id).order_by(Evaluation.created_at.desc()))
        if mv
        else None
    )
    return mv, ev


@router.get("/summary")
def summary(user: User = Depends(current_user), db: Session = Depends(get_db)):
    q = select(func.count(Analysis.id), func.avg(Analysis.inference_ms)).where(Analysis.user_id == user.id)
    total, avg_ms = db.execute(q).one()
    by_class = dict(
        db.execute(
            select(Analysis.predicted_class, func.count()).where(Analysis.user_id == user.id).group_by(Analysis.predicted_class)
        ).all()
    )
    mv, ev = _active_eval(db)
    return {
        # Operational statistics: counts of this user's own uploads. NOT disease prevalence, NOT model accuracy.
        "operational": {
            "total_analyses": total or 0,
            "normal": by_class.get("NORMAL", 0),
            "pneumonia": by_class.get("PNEUMONIA", 0),
            "avg_inference_ms": float(avg_ms) if avg_ms is not None else None,
        },
        # Model-evaluation statistics: from the held-out test split only.
        "model": {
            "active_version": mv.version if mv else None,
            "architecture": mv.architecture if mv else None,
            "evaluated": ev is not None,
            "test_accuracy": ev.accuracy if ev else None,
            "test_roc_auc": ev.roc_auc if ev else None,
            "evaluation_note": None if ev else "Not evaluated - train a model (or run evaluation) to see test metrics.",
        },
    }


@router.get("/activity")
def activity(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.scalars(select(Analysis).where(Analysis.user_id == user.id).order_by(Analysis.created_at.desc()).limit(8)).all()
    return {"items": [to_out(a) for a in rows]}


@router.get("/charts")
def charts(user: User = Depends(current_user), db: Session = Depends(get_db)):
    s = get_settings()
    since = datetime.now(UTC) - timedelta(days=29)
    rows = db.execute(
        select(Analysis.created_at, Analysis.predicted_class).where(Analysis.user_id == user.id, Analysis.created_at >= since)
    ).all()
    days = {
        (since + timedelta(days=i)).strftime("%Y-%m-%d"): {
            "date": (since + timedelta(days=i)).strftime("%Y-%m-%d"),
            "NORMAL": 0,
            "PNEUMONIA": 0,
        }
        for i in range(30)
    }
    for ts, cls in rows:
        key = ts.strftime("%Y-%m-%d")
        if key in days:
            days[key][cls] += 1
    series = [{**d, "total": d["NORMAL"] + d["PNEUMONIA"]} for d in days.values()]
    mv, ev = _active_eval(db)
    curves = None
    if mv:
        hp = s.artifacts_dir / mv.run_dir / "history.json"
        curves = json.loads(hp.read_text()) if hp.exists() else None
    comp_dir = s.artifacts_dir / "comparisons"
    comparison = None
    if comp_dir.exists():
        files = sorted(comp_dir.glob("comparison_*.json"))
        comparison = json.loads(files[-1].read_text()) if files else None
    ds = quick_status(s.dataset_dir)
    dist = [{"split": sp, **c} for sp, c in ds["counts"].items()] if ds["available"] else []
    return {
        "volume_over_time": series,
        "prediction_distribution": [{"name": k, "value": sum(d[k] for d in series)} for k in ("NORMAL", "PNEUMONIA")],
        "dataset_distribution": dist,
        "training_curves": curves,
        "evaluation": ({"confusion": ev.report["metrics"]["confusion"], "curves": ev.report["curves"]} if ev else None),
        "comparison": comparison,
    }
