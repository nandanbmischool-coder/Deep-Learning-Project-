from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import admin_user, audit, current_user
from app.core.config import get_settings
from app.core.errors import AppError
from app.db.models import Evaluation, ModelVersion, User
from app.db.session import get_db
from app.services.model_service import ModelService, get_model_service
from app.services.registry import sync_runs

router = APIRouter(tags=["models"])


def mv_out(m: ModelVersion) -> dict:
    cfg = m.training_config or {}
    return {
        "id": m.id,
        "version": m.version,
        "architecture": m.architecture,
        "pretrained": m.pretrained,
        "pretrained_source": m.pretrained_source,
        "parameter_count": m.parameter_count,
        "model_size_mb": m.model_size_mb,
        "threshold": m.threshold,
        "threshold_method": m.threshold_method,
        "is_active": m.is_active,
        "created_at": m.created_at,
        "training_config": cfg.get("training_args"),
        "best_epoch": cfg.get("best_epoch"),
        "training_seconds": cfg.get("training_seconds"),
        "device": cfg.get("device"),
        "torch": cfg.get("torch"),
        "evaluated": bool(m.evaluations),
    }


def ev_out(e: Evaluation, full: bool = False) -> dict:
    out = {
        "id": e.id,
        "model_version_id": e.model_version_id,
        "model_version": e.model.version,
        "split": e.split,
        "accuracy": e.accuracy,
        "roc_auc": e.roc_auc,
        "created_at": e.created_at,
    }
    if full:
        out["report"] = e.report
    return out


@router.get("/models")
def list_models(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return {"items": [mv_out(m) for m in db.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc()))]}


@router.get("/models/active")
def active_model(user: User = Depends(current_user), db: Session = Depends(get_db)):
    m = db.scalar(select(ModelVersion).where(ModelVersion.is_active.is_(True)))
    if not m:
        return {"active": None, "message": "Model not trained."}
    out = mv_out(m)
    hp = get_settings().artifacts_dir / m.run_dir / "history.json"
    out["history"] = json.loads(hp.read_text()) if hp.exists() else []
    return {"active": out}


@router.post("/models/sync")
def sync(request: Request, admin: User = Depends(admin_user), db: Session = Depends(get_db)):
    n = sync_runs(db)
    audit(db, "models_sync", admin, "", request)
    return {"registered_or_checked": n}


@router.post("/models/{model_id}/activate")
def activate(
    model_id: int,
    request: Request,
    admin: User = Depends(admin_user),
    db: Session = Depends(get_db),
    svc: ModelService = Depends(get_model_service),
):
    m = db.get(ModelVersion, model_id)
    if not m:
        raise AppError(404, "NOT_FOUND", "Model version not found.")
    for other in db.scalars(select(ModelVersion).where(ModelVersion.is_active.is_(True))):
        other.is_active = False
    m.is_active = True
    db.commit()
    svc.invalidate()
    audit(db, "model_activate", admin, m.version, request)
    return mv_out(m)


@router.get("/models/comparisons")
def comparisons(user: User = Depends(current_user)):
    d = get_settings().artifacts_dir / "comparisons"
    files = sorted(d.glob("comparison_*.json")) if d.exists() else []
    return {"items": [json.loads(f.read_text()) for f in files]}


@router.get("/evaluations")
def list_evaluations(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return {"items": [ev_out(e) for e in db.scalars(select(Evaluation).order_by(Evaluation.created_at.desc()))]}


@router.get("/evaluations/{evaluation_id}")
def get_evaluation(evaluation_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    e = db.get(Evaluation, evaluation_id)
    if not e:
        raise AppError(404, "NOT_FOUND", "Evaluation not found.")
    return ev_out(e, full=True)
