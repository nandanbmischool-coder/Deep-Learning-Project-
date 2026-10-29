from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.api.deps import admin_user, current_user
from app.core.config import get_settings
from app.db.models import AuditEvent, ModelVersion, User
from app.db.session import get_db
from app.services.model_service import ModelService, get_model_service

router = APIRouter(tags=["system"])


@router.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "app": s.app_name, "version": s.app_version}


@router.get("/ready")
def ready(db: Session = Depends(get_db), svc: ModelService = Depends(get_model_service)):
    s = get_settings()
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:  # noqa: BLE001
        db_ok = False
    has_model = db.scalar(select(func.count(ModelVersion.id)).where(ModelVersion.is_active.is_(True))) == 1
    return {
        "ready": db_ok,
        "database": db_ok,
        "active_model": has_model,
        "model_loaded_in_memory": svc.is_loaded(),
        "dataset_present": s.dataset_dir.exists(),
        "environment": s.environment,
        "version": s.app_version,
    }


@router.get("/admin/audit")
def audit_log(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    admin: User = Depends(admin_user),
    db: Session = Depends(get_db),
):
    total = db.scalar(select(func.count(AuditEvent.id))) or 0
    rows = db.scalars(
        select(AuditEvent).order_by(AuditEvent.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    ).all()
    return {
        "items": [
            {"id": r.id, "action": r.action, "target": r.target, "user_id": r.user_id, "ip": r.ip, "created_at": r.created_at}
            for r in rows
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/admin/users")
def users(admin: User = Depends(admin_user), db: Session = Depends(get_db)):
    return {
        "items": [
            {"id": u.id, "email": u.email, "full_name": u.full_name, "role": u.role, "created_at": u.created_at}
            for u in db.scalars(select(User).order_by(User.id))
        ]
    }


@router.get("/system/info")
def info(user: User = Depends(current_user), svc: ModelService = Depends(get_model_service)):
    import sys

    import torch
    import torchvision

    s = get_settings()
    return {
        "app_version": s.app_version,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_preference": s.inference_device,
        "max_upload_mb": s.max_upload_mb,
        "environment": s.environment,
        "database": "sqlite" if s.database_url.startswith("sqlite") else "postgresql",
    }
