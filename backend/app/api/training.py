from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import admin_user, audit, current_user
from app.core.config import get_settings
from app.core.errors import AppError
from app.db.models import TrainingJob, User
from app.db.session import get_db
from app.schemas import TrainingJobIn
from app.services.image_io import new_public_id
from app.services.training_manager import read_progress, training_manager

router = APIRouter(prefix="/training", tags=["training"])


def job_out(j: TrainingJob, with_progress: bool = False) -> dict:
    out = {
        "job_id": j.public_id,
        "architecture": j.architecture,
        "state": j.state,
        "config": j.config,
        "error": j.error,
        "created_at": j.created_at,
        "started_at": j.started_at,
        "finished_at": j.finished_at,
        "model_version_id": j.model_version_id,
        "version": j.run_dir.split("/")[-1],
    }
    if with_progress:
        out["progress"] = read_progress(get_settings().artifacts_dir / j.run_dir)
    return out


@router.post("/jobs", status_code=202)
def start_job(body: TrainingJobIn, request: Request, admin: User = Depends(admin_user), db: Session = Depends(get_db)):
    if body.head_epochs > body.epochs:
        raise AppError(422, "INVALID_CONFIG", "head_epochs cannot exceed epochs.")
    job = training_manager.start(db, body, admin.id, new_public_id())
    audit(db, "training_start", admin, job.public_id, request, body.model_dump())
    return job_out(job)


@router.get("/jobs")
def list_jobs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    total = db.scalar(select(func.count(TrainingJob.id))) or 0
    rows = db.scalars(
        select(TrainingJob).order_by(TrainingJob.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    ).all()
    return {"items": [job_out(j) for j in rows], "total": total, "page": page, "page_size": page_size}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    j = db.scalar(select(TrainingJob).where(TrainingJob.public_id == job_id))
    if not j:
        raise AppError(404, "NOT_FOUND", "Training job not found.")
    return job_out(j, with_progress=True)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request, admin: User = Depends(admin_user), db: Session = Depends(get_db)):
    j = db.scalar(select(TrainingJob).where(TrainingJob.public_id == job_id))
    if not j:
        raise AppError(404, "NOT_FOUND", "Training job not found.")
    if j.state not in ("running", "queued"):
        raise AppError(409, "NOT_RUNNING", f"Job is already {j.state}.")
    if not training_manager.cancel(j.id):
        raise AppError(409, "NOT_RUNNING", "This job is not running in the current server process.")
    audit(db, "training_cancel", admin, job_id, request)
    return {"cancelling": True}
