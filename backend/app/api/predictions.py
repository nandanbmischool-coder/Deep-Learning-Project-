from __future__ import annotations

import base64
import html
import shutil
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import audit, current_user
from app.core.config import get_settings
from app.core.errors import AppError
from app.db.models import Analysis, User
from app.db.session import get_db
from app.schemas import DISCLAIMER
from app.services.image_io import (
    decode_image,
    new_public_id,
    read_limited,
    resolve_media,
    safe_display_name,
    save_analysis_images,
)
from app.services.model_service import ModelService, get_model_service

router = APIRouter(prefix="/predictions", tags=["predictions"])
ASSET_KINDS = {"original": "original_path", "heatmap": "heatmap_path", "overlay": "overlay_path", "thumbnail": "thumbnail_path"}


def to_out(a: Analysis) -> dict:
    base = f"/api/v1/predictions/{a.public_id}/assets"
    return {
        "analysis_id": a.public_id,
        "predicted_class": a.predicted_class,
        "probabilities": {"NORMAL": a.prob_normal, "PNEUMONIA": a.prob_pneumonia},
        "threshold": a.threshold,
        "target_class": a.target_class,
        "model_version": a.model_version,
        "inference_ms": a.inference_ms,
        "gradcam_ms": a.gradcam_ms,
        "image_size": {"width": a.image_width, "height": a.image_height},
        "image_validation": a.validation,
        "assets": {k: f"{base}/{k}" for k in ASSET_KINDS},
        "created_at": a.created_at,
        "disclaimer": DISCLAIMER,
        "preprocessing": {
            "steps": ["convert to RGB", "resize 224x224 (bilinear)", "to tensor", "ImageNet mean/std normalisation"],
            "augmentation": "none at inference",
        },
    }


def _owned(db: Session, user: User, public_id: str) -> Analysis:
    a = db.scalar(select(Analysis).where(Analysis.public_id == public_id, Analysis.user_id == user.id))
    if a is None:  # same response for "missing" and "someone else's" so IDs cannot be probed
        raise AppError(404, "NOT_FOUND", "Analysis not found.")
    return a


@router.post("", status_code=201)
def create_prediction(
    file: UploadFile = File(...),
    target_class: str | None = Form(default=None),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
    svc: ModelService = Depends(get_model_service),
):
    s = get_settings()
    if target_class is not None and target_class.upper() not in ("NORMAL", "PNEUMONIA"):
        raise AppError(422, "INVALID_TARGET", "target_class must be NORMAL or PNEUMONIA.")
    svc.active(db)  # fail fast with MODEL_NOT_TRAINED before reading the upload
    data = read_limited(file.file, s.max_upload_mb * 1024 * 1024)
    dec = decode_image(data)
    res = svc.predict(db, dec.image, target_class.upper() if target_class else None)
    pid = new_public_id()
    paths = save_analysis_images(pid, user.id, dec.image, res["heat"], res["overlay"])
    a = Analysis(
        public_id=pid,
        user_id=user.id,
        model_version_id=res["model"].row_id,
        model_version=res["model"].version,
        predicted_class=res["predicted"],
        prob_normal=res["prob_normal"],
        prob_pneumonia=res["prob_pneumonia"],
        threshold=res["model"].threshold,
        target_class=res["target"],
        inference_ms=res["inference_ms"],
        gradcam_ms=res["gradcam_ms"],
        original_filename_hint=safe_display_name(file.filename),
        original_path=paths["original"],
        thumbnail_path=paths["thumbnail"],
        heatmap_path=paths["heatmap"],
        overlay_path=paths["overlay"],
        image_width=dec.width,
        image_height=dec.height,
        validation={**dec.validation, "gradcam_flat_map": res["flat_cam"], "source_format": dec.source_format},
    )
    db.add(a)
    db.commit()
    return to_out(a)


@router.get("")
def list_predictions(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    q: str | None = None,
    predicted_class: str | None = Query(None, pattern="^(NORMAL|PNEUMONIA)$"),
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    stmt = select(Analysis).where(Analysis.user_id == user.id)
    if q:
        like = f"%{q.strip()[:60]}%"
        stmt = stmt.where(Analysis.public_id.ilike(like) | Analysis.original_filename_hint.ilike(like))
    if predicted_class:
        stmt = stmt.where(Analysis.predicted_class == predicted_class)
    if date_from:
        stmt = stmt.where(Analysis.created_at >= date_from)
    if date_to:
        stmt = stmt.where(Analysis.created_at <= date_to)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(stmt.order_by(Analysis.created_at.desc()).offset((page - 1) * page_size).limit(page_size)).all()
    return {
        "items": [{**to_out(a), "filename_hint": a.original_filename_hint} for a in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/{analysis_id}")
def get_prediction(analysis_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    a = _owned(db, user, analysis_id)
    return {**to_out(a), "filename_hint": a.original_filename_hint}


@router.delete("/{analysis_id}", status_code=204)
def delete_prediction(analysis_id: str, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    a = _owned(db, user, analysis_id)
    shutil.rmtree(resolve_media(a.original_path).parent, ignore_errors=True)
    db.delete(a)
    db.commit()
    audit(db, "delete_analysis", user, analysis_id, request)


@router.get("/{analysis_id}/assets/{kind}")
def get_asset(
    analysis_id: str, kind: str, download: bool = False, user: User = Depends(current_user), db: Session = Depends(get_db)
):
    if kind not in ASSET_KINDS:
        raise AppError(404, "NOT_FOUND", "Unknown asset.")
    a = _owned(db, user, analysis_id)
    path = resolve_media(getattr(a, ASSET_KINDS[kind]))
    if not path.exists():
        raise AppError(404, "NOT_FOUND", "File not found.")
    headers = {"Cache-Control": "private, max-age=300", "X-Content-Type-Options": "nosniff"}
    return FileResponse(path, filename=f"{analysis_id}_{kind}{path.suffix}" if download else None, headers=headers)


def _b64(rel: str) -> str:
    return base64.b64encode(resolve_media(rel).read_bytes()).decode()


@router.get("/{analysis_id}/report")
def report(
    analysis_id: str,
    format: str = Query("html", pattern="^(html|json)$"),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    a = _owned(db, user, analysis_id)
    out = to_out(a)
    if format == "json":
        return JSONResponse(jsonable(out), headers={"Content-Disposition": f'attachment; filename="{analysis_id}.json"'})
    imgs = "".join(
        f'<figure><img src="data:image/{"jpeg" if k == "thumbnail" else "png"};base64,{_b64(getattr(a, ASSET_KINDS[k]))}" '
        f'style="max-width:100%"><figcaption>{t}</figcaption></figure>'
        for k, t in (("original", "Original"), ("heatmap", "Grad-CAM heatmap"), ("overlay", "Grad-CAM overlay"))
    )
    warns = "".join(f"<li>{html.escape(w)}</li>" for w in a.validation.get("warnings", [])) or "<li>None</li>"
    body = f"""<!doctype html><html><head><meta charset="utf-8"><title>Analysis {analysis_id}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;color:#12222a}}
.imgs{{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem}}td{{padding:.25rem .8rem .25rem 0}}</style></head><body>
<h1>PneumoVision AI - Analysis report</h1><table>
<tr><td>Analysis ID</td><td>{analysis_id}</td></tr><tr><td>Created</td><td>{a.created_at:%Y-%m-%d %H:%M UTC}</td></tr>
<tr><td>Predicted class</td><td><b>{a.predicted_class}</b> (decision threshold on P(pneumonia) = {a.threshold:.3f})</td></tr>
<tr><td>P(NORMAL) / P(PNEUMONIA)</td><td>{a.prob_normal:.4f} / {a.prob_pneumonia:.4f} (uncalibrated model scores)</td></tr>
<tr><td>Grad-CAM target class</td><td>{a.target_class}</td></tr><tr><td>Model</td><td>{html.escape(a.model_version)}</td></tr>
<tr><td>Inference / Grad-CAM time</td><td>{a.inference_ms:.1f} ms / {a.gradcam_ms:.1f} ms</td></tr>
<tr><td>Image size</td><td>{a.image_width} x {a.image_height}</td></tr></table>
<h3>Image checks (heuristic)</h3><ul>{warns}</ul><div class="imgs">{imgs}</div>
<p><small>Grad-CAM highlights regions that influenced the model's score. It explains model activation and does not prove a
medical finding.</small></p><p><b>{html.escape(DISCLAIMER)}</b></p></body></html>"""
    return HTMLResponse(body, headers={"Content-Disposition": f'attachment; filename="{analysis_id}.html"'})


def jsonable(d: dict) -> dict:
    from fastapi.encoders import jsonable_encoder

    return jsonable_encoder(d)
