"""Loads the active model version once and runs prediction + Grad-CAM with it."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import AppError
from app.db.models import ModelVersion
from app.ml.data import eval_transform
from app.ml.gradcam import GradCAM, render_overlay
from app.ml.model import ARCHS, CLASS_NAMES, load_checkpoint, pick_device


@dataclass
class Loaded:
    row_id: int
    version: str
    architecture: str
    threshold: float
    model: torch.nn.Module
    device: torch.device
    cam: GradCAM
    transform: object


class ModelService:
    def __init__(self) -> None:
        self._loaded: Loaded | None = None
        self._lock = threading.Lock()

    def invalidate(self) -> None:
        with self._lock:
            if self._loaded:
                self._loaded.cam.remove()
            self._loaded = None

    def is_loaded(self) -> bool:
        return self._loaded is not None

    def _load(self, row: ModelVersion) -> Loaded:
        s = get_settings()
        ckpt_path = s.artifacts_dir / row.run_dir / "model.pt"
        if not ckpt_path.exists():
            raise AppError(503, "MODEL_FILE_MISSING", f"Checkpoint file for model '{row.version}' is missing.")
        device = pick_device(s.inference_device)
        model, _ = load_checkpoint(ckpt_path, device)
        layer = ARCHS[row.architecture].cam_layer
        return Loaded(
            row.id,
            row.version,
            row.architecture,
            row.threshold,
            model,
            device,
            GradCAM(model, layer),
            eval_transform(s.image_size),
        )

    def active(self, db: Session) -> Loaded:
        row = db.scalar(select(ModelVersion).where(ModelVersion.is_active.is_(True)))
        if row is None:
            raise AppError(503, "MODEL_NOT_TRAINED", "Model not trained. An administrator must train and activate a model first.")
        with self._lock:
            if self._loaded is None or self._loaded.row_id != row.id:
                if self._loaded:
                    self._loaded.cam.remove()
                self._loaded = self._load(row)
            return self._loaded

    def predict(self, db: Session, rgb: Image.Image, target_class: str | None) -> dict:
        m = self.active(db)
        x = m.transform(rgb).unsqueeze(0).to(m.device)
        with self._lock:  # hooks are per-model state, so serialise inference
            t0 = time.perf_counter()
            with torch.no_grad():
                probs = torch.softmax(m.model(x), dim=1)[0].cpu().numpy()
            inference_ms = (time.perf_counter() - t0) * 1000
            p_pneu = float(probs[1])
            predicted = "PNEUMONIA" if p_pneu >= m.threshold else "NORMAL"
            target = target_class or predicted
            t1 = time.perf_counter()
            res = m.cam(x, CLASS_NAMES.index(target))
            gradcam_ms = (time.perf_counter() - t1) * 1000
        heat, overlay = render_overlay(np.asarray(rgb), res["cam"])
        return {
            "model": m,
            "prob_normal": float(probs[0]),
            "prob_pneumonia": p_pneu,
            "predicted": predicted,
            "target": target,
            "inference_ms": inference_ms,
            "gradcam_ms": gradcam_ms,
            "heat": heat,
            "overlay": overlay,
            "flat_cam": res["flat"],
        }


model_service = ModelService()


def get_model_service() -> ModelService:
    return model_service


def checkpoint_dir(run_dir: str) -> Path:
    return get_settings().artifacts_dir / run_dir
