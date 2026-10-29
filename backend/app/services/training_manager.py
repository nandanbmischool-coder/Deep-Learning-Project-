"""Runs training as a separate OS process (so the API never blocks) with one-job-at-a-time locking."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from app.core.config import BACKEND_DIR, get_settings
from app.core.errors import AppError
from app.db.models import TrainingJob
from app.db.session import SessionLocal
from app.schemas import TrainingJobIn
from app.services.registry import register_run

ACTIVE_STATES = ("queued", "running")


class TrainingManager:
    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._job_id: int | None = None
        self._cancel_requested = False
        self._lock = threading.Lock()

    def running_job_id(self) -> int | None:
        return self._job_id if self._proc and self._proc.poll() is None else None

    def start(self, db, body: TrainingJobIn, user_id: int, public_id: str) -> TrainingJob:
        s = get_settings()
        with self._lock:
            if self.running_job_id() is not None or db.scalar(select(TrainingJob).where(TrainingJob.state.in_(ACTIVE_STATES))):
                raise AppError(409, "TRAINING_BUSY", "A training job is already running. Wait for it to finish or cancel it.")
            if not s.dataset_dir.exists():
                raise AppError(400, "DATASET_MISSING", "Dataset not found. Follow the README to download it first.")
            stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            version = f"{body.architecture}-{stamp}"
            rel = f"runs/{version}"
            run_dir = s.artifacts_dir / rel
            run_dir.mkdir(parents=True, exist_ok=False)  # unique directory per run => checkpoints cannot collide
            cmd = [
                sys.executable,
                "-m",
                "app.ml.train",
                "--arch",
                body.architecture,
                "--epochs",
                str(body.epochs),
                "--head-epochs",
                str(body.head_epochs),
                "--batch-size",
                str(body.batch_size),
                "--lr",
                str(body.learning_rate),
                "--finetune-lr",
                str(body.finetune_learning_rate),
                "--fine-tune-mode",
                body.fine_tune_mode,
                "--patience",
                str(body.patience),
                "--seed",
                str(body.seed),
                "--val-fraction",
                str(body.val_fraction),
                "--num-workers",
                str(body.num_workers),
                "--run-dir",
                str(run_dir),
                "--version",
                version,
            ]
            cmd.append("--pretrained" if body.pretrained else "--no-pretrained")
            if body.horizontal_flip:
                cmd.append("--horizontal-flip")
            if body.max_images_per_class:
                cmd += ["--max-images-per-class", str(body.max_images_per_class)]
            if body.weights_file:
                wp = (s.artifacts_dir / "weights" / body.weights_file).resolve()
                if (s.artifacts_dir / "weights").resolve() not in wp.parents or not wp.is_file():
                    raise AppError(400, "WEIGHTS_NOT_FOUND", "Weights file not found in data/artifacts/weights/.")
                cmd += ["--weights-path", str(wp)]
            job = TrainingJob(
                public_id=public_id,
                created_by=user_id,
                architecture=body.architecture,
                config=body.model_dump(),
                state="running",
                run_dir=rel,
                started_at=datetime.now(UTC),
            )
            db.add(job)
            db.commit()
            log = open(run_dir / "train.log", "ab")
            self._proc = subprocess.Popen(cmd, cwd=BACKEND_DIR, stdout=log, stderr=subprocess.STDOUT)
            self._job_id, self._cancel_requested = job.id, False
            threading.Thread(target=self._watch, args=(self._proc, job.id, run_dir), daemon=True).start()
            return job

    def _watch(self, proc: subprocess.Popen, job_id: int, run_dir: Path) -> None:
        code = proc.wait()
        with SessionLocal() as db:
            job = db.get(TrainingJob, job_id)
            if job is None:
                return
            status = {}
            try:
                status = json.loads((run_dir / "status.json").read_text())
            except Exception:  # noqa: BLE001
                pass
            if self._cancel_requested:
                job.state = "cancelled"
            elif code == 0 and status.get("state") == "completed":
                job.state = "completed"
                row = register_run(db, run_dir)
                job.model_version_id = row.id if row else None
            else:
                job.state, job.error = "failed", status.get("message") or f"Process exited with code {code}"
            job.finished_at = datetime.now(UTC)
            db.commit()
        self._job_id = None

    def cancel(self, job_id: int) -> bool:
        with self._lock:
            if self._job_id == job_id and self._proc and self._proc.poll() is None:
                self._cancel_requested = True
                self._proc.terminate()  # SIGTERM on Linux/macOS (graceful); TerminateProcess on Windows
                return True
            return False

    def mark_interrupted_on_startup(self, db) -> None:
        for job in db.scalars(select(TrainingJob).where(TrainingJob.state.in_(ACTIVE_STATES))):
            job.state, job.error, job.finished_at = (
                "interrupted",
                "Server restarted while the job was running.",
                datetime.now(UTC),
            )
        db.commit()


training_manager = TrainingManager()


def read_progress(run_dir: Path, tail: int = 80) -> dict:
    out: dict = {"status": None, "history": [], "log_tail": []}
    try:
        out["status"] = json.loads((run_dir / "status.json").read_text())
    except Exception:  # noqa: BLE001
        pass
    pj = run_dir / "progress.jsonl"
    if pj.exists():
        out["history"] = [json.loads(line) for line in pj.read_text().splitlines() if line.strip()]
    lg = run_dir / "train.log"
    if lg.exists():
        out["log_tail"] = lg.read_text(errors="replace").splitlines()[-tail:]
    return out
