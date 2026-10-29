from __future__ import annotations

import json
import threading

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.api.deps import audit, current_user
from app.core.config import get_settings
from app.core.errors import AppError
from app.db.models import User
from app.db.session import get_db
from app.ml.dataset_validation import quick_status, validate_dataset

router = APIRouter(prefix="/dataset", tags=["dataset"])
_state = {"running": False, "progress": None, "error": None}
README_HINT = (
    "Download 'Chest X-Ray Images (Pneumonia)' from Kaggle (accept its terms), then place it so that "
    "data/raw/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA} exists. See README.md, section 'Dataset'."
)


def _report_path():
    return get_settings().artifacts_dir / "dataset_report.json"


@router.get("/status")
def status(user: User = Depends(current_user)):
    q = quick_status(get_settings().dataset_dir)
    return {
        **q,
        "validation_running": _state["running"],
        "validation_progress": _state["progress"],
        "validation_error": _state["error"],
        "instructions": README_HINT,
        "has_report": _report_path().exists(),
    }


def _job() -> None:
    try:

        def prog(i, n):
            _state["progress"] = {"done": i, "total": n}

        rep = validate_dataset(get_settings().dataset_dir, progress=prog)
        _report_path().write_text(json.dumps(rep, indent=2))
        _state["error"] = None
    except Exception as exc:  # noqa: BLE001
        _state["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _state["running"], _state["progress"] = False, None


@router.post("/validate", status_code=202)
def validate(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if _state["running"]:
        raise AppError(409, "VALIDATION_RUNNING", "Dataset validation is already running.")
    _state.update(running=True, error=None)
    audit(db, "dataset_validate", user, "", request)
    threading.Thread(target=_job, daemon=True).start()
    return {"started": True}


@router.get("/summary")
def summary(user: User = Depends(current_user)):
    p = _report_path()
    if not p.exists():
        return {"report": None, "message": "No validation report yet. Run validation first."}
    return {"report": json.loads(p.read_text())}
