from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from alembic.config import Config
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from alembic import command
from app.api import auth, dashboard, dataset, models, predictions, system, training
from app.core.config import BACKEND_DIR, get_settings
from app.core.errors import install_error_handlers
from app.db.session import SessionLocal
from app.services.registry import sync_runs
from app.services.training_manager import training_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pneumovision")


def run_migrations() -> None:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    command.upgrade(cfg, "head")


@asynccontextmanager
async def lifespan(_: FastAPI):
    s = get_settings()
    s.resolved_secret_key()  # fail early in production if SECRET_KEY is missing
    run_migrations()
    with SessionLocal() as db:
        training_manager.mark_interrupted_on_startup(db)
        log.info("Registered/checked %d training runs", sync_runs(db))
    yield


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(
        title=s.app_name,
        version=s.app_version,
        lifespan=lifespan,
        description="Explainable pneumonia detection from chest X-rays (research use only).",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "X-CSRF-Token"],
    )
    install_error_handlers(app)

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        rid, t0 = uuid.uuid4().hex[:8], time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        log.info(
            "%s %s %s %d %.0fms", rid, request.method, request.url.path, response.status_code, (time.perf_counter() - t0) * 1000
        )
        return response

    for r in (auth.router, predictions.router, dashboard.router, dataset.router, training.router, models.router, system.router):
        app.include_router(r, prefix="/api/v1")
    return app


app = create_app()
