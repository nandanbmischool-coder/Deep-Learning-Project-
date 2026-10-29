"""Application settings, loaded from environment variables / .env."""

from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    app_name: str = "PneumoVision AI"
    app_version: str = "1.0.0"
    environment: str = "development"  # development | production

    # --- storage -------------------------------------------------------
    database_url: str = f"sqlite:///{(PROJECT_DIR / 'data' / 'app.db').as_posix()}"
    media_dir: Path = PROJECT_DIR / "data" / "media"
    artifacts_dir: Path = PROJECT_DIR / "data" / "artifacts"
    dataset_dir: Path = PROJECT_DIR / "data" / "raw" / "chest_xray"

    # --- security ------------------------------------------------------
    # Leave empty in development: a random key is generated once and stored in data/.secret_key.
    secret_key: str = ""
    access_token_minutes: int = 60
    cookie_secure: bool = False  # set true behind HTTPS (production)
    cookie_samesite: str = "lax"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    login_max_failures: int = 5
    login_window_seconds: int = 900

    # --- uploads -------------------------------------------------------
    max_upload_mb: int = 10
    max_image_side: int = 8000
    min_image_side: int = 64
    max_image_pixels: int = 40_000_000

    # --- inference -----------------------------------------------------
    inference_device: str = "auto"  # auto | cpu | cuda
    image_size: int = 224

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    def resolved_secret_key(self) -> str:
        if self.secret_key:
            return self.secret_key
        if self.is_production:
            raise RuntimeError("SECRET_KEY must be set in production.")
        key_file = PROJECT_DIR / "data" / ".secret_key"
        key_file.parent.mkdir(parents=True, exist_ok=True)
        if not key_file.exists():
            key_file.write_text(secrets.token_urlsafe(64))
        return key_file.read_text().strip()


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.media_dir.mkdir(parents=True, exist_ok=True)
    s.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return s
