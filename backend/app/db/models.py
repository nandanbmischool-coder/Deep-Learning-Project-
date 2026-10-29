"""SQLAlchemy models. Images live on disk (media dir); only relative paths are stored here."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="user")  # user | admin
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    analyses: Mapped[list[Analysis]] = relationship(back_populates="user", cascade="all, delete-orphan")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    architecture: Mapped[str] = mapped_column(String(40))
    pretrained: Mapped[bool] = mapped_column(Boolean)
    pretrained_source: Mapped[str] = mapped_column(String(120), default="")
    run_dir: Mapped[str] = mapped_column(String(255))  # relative to artifacts dir
    parameter_count: Mapped[int] = mapped_column(Integer, default=0)
    model_size_mb: Mapped[float] = mapped_column(Float, default=0.0)
    threshold: Mapped[float] = mapped_column(Float, default=0.5)
    threshold_method: Mapped[str] = mapped_column(String(200), default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    training_config: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    evaluations: Mapped[list[Evaluation]] = relationship(back_populates="model", cascade="all, delete-orphan")


class Evaluation(Base):
    __tablename__ = "evaluations"
    id: Mapped[int] = mapped_column(primary_key=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id", ondelete="CASCADE"), index=True)
    split: Mapped[str] = mapped_column(String(20), default="test")
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    roc_auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    report: Mapped[dict] = mapped_column(JSON)  # full evaluation JSON (metrics, curves, CIs, ...)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    model: Mapped[ModelVersion] = relationship(back_populates="evaluations")


class Analysis(Base):
    __tablename__ = "analyses"
    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True)
    model_version: Mapped[str] = mapped_column(String(80))
    predicted_class: Mapped[str] = mapped_column(String(20), index=True)
    prob_normal: Mapped[float] = mapped_column(Float)
    prob_pneumonia: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    target_class: Mapped[str] = mapped_column(String(20))
    inference_ms: Mapped[float] = mapped_column(Float)
    gradcam_ms: Mapped[float] = mapped_column(Float)
    original_filename_hint: Mapped[str] = mapped_column(String(120), default="")  # sanitized, display only
    original_path: Mapped[str] = mapped_column(String(255))
    thumbnail_path: Mapped[str] = mapped_column(String(255))
    heatmap_path: Mapped[str] = mapped_column(String(255))
    overlay_path: Mapped[str] = mapped_column(String(255))
    image_width: Mapped[int] = mapped_column(Integer)
    image_height: Mapped[int] = mapped_column(Integer)
    validation: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    user: Mapped[User] = relationship(back_populates="analyses")


class TrainingJob(Base):
    __tablename__ = "training_jobs"
    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    architecture: Mapped[str] = mapped_column(String(40))
    config: Mapped[dict] = mapped_column(JSON)
    state: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    # queued | running | completed | failed | cancelled | interrupted
    run_dir: Mapped[str] = mapped_column(String(255))
    error: Mapped[str] = mapped_column(Text, default="")
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action: Mapped[str] = mapped_column(String(60), index=True)
    target: Mapped[str] = mapped_column(String(120), default="")
    ip: Mapped[str] = mapped_column(String(64), default="")
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
