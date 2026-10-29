"""Pydantic request/response schemas."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

DISCLAIMER = (
    "Research and education use only. This tool is not a clinically validated diagnostic device and "
    "must not be used to make medical decisions. Model probabilities are uncalibrated scores unless stated otherwise."
)


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ----- auth -------------------------------------------------------------
class RegisterIn(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=2, max_length=120)
    password: str = Field(min_length=10, max_length=128)

    @field_validator("password")
    @classmethod
    def _strong(cls, v: str) -> str:
        if not (re.search(r"[A-Za-z]", v) and re.search(r"\d", v)):
            raise ValueError("Password must contain letters and at least one digit.")
        return v


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserOut(ORM):
    id: int
    email: str
    full_name: str
    role: str
    created_at: datetime


class ProfileUpdate(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)


class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(min_length=10, max_length=128)

    @field_validator("new_password")
    @classmethod
    def _strong(cls, v: str) -> str:
        if not (re.search(r"[A-Za-z]", v) and re.search(r"\d", v)):
            raise ValueError("Password must contain letters and at least one digit.")
        return v


# ----- predictions ------------------------------------------------------
class ValidationInfo(BaseModel):
    status: Literal["ok", "warning"]
    warnings: list[str]
    checks: dict
    note: str


class PredictionOut(ORM):
    analysis_id: str
    predicted_class: str
    probabilities: dict[str, float]
    threshold: float
    target_class: str
    model_version: str
    inference_ms: float
    gradcam_ms: float
    image_size: dict[str, int]
    image_validation: dict
    assets: dict[str, str]
    created_at: datetime
    disclaimer: str = DISCLAIMER


class Page(BaseModel):
    items: list
    total: int
    page: int
    page_size: int


# ----- training ---------------------------------------------------------
class TrainingJobIn(BaseModel):
    architecture: Literal["densenet121", "resnet50", "mobilenet_v3_large"] = "densenet121"
    epochs: int = Field(default=12, ge=1, le=200, description="Total epochs (head + fine-tune)")
    head_epochs: int = Field(default=4, ge=0, le=100)
    batch_size: int = Field(default=32, ge=2, le=256)
    learning_rate: float = Field(default=1e-3, gt=0, le=1.0)
    finetune_learning_rate: float = Field(default=1e-4, gt=0, le=1.0)
    fine_tune_mode: Literal["head_only", "last_block", "full"] = "last_block"
    patience: int = Field(default=4, ge=1, le=50)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    val_fraction: float = Field(default=0.15, gt=0.05, lt=0.4)
    horizontal_flip: bool = False
    pretrained: bool = True
    weights_file: str | None = Field(default=None, description="File name inside data/artifacts/weights/")
    max_images_per_class: int | None = Field(default=None, ge=8, description="Smoke-test subset size")
    num_workers: int = Field(default=0, ge=0, le=16)

    @field_validator("weights_file")
    @classmethod
    def _plain_name(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", v):
            raise ValueError("weights_file must be a plain file name (letters, digits, . _ -).")
        return v
