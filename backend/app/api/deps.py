"""Shared dependencies: DB session, authentication, CSRF protection, audit logging."""

from __future__ import annotations

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.core.errors import AppError
from app.core.security import ACCESS_COOKIE, CSRF_COOKIE, CSRF_HEADER, csrf_matches, decode_access_token
from app.db.models import AuditEvent, User
from app.db.session import get_db

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def client_ip(request: Request) -> str:
    return request.client.host if request.client else ""


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = request.cookies.get(ACCESS_COOKIE)
    payload = decode_access_token(token) if token else None
    if not payload:
        raise AppError(401, "NOT_AUTHENTICATED", "Please sign in to continue.")
    user = db.get(User, int(payload["sub"]))
    if not user or not user.is_active:
        raise AppError(401, "NOT_AUTHENTICATED", "Please sign in to continue.")
    # CSRF double-submit check for state-changing requests (cookie auth).
    if request.method not in SAFE_METHODS and not csrf_matches(
        request.cookies.get(CSRF_COOKIE), request.headers.get(CSRF_HEADER)
    ):
        raise AppError(403, "CSRF_FAILED", "Security token missing or invalid. Reload the page and try again.")
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise AppError(403, "FORBIDDEN", "This action requires an administrator account.")
    return user


def audit(
    db: Session,
    action: str,
    user: User | None = None,
    target: str = "",
    request: Request | None = None,
    detail: dict | None = None,
) -> None:
    db.add(
        AuditEvent(
            user_id=user.id if user else None,
            action=action,
            target=target,
            ip=client_ip(request) if request else "",
            detail=detail or {},
        )
    )
    db.commit()
