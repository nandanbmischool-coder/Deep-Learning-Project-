from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import audit, client_ip, current_user
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.security import (
    ACCESS_COOKIE,
    CSRF_COOKIE,
    create_access_token,
    hash_password,
    login_limiter,
    new_csrf_token,
    verify_password,
)
from app.db.models import User
from app.db.session import get_db
from app.schemas import LoginIn, PasswordChange, ProfileUpdate, RegisterIn, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])

# Hash of a throwaway password, used so unknown-email logins cost the same as wrong-password logins.
_DUMMY_HASH = hash_password("not-a-real-password-1")


def _set_session(response: Response, user: User) -> None:
    s = get_settings()
    max_age = s.access_token_minutes * 60
    common = {"max_age": max_age, "secure": s.cookie_secure, "samesite": s.cookie_samesite, "path": "/"}
    response.set_cookie(ACCESS_COOKIE, create_access_token(user.id, user.role), httponly=True, **common)
    # CSRF cookie is readable by JS on purpose (double-submit pattern).
    response.set_cookie(CSRF_COOKIE, new_csrf_token(), httponly=False, **common)


@router.post("/register", response_model=UserOut, status_code=201)
def register(body: RegisterIn, request: Request, response: Response, db: Session = Depends(get_db)):
    email = body.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise AppError(409, "EMAIL_TAKEN", "An account with this email already exists.")
    first = db.scalar(select(func.count(User.id))) == 0
    user = User(
        email=email,
        full_name=body.full_name.strip(),
        password_hash=hash_password(body.password),
        role="admin" if first else "user",
    )  # the first registered account becomes administrator
    db.add(user)
    db.commit()
    audit(db, "register", user, f"user:{user.id}", request, {"role": user.role})
    _set_session(response, user)
    return user


@router.post("/login", response_model=UserOut)
def login(body: LoginIn, request: Request, response: Response, db: Session = Depends(get_db)):
    email = body.email.lower()
    key = f"{client_ip(request)}|{email}"
    if login_limiter.blocked(key):
        raise AppError(429, "TOO_MANY_ATTEMPTS", "Too many failed sign-in attempts. Try again later.")
    user = db.scalar(select(User).where(User.email == email))
    ok = verify_password(body.password, user.password_hash if user else _DUMMY_HASH)
    if not user or not ok or not user.is_active:
        login_limiter.record_failure(key)
        audit(db, "login_failed", None, email[:120], request)
        raise AppError(401, "INVALID_CREDENTIALS", "Email or password is incorrect.")
    login_limiter.reset(key)
    audit(db, "login", user, f"user:{user.id}", request)
    _set_session(response, user)
    return user


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    # Logout only clears cookies; it is safe without CSRF because it cannot change data.
    response.delete_cookie(ACCESS_COOKIE, path="/")
    response.delete_cookie(CSRF_COOKIE, path="/")
    audit(db, "logout", None, "", request)
    response.status_code = 204
    return response


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return user


@router.patch("/me", response_model=UserOut)
def update_me(body: ProfileUpdate, user: User = Depends(current_user), db: Session = Depends(get_db)):
    user.full_name = body.full_name.strip()
    db.commit()
    return user


@router.post("/change-password", status_code=204)
def change_password(body: PasswordChange, request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if not verify_password(body.current_password, user.password_hash):
        raise AppError(400, "WRONG_PASSWORD", "Current password is incorrect.")
    user.password_hash = hash_password(body.new_password)
    db.commit()
    audit(db, "password_change", user, f"user:{user.id}", request)
    return Response(status_code=204)
