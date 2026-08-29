"""Authentication (JWT), password hashing and role-based access control."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Iterable

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .database import get_db
from .models import Document, RoutingResult, User
from .taxonomy import (
    ORG_WIDE_ROLES,
    ROLE_ADMIN,
    ROLE_DEPARTMENT_HEAD,
    ROLE_EXECUTIVE,
)

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.api_prefix}/auth/login", auto_error=False
)

CREDENTIALS_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


# ------------------------------------------------------------------ passwords
def hash_password(password: str) -> str:
    # bcrypt silently truncates beyond 72 bytes; cut explicitly so a long
    # password never hashes to the same value as its prefix by surprise.
    return pwd_context.hash(password[:72])


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(plain[:72], hashed)
    except Exception:  # noqa: BLE001 - malformed hash must not 500
        return False


# --------------------------------------------------------------------- tokens
def create_access_token(user: User, expires_minutes: int | None = None) -> tuple[str, int]:
    """Return ``(token, expires_in_seconds)`` for a signed access token."""
    minutes = expires_minutes or settings.kmrl_access_token_minutes
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=minutes)
    payload = {
        "sub": user.id,
        "email": user.email,
        "role": user.role_name,
        "department_id": user.department_id or "",
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.kmrl_jwt_algorithm)
    return token, minutes * 60


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.kmrl_jwt_algorithm]
        )
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except jwt.PyJWTError as exc:
        raise CREDENTIALS_ERROR from exc


# --------------------------------------------------------------- dependencies
def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    if not token:
        raise CREDENTIALS_ERROR
    payload = decode_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise CREDENTIALS_ERROR
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise CREDENTIALS_ERROR
    return user


def get_optional_user(
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User | None:
    if not token:
        return None
    try:
        return get_current_user(token=token, db=db)
    except HTTPException:
        return None


def require_roles(*roles: str):
    """Dependency factory: allow only the listed roles."""
    allowed = set(roles)

    def dependency(user: User = Depends(get_current_user)) -> User:
        if user.role_name not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Your role does not permit this operation "
                    f"(requires one of: {', '.join(sorted(allowed))})."
                ),
            )
        return user

    return dependency


require_admin = require_roles(ROLE_ADMIN)
require_admin_or_head = require_roles(ROLE_ADMIN, ROLE_DEPARTMENT_HEAD)
require_leadership = require_roles(ROLE_ADMIN, ROLE_EXECUTIVE, ROLE_DEPARTMENT_HEAD)


# ------------------------------------------------------------ document access
def visible_department_ids(db: Session, user: User) -> set[str]:
    """Departments whose documents this user may read."""
    return {user.department_id} if user.department_id else set()


def can_read_document(db: Session, user: User, document: Document) -> bool:
    """Admins and executives see everything; everyone else needs a connection.

    A "connection" is: they uploaded it, it belongs to their department, it was
    routed to their department, or an action on it is assigned to them.
    """
    if user.role_name in ORG_WIDE_ROLES:
        return True
    if document.uploaded_by_id == user.id:
        return True
    if user.department_id:
        if document.department_id == user.department_id:
            return True
        if document.declared_department_id == user.department_id:
            return True
        routed = db.execute(
            select(RoutingResult.id).where(
                RoutingResult.document_id == document.id,
                RoutingResult.department_id == user.department_id,
            )
        ).first()
        if routed is not None:
            return True
    if document.confidentiality == "Public":
        return True
    return any(action.assigned_to_id == user.id for action in document.actions)


def ensure_can_read(db: Session, user: User, document: Document) -> None:
    if not can_read_document(db, user, document):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this document.",
        )


def can_modify_document(db: Session, user: User, document: Document) -> bool:
    if user.role_name == ROLE_ADMIN:
        return True
    if user.role_name == ROLE_DEPARTMENT_HEAD:
        return can_read_document(db, user, document)
    return document.uploaded_by_id == user.id


def ensure_can_modify(db: Session, user: User, document: Document) -> None:
    if not can_modify_document(db, user, document):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to modify this document.",
        )


def client_context(request: Request) -> tuple[str, str]:
    """(ip, user-agent) for the audit trail."""
    ip = request.client.host if request.client else ""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    return ip[:64], (request.headers.get("user-agent") or "")[:256]


def roles_that_see_everything() -> Iterable[str]:
    return ORG_WIDE_ROLES
