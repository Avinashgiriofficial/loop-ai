from __future__ import annotations

import asyncio
import base64
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
import hashlib
import hmac
import html as html_lib
import io
import json
import logging
import math
from pathlib import Path
import re
import secrets
import socket
import struct
import time
from typing import Any, Literal
from urllib.parse import urlencode, urlparse
import uuid
import zipfile
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import (
    BackgroundTasks,
    Cookie,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx
import jwt
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from pypdf import PdfReader
import pytesseract
from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    event,
    or_,
    select,
    update,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

# =============================================================================
# ===== 1. CONFIG & ENV (read from environment variables, never hard-code secrets)
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_FILE = BASE_DIR / "loop-personal-archive.html"
STATIC_DIR = BASE_DIR / "static"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("loop")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = "development"
    app_origin: str = "http://localhost:8000"
    database_url: str = "sqlite:///./loop.db"

    supabase_url: str = ""
    supabase_publishable_key: str = ""
    supabase_service_role_key: str = ""

    cookie_secure: bool = False
    storage_dir: str = "./loop_storage"
    storage_signing_secret: str = ""

    dev_demo_mode: bool = False
    dev_demo_jwt_secret: str = ""

    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_api_key: str = ""
    llm_model: str = "gemini-3.5-flash-lite"
    embedding_model: str = "gemini-embedding-001"
    ask_similarity_threshold: float = 0.35
    transcribe_model: str = "whisper-1"
    llm_timeout_sec: float = 30.0

    clamav_host: str = ""
    clamav_port: int = 3310

    resend_api_key: str = ""
    email_from: str = "LOOP Reminders <onboarding@resend.dev>"
    reminder_poll_interval_sec: int = 15

    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB

    @property
    def normalized_database_url(self) -> str:
        url = self.database_url.strip()
        if url.startswith("postgres://"):
            return "postgresql+psycopg://" + url[len("postgres://") :]
        if url.startswith("postgresql://"):
            return "postgresql+psycopg://" + url[len("postgresql://") :]
        return url

    @property
    def is_dev_demo_enabled(self) -> bool:
        return self.env == "development" and self.dev_demo_mode and len(self.dev_demo_jwt_secret) >= 32

    @property
    def jwks_url(self) -> str:
        return f"{self.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"


settings = Settings()
_ephemeral_storage_secret = secrets.token_hex(32)


def get_storage_signing_secret() -> str:
    if settings.storage_signing_secret:
        return settings.storage_signing_secret
    if settings.env != "development":
        raise RuntimeError("STORAGE_SIGNING_SECRET must be set in production")
    return _ephemeral_storage_secret


def validate_production_config() -> None:
    if settings.env == "development":
        return
    missing = []
    if not settings.cookie_secure:
        missing.append("COOKIE_SECURE=true")
    if settings.dev_demo_mode:
        missing.append("DEV_DEMO_MODE=false")
    for field_name in (
        "supabase_url",
        "supabase_publishable_key",
        "storage_signing_secret",
        "llm_api_key",
        "resend_api_key",
    ):
        if not getattr(settings, field_name):
            missing.append(field_name.upper())
    if missing:
        raise RuntimeError(f"Missing required production config: {', '.join(missing)}")


# =============================================================================
# ===== 2. DATABASE (schema as SQL strings or SQLAlchemy models, auto-created on startup)
# =============================================================================


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


_db_url = settings.normalized_database_url
connect_args = {"check_same_thread": False} if _db_url.startswith("sqlite") else {}
engine = create_engine(_db_url, connect_args=connect_args, pool_pre_ping=True, future=True)

if _db_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _enable_sqlite_fks(dbapi_conn, _conn_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    auth_provider_sub: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    plan: Mapped[str] = mapped_column(String(32), default="free", nullable=False)
    permissions_granted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)


class UserSettings(Base):
    __tablename__ = "user_settings"
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    timezone: Mapped[str] = mapped_column(String(64), default="Asia/Kolkata", nullable=False)
    allow_document_parsing: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    allow_notifications: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    allow_microphone: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    default_lead_days: Mapped[int] = mapped_column(Integer, default=2, nullable=False)
    email_digest: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    push_notifications: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    sms_fallback: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    consent_local_parsing: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    consent_pattern_insights: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    consent_model_training: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (Index("idx_documents_user_status", "user_id", "processing_status"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    storage_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    sha256_checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)
    processing_status: Mapped[str] = mapped_column(String(32), default="queued", nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)


class Extraction(Base):
    __tablename__ = "extractions"
    __table_args__ = (Index("idx_extractions_user_status", "user_id", "review_status"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=
