"""Database engine, session factory, declarative base and pgvector detection."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


def _make_engine(url: str):
    if url.startswith("sqlite"):
        engine = create_engine(
            url, connect_args={"check_same_thread": False}, future=True
        )

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, _record):  # pragma: no cover - trivial
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine

    return create_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=20, future=True)


engine = _make_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

_pgvector_ready: bool | None = None


def is_postgres() -> bool:
    return engine.dialect.name == "postgresql"


def pgvector_available() -> bool:
    """True when running on PostgreSQL with the ``vector`` extension enabled.

    Probed once and cached; the result decides whether similarity search runs
    as an SQL ANN query or as the portable NumPy scan.
    """
    global _pgvector_ready
    if _pgvector_ready is not None:
        return _pgvector_ready
    if not is_postgres() or settings.kmrl_vector_backend == "numpy":
        _pgvector_ready = False
        return False
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        _pgvector_ready = True
        logger.info("pgvector extension is available; using SQL vector search.")
    except Exception as exc:  # noqa: BLE001 - any failure means "not available"
        logger.info("pgvector unavailable (%s); using the NumPy vector index.", exc)
        _pgvector_ready = False
    return _pgvector_ready


def _ensure_pgvector_column() -> None:
    """Add the pgvector column and an IVFFlat index when running on Postgres."""
    dim = settings.kmrl_embedding_dim
    statements = [
        f"ALTER TABLE document_embeddings ADD COLUMN IF NOT EXISTS embedding vector({dim})",
        "CREATE INDEX IF NOT EXISTS ix_embeddings_vector "
        "ON document_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
    ]
    try:
        with engine.begin() as connection:
            for statement in statements:
                connection.execute(text(statement))
    except Exception as exc:  # noqa: BLE001 - degrade to the NumPy backend
        logger.warning("Could not prepare the pgvector column: %s", exc)
        global _pgvector_ready
        _pgvector_ready = False


def init_db() -> None:
    """Create tables for every registered model (and the pgvector bits)."""
    from . import models  # noqa: F401  (import registers the mappers)

    Base.metadata.create_all(bind=engine)
    if pgvector_available():
        _ensure_pgvector_column()


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
