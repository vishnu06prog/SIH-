"""Application configuration.

Every setting can be overridden through environment variables or a ``.env``
file at the repository root or inside ``backend/`` (see ``.env.example``).
Secrets are NEVER hard-coded: they are read from the environment only.
"""

from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_prefix="",
        extra="ignore",
    )

    app_name: str = "KMRL Intelligent Document Management & Decision Support System"
    app_short_name: str = "KMRL DMS"
    api_prefix: str = "/api"
    environment: str = "development"

    # --- storage -----------------------------------------------------------
    kmrl_data_dir: str = str(BACKEND_ROOT / "data")
    kmrl_database_url: str = ""

    # --- uploads -----------------------------------------------------------
    kmrl_max_upload_mb: int = 40
    kmrl_allowed_extensions: str = ".pdf,.docx,.doc,.txt,.md,.png,.jpg,.jpeg,.tiff,.bmp,.webp"

    # --- security ----------------------------------------------------------
    kmrl_jwt_secret: str = ""
    kmrl_jwt_algorithm: str = "HS256"
    kmrl_access_token_minutes: int = 12 * 60
    kmrl_cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173"
    # Password used only when bootstrapping the demo accounts.
    kmrl_demo_password: str = "kmrl@2026"
    kmrl_seed_demo_users: bool = True

    # --- OCR ---------------------------------------------------------------
    kmrl_ocr_enabled: bool = True
    kmrl_ocr_languages: str = "eng+mal"
    kmrl_ocr_max_pages: int = 25
    kmrl_ocr_dpi: int = 200

    # --- LLM provider ------------------------------------------------------
    # "auto" picks Anthropic when a key is present, otherwise the local engine.
    kmrl_llm_provider: str = "auto"
    anthropic_api_key: str = ""
    kmrl_analysis_model: str = "claude-opus-5"
    kmrl_analysis_max_chars: int = 60_000
    kmrl_rag_model: str = "claude-opus-5"

    # --- embeddings / vector search ---------------------------------------
    # "auto" uses sentence-transformers when installed, else the built-in
    # deterministic hashing embedder (no downloads, works offline).
    kmrl_embedding_provider: str = "auto"
    kmrl_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    kmrl_embedding_dim: int = 384
    kmrl_chunk_chars: int = 1200
    kmrl_chunk_overlap: int = 180
    kmrl_vector_backend: str = "auto"  # auto | pgvector | numpy
    kmrl_semantic_top_k: int = 8
    kmrl_hybrid_alpha: float = 0.6  # weight of the semantic score in hybrid search

    # --- duplicates / versions --------------------------------------------
    kmrl_duplicate_threshold: float = 0.92
    kmrl_version_threshold: float = 0.78

    # --- search ------------------------------------------------------------
    kmrl_default_page_size: int = 20
    kmrl_max_page_size: int = 100

    # --- deadlines ---------------------------------------------------------
    kmrl_deadline_warning_days: int = 7

    # ---------------------------------------------------------------- helpers
    @property
    def data_dir(self) -> Path:
        path = Path(self.kmrl_data_dir)
        if not path.is_absolute():
            path = (BACKEND_ROOT / path).resolve()
        return path

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def sample_dir(self) -> Path:
        return self.data_dir / "samples"

    @property
    def database_url(self) -> str:
        if self.kmrl_database_url:
            return self.kmrl_database_url
        return f"sqlite:///{self.data_dir / 'kmrl.db'}"

    @property
    def max_upload_bytes(self) -> int:
        return self.kmrl_max_upload_mb * 1024 * 1024

    @property
    def allowed_extensions(self) -> set[str]:
        return {
            ext.strip().lower()
            for ext in self.kmrl_allowed_extensions.split(",")
            if ext.strip()
        }

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.kmrl_cors_origins.split(",") if o.strip()]

    @property
    def llm_enabled(self) -> bool:
        provider = self.kmrl_llm_provider.strip().lower()
        if provider == "none":
            return False
        return bool(self.anthropic_api_key.strip())

    @property
    def llm_provider_name(self) -> str:
        return "anthropic" if self.llm_enabled else "local"

    @property
    def jwt_secret(self) -> str:
        """The signing key. Generated per-process when not configured.

        A generated key means tokens do not survive a restart — fine for local
        development, and loudly documented in ``.env.example`` for deployment.
        """
        if self.kmrl_jwt_secret.strip():
            return self.kmrl_jwt_secret.strip()
        return _ephemeral_secret()

    @property
    def jwt_secret_is_ephemeral(self) -> bool:
        return not self.kmrl_jwt_secret.strip()

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def _ephemeral_secret() -> str:
    return secrets.token_urlsafe(48)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


settings = get_settings()
