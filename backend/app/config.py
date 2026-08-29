"""Application configuration.

All settings can be overridden through environment variables or a ``.env`` file
placed next to the ``backend`` package (see ``.env.example``).
"""

from __future__ import annotations

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

    app_name: str = "KMRL Document Intelligence"
    api_prefix: str = "/api"

    # --- storage -----------------------------------------------------------
    kmrl_data_dir: str = str(BACKEND_ROOT / "data")
    kmrl_database_url: str = ""

    # --- uploads -----------------------------------------------------------
    kmrl_max_upload_mb: int = 40

    # --- OCR ---------------------------------------------------------------
    kmrl_ocr_enabled: bool = True
    kmrl_ocr_languages: str = "eng+mal"
    kmrl_ocr_max_pages: int = 25
    kmrl_ocr_dpi: int = 200

    # --- AI ----------------------------------------------------------------
    anthropic_api_key: str = ""
    kmrl_analysis_model: str = "claude-opus-5"
    kmrl_analysis_max_chars: int = 60_000

    # --- search ------------------------------------------------------------
    kmrl_default_page_size: int = 20
    kmrl_max_page_size: int = 100

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
    def database_url(self) -> str:
        if self.kmrl_database_url:
            return self.kmrl_database_url
        return f"sqlite:///{self.data_dir / 'kmrl.db'}"

    @property
    def max_upload_bytes(self) -> int:
        return self.kmrl_max_upload_mb * 1024 * 1024

    @property
    def ai_enabled(self) -> bool:
        return bool(self.anthropic_api_key.strip())

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


settings = get_settings()
