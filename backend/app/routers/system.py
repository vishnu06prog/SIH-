"""Health and capability endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ..config import settings
from ..services import extraction as extraction_service

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "app": settings.app_name,
        "capabilities": {
            "ai_analysis": "claude" if settings.ai_enabled else "rule-based",
            "analysis_model": settings.kmrl_analysis_model if settings.ai_enabled else None,
            "ocr": extraction_service.ocr_available(),
            "ocr_engine": extraction_service.ocr_engine_version(),
            "ocr_languages": settings.kmrl_ocr_languages,
            "pdf_rendering": extraction_service.poppler_available(),
        },
        "limits": {
            "max_upload_mb": settings.kmrl_max_upload_mb,
            "default_page_size": settings.kmrl_default_page_size,
            "max_page_size": settings.kmrl_max_page_size,
        },
    }
