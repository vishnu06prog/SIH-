"""Pydantic request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    """Compact shape used by the library grid — never includes full text."""

    id: int
    filename: str
    title: str = ""
    file_type: str
    file_ext: str = ""
    size_bytes: int = 0
    uploaded_at: datetime
    processed_at: datetime | None = None
    status: str
    status_label: str
    status_detail: str = ""
    page_count: int | None = None
    word_count: int = 0
    language: str = "Unknown"
    ocr_used: bool = False
    ocr_quality: float | None = None
    document_type: str = ""
    department: str = ""
    priority: str = "Medium"
    summary: str = ""
    keywords: list[str] = Field(default_factory=list)
    action_count: int = 0
    deadline_count: int = 0
    analysis_engine: str = ""
    snippet: str | None = None
    score: float | None = None


class ActionItem(BaseModel):
    action: str
    owner: str = ""
    due_date: str = ""
    priority: str = ""


class DeadlineItem(BaseModel):
    date: str
    description: str = ""


class DocumentDetail(DocumentSummary):
    """Full intelligence payload for a single document."""

    secondary_departments: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    deadlines: list[dict[str, Any]] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    compliance_flags: list[str] = Field(default_factory=list)
    analysis_model: str = ""
    analysis_confidence: float | None = None
    analysis_error: str = ""
    error_message: str = ""
    extraction_method: str = ""
    char_count: int = 0
    uploaded_by: str = ""
    text_preview: str = ""


class SearchFacetValue(BaseModel):
    value: str
    label: str
    count: int


class SearchFacets(BaseModel):
    departments: list[SearchFacetValue] = Field(default_factory=list)
    priorities: list[SearchFacetValue] = Field(default_factory=list)
    file_types: list[SearchFacetValue] = Field(default_factory=list)
    statuses: list[SearchFacetValue] = Field(default_factory=list)
    document_types: list[SearchFacetValue] = Field(default_factory=list)


class DocumentSearchResponse(BaseModel):
    items: list[DocumentSummary]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool
    query: str = ""
    applied_filters: dict[str, Any] = Field(default_factory=dict)
    sort: str = "relevance"
    took_ms: int = 0
    message: str | None = None
    suggestion: str | None = None


class UploadResponse(BaseModel):
    document: DocumentSummary
    message: str


class LibraryStats(BaseModel):
    total_documents: int
    analysis_complete: int
    in_progress: int
    failed: int
    high_priority: int
    departments_covered: int
    total_pages: int
    open_actions: int
    ai_engine: str


class FilterOptions(BaseModel):
    departments: list[str]
    priorities: list[str]
    file_types: list[str]
    document_types: list[str]
    statuses: list[dict[str, str]]
    sort_options: list[dict[str, str]]
    date_ranges: list[dict[str, str]]
