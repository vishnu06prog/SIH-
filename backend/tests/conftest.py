"""Pytest fixtures — every test runs against an isolated temporary database."""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

# Point the app at a throwaway data directory *before* importing it, so the
# developer's real library is never touched by the test run.
_TEST_DIR = tempfile.mkdtemp(prefix="kmrl-tests-")
os.environ["KMRL_DATA_DIR"] = _TEST_DIR
os.environ["KMRL_DATABASE_URL"] = f"sqlite:///{Path(_TEST_DIR) / 'test.db'}"
os.environ["ANTHROPIC_API_KEY"] = ""

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.database import Base, SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Document  # noqa: E402
from app.services.search import build_search_blob  # noqa: E402
from app.taxonomy import STATUS_COMPLETE, STATUS_FAILED, STATUS_PROCESSING  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_database():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def make_document(session, **overrides) -> Document:
    """Insert a fully-analysed document without running the pipeline."""
    defaults = dict(
        filename="document.pdf",
        stored_filename="stored_document.pdf",
        file_ext="pdf",
        file_type="PDF",
        size_bytes=1024,
        uploaded_at=datetime.utcnow(),
        processed_at=datetime.utcnow(),
        status=STATUS_COMPLETE,
        status_detail="Analysed with rule engine",
        extracted_text="",
        page_count=3,
        word_count=200,
        char_count=1200,
        extraction_method="pdf-text",
        language="English",
        title="Document",
        document_type="Other",
        department="Operations",
        priority="Medium",
        summary="",
        analysis_engine="rule-based",
    )
    keywords = overrides.pop("keywords", [])
    actions = overrides.pop("actions", [])
    deadlines = overrides.pop("deadlines", [])
    defaults.update(overrides)

    document = Document(**defaults)
    document.set_json_field("keywords", keywords)
    document.set_json_field("actions", actions)
    document.set_json_field("deadlines", deadlines)
    document.search_blob = build_search_blob(document)
    session.add(document)
    session.commit()
    session.refresh(document)
    return document


@pytest.fixture
def library(db):
    """A small, representative KMRL library used across the search tests."""
    docs = [
        make_document(
            db,
            filename="Safety_Circular_14_Platform_Screen_Doors.pdf",
            title="Safety Circular 14 - Platform Screen Doors",
            document_type="Safety Circular",
            department="Safety",
            priority="High",
            summary=(
                "All station controllers must complete an inspection of the platform "
                "screen doors and report non-conformities to the Safety department."
            ),
            extracted_text=(
                "SAFETY CIRCULAR 14/2025. Mandatory inspection of platform screen doors "
                "at Aluva and Edappally stations before 30/09/2025."
            ),
            keywords=["safety", "inspection", "platform", "screen doors"],
            actions=[{"action": "Complete inspection", "owner": "Safety", "due_date": "30/09/2025"}],
            deadlines=[{"date": "30/09/2025", "description": "Inspection due"}],
            uploaded_at=datetime.utcnow() - timedelta(days=1),
        ),
        make_document(
            db,
            filename="Tender_Spare_Parts_Bogie.docx",
            file_type="DOCX",
            file_ext="docx",
            title="Tender for bogie spare parts",
            document_type="Tender / Contract",
            department="Procurement",
            priority="Medium",
            summary="Procurement of bogie spare parts for the rolling stock fleet.",
            extracted_text="Tender document for supply of bogie spare parts. Bid due 15/10/2025.",
            keywords=["tender", "bogie", "spare parts"],
            uploaded_at=datetime.utcnow() - timedelta(days=2),
        ),
        make_document(
            db,
            filename="ML_Unit-2.pdf",
            title="Machine Learning Unit 2",
            document_type="Training Material",
            department="IT",
            priority="Low",
            summary="Course notes covering supervised machine learning algorithms.",
            extracted_text=(
                "Unit 2 covers supervised learning, regression and decision trees for "
                "machine learning practitioners."
            ),
            keywords=["machine", "learning", "regression"],
            uploaded_at=datetime.utcnow() - timedelta(days=3),
        ),
        make_document(
            db,
            filename="Track_Inspection_Report_Q3.pdf",
            title="Track inspection report Q3",
            document_type="Technical Report",
            department="Civil / Track",
            priority="High",
            summary="Quarterly track geometry inspection findings for the Aluva corridor.",
            extracted_text="Track inspection identified two alignment defects requiring tamping.",
            keywords=["track", "inspection", "alignment"],
            uploaded_at=datetime.utcnow() - timedelta(days=4),
        ),
        make_document(
            db,
            filename="Invoice_8842_Vendor.txt",
            file_type="TXT",
            file_ext="txt",
            title="Invoice 8842",
            document_type="Invoice",
            department="Finance",
            priority="Medium",
            summary="Vendor invoice for spare parts supplied in August.",
            extracted_text="Tax invoice 8842. Amount payable Rs 4,50,000 for spare parts.",
            keywords=["invoice", "payment"],
            uploaded_at=datetime.utcnow() - timedelta(days=40),
        ),
        make_document(
            db,
            filename="Scanned_Notice.png",
            file_type="IMAGE",
            file_ext="png",
            title="Scanned notice",
            document_type="Correspondence",
            department="HR",
            priority="Low",
            summary="Scanned staff notice about refresher training.",
            extracted_text="Refresher training for station staff scheduled next month.",
            keywords=["training", "staff"],
            ocr_used=True,
            ocr_quality=54.0,
            uploaded_at=datetime.utcnow() - timedelta(days=5),
        ),
        make_document(
            db,
            filename="Pending_Upload.pdf",
            title="Pending upload",
            status=STATUS_PROCESSING,
            status_detail="Extracting text",
            processed_at=None,
            document_type="",
            department="Unassigned",
            summary="",
            extracted_text="",
        ),
        make_document(
            db,
            filename="Broken_File.pdf",
            title="Broken file",
            status=STATUS_FAILED,
            status_detail="Text extraction failed",
            processed_at=None,
            document_type="",
            department="Unassigned",
            summary="",
            extracted_text="",
        ),
    ]
    return docs
