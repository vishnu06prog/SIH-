"""Unit tests for the extraction and analysis services."""

from __future__ import annotations

import pytest

from app.services import extraction, processing
from app.services.analysis import analyse_with_rules

from .factories import make_docx_bytes, make_pdf_bytes


def test_classify_file_type():
    assert extraction.classify_file_type("a.PDF") == "PDF"
    assert extraction.classify_file_type("a.docx") == "DOCX"
    assert extraction.classify_file_type("a.png") == "IMAGE"
    assert extraction.classify_file_type("a.txt") == "TXT"


def test_is_supported():
    assert extraction.is_supported("notes.md")
    assert not extraction.is_supported("archive.zip")


def test_detect_language():
    assert extraction.detect_language("Platform screen door inspection") == "English"
    assert extraction.detect_language("സുരക്ഷാ പരിശോധന നടത്തണം") == "Malayalam"
    mixed = "Safety circular / സുരക്ഷാ സർക്കുലർ - inspection പരിശോധന നടത്തണം"
    assert extraction.detect_language(mixed) == "Bilingual (EN/ML)"


def test_extract_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    path.write_bytes(make_pdf_bytes(["Track alignment defect found at chainage 12."]))
    result = extraction.extract_document(path)
    assert result.page_count == 1
    assert "alignment defect" in result.text
    assert result.method.startswith("pdf-text")
    assert len(result.pages) == 1


def test_extract_docx(tmp_path):
    path = tmp_path / "sample.docx"
    path.write_bytes(make_docx_bytes(["Vendor invoice for spare parts.", "Amount payable."]))
    result = extraction.extract_document(path)
    assert "Vendor invoice" in result.text
    assert result.method == "docx"


def test_extract_txt(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("Board meeting minutes for March.", encoding="utf-8")
    result = extraction.extract_document(path)
    assert result.text == "Board meeting minutes for March."
    assert result.page_count == 1


def test_extract_unsupported(tmp_path):
    path = tmp_path / "sample.zip"
    path.write_bytes(b"PK")
    with pytest.raises(ValueError):
        extraction.extract_document(path)


@pytest.mark.skipif(not extraction.ocr_available(), reason="tesseract not installed")
def test_ocr_reads_an_image(tmp_path):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (900, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 80), "SAFETY INSPECTION NOTICE", fill="black")
    path = tmp_path / "notice.png"
    image.save(path)

    result = extraction.extract_document(path)
    assert result.ocr_used is True
    assert "SAFETY" in result.text.upper()


# ------------------------------------------------------------------- analysis
def test_rule_analysis_detects_safety_department():
    text = (
        "Safety circular: all station controllers shall complete the platform screen "
        "door inspection immediately. Report any hazard to the Commissioner of Metro "
        "Rail Safety by 30/09/2025. Non-compliance may attract a penalty."
    )
    result = analyse_with_rules(text, "Safety_Circular.pdf")
    assert result.department == "Safety"
    assert result.priority == "High"
    assert result.document_type == "Safety Circular"
    assert result.actions
    assert result.deadlines[0]["date"] == "30/09/2025"
    assert result.risks
    assert "Cmrs" in result.compliance_flags or "Compliance" in result.compliance_flags


def test_rule_analysis_detects_procurement():
    text = (
        "Tender document for the supply of bogie spare parts. Vendors must submit a "
        "quotation with earnest money deposit. Purchase order will follow the bid award."
    )
    result = analyse_with_rules(text, "tender.docx")
    assert result.department == "Procurement"
    assert result.document_type in {"Tender / Contract", "Purchase Order"}


def test_rule_analysis_handles_empty_text():
    result = analyse_with_rules("", "ML_Unit-2.pdf")
    assert result.summary
    assert result.title == "Ml Unit 2"
    assert result.confidence == 0.15


def test_rule_analysis_never_invents_a_department():
    result = analyse_with_rules("Lorem ipsum dolor sit amet consectetur.", "notes.txt")
    assert result.department == "Unassigned"


def test_low_quality_ocr_detection(db):
    from .conftest import make_document

    good = make_document(db, filename="good.png", ocr_used=True, ocr_quality=91.0)
    poor = make_document(db, filename="poor.png", ocr_used=True, ocr_quality=41.0)
    native = make_document(db, filename="native.pdf", ocr_used=False, ocr_quality=None)

    assert processing.is_low_quality_ocr(good) is False
    assert processing.is_low_quality_ocr(poor) is True
    assert processing.is_low_quality_ocr(native) is False


def test_safe_filename_strips_path_traversal():
    assert processing.safe_filename("../../etc/passwd") == "passwd"
    assert processing.safe_filename("report v2 (final).pdf") == "report v2 _final_.pdf"
