"""Document library search, filtering, ranking and pagination."""

from __future__ import annotations

from .conftest import make_document

SEARCH = "/api/documents/search"


def _filenames(payload) -> list[str]:
    return [item["filename"] for item in payload["items"]]


def test_search_by_filename(client, library):
    response = client.get(SEARCH, params={"q": "ML_Unit-2"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["filename"] == "ML_Unit-2.pdf"


def test_search_by_extracted_content(client, library):
    """A word that exists only in the document body must still be found."""
    response = client.get(SEARCH, params={"q": "tamping"})
    payload = response.json()
    assert _filenames(payload) == ["Track_Inspection_Report_Q3.pdf"]


def test_search_by_ai_summary(client, library):
    response = client.get(SEARCH, params={"q": "quarterly"})
    payload = response.json()
    assert _filenames(payload) == ["Track_Inspection_Report_Q3.pdf"]


def test_search_by_keyword_list(client, library):
    response = client.get(SEARCH, params={"q": "regression"})
    assert _filenames(response.json()) == ["ML_Unit-2.pdf"]


def test_search_by_department_term(client, library):
    response = client.get(SEARCH, params={"q": "procurement"})
    assert "Tender_Spare_Parts_Bogie.docx" in _filenames(response.json())


def test_search_is_case_insensitive(client, library):
    lower = client.get(SEARCH, params={"q": "safety"}).json()
    upper = client.get(SEARCH, params={"q": "SAFETY"}).json()
    assert lower["total"] == upper["total"] > 0


def test_multi_term_search_requires_all_terms(client, library):
    """"machine learning" must not match a document holding only one word."""
    payload = client.get(SEARCH, params={"q": "machine learning"}).json()
    assert _filenames(payload) == ["ML_Unit-2.pdf"]


def test_ranking_prefers_filename_over_body(client, db):
    make_document(
        db,
        filename="Body_Mention.pdf",
        title="Body mention",
        summary="",
        extracted_text="This report references the inspection process in passing.",
        department="Operations",
    )
    make_document(
        db,
        filename="Inspection_Checklist.pdf",
        title="Inspection checklist",
        summary="",
        extracted_text="Checklist contents.",
        department="Operations",
    )
    payload = client.get(SEARCH, params={"q": "inspection"}).json()
    assert payload["items"][0]["filename"] == "Inspection_Checklist.pdf"
    assert payload["items"][0]["score"] > payload["items"][1]["score"]


def test_search_returns_snippet(client, library):
    payload = client.get(SEARCH, params={"q": "tamping"}).json()
    assert "tamping" in payload["items"][0]["snippet"].lower()


def test_search_never_returns_full_text(client, library):
    payload = client.get(SEARCH, params={"q": "safety"}).json()
    assert "extracted_text" not in payload["items"][0]


# ------------------------------------------------------------------ filters
def test_department_filter(client, library):
    payload = client.get(SEARCH, params={"department": "Safety"}).json()
    assert payload["total"] == 1
    assert payload["items"][0]["department"] == "Safety"


def test_priority_filter(client, library):
    payload = client.get(SEARCH, params={"priority": "High"}).json()
    assert payload["total"] == 2
    assert {item["priority"] for item in payload["items"]} == {"High"}


def test_file_type_filter(client, library):
    payload = client.get(SEARCH, params={"file_type": "DOCX"}).json()
    assert _filenames(payload) == ["Tender_Spare_Parts_Bogie.docx"]


def test_status_filter_accepts_label(client, library):
    payload = client.get(SEARCH, params={"status": "AI Analysis Complete"}).json()
    assert payload["total"] == 6
    payload = client.get(SEARCH, params={"status": "failed"}).json()
    assert _filenames(payload) == ["Broken_File.pdf"]


def test_document_type_filter(client, library):
    payload = client.get(SEARCH, params={"document_type": "Invoice"}).json()
    assert _filenames(payload) == ["Invoice_8842_Vendor.txt"]


def test_multi_value_filter(client, library):
    payload = client.get(SEARCH, params={"department": ["Safety", "Finance"]}).json()
    assert payload["total"] == 2


def test_comma_separated_filter(client, library):
    payload = client.get(SEARCH, params={"department": "Safety,Finance"}).json()
    assert payload["total"] == 2


def test_date_range_filters(client, library):
    recent = client.get(SEARCH, params={"date_range": "recent"}).json()
    older = client.get(SEARCH, params={"date_range": "older"}).json()
    assert _filenames(older) == ["Invoice_8842_Vendor.txt"]
    assert "Invoice_8842_Vendor.txt" not in _filenames(recent)


def test_combined_search_and_filters(client, library):
    """Department = Safety + Priority = High + q=inspection -> one document."""
    payload = client.get(
        SEARCH,
        params={"q": "inspection", "department": "Safety", "priority": "High"},
    ).json()
    assert payload["total"] == 1
    assert payload["items"][0]["filename"] == "Safety_Circular_14_Platform_Screen_Doors.pdf"
    assert payload["applied_filters"]["department"] == ["Safety"]


def test_combined_filters_can_exclude_everything(client, library):
    payload = client.get(
        SEARCH, params={"q": "inspection", "department": "Finance"}
    ).json()
    assert payload["total"] == 0
    assert payload["message"] == "No documents found"


# --------------------------------------------------------------- empty states
def test_nonexistent_search_term(client, library):
    payload = client.get(SEARCH, params={"q": "zzzznotarealterm"}).json()
    assert payload["total"] == 0
    assert payload["items"] == []
    assert payload["message"] == "No documents found"
    assert "different keyword" in payload["suggestion"]


def test_empty_library_returns_helpful_state(client):
    payload = client.get(SEARCH).json()
    assert payload["total"] == 0
    assert payload["message"] == "No documents found"
    assert "Upload a document" in payload["suggestion"]


def test_blank_query_lists_everything(client, library):
    payload = client.get(SEARCH, params={"q": "   "}).json()
    assert payload["total"] == len(library)
    assert payload["sort"] == "newest"


# ---------------------------------------------------------------- pagination
def test_pagination(client, library):
    first = client.get(SEARCH, params={"page_size": 3, "page": 1, "sort": "newest"}).json()
    assert len(first["items"]) == 3
    assert first["total"] == 8
    assert first["total_pages"] == 3
    assert first["has_next"] is True
    assert first["has_prev"] is False

    second = client.get(SEARCH, params={"page_size": 3, "page": 2, "sort": "newest"}).json()
    assert second["has_prev"] is True
    assert set(_filenames(first)).isdisjoint(_filenames(second))

    last = client.get(SEARCH, params={"page_size": 3, "page": 3, "sort": "newest"}).json()
    assert last["has_next"] is False
    assert len(last["items"]) == 2


def test_page_beyond_the_end_is_empty_not_an_error(client, library):
    payload = client.get(SEARCH, params={"page": 99}).json()
    assert payload["items"] == []
    assert payload["total"] == 8


def test_page_size_is_capped(client, library):
    response = client.get(SEARCH, params={"page_size": 5000})
    assert response.status_code == 422


# --------------------------------------------------------------------- sorting
def test_sort_by_name(client, library):
    payload = client.get(SEARCH, params={"sort": "name"}).json()
    assert _filenames(payload) == sorted(_filenames(payload), key=str.lower)


def test_sort_by_priority(client, library):
    payload = client.get(SEARCH, params={"sort": "priority"}).json()
    assert payload["items"][0]["priority"] == "High"


def test_sort_oldest_first(client, library):
    payload = client.get(SEARCH, params={"sort": "oldest"}).json()
    assert payload["items"][0]["filename"] == "Invoice_8842_Vendor.txt"


# ---------------------------------------------------------------------- facets
def test_facets_reflect_the_query(client, library):
    payload = client.get("/api/documents/facets", params={"q": "inspection"}).json()
    departments = {row["value"]: row["count"] for row in payload["departments"]}
    assert departments.get("Safety") == 1
    assert departments.get("Civil / Track") == 1
    assert "Finance" not in departments


def test_facets_ignore_their_own_filter(client, library):
    payload = client.get(
        "/api/documents/facets", params={"department": "Safety"}
    ).json()
    departments = {row["value"] for row in payload["departments"]}
    assert {"Safety", "Finance", "IT"} <= departments


def test_filter_options_endpoint(client):
    payload = client.get("/api/documents/filters").json()
    assert "Rolling Stock" in payload["departments"]
    assert payload["priorities"] == ["High", "Medium", "Low"]
    assert {"value": "analysis_complete", "label": "AI Analysis Complete"} in payload["statuses"]


def test_library_stats(client, library):
    payload = client.get("/api/documents/stats").json()
    assert payload["total_documents"] == 8
    assert payload["analysis_complete"] == 6
    assert payload["failed"] == 1
    assert payload["in_progress"] == 1
    assert payload["high_priority"] == 2


def test_snippet_starts_on_a_word_boundary(client, db):
    make_document(
        db,
        filename="Long_Report.pdf",
        summary="",
        extracted_text=(
            "The quarterly engineering review opens with a long preamble about the "
            "corridor, the depots and the stabling lines before it finally records "
            "that tamping was carried out at chainage twelve during the night block."
        ),
    )
    payload = client.get(SEARCH, params={"q": "tamping"}).json()
    snippet = payload["items"][0]["snippet"]
    assert "tamping" in snippet
    assert snippet.startswith("…")
    assert not snippet.startswith("…he ")


def test_snippet_from_the_start_is_not_clipped(client, db):
    make_document(
        db,
        filename="Short_Note.pdf",
        summary="",
        extracted_text="Tamping is scheduled for the Aluva corridor this weekend.",
    )
    payload = client.get(SEARCH, params={"q": "tamping"}).json()
    assert payload["items"][0]["snippet"].startswith("Tamping is scheduled")
