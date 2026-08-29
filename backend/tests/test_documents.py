"""Upload → extraction → AI analysis → intelligence, end to end."""

from __future__ import annotations

from app.taxonomy import STATUS_COMPLETE

from .factories import make_docx_bytes, make_pdf_bytes

UPLOAD = "/api/documents/upload"

CIRCULAR_LINES = [
    "KOCHI METRO RAIL LIMITED - SAFETY CIRCULAR 21/2025",
    "Subject: Mandatory inspection of platform screen doors at Aluva station.",
    "All station controllers shall complete the inspection by 30/09/2025.",
    "The Safety department is required to submit the compliance report to the",
    "Commissioner of Metro Rail Safety within 15 days. Failure to comply may",
    "attract a penalty and an audit non-conformity.",
]


def _upload(client, name: str, data: bytes, content_type: str):
    return client.post(
        UPLOAD, files={"file": (name, data, content_type)}, data={"uploaded_by": "Tester"}
    )


def test_upload_pdf_runs_the_full_pipeline(client):
    response = _upload(client, "Safety_Circular_21.pdf", make_pdf_bytes(CIRCULAR_LINES), "application/pdf")
    assert response.status_code == 201
    document_id = response.json()["document"]["id"]

    detail = client.get(f"/api/documents/{document_id}").json()
    assert detail["status"] == STATUS_COMPLETE
    assert detail["status_label"] == "AI Analysis Complete"
    assert detail["file_type"] == "PDF"
    assert detail["page_count"] == 1
    assert detail["word_count"] > 20
    assert detail["department"] == "Safety"
    assert detail["priority"] == "High"
    assert detail["summary"]
    assert detail["keywords"]
    assert detail["actions"]
    assert detail["deadlines"][0]["date"] == "30/09/2025"
    assert detail["analysis_engine"] == "rule-based"  # no API key in the test env


def test_uploaded_document_appears_in_the_library_and_is_searchable(client):
    _upload(client, "Safety_Circular_21.pdf", make_pdf_bytes(CIRCULAR_LINES), "application/pdf")

    library = client.get("/api/documents").json()
    assert library["total"] == 1

    by_content = client.get("/api/documents/search", params={"q": "platform screen doors"}).json()
    assert by_content["total"] == 1

    by_filter = client.get(
        "/api/documents/search", params={"q": "inspection", "department": "Safety", "priority": "High"}
    ).json()
    assert by_filter["total"] == 1


def test_upload_docx(client):
    data = make_docx_bytes(
        [
            "Tender document for supply of bogie spare parts.",
            "Vendors must submit bids to the procurement department by 15/10/2025.",
        ]
    )
    response = _upload(
        client,
        "Tender_Bogie_Spares.docx",
        data,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    assert response.status_code == 201
    detail = client.get(f"/api/documents/{response.json()['document']['id']}").json()
    assert detail["file_type"] == "DOCX"
    assert detail["status"] == STATUS_COMPLETE
    assert detail["department"] in {"Procurement", "Rolling Stock"}


def test_upload_txt(client):
    response = _upload(
        client,
        "hr_notice.txt",
        b"Refresher training for all station staff. Payroll and leave policy updates apply.",
        "text/plain",
    )
    assert response.status_code == 201
    detail = client.get(f"/api/documents/{response.json()['document']['id']}").json()
    assert detail["file_type"] == "TXT"
    assert detail["department"] == "HR"


def test_upload_rejects_unsupported_type(client):
    response = _upload(client, "archive.zip", b"PK\x03\x04", "application/zip")
    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_rejects_empty_file(client):
    response = _upload(client, "empty.txt", b"", "text/plain")
    assert response.status_code == 400


def test_document_detail_404(client):
    response = client.get("/api/documents/999999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_document_text_endpoint_paginates(client):
    _upload(client, "notes.txt", b"KMRL " * 400, "text/plain")
    document_id = client.get("/api/documents").json()["items"][0]["id"]

    payload = client.get(f"/api/documents/{document_id}/text", params={"limit": 50}).json()
    assert len(payload["text"]) == 50
    assert payload["has_more"] is True
    assert payload["total_chars"] > 50


def test_download_returns_the_original_file(client):
    data = make_pdf_bytes(["Board minutes for the March meeting."])
    _upload(client, "Board_Minutes.pdf", data, "application/pdf")
    document_id = client.get("/api/documents").json()["items"][0]["id"]

    response = client.get(f"/api/documents/{document_id}/file")
    assert response.status_code == 200
    assert response.content == data


def test_reprocess_reruns_the_pipeline(client):
    _upload(client, "circular.pdf", make_pdf_bytes(CIRCULAR_LINES), "application/pdf")
    document_id = client.get("/api/documents").json()["items"][0]["id"]

    response = client.post(f"/api/documents/{document_id}/reprocess")
    assert response.status_code == 200

    detail = client.get(f"/api/documents/{document_id}").json()
    assert detail["status"] == STATUS_COMPLETE
    assert detail["department"] == "Safety"


def test_delete_document(client):
    _upload(client, "circular.pdf", make_pdf_bytes(CIRCULAR_LINES), "application/pdf")
    document_id = client.get("/api/documents").json()["items"][0]["id"]

    assert client.delete(f"/api/documents/{document_id}").status_code == 204
    assert client.get(f"/api/documents/{document_id}").status_code == 404
    assert client.get("/api/documents").json()["total"] == 0


def test_health_reports_capabilities(client):
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["capabilities"]["ai_analysis"] in {"claude", "rule-based"}
    assert "ocr" in payload["capabilities"]
