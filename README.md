# KMRL Document Intelligence

A full-stack document intelligence platform for **Kochi Metro Rail Limited** — it ingests the
thousands of pages KMRL receives every business day (engineering drawings, maintenance job
cards, incident reports, invoices, purchase orders, regulatory directives, safety circulars,
HR policies, legal opinions, board minutes), reads them (including scanned and
English/Malayalam bilingual files), and turns them into a searchable, routed, action-oriented
library.

```
                KMRL DOCUMENT INTELLIGENCE
                         │
          ┌──────────────┴──────────────┐
          │                             │
    DOCUMENT LIBRARY              UPLOAD DOCUMENT
          │                             │
    Search / Filters                    ▼
          │                     Extraction / OCR
          │                             │
          │                             ▼
          │                       AI Analysis
          │                             │
          └──────────────┬──────────────┘
                         ▼
                 DOCUMENT INTELLIGENCE
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
          Summary     Actions    Deadlines
```

The roadmap continues with semantic search, "Ask this Document", cross-document RAG, an action
and deadline dashboard, department routing and notifications. **This repository currently
implements the foundation plus Step 1 of that plan** (see [Status](#status)).

---

## Quick start

```bash
git clone <this repo> && cd SIH-
./start.sh --seed          # installs, builds the UI, serves on :8000, loads demo documents
```

Then open **http://127.0.0.1:8000**.

<details>
<summary>Manual setup</summary>

```bash
# 1. Backend
python3 -m pip install -r backend/requirements.txt
cd backend && python3 -m uvicorn app.main:app --reload --port 8000

# 2. Frontend (new terminal)
cd frontend && npm install && npm run dev      # http://127.0.0.1:5173, proxies /api

# or build once and let FastAPI serve it:
cd frontend && npm run build                   # served at http://127.0.0.1:8000
```
</details>

### Optional system packages (OCR)

Scanned PDFs and image uploads are read with Tesseract. Without these the app still runs —
it simply reports that OCR is unavailable for a scanned file.

```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-mal poppler-utils
```

### Configuration

Copy `backend/.env.example` to `backend/.env`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | *(empty)* | When set, document analysis runs on Claude. When empty the app automatically uses the built-in KMRL rule engine, so nothing breaks without a key. |
| `KMRL_ANALYSIS_MODEL` | `claude-opus-5` | Model used for analysis. |
| `KMRL_DATA_DIR` | `backend/data` | Uploads + SQLite database location. |
| `KMRL_MAX_UPLOAD_MB` | `40` | Upload size limit. |
| `KMRL_OCR_ENABLED` | `true` | Turn the OCR fallback on/off. |
| `KMRL_OCR_LANGUAGES` | `eng+mal` | Tesseract language packs (English + Malayalam). |
| `KMRL_OCR_MAX_PAGES` | `25` | OCR budget per document. |

`GET /api/health` reports exactly which engines are live:

```json
{"capabilities": {"ai_analysis": "claude", "ocr": true, "ocr_languages": "eng+mal", ...}}
```

---

## Architecture

```
SIH-/
├── backend/
│   ├── app/
│   │   ├── main.py               FastAPI app; also serves the built SPA
│   │   ├── config.py             Settings (.env driven)
│   │   ├── database.py           SQLAlchemy engine/session (SQLite + WAL)
│   │   ├── models.py             Document, DocumentPage
│   │   ├── schemas.py            Pydantic request/response models
│   │   ├── taxonomy.py           KMRL departments, document types, priorities, statuses
│   │   ├── routers/
│   │   │   ├── documents.py      Upload, library, search, facets, intelligence
│   │   │   └── system.py         Health & capabilities
│   │   └── services/
│   │       ├── extraction.py     PDF/DOCX/TXT/image text + OCR fallback + language detection
│   │       ├── analysis.py       Claude analysis (JSON schema) + rule-based fallback
│   │       ├── search.py         Ranking, filters, facets, pagination — all in SQL
│   │       └── processing.py     Upload storage and the background pipeline
│   ├── scripts/seed_demo.py      Generates and loads a realistic KMRL demo library
│   └── tests/                    61 pytest tests
└── frontend/
    ├── src/pages/                LibraryPage, UploadPage, IntelligencePage
    ├── src/components/           Search bar, filter rail, document card, pagination, toasts…
    ├── src/lib/                  Typed API client, formatting helpers
    └── src/styles/               Design tokens + hand-written CSS (light & dark)
```

**Stack:** FastAPI · SQLAlchemy 2 · SQLite · pypdf / python-docx / Tesseract · Anthropic Claude ·
React 18 · TypeScript · Vite · hand-written CSS design system (no UI framework).

### Processing pipeline

```
upload → stored + checksummed → text extraction (native first)
       → OCR fallback for scanned pages/images (eng+mal, per-page confidence)
       → language detection (English / Malayalam / Bilingual)
       → AI analysis: title, type, department, priority, summary, key points,
         keywords, actions, deadlines, risks, compliance flags
       → search blob built → status: AI Analysis Complete
```

Processing runs in the background, so the upload response returns immediately and the UI
follows the status live (`Processing → OCR Processing → AI Analysis → AI Analysis Complete`).

### Analysis engines

| | Claude (`ANTHROPIC_API_KEY` set) | Rule engine (fallback) |
| --- | --- | --- |
| Classification | Structured JSON output constrained to the KMRL taxonomy | Weighted, whole-word KMRL vocabulary scoring |
| Summary | 3–5 sentences written for a KMRL manager | Extractive: line-reflow, letterhead filtering, keyword-density ranking |
| Grounding | System prompt forbids inventing facts, dates or departments | Only quotes sentences from the document |

Both engines emit the same shape, so the UI and API never branch on which one ran — the
document records `analysis_engine` and the model used, and the Intelligence page shows it.

---

## API

Base path `/api`. Interactive docs at `/docs`.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Status + live engine capabilities |
| `GET` | `/documents` | Document library (same handler as search) |
| `GET` | `/documents/search` | Keyword search + filters + sort + pagination |
| `GET` | `/documents/facets` | Result counts per filter value |
| `GET` | `/documents/filters` | Filter vocabulary (departments, priorities, types, statuses, sorts, date ranges) |
| `GET` | `/documents/stats` | Library KPIs |
| `POST` | `/documents/upload` | Upload a document (multipart) |
| `GET` | `/documents/{id}` | Full intelligence payload |
| `GET` | `/documents/{id}/text` | Extracted text (offset/limit paginated) |
| `GET` | `/documents/{id}/file` | Download the original file |
| `POST` | `/documents/{id}/reprocess` | Re-run extraction + analysis |
| `DELETE` | `/documents/{id}` | Remove a document |

### Search

```
GET /api/documents/search?q=inspection&department=Safety&priority=High&page=1&page_size=20
```

Query parameters: `q`, `department`, `priority`, `file_type`, `status`, `document_type`
(all repeatable **and** comma-separated), `date_range` (`any|today|week|recent|older`),
`date_from`, `date_to`, `sort` (`relevance|newest|oldest|name|priority`), `page`, `page_size`.

* Every term must appear in the document (AND across terms); `"quoted phrases"` are supported.
* Ranking weights, highest first: **filename → document type → title → department → keywords →
  AI summary → extracted content**. This is deliberate keyword ranking, *not* semantic search —
  embeddings arrive in Step 2.
* Matching, ranking, filtering, faceting and pagination all execute in SQL. Excerpts are cut
  out with `substr`/`instr` in the database, so the API never loads whole documents to build a
  result page, and never returns raw document text in the library payload.

Response:

```json
{
  "items": [{ "id": 1, "filename": "...", "department": "Safety", "priority": "High",
              "summary": "...", "snippet": "…mandatory inspection of platform…", "score": 90 }],
  "total": 1, "page": 1, "page_size": 20, "total_pages": 1,
  "has_next": false, "has_prev": false,
  "query": "inspection", "applied_filters": {"department": ["Safety"], "priority": ["High"]},
  "sort": "relevance", "took_ms": 15, "message": null, "suggestion": null
}
```

When nothing matches, `message` is `"No documents found"` and `suggestion` explains what to try
next — the UI renders that, never an error.

---

## Demo data

```bash
cd backend && python3 scripts/seed_demo.py          # writes samples/ and uploads them
python3 scripts/seed_demo.py --files-only           # just generate the files
```

Fifteen realistic KMRL documents across every department: a safety circular, a Maximo job
card, a vendor invoice, a tender notice, a CMRS regulatory directive, an HR training circular,
an incident report, a track inspection report, a CBTC upgrade note, an OHE shutdown notice, a
**bilingual English/Malayalam** station notice, a **scanned image** notice (exercises OCR), an
IT/UNS architecture note, a legal opinion and board minutes.

## Tests

```bash
cd backend && python3 -m pytest
```

61 tests covering search by filename / extracted content / AI summary / keywords, every filter,
combined filters, ranking order, empty results, pagination, snippets, the full
upload → extraction → OCR → analysis → intelligence path, and the extraction and analysis
services (including a real Tesseract OCR round-trip when Tesseract is installed).

---

## Status

**Done — foundation + Step 1**

- [x] Upload, storage, background processing pipeline
- [x] PDF / DOCX / TXT / image extraction, OCR fallback (English + Malayalam), language detection
- [x] AI analysis — Claude with a rule-based fallback; department, priority, type, summary,
      key points, keywords, actions, deadlines, risks, compliance flags
- [x] Document Intelligence page (summary, key points, actions, deadlines, risks, compliance,
      keywords, provenance, source text)
- [x] **Document library + keyword search + filters + ranking + pagination** (Step 1)

**Next**

- [ ] Step 2 — semantic search with embeddings
- [ ] Step 3 — Ask this Document (grounded Q&A with page citations)
- [ ] Step 4 — cross-document RAG
- [ ] Step 5 — action & deadline dashboard
- [ ] Step 6 — department routing
- [ ] Step 7 — notifications / workflow
