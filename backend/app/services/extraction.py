"""Multilingual text extraction for PDF / DOCX / TXT / image uploads.

The pipeline picks the cheapest route that actually works, per page:

    text PDF     -> native text layer (pdfplumber, then pypdf)
    scanned PDF  -> render the page (PyMuPDF, then poppler) -> Tesseract
    mixed PDF    -> native text where present, OCR only for the image pages
    image        -> Tesseract directly
    DOCX         -> paragraphs + tables, OCR for embedded images if needed
    TXT          -> read as-is

OCR runs with ``eng+mal`` so Malayalam and bilingual scans are read properly,
and every page carries its own language, method and confidence so the rest of
the system can cite it.
"""

from __future__ import annotations

import io
import logging
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ..config import settings

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".json"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
DOC_EXTENSIONS = {".docx", ".doc"}
PDF_EXTENSIONS = {".pdf"}

SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | DOC_EXTENSIONS | IMAGE_EXTENSIONS | TEXT_EXTENSIONS

# A page with fewer usable characters than this is treated as scanned.
MIN_NATIVE_CHARS_PER_PAGE = 40

# Below this mean Tesseract confidence we flag the document for human review.
LOW_OCR_CONFIDENCE = 70.0

MALAYALAM_RANGE = re.compile(r"[ഀ-ൿ]")
LATIN_RANGE = re.compile(r"[A-Za-z]")

ENGLISH = "English"
MALAYALAM = "Malayalam"
BILINGUAL = "Bilingual (EN/ML)"
UNKNOWN = "Unknown"

METHOD_PDF_TEXT = "pdf-text"
METHOD_OCR = "ocr"
METHOD_DOCX = "docx"
METHOD_DOCX_OCR = "docx-image-ocr"
METHOD_PLAIN = "plain-text"


@dataclass
class PageResult:
    page_number: int
    text: str = ""
    language: str = UNKNOWN
    extraction_method: str = METHOD_PDF_TEXT
    ocr_used: bool = False
    ocr_confidence: float | None = None
    tables: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def table_count(self) -> int:
        return len(self.tables)


@dataclass
class ExtractionResult:
    text: str = ""
    pages: list[PageResult] = field(default_factory=list)
    page_count: int | None = None
    method: str = ""
    ocr_used: bool = False
    ocr_quality: float | None = None
    ocr_page_count: int = 0
    ocr_languages: str = ""
    language: str = UNKNOWN
    languages_detected: list[str] = field(default_factory=list)
    image_count: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def table_count(self) -> int:
        return sum(page.table_count for page in self.pages)

    @property
    def needs_review(self) -> bool:
        """Low-confidence OCR must never feed silent AI conclusions."""
        return bool(
            self.ocr_used
            and self.ocr_quality is not None
            and self.ocr_quality < LOW_OCR_CONFIDENCE
        )


# --------------------------------------------------------------- file typing
def classify_file_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in PDF_EXTENSIONS:
        return "PDF"
    if ext in DOC_EXTENSIONS:
        return "DOCX"
    if ext in IMAGE_EXTENSIONS:
        return "IMAGE"
    return "TXT"


def is_supported(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------- OCR probes
@lru_cache(maxsize=1)
def tesseract_languages() -> tuple[str, ...]:
    """Language packs Tesseract can actually use on this machine."""
    if shutil.which("tesseract") is None:
        return ()
    try:
        output = subprocess.run(
            ["tesseract", "--list-langs"], capture_output=True, text=True, timeout=15
        )
    except Exception:  # pragma: no cover - environment probe
        return ()
    lines = [line.strip() for line in output.stdout.splitlines()[1:] if line.strip()]
    return tuple(sorted(lines))


def ocr_available() -> bool:
    if not settings.kmrl_ocr_enabled or shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract  # noqa: F401
    except ImportError:  # pragma: no cover - depends on environment
        return False
    return True


def malayalam_ocr_available() -> bool:
    return ocr_available() and "mal" in tesseract_languages()


def effective_ocr_languages() -> str:
    """``eng+mal`` when both packs exist, narrowed to what is installed."""
    installed = set(tesseract_languages())
    requested = [
        lang.strip()
        for lang in settings.kmrl_ocr_languages.replace(",", "+").split("+")
        if lang.strip()
    ]
    usable = [lang for lang in requested if lang in installed]
    if not usable:
        usable = ["eng"] if "eng" in installed else list(installed)[:1]
    return "+".join(usable)


def ocr_engine_version() -> str | None:  # pragma: no cover - environment probe
    if shutil.which("tesseract") is None:
        return None
    try:
        output = subprocess.run(
            ["tesseract", "--version"], capture_output=True, text=True, timeout=10
        )
        return output.stdout.splitlines()[0].strip() if output.stdout else None
    except Exception:
        return None


def poppler_available() -> bool:
    return shutil.which("pdftoppm") is not None


def pymupdf_available() -> bool:
    try:
        import pymupdf  # noqa: F401
    except ImportError:
        try:
            import fitz  # noqa: F401
        except ImportError:
            return False
    return True


# ------------------------------------------------------- language detection
def detect_language(text: str) -> str:
    """Script-ratio language detection — reliable for Malayalam vs Latin."""
    sample = (text or "")[:20_000]
    malayalam = len(MALAYALAM_RANGE.findall(sample))
    latin = len(LATIN_RANGE.findall(sample))
    if malayalam == 0 and latin == 0:
        return UNKNOWN
    if malayalam == 0:
        return ENGLISH
    if latin == 0:
        return MALAYALAM
    ratio = malayalam / (malayalam + latin)
    if ratio > 0.75:
        return MALAYALAM
    if ratio > 0.08:
        return BILINGUAL
    return ENGLISH


def has_malayalam(text: str) -> bool:
    return bool(MALAYALAM_RANGE.search(text or ""))


def summarise_languages(pages: list[PageResult]) -> list[str]:
    """Distinct page languages, most common first — for the bilingual badge."""
    order = [ENGLISH, MALAYALAM, BILINGUAL, UNKNOWN]
    present = {page.language for page in pages if page.language and page.language != UNKNOWN}
    return [language for language in order if language in present]


# Tesseract emits chillu letters in their decomposed form (consonant + virama
# + ZWJ). Recomposing them is what makes OCR output match the glossary keys and
# compare equal to natively-typed Malayalam.
CHILLU_RECOMPOSE = {
    "\u0d23\u0d4d\u200d": "\u0d7a",  # ṇ
    "\u0d28\u0d4d\u200d": "\u0d7b",  # n
    "\u0d30\u0d4d\u200d": "\u0d7c",  # r
    "\u0d32\u0d4d\u200d": "\u0d7d",  # l
    "\u0d33\u0d4d\u200d": "\u0d7e",  # ḷ
    "\u0d15\u0d4d\u200d": "\u0d7f",  # k
}


def normalise_malayalam(text: str) -> str:
    """Recompose chillus and drop stray joiners so text compares consistently."""
    if not MALAYALAM_RANGE.search(text or ""):
        return text
    for decomposed, composed in CHILLU_RECOMPOSE.items():
        text = text.replace(decomposed, composed)
    # A ZWNJ after a virama is a rendering hint, not content.
    return text.replace("\u0d4d\u200c", "\u0d4d")


def clean_text(text: str) -> str:
    text = (text or "").replace("\x00", " ")
    text = normalise_malayalam(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------- OCR
def ocr_image(image, languages: str | None = None) -> tuple[str, float | None]:
    """Run Tesseract on a PIL image -> (text, mean confidence 0-100)."""
    import pytesseract

    langs = languages or effective_ocr_languages()
    try:
        data = pytesseract.image_to_data(
            image, lang=langs, output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractError:
        # A requested pack is missing — retry with English only rather than fail.
        logger.warning("Tesseract rejected languages %r; retrying with eng.", langs)
        data = pytesseract.image_to_data(
            image, lang="eng", output_type=pytesseract.Output.DICT
        )

    words: list[str] = []
    confidences: list[float] = []
    for word, conf in zip(data.get("text", []), data.get("conf", [])):
        token = (word or "").strip()
        if not token:
            continue
        words.append(token)
        try:
            value = float(conf)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            confidences.append(value)

    text = " ".join(words)
    confidence = round(sum(confidences) / len(confidences), 1) if confidences else None
    return text, confidence


def render_pdf_page(path: Path, page_number: int):
    """Rasterise one PDF page to a PIL image.

    PyMuPDF is tried first because it ships its own renderer (no poppler
    system package needed); poppler is the fallback where it is installed.
    """
    dpi = settings.kmrl_ocr_dpi
    try:
        import pymupdf
    except ImportError:  # pragma: no cover - older wheels expose `fitz`
        try:
            import fitz as pymupdf  # type: ignore
        except ImportError:
            pymupdf = None  # type: ignore[assignment]

    if pymupdf is not None:
        from PIL import Image

        with pymupdf.open(str(path)) as doc:
            if page_number > doc.page_count:
                return None
            pixmap = doc.load_page(page_number - 1).get_pixmap(dpi=dpi)
            return Image.open(io.BytesIO(pixmap.tobytes("png")))

    if poppler_available():  # pragma: no cover - depends on system packages
        from pdf2image import convert_from_path

        images = convert_from_path(
            str(path), dpi=dpi, first_page=page_number, last_page=page_number
        )
        return images[0] if images else None
    return None


# --------------------------------------------------------------- PDF tables
def _format_table(rows: list[list[str | None]]) -> str:
    """Render an extracted table as pipe-delimited text the AI can read."""
    lines: list[str] = []
    for row in rows:
        cells = [(cell or "").strip().replace("\n", " ") for cell in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _pdf_page_payload(page) -> tuple[str, list[str]]:
    """(text, tables) for a pdfplumber page."""
    try:
        text = clean_text(page.extract_text() or "")
    except Exception:  # pragma: no cover - malformed page
        text = ""
    tables: list[str] = []
    try:
        for raw in page.extract_tables() or []:
            rendered = _format_table(raw)
            if rendered:
                tables.append(rendered)
    except Exception:  # pragma: no cover - pdfplumber table heuristics
        pass
    return text, tables


# --------------------------------------------------------------- extractors
def _extract_pdf(path: Path) -> ExtractionResult:
    result = ExtractionResult(method=METHOD_PDF_TEXT)
    native_pages: list[tuple[str, list[str]]] = []

    try:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                native_pages.append(_pdf_page_payload(page))
                result.image_count += len(getattr(page, "images", []) or [])
    except Exception as exc:  # noqa: BLE001 - fall back to pypdf
        logger.info("pdfplumber could not read %s (%s); using pypdf.", path.name, exc)
        native_pages = _pypdf_pages(path)

    if not native_pages:
        native_pages = _pypdf_pages(path)

    result.page_count = len(native_pages)
    use_ocr = ocr_available()
    languages = effective_ocr_languages() if use_ocr else ""
    result.ocr_languages = languages
    ocr_budget = settings.kmrl_ocr_max_pages
    ocr_confidences: list[float] = []

    for index, (native, tables) in enumerate(native_pages, start=1):
        page = PageResult(
            page_number=index,
            text=native,
            extraction_method=METHOD_PDF_TEXT,
            tables=tables,
        )

        # Scanned page: no usable text layer, so rasterise it and read it.
        if len(native) < MIN_NATIVE_CHARS_PER_PAGE and use_ocr and ocr_budget > 0:
            try:
                image = render_pdf_page(path, index)
                if image is not None:
                    ocr_text, confidence = clean_text_and_confidence(
                        *ocr_image(image, languages)
                    )
                    if len(ocr_text) > len(native):
                        page.text = ocr_text
                        page.ocr_used = True
                        page.extraction_method = METHOD_OCR
                        page.ocr_confidence = confidence
                        if confidence is not None:
                            ocr_confidences.append(confidence)
                    ocr_budget -= 1
            except Exception as exc:  # noqa: BLE001 - depends on renderer
                logger.warning("OCR failed for %s page %s: %s", path.name, index, exc)
                result.warnings.append(f"OCR failed on page {index}.")

        if page.tables:
            page.text = f"{page.text}\n\n{chr(10).join(page.tables)}".strip()
        page.language = detect_language(page.text)
        result.pages.append(page)

    scanned = [page for page in result.pages if page.ocr_used]
    result.ocr_page_count = len(scanned)
    if scanned:
        result.ocr_used = True
        result.method = METHOD_OCR if len(scanned) == len(result.pages) else "pdf-text+ocr"
        if ocr_confidences:
            result.ocr_quality = round(sum(ocr_confidences) / len(ocr_confidences), 1)

    result.text = clean_text("\n\n".join(page.text for page in result.pages if page.text))

    if not result.text:
        if use_ocr:
            result.warnings.append("No text could be extracted from this PDF, even with OCR.")
        else:
            result.warnings.append(
                "This PDF has no text layer and OCR is unavailable on this server. "
                "Install Tesseract (with the 'mal' language pack) to read scanned documents."
            )
    return result


def clean_text_and_confidence(text: str, confidence: float | None) -> tuple[str, float | None]:
    return clean_text(text), confidence


def _pypdf_pages(path: Path) -> list[tuple[str, list[str]]]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:  # pragma: no cover - depends on file
                pass
        pages: list[tuple[str, list[str]]] = []
        for page in reader.pages:
            try:
                pages.append((clean_text(page.extract_text() or ""), []))
            except Exception:  # pragma: no cover - malformed page
                pages.append(("", []))
        return pages
    except Exception as exc:
        raise ValueError(f"Unable to read PDF: {exc}") from exc


def _extract_docx(path: Path) -> ExtractionResult:
    from docx import Document as DocxDocument

    try:
        document = DocxDocument(str(path))
    except Exception as exc:
        raise ValueError(f"Unable to read Word document: {exc}") from exc

    blocks: list[str] = []
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            style = (paragraph.style.name or "") if paragraph.style else ""
            # Keep heading structure — it gives the chunker natural sections.
            prefix = "## " if style.lower().startswith("heading") else ""
            blocks.append(f"{prefix}{paragraph.text.strip()}")

    tables: list[str] = []
    for table in document.tables:
        rows = [[cell.text for cell in row.cells] for row in table.rows]
        rendered = _format_table(rows)
        if rendered:
            tables.append(rendered)
            blocks.append(rendered)

    text = clean_text("\n".join(blocks))
    result = ExtractionResult(method=METHOD_DOCX)

    # A DOCX whose content is a scan pasted as pictures still has to be read.
    embedded = _docx_images(path)
    result.image_count = len(embedded)
    if len(text) < MIN_NATIVE_CHARS_PER_PAGE and embedded and ocr_available():
        languages = effective_ocr_languages()
        result.ocr_languages = languages
        confidences: list[float] = []
        ocr_blocks: list[str] = []
        from PIL import Image

        for payload in embedded[: settings.kmrl_ocr_max_pages]:
            try:
                with Image.open(io.BytesIO(payload)) as image:
                    ocr_text, confidence = ocr_image(image, languages)
            except Exception as exc:  # noqa: BLE001 - skip unreadable media
                logger.debug("Embedded image OCR failed in %s: %s", path.name, exc)
                continue
            ocr_text = clean_text(ocr_text)
            if ocr_text:
                ocr_blocks.append(ocr_text)
                if confidence is not None:
                    confidences.append(confidence)
        if ocr_blocks:
            text = clean_text("\n\n".join([text, *ocr_blocks]))
            result.ocr_used = True
            result.method = METHOD_DOCX_OCR
            result.ocr_page_count = len(ocr_blocks)
            if confidences:
                result.ocr_quality = round(sum(confidences) / len(confidences), 1)

    result.text = text
    # Word has no fixed pagination outside a renderer; approximate for display.
    result.page_count = max(1, round(len(text.split()) / 500)) if text else 1
    if text:
        result.pages = [
            PageResult(
                page_number=1,
                text=text,
                language=detect_language(text),
                extraction_method=result.method,
                ocr_used=result.ocr_used,
                ocr_confidence=result.ocr_quality,
                tables=tables,
            )
        ]
    if not text:
        result.warnings.append("This Word document contains no readable text.")
    return result


def _docx_images(path: Path) -> list[bytes]:
    """Raw bytes of every picture embedded in a .docx (it is a zip)."""
    payloads: list[bytes] = []
    try:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.startswith("word/media/") and Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                    payloads.append(archive.read(name))
    except Exception as exc:  # noqa: BLE001 - .doc or corrupt archive
        logger.debug("Could not list media in %s: %s", path.name, exc)
    return payloads


def _extract_text_file(path: Path) -> ExtractionResult:
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = clean_text(raw)
    pages = (
        [
            PageResult(
                page_number=1,
                text=text,
                language=detect_language(text),
                extraction_method=METHOD_PLAIN,
            )
        ]
        if text
        else []
    )
    return ExtractionResult(text=text, pages=pages, page_count=1, method=METHOD_PLAIN)


def _extract_image(path: Path) -> ExtractionResult:
    result = ExtractionResult(method=METHOD_OCR, page_count=1, image_count=1)
    if not ocr_available():
        result.warnings.append(
            "OCR is unavailable on this server, so the text in this image was not read. "
            "Install Tesseract with the 'eng' and 'mal' language packs."
        )
        return result

    from PIL import Image

    languages = effective_ocr_languages()
    result.ocr_languages = languages
    with Image.open(path) as image:
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        text, confidence = ocr_image(image, languages)

    text = clean_text(text)
    result.text = text
    result.ocr_used = True
    result.ocr_quality = confidence
    result.ocr_page_count = 1
    result.pages = [
        PageResult(
            page_number=1,
            text=text,
            language=detect_language(text),
            extraction_method=METHOD_OCR,
            ocr_used=True,
            ocr_confidence=confidence,
        )
    ]
    if not text:
        result.warnings.append("No readable text was found in this image.")
    elif has_malayalam(text) and "mal" not in languages:
        result.warnings.append(
            "Malayalam script was detected but the 'mal' Tesseract pack is not installed; "
            "accuracy will be poor."
        )
    return result


def extract_document(path: Path, filename: str | None = None) -> ExtractionResult:
    """Extract text from ``path``, dispatching on the file extension."""
    name = filename or path.name
    ext = Path(name).suffix.lower()

    if ext in PDF_EXTENSIONS:
        result = _extract_pdf(path)
    elif ext in DOC_EXTENSIONS:
        result = _extract_docx(path)
    elif ext in IMAGE_EXTENSIONS:
        result = _extract_image(path)
    elif ext in TEXT_EXTENSIONS:
        result = _extract_text_file(path)
    else:
        raise ValueError(f"Unsupported file type: {ext or 'unknown'}")

    result.language = detect_language(result.text)
    result.languages_detected = summarise_languages(result.pages)

    if has_malayalam(result.text) and result.ocr_used and not malayalam_ocr_available():
        result.warnings.append(
            "Malayalam text detected without the 'mal' Tesseract language pack — "
            "install it for accurate Malayalam OCR."
        )
    return result


def ocr_page(path: Path, page_number: int, languages: str | None = None) -> PageResult:
    """Re-run OCR for a single page — used by the 'Re-run OCR' action."""
    langs = languages or effective_ocr_languages()
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        from PIL import Image

        with Image.open(path) as image:
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            text, confidence = ocr_image(image, langs)
    else:
        image = render_pdf_page(path, page_number)
        if image is None:
            raise ValueError(f"Page {page_number} could not be rendered for OCR.")
        text, confidence = ocr_image(image, langs)

    text = clean_text(text)
    return PageResult(
        page_number=page_number,
        text=text,
        language=detect_language(text),
        extraction_method=METHOD_OCR,
        ocr_used=True,
        ocr_confidence=confidence,
    )


def ocr_capabilities() -> dict[str, object]:
    """What the health endpoint reports about OCR readiness."""
    languages = tesseract_languages()
    return {
        "available": ocr_available(),
        "engine": ocr_engine_version(),
        "installed_languages": list(languages),
        "configured_languages": settings.kmrl_ocr_languages,
        "effective_languages": effective_ocr_languages() if ocr_available() else "",
        "english": "eng" in languages,
        "malayalam": "mal" in languages,
        "pdf_rendering": pymupdf_available() or poppler_available(),
        "renderer": "pymupdf" if pymupdf_available() else ("poppler" if poppler_available() else None),
        "max_pages": settings.kmrl_ocr_max_pages,
        "dpi": settings.kmrl_ocr_dpi,
    }
