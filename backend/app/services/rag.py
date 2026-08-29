"""Retrieval-augmented question answering over the KMRL knowledge base.

The answer is grounded ONLY in retrieved chunks. Two answerers share the same
contract:

``claude``     synthesises an answer from the retrieved passages, told
               explicitly to refuse rather than guess.
``extractive`` the no-API-key fallback: it does not generate prose at all, it
               ranks and quotes the retrieved passages. It cannot hallucinate
               because it never writes a sentence the documents do not contain.

Either way, when nothing relevant is retrieved the system says so instead of
answering. Citations carry document, page, section, the English passage AND —
for Malayalam sources — the original text the answer actually rests on.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import Document
from ..taxonomy import ORG_WIDE_ROLES
from .embeddings import search_similar_chunks
from .search import SearchParams, _allowed_document_ids

logger = logging.getLogger(__name__)

NO_EVIDENCE = "I could not find sufficient information in the KMRL document repository."

# A retrieved chunk below this cosine similarity is not treated as evidence.
MIN_EVIDENCE_SCORE = 0.12
MAX_CONTEXT_CHARS = 14_000


@dataclass
class Citation:
    document_id: str
    document_title: str
    filename: str
    page_number: int | None
    section: str = ""
    text: str = ""
    original_text: str = ""
    language: str = "English"
    score: float = 0.0
    department: str = ""
    document_type: str = ""
    uploaded_at: str = ""


@dataclass
class RagAnswer:
    answer: str = ""
    has_evidence: bool = False
    citations: list[Citation] = field(default_factory=list)
    engine: str = "extractive"
    model: str = ""
    confidence: float | None = None
    query: str = ""
    retrieved: int = 0


def retrieve(
    db: Session,
    question: str,
    user=None,
    top_k: int | None = None,
    document_ids: list[str] | None = None,
) -> list[Citation]:
    """Fetch the passages that could answer ``question``, RBAC-filtered."""
    top_k = top_k or settings.kmrl_semantic_top_k

    scope = document_ids
    if scope is None and user is not None and user.role_name not in ORG_WIDE_ROLES:
        params = SearchParams(visible_department_ids=[user.department_id] if user.department_id else [])
        params.visible_uploader_id = user.id
        scope = _allowed_document_ids(db, params)
        if not scope:
            return []

    hits = search_similar_chunks(
        db, question, top_k=top_k * 2, document_ids=scope, min_score=MIN_EVIDENCE_SCORE
    )
    if not hits:
        return []

    documents = {
        row.id: row
        for row in db.execute(
            select(Document).where(Document.id.in_({hit.document_id for hit in hits}))
        ).scalars()
    }

    citations: list[Citation] = []
    seen: set[tuple[str, int | None]] = set()
    for hit in hits:
        document = documents.get(hit.document_id)
        if document is None:
            continue
        key = (hit.document_id, hit.page_number)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                document_id=document.id,
                document_title=document.title or document.filename,
                filename=document.filename,
                page_number=hit.page_number,
                section=hit.section,
                text=hit.text.strip(),
                original_text=(hit.original_text or "").strip(),
                language=hit.language,
                score=hit.score,
                department=document.department_name,
                document_type=document.document_type,
                uploaded_at=document.uploaded_at.isoformat() if document.uploaded_at else "",
            )
        )
        if len(citations) >= top_k:
            break
    return citations


# ------------------------------------------------------------ extractive mode
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
STOP = {
    "what", "which", "who", "whom", "when", "where", "why", "how", "the", "is",
    "are", "was", "were", "show", "find", "list", "all", "any", "of", "in",
    "on", "for", "to", "and", "or", "me", "my", "our", "please", "give", "tell",
    "about", "there", "does", "do", "did", "a", "an", "that", "this", "with",
}


def _keywords(question: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9]{3,}", (question or "").lower())
        if word not in STOP
    }


def answer_extractively(question: str, citations: list[Citation]) -> RagAnswer:
    """Quote the retrieved evidence instead of generating prose.

    This is the fallback used with no API key. It is deliberately not a
    "summary": every sentence it prints came verbatim from an indexed document,
    so it is incapable of inventing a fact.
    """
    if not citations:
        return RagAnswer(answer=NO_EVIDENCE, has_evidence=False, query=question)

    keywords = _keywords(question)
    lines: list[str] = []
    for position, citation in enumerate(citations[:4], start=1):
        sentences = [s.strip() for s in SENTENCE_SPLIT.split(citation.text) if len(s.strip()) > 30]
        if not sentences:
            sentences = [citation.text[:300]]
        best = max(
            sentences,
            key=lambda sentence: sum(1 for word in keywords if word in sentence.lower()),
        )
        where = f"page {citation.page_number}" if citation.page_number else "the document"
        lines.append(
            f"[{position}] {citation.document_title} ({where}): “{best.strip()}”"
        )

    header = (
        f"Found {len(citations)} relevant passage(s) in the KMRL repository. "
        "The following text is quoted directly from the indexed documents:"
    )
    return RagAnswer(
        answer=header + "\n\n" + "\n\n".join(lines),
        has_evidence=True,
        citations=citations,
        engine="extractive",
        model="kmrl-extractive-qa-v1",
        confidence=round(min(0.9, citations[0].score + 0.25), 2),
        query=question,
        retrieved=len(citations),
    )


# ---------------------------------------------------------------- LLM mode
RAG_SYSTEM_PROMPT = """You answer questions about Kochi Metro Rail Limited's (KMRL) document repository.

You are given numbered passages retrieved from indexed documents. Those passages
are your ONLY source of truth.

Rules:
- Answer using ONLY the supplied passages. Never use outside knowledge, and never
  infer facts the passages do not state.
- Cite the passages you used inline as [1], [2] … matching the numbering given.
- If the passages do not contain enough information to answer, reply with exactly:
  "I could not find sufficient information in the KMRL document repository."
  Do not pad that reply with guesses or suggestions of what might be true.
- Some passages were translated from Malayalam. Answer in English, and say when a
  fact comes from a Malayalam source so the reader knows to check the original.
- Be concise and specific: name the department, the date, the reference number and
  the obligation exactly as the passages state them.
- Never invent a document name, page number, date or deadline."""


def answer_with_llm(question: str, citations: list[Citation]) -> RagAnswer:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    blocks: list[str] = []
    used = 0
    for position, citation in enumerate(citations, start=1):
        page = f", page {citation.page_number}" if citation.page_number else ""
        language = (
            f" (translated from {citation.language})"
            if citation.original_text and citation.language != "English"
            else ""
        )
        block = (
            f"[{position}] Document: {citation.document_title}{page}{language}\n"
            f"Type: {citation.document_type} | Department: {citation.department}\n"
            f"Passage: {citation.text}\n"
        )
        if used + len(block) > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        used += len(block)

    response = client.messages.create(
        model=settings.kmrl_rag_model,
        max_tokens=2000,
        system=RAG_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    "Retrieved passages:\n\n"
                    + "\n".join(blocks)
                    + f"\n\nQuestion: {question}\n\nAnswer using only the passages above."
                ),
            }
        ],
    )
    text = "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()
    if not text:
        raise ValueError("The model returned an empty answer.")

    refused = NO_EVIDENCE.lower()[:40] in text.lower()
    return RagAnswer(
        answer=text,
        has_evidence=not refused,
        citations=[] if refused else citations,
        engine="claude",
        model=settings.kmrl_rag_model,
        confidence=None if refused else round(min(0.95, citations[0].score + 0.35), 2),
        query=question,
        retrieved=len(citations),
    )


def ask(
    db: Session,
    question: str,
    user=None,
    top_k: int | None = None,
    document_ids: list[str] | None = None,
) -> RagAnswer:
    """Answer ``question`` from indexed documents only."""
    question = (question or "").strip()
    if not question:
        return RagAnswer(answer="Please enter a question.", has_evidence=False)

    citations = retrieve(db, question, user=user, top_k=top_k, document_ids=document_ids)
    if not citations:
        return RagAnswer(
            answer=NO_EVIDENCE,
            has_evidence=False,
            engine="claude" if settings.llm_enabled else "extractive",
            query=question,
            retrieved=0,
        )

    if settings.llm_enabled:
        try:
            return answer_with_llm(question, citations)
        except Exception as exc:  # noqa: BLE001 - degrade to quoting evidence
            logger.warning("LLM answering failed (%s); quoting the evidence instead.", exc)

    return answer_extractively(question, citations)


def suggested_questions() -> list[str]:
    return [
        "What are the latest directives regarding platform safety?",
        "Which regulatory deadlines are due next week?",
        "Show all safety circulars related to emergency braking.",
        "Which documents mention contractor safety training?",
        "Find previous incidents involving platform screen doors.",
        "What actions are pending with the Rolling Stock department?",
    ]
