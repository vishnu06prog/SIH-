"""Embeddings and vector similarity search.

Two interchangeable embedders produce vectors of the same dimension:

``sentence-transformers``  a real multilingual model (used when the package and
                           model are available locally). Preferred, because it
                           gives genuine cross-lingual semantics.
``hashed-lexical``         a deterministic, dependency-free fallback: signed
                           feature hashing over word unigrams, bigrams and
                           character 4-grams, sub-linear TF weighted and L2
                           normalised. It is a lexical model, not a semantic
                           one — but it needs no download, works offline, and
                           because Malayalam pages are indexed through their
                           English translation, an English query still retrieves
                           them.

Two interchangeable vector stores answer the same query:

``pgvector``  SQL ANN search when running on PostgreSQL with the extension.
``numpy``     a brute-force cosine scan over the stored float32 vectors, which
              is exact and fast enough for the corpus sizes KMRL works with.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..database import pgvector_available
from ..models import Document, DocumentChunk, DocumentEmbedding

logger = logging.getLogger(__name__)

WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")

PROVIDER_ST = "sentence-transformers"
PROVIDER_HASHED = "hashed-lexical"


# --------------------------------------------------------------- embedders
@lru_cache(maxsize=1)
def _sentence_transformer():
    """Load the sentence-transformers model, or return None if unavailable."""
    if settings.kmrl_embedding_provider.strip().lower() == "hashed":
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.info(
            "sentence-transformers is not installed; using the hashed lexical embedder."
        )
        return None
    try:
        model = SentenceTransformer(settings.kmrl_embedding_model)
        logger.info("Loaded embedding model %s", settings.kmrl_embedding_model)
        return model
    except Exception as exc:  # noqa: BLE001 - no network, no local cache, etc.
        logger.warning(
            "Could not load %s (%s); using the hashed lexical embedder.",
            settings.kmrl_embedding_model,
            exc,
        )
        return None


def embedding_provider() -> str:
    return PROVIDER_ST if _sentence_transformer() is not None else PROVIDER_HASHED


def embedding_model_name() -> str:
    if embedding_provider() == PROVIDER_ST:
        return settings.kmrl_embedding_model
    return "kmrl-hashed-lexical-v1"


def embedding_dimension() -> int:
    model = _sentence_transformer()
    if model is not None:
        try:
            return int(model.get_sentence_embedding_dimension())
        except Exception:  # pragma: no cover - defensive
            pass
    return settings.kmrl_embedding_dim


def _features(text: str) -> dict[str, float]:
    """Word unigrams, bigrams and character 4-grams with sub-linear TF."""
    words = WORD_RE.findall((text or "").lower())
    counts: dict[str, float] = {}

    def bump(key: str, weight: float) -> None:
        counts[key] = counts.get(key, 0.0) + weight

    for word in words:
        bump(f"w:{word}", 1.0)
    for first, second in zip(words, words[1:]):
        bump(f"b:{first}_{second}", 0.7)

    # Character n-grams give partial credit for morphology and OCR noise, and
    # they carry Malayalam script through when a passage was never translated.
    compact = re.sub(r"\s+", " ", (text or "").lower())
    for start in range(0, max(0, len(compact) - 3), 2):
        bump(f"c:{compact[start : start + 4]}", 0.25)

    return {key: 1.0 + math.log(value) for key, value in counts.items() if value > 0}


def _hash_bucket(key: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dim, 1.0 if (value >> 63) & 1 else -1.0


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch, returning an L2-normalised ``(n, dim)`` float32 array."""
    if not texts:
        return np.zeros((0, embedding_dimension()), dtype=np.float32)

    model = _sentence_transformer()
    if model is not None:
        vectors = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )
        return np.asarray(vectors, dtype=np.float32)

    dim = settings.kmrl_embedding_dim
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        for key, weight in _features(text).items():
            bucket, sign = _hash_bucket(key, dim)
            matrix[row, bucket] += sign * weight
    return normalise(matrix)


def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


# ------------------------------------------------------------- (de)serialise
def to_bytes(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_bytes(payload: bytes, dim: int) -> np.ndarray:
    array = np.frombuffer(payload, dtype=np.float32)
    if array.size != dim:  # a model change left stale rows behind
        return np.zeros(dim, dtype=np.float32)
    return array


# ------------------------------------------------------------- vector store
@dataclass
class VectorHit:
    chunk_id: str
    document_id: str
    score: float
    chunk_index: int = 0
    page_number: int | None = None
    text: str = ""
    original_text: str = ""
    language: str = "English"
    section: str = ""


def vector_backend() -> str:
    configured = settings.kmrl_vector_backend.strip().lower()
    if configured == "numpy":
        return "numpy"
    if configured == "pgvector":
        return "pgvector" if pgvector_available() else "numpy"
    return "pgvector" if pgvector_available() else "numpy"


def store_embeddings(db: Session, document: Document, chunks: list[DocumentChunk]) -> int:
    """Embed every chunk of a document and persist the vectors."""
    if not chunks:
        return 0

    vectors = embed_texts([chunk.text for chunk in chunks])
    model = embedding_model_name()
    dim = vectors.shape[1]

    db.query(DocumentEmbedding).filter(
        DocumentEmbedding.document_id == document.id
    ).delete(synchronize_session=False)

    for chunk, vector in zip(chunks, vectors):
        db.add(
            DocumentEmbedding(
                chunk_id=chunk.id,
                document_id=document.id,
                model=model,
                dim=dim,
                vector=to_bytes(vector),
            )
        )
    db.flush()

    if vector_backend() == "pgvector":
        _sync_pgvector(db, document.id)
    return len(chunks)


def _sync_pgvector(db: Session, document_id: str) -> None:
    """Mirror the stored bytes into the pgvector column for ANN search."""
    from sqlalchemy import text as sql_text

    try:
        rows = db.execute(
            select(DocumentEmbedding.id, DocumentEmbedding.vector, DocumentEmbedding.dim).where(
                DocumentEmbedding.document_id == document_id
            )
        ).all()
        for row_id, payload, dim in rows:
            vector = from_bytes(payload, dim)
            literal = "[" + ",".join(f"{value:.6f}" for value in vector.tolist()) + "]"
            db.execute(
                sql_text(
                    "UPDATE document_embeddings SET embedding = CAST(:v AS vector) WHERE id = :id"
                ),
                {"v": literal, "id": row_id},
            )
    except Exception as exc:  # noqa: BLE001 - degrade to the NumPy scan
        logger.warning("pgvector sync failed (%s); NumPy search still works.", exc)


def search_similar_chunks(
    db: Session,
    query: str,
    top_k: int | None = None,
    document_ids: list[str] | None = None,
    min_score: float = 0.0,
) -> list[VectorHit]:
    """Nearest chunks to ``query`` by cosine similarity."""
    top_k = top_k or settings.kmrl_semantic_top_k
    query_vector = embed_text(query)

    if vector_backend() == "pgvector":
        hits = _search_pgvector(db, query_vector, top_k, document_ids)
        if hits is not None:
            return [hit for hit in hits if hit.score >= min_score]

    return [
        hit
        for hit in _search_numpy(db, query_vector, top_k, document_ids)
        if hit.score >= min_score
    ]


def _chunk_rows(db: Session, chunk_ids: list[str]) -> dict[str, DocumentChunk]:
    if not chunk_ids:
        return {}
    rows = db.execute(
        select(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids))
    ).scalars().all()
    return {row.id: row for row in rows}


def _search_numpy(
    db: Session,
    query_vector: np.ndarray,
    top_k: int,
    document_ids: list[str] | None,
) -> list[VectorHit]:
    stmt = select(
        DocumentEmbedding.chunk_id,
        DocumentEmbedding.document_id,
        DocumentEmbedding.vector,
        DocumentEmbedding.dim,
    )
    if document_ids:
        stmt = stmt.where(DocumentEmbedding.document_id.in_(document_ids))
    rows = db.execute(stmt).all()
    if not rows:
        return []

    dim = query_vector.shape[0]
    matrix = np.vstack(
        [from_bytes(payload, row_dim if row_dim == dim else dim) for _, _, payload, row_dim in rows]
    )
    scores = matrix @ query_vector
    order = np.argsort(-scores)[:top_k]

    chunk_ids = [rows[int(i)][0] for i in order]
    chunks = _chunk_rows(db, chunk_ids)

    hits: list[VectorHit] = []
    for position in order:
        chunk_id, document_id, _, _ = rows[int(position)]
        chunk = chunks.get(chunk_id)
        hits.append(
            VectorHit(
                chunk_id=chunk_id,
                document_id=document_id,
                score=round(float(scores[int(position)]), 4),
                chunk_index=chunk.chunk_index if chunk else 0,
                page_number=chunk.page_number if chunk else None,
                text=chunk.text if chunk else "",
                original_text=chunk.original_text if chunk else "",
                language=chunk.language if chunk else "English",
                section=chunk.section if chunk else "",
            )
        )
    return hits


def _search_pgvector(
    db: Session,
    query_vector: np.ndarray,
    top_k: int,
    document_ids: list[str] | None,
) -> list[VectorHit] | None:
    from sqlalchemy import text as sql_text

    literal = "[" + ",".join(f"{value:.6f}" for value in query_vector.tolist()) + "]"
    filter_sql = ""
    params: dict[str, object] = {"v": literal, "k": top_k}
    if document_ids:
        filter_sql = "AND e.document_id = ANY(:doc_ids)"
        params["doc_ids"] = document_ids

    statement = sql_text(
        f"""
        SELECT c.id, e.document_id, c.chunk_index, c.page_number, c.text,
               c.original_text, c.language, c.section,
               1 - (e.embedding <=> CAST(:v AS vector)) AS score
        FROM document_embeddings e
        JOIN document_chunks c ON c.id = e.chunk_id
        WHERE e.embedding IS NOT NULL {filter_sql}
        ORDER BY e.embedding <=> CAST(:v AS vector)
        LIMIT :k
        """
    )
    try:
        rows = db.execute(statement, params).all()
    except Exception as exc:  # noqa: BLE001 - fall back to the NumPy scan
        logger.warning("pgvector query failed (%s); using the NumPy index.", exc)
        return None

    return [
        VectorHit(
            chunk_id=row[0],
            document_id=row[1],
            chunk_index=row[2] or 0,
            page_number=row[3],
            text=row[4] or "",
            original_text=row[5] or "",
            language=row[6] or "English",
            section=row[7] or "",
            score=round(float(row[8]), 4),
        )
        for row in rows
    ]


def document_vector(db: Session, document_id: str) -> np.ndarray | None:
    """Mean of a document's chunk vectors — its position in embedding space."""
    rows = db.execute(
        select(DocumentEmbedding.vector, DocumentEmbedding.dim).where(
            DocumentEmbedding.document_id == document_id
        )
    ).all()
    if not rows:
        return None
    dim = rows[0][1] or embedding_dimension()
    matrix = np.vstack([from_bytes(payload, dim) for payload, _ in rows])
    mean = matrix.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    return (mean / norm).astype(np.float32) if norm else mean.astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def embedding_capabilities() -> dict[str, object]:
    return {
        "provider": embedding_provider(),
        "model": embedding_model_name(),
        "dimensions": embedding_dimension(),
        "vector_backend": vector_backend(),
        "pgvector": pgvector_available(),
    }
