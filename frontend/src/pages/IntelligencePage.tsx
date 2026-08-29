import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { DepartmentBadge, EngineBadge, FileTypeTile, PriorityBadge, StatusBadge } from "../components/Badges";
import { EmptyState } from "../components/EmptyState";
import {
  AlertIcon,
  ChevronLeft,
  DownloadIcon,
  FileIcon,
  RefreshIcon,
  ShieldIcon,
  SparkIcon,
  TrashIcon,
} from "../components/Icons";
import { useToast } from "../components/Toast";
import { api, ApiError } from "../lib/api";
import { formatBytes, formatDateTime, formatNumber } from "../lib/format";
import type { DocumentDetail } from "../lib/types";

const PROCESSING = new Set(["queued", "processing", "ocr_processing", "ai_analysis"]);

export function IntelligencePage() {
  const { documentId } = useParams();
  const id = Number(documentId);
  const navigate = useNavigate();
  const { notify } = useToast();

  const [document, setDocument] = useState<DocumentDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showSource, setShowSource] = useState(false);
  const [fullText, setFullText] = useState<string | null>(null);

  const load = useCallback(
    async (signal?: AbortSignal) => {
      try {
        const result = await api.document(id, signal);
        if (signal?.aborted) return;
        setDocument(result);
        setError(null);
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        setError(err instanceof ApiError ? err.message : "Could not load this document.");
      } finally {
        if (!signal?.aborted) setLoading(false);
      }
    },
    [id],
  );

  useEffect(() => {
    if (!Number.isFinite(id)) {
      setError("Invalid document reference.");
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    load(controller.signal);
    return () => controller.abort();
  }, [id, load]);

  // Refresh while the pipeline is still running.
  useEffect(() => {
    if (!document || !PROCESSING.has(document.status)) return;
    const timer = window.setInterval(() => load(), 2000);
    return () => window.clearInterval(timer);
  }, [document, load]);

  const openSource = useCallback(async () => {
    setShowSource((visible) => !visible);
    if (fullText === null) {
      try {
        const response = await api.documentText(id, 0, 60000);
        setFullText(response.text);
      } catch {
        setFullText("");
      }
    }
  }, [fullText, id]);

  if (loading) {
    return (
      <div className="col gap-md">
        <div className="skeleton" style={{ height: 140, borderRadius: 14 }} />
        <div className="skeleton" style={{ height: 220, borderRadius: 14 }} />
      </div>
    );
  }

  if (error || !document) {
    return (
      <EmptyState
        icon={<AlertIcon size={24} />}
        title="Document not available"
        description={error ?? "This document no longer exists in the KMRL library."}
        action={
          <Link className="btn btn-primary btn-sm" to="/">
            Back to library
          </Link>
        }
      />
    );
  }

  const analysing = PROCESSING.has(document.status);
  const lowOcr = document.ocr_used && document.ocr_quality !== null && document.ocr_quality < 70;

  return (
    <div>
      <div className="row gap-sm" style={{ marginBottom: 12 }}>
        <Link to="/" className="btn btn-ghost btn-sm">
          <ChevronLeft size={14} />
          Back to library
        </Link>
      </div>

      <header className="intel-head">
        <div className="intel-title-row">
          <FileTypeTile fileType={document.file_type} ext={document.file_ext} />
          <div className="intel-title">
            <h1>{document.title || document.filename}</h1>
            <div className="intel-filename">{document.filename}</div>
            <div className="doc-badges" style={{ marginTop: 10 }}>
              <StatusBadge status={document.status} label={document.status_label} />
              <DepartmentBadge department={document.department} />
              <PriorityBadge priority={document.priority} />
              {document.document_type ? (
                <span className="badge badge-neutral">{document.document_type}</span>
              ) : null}
              <EngineBadge engine={document.analysis_engine} />
              {document.secondary_departments.map((dept) => (
                <span key={dept} className="badge badge-neutral" title="Also relevant to">
                  cc: {dept}
                </span>
              ))}
            </div>
          </div>
          <div className="row gap-sm" style={{ flexWrap: "wrap", justifyContent: "flex-end" }}>
            <a className="btn btn-sm" href={api.downloadUrl(document.id)}>
              <DownloadIcon size={13} />
              Original
            </a>
            <button
              type="button"
              className="btn btn-sm"
              onClick={async () => {
                try {
                  await api.reprocess(document.id);
                  notify("success", "Re-processing started", document.filename);
                  load();
                } catch (err) {
                  notify("error", "Could not re-process", (err as Error).message);
                }
              }}
            >
              <RefreshIcon size={13} />
              Re-analyse
            </button>
            <button
              type="button"
              className="btn btn-sm btn-danger"
              onClick={async () => {
                if (!window.confirm(`Delete “${document.filename}” from the library?`)) return;
                try {
                  await api.remove(document.id);
                  notify("success", "Document deleted", document.filename);
                  navigate("/");
                } catch (err) {
                  notify("error", "Could not delete", (err as Error).message);
                }
              }}
            >
              <TrashIcon size={13} />
              Delete
            </button>
          </div>
        </div>

        <div className="intel-facts">
          <div>
            <div className="fact-label">Uploaded</div>
            <div className="fact-value">{formatDateTime(document.uploaded_at)}</div>
          </div>
          <div>
            <div className="fact-label">Pages</div>
            <div className="fact-value">{document.page_count ?? "—"}</div>
          </div>
          <div>
            <div className="fact-label">Words</div>
            <div className="fact-value">{formatNumber(document.word_count)}</div>
          </div>
          <div>
            <div className="fact-label">Language</div>
            <div className="fact-value">{document.language}</div>
          </div>
          <div>
            <div className="fact-label">Size</div>
            <div className="fact-value">{formatBytes(document.size_bytes)}</div>
          </div>
          <div>
            <div className="fact-label">Extraction</div>
            <div className="fact-value">{document.extraction_method || "—"}</div>
          </div>
        </div>
      </header>

      {analysing ? (
        <div className="notice info" style={{ marginBottom: 16 }}>
          <SparkIcon size={16} />
          <span>
            {document.status_detail || "Processing"} — this page refreshes automatically when the
            analysis is ready.
          </span>
        </div>
      ) : null}

      {document.status === "failed" ? (
        <div className="notice danger" style={{ marginBottom: 16 }}>
          <AlertIcon size={16} />
          <span>{document.error_message || "This document could not be processed."}</span>
        </div>
      ) : null}

      {lowOcr ? (
        <div className="notice" style={{ marginBottom: 16 }}>
          <AlertIcon size={16} />
          <span>
            Source document OCR quality is low ({document.ocr_quality?.toFixed(0)}% confidence).
            Please verify this information against the original document.
          </span>
        </div>
      ) : null}

      {document.analysis_error ? (
        <div className="notice" style={{ marginBottom: 16 }}>
          <AlertIcon size={16} />
          <span>{document.analysis_error}</span>
        </div>
      ) : null}

      <div className="intel-layout">
        <div className="intel-col">
          <section className="card">
            <div className="card-head">
              <SparkIcon size={15} style={{ color: "var(--brand-600)" }} />
              <h3>Executive summary</h3>
              {document.analysis_confidence !== null ? (
                <div className="card-head-actions">
                  <span className="small muted">
                    confidence {(document.analysis_confidence * 100).toFixed(0)}%
                  </span>
                </div>
              ) : null}
            </div>
            <div className="card-body col gap-md">
              {document.summary ? (
                <p className="summary-text">{document.summary}</p>
              ) : (
                <p className="muted small">No summary is available for this document yet.</p>
              )}
              {document.analysis_confidence !== null ? (
                <div className="confidence-bar">
                  <span style={{ width: `${Math.round(document.analysis_confidence * 100)}%` }} />
                </div>
              ) : null}
            </div>
          </section>

          {document.key_points.length ? (
            <section className="card">
              <div className="card-head">
                <h3>Key points</h3>
              </div>
              <div className="card-body">
                <ul className="point-list">
                  {document.key_points.map((point, index) => (
                    <li key={index}>
                      <span className="point-marker">●</span>
                      <span>{point}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </section>
          ) : null}

          <section className="card">
            <div className="card-head">
              <h3>Required actions</h3>
              <div className="card-head-actions">
                <span className="badge badge-neutral">{document.actions.length}</span>
              </div>
            </div>
            <div className="card-body">
              {document.actions.length ? (
                document.actions.map((action, index) => (
                  <div
                    key={index}
                    className={`action-item ${(action.priority || document.priority).toLowerCase()}`}
                  >
                    <div className="action-text">{action.action}</div>
                    <div className="action-meta">
                      {action.owner ? <span>Owner: {action.owner}</span> : null}
                      {action.due_date ? <span>Due: {action.due_date}</span> : null}
                      {action.priority ? <span>Priority: {action.priority}</span> : null}
                    </div>
                  </div>
                ))
              ) : (
                <p className="muted small">
                  No explicit action items were identified in this document.
                </p>
              )}
            </div>
          </section>

          <section className="card">
            <div className="card-head">
              <FileIcon size={15} />
              <h3>Source text</h3>
              <div className="card-head-actions">
                <button type="button" className="btn btn-sm" onClick={openSource}>
                  {showSource ? "Hide" : "Show extracted text"}
                </button>
              </div>
            </div>
            {showSource ? (
              <div className="card-body">
                <pre className="source-text">
                  {fullText ?? document.text_preview ?? "Loading…"}
                </pre>
                <p className="card-note" style={{ marginTop: 8 }}>
                  {formatNumber(document.char_count)} characters extracted via{" "}
                  {document.extraction_method || "text extraction"}.
                </p>
              </div>
            ) : null}
          </section>
        </div>

        <div className="intel-col">
          <section className="card">
            <div className="card-head">
              <h3>Deadlines</h3>
              <div className="card-head-actions">
                <span className="badge badge-neutral">{document.deadlines.length}</span>
              </div>
            </div>
            <div className="card-body">
              {document.deadlines.length ? (
                document.deadlines.map((deadline, index) => (
                  <div key={index} className="deadline-item">
                    <span className="deadline-date">{deadline.date}</span>
                    <span className="deadline-desc">{deadline.description}</span>
                  </div>
                ))
              ) : (
                <p className="muted small">No dated obligations were found.</p>
              )}
            </div>
          </section>

          {document.risks.length ? (
            <section className="card">
              <div className="card-head">
                <AlertIcon size={15} style={{ color: "var(--danger-600)" }} />
                <h3>Risks flagged</h3>
              </div>
              <div className="card-body">
                <ul className="point-list">
                  {document.risks.map((risk, index) => (
                    <li key={index}>
                      <span className="point-marker" style={{ color: "var(--danger-600)" }}>
                        ●
                      </span>
                      <span>{risk}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </section>
          ) : null}

          {document.compliance_flags.length ? (
            <section className="card">
              <div className="card-head">
                <ShieldIcon size={15} style={{ color: "var(--brand-600)" }} />
                <h3>Compliance signals</h3>
              </div>
              <div className="card-body chip-row">
                {document.compliance_flags.map((flag) => (
                  <span key={flag} className="badge badge-brand">
                    {flag}
                  </span>
                ))}
              </div>
            </section>
          ) : null}

          {document.keywords.length ? (
            <section className="card">
              <div className="card-head">
                <h3>Keywords</h3>
              </div>
              <div className="card-body chip-row">
                {document.keywords.map((keyword) => (
                  <Link
                    key={keyword}
                    className="chip"
                    to={`/?q=${encodeURIComponent(keyword)}`}
                    title={`Search the library for “${keyword}”`}
                  >
                    {keyword}
                  </Link>
                ))}
              </div>
            </section>
          ) : null}

          <section className="card">
            <div className="card-head">
              <h3>Provenance</h3>
            </div>
            <div className="card-body col gap-sm small muted">
              <div className="row-between">
                <span>Uploaded by</span>
                <span style={{ color: "var(--text)" }}>{document.uploaded_by}</span>
              </div>
              <div className="row-between">
                <span>Analysed at</span>
                <span style={{ color: "var(--text)" }}>{formatDateTime(document.processed_at)}</span>
              </div>
              <div className="row-between">
                <span>Engine</span>
                <span style={{ color: "var(--text)" }}>
                  {document.analysis_engine === "claude" ? "Claude" : "KMRL rule engine"}
                </span>
              </div>
              <div className="row-between">
                <span>Model</span>
                <span style={{ color: "var(--text)" }} className="mono">
                  {document.analysis_model || "—"}
                </span>
              </div>
              {document.ocr_used ? (
                <div className="row-between">
                  <span>OCR confidence</span>
                  <span style={{ color: "var(--text)" }}>
                    {document.ocr_quality !== null ? `${document.ocr_quality.toFixed(0)}%` : "—"}
                  </span>
                </div>
              ) : null}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
