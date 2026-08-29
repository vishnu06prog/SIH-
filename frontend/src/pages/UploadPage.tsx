import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { AlertIcon, CheckIcon, SparkIcon, UploadIcon } from "../components/Icons";
import { StatusBadge } from "../components/Badges";
import { useToast } from "../components/Toast";
import { api, ApiError } from "../lib/api";
import { formatBytes } from "../lib/format";
import type { DocumentDetail } from "../lib/types";

interface QueueEntry {
  key: string;
  name: string;
  size: number;
  state: "uploading" | "processing" | "done" | "error";
  documentId?: number;
  status?: string;
  statusLabel?: string;
  detail?: string;
  error?: string;
}

const ACCEPT = ".pdf,.docx,.doc,.txt,.md,.csv,.png,.jpg,.jpeg,.tiff,.bmp,.webp";

const PIPELINE = [
  { title: "Upload", desc: "The original file is stored and checksummed." },
  { title: "Text extraction", desc: "Native PDF/DOCX text is read page by page." },
  { title: "OCR fallback", desc: "Scanned pages and images go through Tesseract (English + Malayalam)." },
  { title: "AI analysis", desc: "Department, priority, summary, actions and deadlines are extracted." },
  { title: "Indexed", desc: "The document becomes searchable in the library." },
];

export function UploadPage() {
  const [queue, setQueue] = useState<QueueEntry[]>([]);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const { notify } = useToast();

  const patch = useCallback((key: string, changes: Partial<QueueEntry>) => {
    setQueue((current) =>
      current.map((entry) => (entry.key === key ? { ...entry, ...changes } : entry)),
    );
  }, []);

  const upload = useCallback(
    async (files: FileList | File[]) => {
      for (const file of Array.from(files)) {
        const key = `${file.name}-${Date.now()}-${Math.random()}`;
        setQueue((current) => [
          { key, name: file.name, size: file.size, state: "uploading" },
          ...current,
        ]);
        try {
          const response = await api.upload(file);
          patch(key, {
            state: "processing",
            documentId: response.document.id,
            status: response.document.status,
            statusLabel: response.document.status_label,
            detail: response.document.status_detail,
          });
        } catch (err) {
          const message =
            err instanceof ApiError ? err.message : "Upload failed. Please try again.";
          patch(key, { state: "error", error: message });
          notify("error", `Could not upload ${file.name}`, message);
        }
      }
    },
    [notify, patch],
  );

  // Poll the documents that are still being processed.
  useEffect(() => {
    const pending = queue.filter((entry) => entry.state === "processing" && entry.documentId);
    if (!pending.length) return;

    const timer = window.setInterval(async () => {
      for (const entry of pending) {
        try {
          const document: DocumentDetail = await api.document(entry.documentId!);
          if (document.status === "analysis_complete") {
            patch(entry.key, {
              state: "done",
              status: document.status,
              statusLabel: document.status_label,
              detail: `${document.department} · ${document.priority} priority`,
            });
            notify("success", `${document.filename} analysed`, document.status_detail);
          } else if (document.status === "failed") {
            patch(entry.key, {
              state: "error",
              status: document.status,
              statusLabel: document.status_label,
              error: document.error_message || "Processing failed.",
            });
          } else {
            patch(entry.key, {
              status: document.status,
              statusLabel: document.status_label,
              detail: document.status_detail,
            });
          }
        } catch {
          /* keep polling — a transient error should not stop the queue */
        }
      }
    }, 1600);

    return () => window.clearInterval(timer);
  }, [queue, patch, notify]);

  return (
    <div className="upload-layout">
      <div className="col gap-md">
        <div
          className={`dropzone${dragging ? " dragging" : ""}`}
          role="button"
          tabIndex={0}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
          }}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            if (event.dataTransfer.files?.length) upload(event.dataTransfer.files);
          }}
        >
          <div className="dropzone-icon">
            <UploadIcon size={24} />
          </div>
          <h3>Drop documents here, or click to browse</h3>
          <p>
            PDF, Word, text and scanned images. Bilingual English/Malayalam files are supported —
            scanned pages are read with OCR automatically.
          </p>
          <span className="btn btn-primary btn-lg">Select files</span>
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPT}
            multiple
            className="visually-hidden"
            onChange={(event) => {
              if (event.target.files?.length) upload(event.target.files);
              event.target.value = "";
            }}
          />
        </div>

        {queue.length ? (
          <div className="card">
            <div className="card-head">
              <h3>Upload queue</h3>
              <div className="card-head-actions">
                <button
                  type="button"
                  className="btn btn-ghost btn-sm"
                  onClick={() => setQueue((c) => c.filter((e) => e.state !== "done"))}
                >
                  Clear completed
                </button>
              </div>
            </div>
            <div className="card-body">
              {queue.map((entry) => (
                <div key={entry.key} className="queue-item">
                  <div
                    style={{
                      width: 34,
                      height: 34,
                      borderRadius: 9,
                      display: "grid",
                      placeItems: "center",
                      background: "var(--surface-sunken)",
                      border: "1px solid var(--border)",
                      color:
                        entry.state === "error"
                          ? "var(--danger-600)"
                          : entry.state === "done"
                            ? "var(--ok-700)"
                            : "var(--brand-600)",
                    }}
                  >
                    {entry.state === "error" ? (
                      <AlertIcon size={16} />
                    ) : entry.state === "done" ? (
                      <CheckIcon size={16} />
                    ) : (
                      <SparkIcon size={16} />
                    )}
                  </div>
                  <div style={{ minWidth: 0 }}>
                    <div className="queue-name truncate">{entry.name}</div>
                    <div className="queue-status">
                      {entry.state === "uploading"
                        ? "Uploading…"
                        : entry.state === "error"
                          ? entry.error
                          : entry.detail || entry.statusLabel}
                      <span className="subtle"> · {formatBytes(entry.size)}</span>
                    </div>
                    {entry.state === "uploading" || entry.state === "processing" ? (
                      <div className="progress indeterminate" style={{ marginTop: 6 }}>
                        <span />
                      </div>
                    ) : null}
                  </div>
                  <div className="row gap-sm">
                    {entry.status ? (
                      <StatusBadge status={entry.status} label={entry.statusLabel ?? ""} />
                    ) : null}
                    {entry.documentId && entry.state === "done" ? (
                      <Link className="btn btn-sm btn-primary" to={`/documents/${entry.documentId}`}>
                        View Intelligence
                      </Link>
                    ) : null}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </div>

      <div className="card">
        <div className="card-head">
          <h3>What happens next</h3>
        </div>
        <div className="card-body">
          <div className="pipeline">
            {PIPELINE.map((step, index) => (
              <div key={step.title} className="pipeline-step done">
                <div className="pipeline-rail">
                  <div className="pipeline-dot" />
                  {index < PIPELINE.length - 1 ? <div className="pipeline-line" /> : null}
                </div>
                <div className="pipeline-body">
                  <div className="pipeline-title">{step.title}</div>
                  <div className="pipeline-desc">{step.desc}</div>
                </div>
              </div>
            ))}
          </div>
          <hr className="divider" style={{ margin: "6px 0 14px" }} />
          <p className="card-note">
            Processing runs in the background — you can keep uploading or go back to the library
            while a document is being analysed.
          </p>
        </div>
      </div>
    </div>
  );
}
