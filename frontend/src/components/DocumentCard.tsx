import { Link } from "react-router-dom";
import type { DocumentSummary } from "../lib/types";
import { formatBytes, formatNumber, relativeTime, splitHighlight } from "../lib/format";
import { DepartmentBadge, FileTypeTile, PriorityBadge, StatusBadge } from "./Badges";
import { ClockIcon, SparkIcon } from "./Icons";

interface Props {
  document: DocumentSummary;
  terms: string[];
}

export function DocumentCard({ document, terms }: Props) {
  const analysed = document.status === "analysis_complete";
  // The snippet is the evidence for *why* a document matched — it only earns
  // its space when there is a query to match against.
  const snippet = terms.length ? document.snippet?.trim() : "";

  return (
    <article className="doc-card">
      <FileTypeTile fileType={document.file_type} ext={document.file_ext} />

      <div className="doc-main">
        <Link to={`/documents/${document.id}`} className="doc-title" title={document.filename}>
          {document.filename}
        </Link>

        <div className="doc-meta">
          <span>{document.file_type}</span>
          <span className="sep" />
          <span>{formatBytes(document.size_bytes)}</span>
          {document.page_count ? (
            <>
              <span className="sep" />
              <span>{document.page_count} page{document.page_count === 1 ? "" : "s"}</span>
            </>
          ) : null}
          {document.word_count ? (
            <>
              <span className="sep" />
              <span>{formatNumber(document.word_count)} words</span>
            </>
          ) : null}
          <span className="sep" />
          <span title={document.uploaded_at}>
            <ClockIcon size={11} style={{ verticalAlign: -1, marginRight: 4 }} />
            {relativeTime(document.uploaded_at)}
          </span>
          {document.language && document.language !== "Unknown" ? (
            <>
              <span className="sep" />
              <span>{document.language}</span>
            </>
          ) : null}
        </div>

        <div className="doc-badges">
          <StatusBadge status={document.status} label={document.status_label} />
          {analysed ? <DepartmentBadge department={document.department} /> : null}
          {analysed ? <PriorityBadge priority={document.priority} /> : null}
          {document.document_type ? (
            <span className="badge badge-neutral">{document.document_type}</span>
          ) : null}
          {document.action_count > 0 ? (
            <span className="badge badge-neutral" title="Action items identified">
              {document.action_count} action{document.action_count === 1 ? "" : "s"}
            </span>
          ) : null}
          {document.deadline_count > 0 ? (
            <span className="badge badge-medium" title="Deadlines identified">
              {document.deadline_count} deadline{document.deadline_count === 1 ? "" : "s"}
            </span>
          ) : null}
        </div>

        {document.summary ? <p className="doc-summary">{document.summary}</p> : null}

        {snippet ? (
          <p className="doc-snippet">
            {splitHighlight(snippet, terms).map((part, index) =>
              part.hit ? <mark key={index}>{part.text}</mark> : <span key={index}>{part.text}</span>,
            )}
          </p>
        ) : null}
      </div>

      <div className="doc-actions">
        <Link className="btn btn-primary btn-sm" to={`/documents/${document.id}`}>
          <SparkIcon size={13} />
          View Intelligence
        </Link>
        {typeof document.score === "number" && document.score > 0 ? (
          <span className="small subtle" title="Relevance score for this search">
            match {document.score.toFixed(0)}
          </span>
        ) : null}
      </div>
    </article>
  );
}
