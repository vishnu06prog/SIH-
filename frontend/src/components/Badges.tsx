import { priorityClass } from "../lib/format";

export function PriorityBadge({ priority }: { priority: string }) {
  if (!priority) return null;
  return (
    <span className={`badge ${priorityClass(priority)}`} title={`Priority: ${priority}`}>
      <span className="dot" />
      {priority}
    </span>
  );
}

const PROCESSING = new Set(["queued", "processing", "ocr_processing", "ai_analysis"]);

export function StatusBadge({ status, label }: { status: string; label: string }) {
  if (status === "analysis_complete") {
    return (
      <span className="badge badge-ok">
        <span className="dot" />
        {label}
      </span>
    );
  }
  if (status === "failed") {
    return (
      <span className="badge badge-danger">
        <span className="dot" />
        {label}
      </span>
    );
  }
  if (PROCESSING.has(status)) {
    return (
      <span className="badge badge-brand">
        <span className="dot dot-pulse" />
        {label}
      </span>
    );
  }
  return <span className="badge badge-neutral">{label}</span>;
}

export function DepartmentBadge({ department }: { department: string }) {
  if (!department) return null;
  const unassigned = department === "Unassigned";
  return (
    <span className={`badge ${unassigned ? "badge-neutral" : "badge-brand"}`}>{department}</span>
  );
}

export function FileTypeTile({ fileType, ext }: { fileType: string; ext?: string }) {
  const kind = (fileType || "txt").toLowerCase();
  return (
    <div className={`filetype ${kind}`} aria-hidden="true">
      <span>{(ext || fileType || "FILE").toUpperCase().slice(0, 4)}</span>
    </div>
  );
}

export function EngineBadge({ engine }: { engine: string }) {
  if (!engine) return null;
  const isClaude = engine === "claude";
  return (
    <span
      className="badge badge-neutral"
      title={
        isClaude
          ? "Analysed by Claude"
          : "Analysed by the built-in KMRL rule engine (no API key configured)"
      }
    >
      {isClaude ? "Claude" : "Rule engine"}
    </span>
  );
}
