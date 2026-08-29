export function formatBytes(bytes: number): string {
  if (!bytes) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 10 || unit === 0 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`;
}

export function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-IN").format(value);
}

/** The API returns naive UTC timestamps; treat them as UTC explicitly. */
function toDate(value: string): Date {
  return new Date(/[zZ]|[+-]\d{2}:?\d{2}$/.test(value) ? value : `${value}Z`);
}

export function formatDate(value: string | null): string {
  if (!value) return "—";
  return toDate(value).toLocaleDateString("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

export function formatDateTime(value: string | null): string {
  if (!value) return "—";
  return toDate(value).toLocaleString("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function relativeTime(value: string | null): string {
  if (!value) return "—";
  const then = toDate(value).getTime();
  const diff = Date.now() - then;
  if (Number.isNaN(diff)) return "—";
  const minutes = Math.round(diff / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} hr ago`;
  const days = Math.round(hours / 24);
  if (days < 30) return `${days} day${days === 1 ? "" : "s"} ago`;
  return formatDate(value);
}

export function priorityClass(priority: string): string {
  switch (priority) {
    case "High":
      return "badge-high";
    case "Medium":
      return "badge-medium";
    case "Low":
      return "badge-low";
    default:
      return "badge-neutral";
  }
}

/** Highlight query terms inside a snippet without using dangerouslySetInnerHTML. */
export function splitHighlight(text: string, terms: string[]): { text: string; hit: boolean }[] {
  const cleaned = terms.map((t) => t.trim().toLowerCase()).filter((t) => t.length > 1);
  if (!cleaned.length) return [{ text, hit: false }];
  const escaped = cleaned.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const pattern = new RegExp(`(${escaped.join("|")})`, "gi");
  return text
    .split(pattern)
    .filter((part) => part !== "")
    .map((part) => ({ text: part, hit: cleaned.includes(part.toLowerCase()) }));
}

export function parseTerms(query: string): string[] {
  return (query.match(/"[^"]+"|\S+/g) || [])
    .map((term) => term.replace(/"/g, "").trim())
    .filter((term) => term.length > 1);
}
