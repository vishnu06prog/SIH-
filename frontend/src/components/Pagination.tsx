import { ChevronLeft, ChevronRight } from "./Icons";

interface Props {
  page: number;
  totalPages: number;
  total: number;
  pageSize: number;
  onPage: (page: number) => void;
}

function pageWindow(page: number, totalPages: number): (number | "gap")[] {
  if (totalPages <= 7) return Array.from({ length: totalPages }, (_, i) => i + 1);
  const pages = new Set<number>([1, totalPages, page, page - 1, page + 1]);
  const sorted = [...pages].filter((p) => p >= 1 && p <= totalPages).sort((a, b) => a - b);
  const result: (number | "gap")[] = [];
  sorted.forEach((value, index) => {
    if (index > 0 && value - (sorted[index - 1] as number) > 1) result.push("gap");
    result.push(value);
  });
  return result;
}

export function Pagination({ page, totalPages, total, pageSize, onPage }: Props) {
  if (totalPages <= 1) {
    return (
      <div className="pagination">
        <span className="pagination-info">
          {total} document{total === 1 ? "" : "s"}
        </span>
      </div>
    );
  }

  const first = (page - 1) * pageSize + 1;
  const last = Math.min(page * pageSize, total);

  return (
    <nav className="pagination" aria-label="Pagination">
      <span className="pagination-info">
        Showing {first}–{last} of {total}
      </span>
      <div className="pagination-controls">
        <button
          className="page-btn"
          onClick={() => onPage(page - 1)}
          disabled={page <= 1}
          aria-label="Previous page"
        >
          <ChevronLeft size={14} />
        </button>
        {pageWindow(page, totalPages).map((entry, index) =>
          entry === "gap" ? (
            <span key={`gap-${index}`} className="page-ellipsis">
              …
            </span>
          ) : (
            <button
              key={entry}
              className={`page-btn${entry === page ? " current" : ""}`}
              onClick={() => onPage(entry)}
              aria-current={entry === page ? "page" : undefined}
            >
              {entry}
            </button>
          ),
        )}
        <button
          className="page-btn"
          onClick={() => onPage(page + 1)}
          disabled={page >= totalPages}
          aria-label="Next page"
        >
          <ChevronRight size={14} />
        </button>
      </div>
    </nav>
  );
}
