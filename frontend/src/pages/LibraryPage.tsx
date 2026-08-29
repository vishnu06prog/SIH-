import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { DocumentCard } from "../components/DocumentCard";
import { EmptyState } from "../components/EmptyState";
import { FilterRail, type FilterKey } from "../components/FilterRail";
import { CloseIcon, SearchIcon, UploadIcon } from "../components/Icons";
import { Pagination } from "../components/Pagination";
import { SearchBar } from "../components/SearchBar";
import { api, ApiError } from "../lib/api";
import { formatNumber, parseTerms } from "../lib/format";
import type {
  Facets,
  FilterOptions,
  LibraryStats,
  SearchQuery,
  SearchResponse,
} from "../lib/types";

const SUGGESTIONS = [
  "safety",
  "inspection",
  "procurement",
  "invoice",
  "maintenance",
  "training",
];

const PROCESSING_STATUSES = new Set(["queued", "processing", "ocr_processing", "ai_analysis"]);

const PARAM_KEYS: Record<FilterKey, string> = {
  departments: "dept",
  priorities: "priority",
  fileTypes: "type",
  statuses: "status",
};

function readQuery(params: URLSearchParams): SearchQuery {
  return {
    q: params.get("q") ?? "",
    departments: params.getAll("dept"),
    priorities: params.getAll("priority"),
    fileTypes: params.getAll("type"),
    statuses: params.getAll("status"),
    documentTypes: params.getAll("doctype"),
    dateRange: params.get("date") ?? "any",
    sort: params.get("sort") ?? "relevance",
    page: Math.max(1, Number(params.get("page") ?? 1) || 1),
    pageSize: Math.min(100, Math.max(1, Number(params.get("size") ?? 20) || 20)),
  };
}

function writeQuery(query: SearchQuery): URLSearchParams {
  const params = new URLSearchParams();
  if (query.q.trim()) params.set("q", query.q.trim());
  query.departments.forEach((v) => params.append("dept", v));
  query.priorities.forEach((v) => params.append("priority", v));
  query.fileTypes.forEach((v) => params.append("type", v));
  query.statuses.forEach((v) => params.append("status", v));
  query.documentTypes.forEach((v) => params.append("doctype", v));
  if (query.dateRange !== "any") params.set("date", query.dateRange);
  if (query.sort !== "relevance") params.set("sort", query.sort);
  if (query.page > 1) params.set("page", String(query.page));
  if (query.pageSize !== 20) params.set("size", String(query.pageSize));
  return params;
}

export function LibraryPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const query = useMemo(() => readQuery(searchParams), [searchParams]);

  const [results, setResults] = useState<SearchResponse | null>(null);
  const [facets, setFacets] = useState<Facets | null>(null);
  const [options, setOptions] = useState<FilterOptions | null>(null);
  const [stats, setStats] = useState<LibraryStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const firstLoad = useRef(true);

  const update = useCallback(
    (patch: Partial<SearchQuery>, resetPage = true) => {
      const next: SearchQuery = { ...query, ...patch };
      if (resetPage && patch.page === undefined) next.page = 1;
      setSearchParams(writeQuery(next), { replace: false });
    },
    [query, setSearchParams],
  );

  useEffect(() => {
    api.filters().then(setOptions).catch(() => setOptions(null));
  }, []);

  const load = useCallback(
    async (signal?: AbortSignal, quiet = false) => {
      if (!quiet) setLoading(true);
      try {
        const [searchResult, facetResult, statsResult] = await Promise.all([
          api.search(query, signal),
          api.facets(query, signal),
          api.stats(),
        ]);
        if (signal?.aborted) return;
        setResults(searchResult);
        setFacets(facetResult);
        setStats(statsResult);
        setError(null);
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        setError(err instanceof ApiError ? err.message : "Something went wrong loading documents.");
      } finally {
        if (!signal?.aborted) {
          setLoading(false);
          firstLoad.current = false;
        }
      }
    },
    [query],
  );

  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal);
    return () => controller.abort();
  }, [load]);

  // Keep the library live while documents are still being processed.
  const hasProcessing = useMemo(
    () => (results?.items ?? []).some((item) => PROCESSING_STATUSES.has(item.status)),
    [results],
  );

  useEffect(() => {
    if (!hasProcessing) return;
    const timer = window.setInterval(() => load(undefined, true), 3500);
    return () => window.clearInterval(timer);
  }, [hasProcessing, load]);

  const toggleFilter = useCallback(
    (key: FilterKey, value: string) => {
      const current = query[key];
      const next = current.includes(value)
        ? current.filter((entry) => entry !== value)
        : [...current, value];
      update({ [key]: next } as Partial<SearchQuery>);
    },
    [query, update],
  );

  const resetFilters = useCallback(() => {
    update({
      departments: [],
      priorities: [],
      fileTypes: [],
      statuses: [],
      documentTypes: [],
      dateRange: "any",
    });
  }, [update]);

  const terms = useMemo(() => parseTerms(query.q), [query.q]);

  const activeFilters: { key: FilterKey | "dateRange"; value: string; label: string }[] = [];
  (Object.keys(PARAM_KEYS) as FilterKey[]).forEach((key) => {
    query[key].forEach((value) => {
      const label =
        key === "statuses"
          ? options?.statuses.find((status) => status.value === value)?.label ?? value
          : value;
      activeFilters.push({ key, value, label });
    });
  });
  if (query.dateRange !== "any") {
    activeFilters.push({
      key: "dateRange",
      value: query.dateRange,
      label: options?.date_ranges.find((r) => r.value === query.dateRange)?.label ?? query.dateRange,
    });
  }

  return (
    <div>
      <div className="library-head">
        {stats ? (
          <div className="stat-strip">
            <div className="stat">
              <div className="stat-label">Documents</div>
              <div className="stat-value">{formatNumber(stats.total_documents)}</div>
              <div className="stat-hint">{formatNumber(stats.total_pages)} pages indexed</div>
            </div>
            <div className="stat accent-ok">
              <div className="stat-label">Analysed</div>
              <div className="stat-value">{formatNumber(stats.analysis_complete)}</div>
              <div className="stat-hint">
                {stats.in_progress > 0 ? `${stats.in_progress} in progress` : "All caught up"}
              </div>
            </div>
            <div className="stat accent-danger">
              <div className="stat-label">High priority</div>
              <div className="stat-value">{formatNumber(stats.high_priority)}</div>
              <div className="stat-hint">need attention first</div>
            </div>
            <div className="stat accent-warn">
              <div className="stat-label">Action items</div>
              <div className="stat-value">{formatNumber(stats.open_actions)}</div>
              <div className="stat-hint">extracted across the library</div>
            </div>
            <div className="stat">
              <div className="stat-label">Departments</div>
              <div className="stat-value">{formatNumber(stats.departments_covered)}</div>
              <div className="stat-hint">routed automatically</div>
            </div>
          </div>
        ) : (
          <div className="stat-strip">
            {[0, 1, 2, 3, 4].map((key) => (
              <div key={key} className="skeleton" style={{ height: 78, borderRadius: 14 }} />
            ))}
          </div>
        )}

        <div className="search-panel">
          <SearchBar
            value={query.q}
            onChange={(value) => update({ q: value })}
            placeholder="Search documents… filename, department, summary, keywords or content"
            autoFocus
          />
          <div className="row-between wrap gap-sm">
            <div className="search-suggestions">
              <span>Try:</span>
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  className="suggest-chip"
                  onClick={() => update({ q: suggestion })}
                >
                  {suggestion}
                </button>
              ))}
            </div>
            <div className="row gap-sm">
              <label className="small muted" htmlFor="sort">
                Sort
              </label>
              <select
                id="sort"
                className="select"
                style={{ width: 168 }}
                value={query.sort}
                onChange={(event) => update({ sort: event.target.value })}
              >
                {(options?.sort_options ?? [{ value: "relevance", label: "Best match" }]).map(
                  (option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ),
                )}
              </select>
              <select
                className="select"
                style={{ width: 108 }}
                value={query.pageSize}
                aria-label="Documents per page"
                onChange={(event) => update({ pageSize: Number(event.target.value) })}
              >
                {[10, 20, 50, 100].map((size) => (
                  <option key={size} value={size}>
                    {size} / page
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </div>

      <div className="library-body">
        <FilterRail
          options={options}
          facets={facets}
          query={query}
          onToggle={toggleFilter}
          onDateRange={(value) => update({ dateRange: value })}
          onReset={resetFilters}
        />

        <section>
          {activeFilters.length ? (
            <div className="active-filters">
              {activeFilters.map((filter) => (
                <span key={`${filter.key}-${filter.value}`} className="active-filter">
                  {filter.label}
                  <button
                    type="button"
                    aria-label={`Remove filter ${filter.label}`}
                    onClick={() =>
                      filter.key === "dateRange"
                        ? update({ dateRange: "any" })
                        : toggleFilter(filter.key, filter.value)
                    }
                  >
                    <CloseIcon size={11} />
                  </button>
                </span>
              ))}
              <button type="button" className="btn btn-ghost btn-sm" onClick={resetFilters}>
                Clear all
              </button>
            </div>
          ) : null}

          <div className="results-head">
            <span className="results-count">
              {loading && firstLoad.current ? (
                "Searching…"
              ) : results ? (
                <>
                  <strong>{formatNumber(results.total)}</strong> document
                  {results.total === 1 ? "" : "s"}
                  {results.query ? (
                    <>
                      {" "}
                      for <strong>“{results.query}”</strong>
                    </>
                  ) : null}
                  <span className="subtle"> · {results.took_ms} ms</span>
                </>
              ) : null}
            </span>
            <Link to="/upload" className="btn btn-sm">
              <UploadIcon size={13} />
              Upload document
            </Link>
          </div>

          {error ? (
            <div className="notice danger" style={{ marginBottom: 12 }}>
              <span>{error}</span>
            </div>
          ) : null}

          {loading && !results ? (
            <div className="doc-grid">
              {[0, 1, 2, 3, 4].map((key) => (
                <div key={key} className="skeleton skeleton-card" />
              ))}
            </div>
          ) : results && results.items.length ? (
            <>
              <div className="doc-grid" style={loading ? { opacity: 0.55 } : undefined}>
                {results.items.map((document) => (
                  <DocumentCard key={document.id} document={document} terms={terms} />
                ))}
              </div>
              <div style={{ marginTop: 16 }}>
                <Pagination
                  page={results.page}
                  totalPages={results.total_pages}
                  total={results.total}
                  pageSize={results.page_size}
                  onPage={(page) => {
                    update({ page }, false);
                    window.scrollTo({ top: 0, behavior: "smooth" });
                  }}
                />
              </div>
            </>
          ) : (
            <EmptyState
              icon={<SearchIcon size={24} />}
              title={results?.message ?? "No documents found"}
              description={results?.suggestion ?? "Try a different keyword or remove a filter."}
              action={
                activeFilters.length || query.q ? (
                  <button
                    type="button"
                    className="btn btn-primary btn-sm"
                    onClick={() =>
                      update({
                        q: "",
                        departments: [],
                        priorities: [],
                        fileTypes: [],
                        statuses: [],
                        documentTypes: [],
                        dateRange: "any",
                      })
                    }
                  >
                    Clear search and filters
                  </button>
                ) : (
                  <Link to="/upload" className="btn btn-primary btn-sm">
                    <UploadIcon size={14} />
                    Upload your first document
                  </Link>
                )
              }
            />
          )}
        </section>
      </div>
    </div>
  );
}
