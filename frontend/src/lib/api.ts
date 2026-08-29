import type {
  DocumentDetail,
  Facets,
  FilterOptions,
  HealthInfo,
  LibraryStats,
  SearchQuery,
  SearchResponse,
} from "./types";

const API_BASE = "/api";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new ApiError(
      "Cannot reach the KMRL service. Check that the backend is running.",
      0,
    );
  }

  if (response.status === 204) return undefined as T;

  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      (payload && (payload.detail || payload.message)) ||
      `Request failed with status ${response.status}`;
    throw new ApiError(
      typeof detail === "string" ? detail : JSON.stringify(detail),
      response.status,
    );
  }
  return payload as T;
}

export function buildSearchParams(query: Partial<SearchQuery>): URLSearchParams {
  const params = new URLSearchParams();
  if (query.q?.trim()) params.set("q", query.q.trim());
  query.departments?.forEach((value) => params.append("department", value));
  query.priorities?.forEach((value) => params.append("priority", value));
  query.fileTypes?.forEach((value) => params.append("file_type", value));
  query.statuses?.forEach((value) => params.append("status", value));
  query.documentTypes?.forEach((value) => params.append("document_type", value));
  if (query.dateRange && query.dateRange !== "any") params.set("date_range", query.dateRange);
  if (query.sort) params.set("sort", query.sort);
  if (query.page) params.set("page", String(query.page));
  if (query.pageSize) params.set("page_size", String(query.pageSize));
  return params;
}

export const api = {
  health: () => request<HealthInfo>("/health"),

  stats: () => request<LibraryStats>("/documents/stats"),

  filters: () => request<FilterOptions>("/documents/filters"),

  search: (query: Partial<SearchQuery>, signal?: AbortSignal) =>
    request<SearchResponse>(`/documents/search?${buildSearchParams(query)}`, { signal }),

  facets: (query: Partial<SearchQuery>, signal?: AbortSignal) => {
    const params = buildSearchParams({ ...query, page: undefined, pageSize: undefined, sort: undefined });
    return request<Facets>(`/documents/facets?${params}`, { signal });
  },

  document: (id: number, signal?: AbortSignal) =>
    request<DocumentDetail>(`/documents/${id}`, { signal }),

  documentText: (id: number, offset = 0, limit = 20000) =>
    request<{ text: string; total_chars: number; has_more: boolean }>(
      `/documents/${id}/text?offset=${offset}&limit=${limit}`,
    ),

  upload: (file: File, uploadedBy = "KMRL User") => {
    const body = new FormData();
    body.append("file", file);
    body.append("uploaded_by", uploadedBy);
    return request<{ document: DocumentDetail; message: string }>("/documents/upload", {
      method: "POST",
      body,
    });
  },

  reprocess: (id: number) =>
    request<{ message: string }>(`/documents/${id}/reprocess`, { method: "POST" }),

  remove: (id: number) => request<void>(`/documents/${id}`, { method: "DELETE" }),

  downloadUrl: (id: number) => `${API_BASE}/documents/${id}/file`,
};
