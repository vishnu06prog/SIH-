export type Priority = "High" | "Medium" | "Low";

export interface DocumentSummary {
  id: number;
  filename: string;
  title: string;
  file_type: string;
  file_ext: string;
  size_bytes: number;
  uploaded_at: string;
  processed_at: string | null;
  status: string;
  status_label: string;
  status_detail: string;
  page_count: number | null;
  word_count: number;
  language: string;
  ocr_used: boolean;
  ocr_quality: number | null;
  document_type: string;
  department: string;
  priority: string;
  summary: string;
  keywords: string[];
  action_count: number;
  deadline_count: number;
  analysis_engine: string;
  snippet?: string | null;
  score?: number | null;
}

export interface ActionItem {
  action: string;
  owner?: string;
  due_date?: string;
  priority?: string;
}

export interface DeadlineItem {
  date: string;
  description?: string;
}

export interface DocumentDetail extends DocumentSummary {
  secondary_departments: string[];
  key_points: string[];
  actions: ActionItem[];
  deadlines: DeadlineItem[];
  risks: string[];
  compliance_flags: string[];
  analysis_model: string;
  analysis_confidence: number | null;
  analysis_error: string;
  error_message: string;
  extraction_method: string;
  char_count: number;
  uploaded_by: string;
  text_preview: string;
}

export interface SearchResponse {
  items: DocumentSummary[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
  query: string;
  applied_filters: Record<string, unknown>;
  sort: string;
  took_ms: number;
  message: string | null;
  suggestion: string | null;
}

export interface FacetValue {
  value: string;
  label: string;
  count: number;
}

export interface Facets {
  departments: FacetValue[];
  priorities: FacetValue[];
  file_types: FacetValue[];
  statuses: FacetValue[];
  document_types: FacetValue[];
}

export interface FilterOptions {
  departments: string[];
  priorities: string[];
  file_types: string[];
  document_types: string[];
  statuses: { value: string; label: string }[];
  sort_options: { value: string; label: string }[];
  date_ranges: { value: string; label: string }[];
}

export interface LibraryStats {
  total_documents: number;
  analysis_complete: number;
  in_progress: number;
  failed: number;
  high_priority: number;
  departments_covered: number;
  total_pages: number;
  open_actions: number;
  ai_engine: string;
}

export interface HealthInfo {
  status: string;
  app: string;
  capabilities: {
    ai_analysis: string;
    analysis_model: string | null;
    ocr: boolean;
    ocr_engine: string | null;
    ocr_languages: string;
    pdf_rendering: boolean;
  };
  limits: { max_upload_mb: number; default_page_size: number; max_page_size: number };
}

export interface SearchQuery {
  q: string;
  departments: string[];
  priorities: string[];
  fileTypes: string[];
  statuses: string[];
  documentTypes: string[];
  dateRange: string;
  sort: string;
  page: number;
  pageSize: number;
}

export const EMPTY_QUERY: SearchQuery = {
  q: "",
  departments: [],
  priorities: [],
  fileTypes: [],
  statuses: [],
  documentTypes: [],
  dateRange: "any",
  sort: "relevance",
  page: 1,
  pageSize: 20,
};
