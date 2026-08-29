"""KMRL domain vocabulary shared by the analyser, search layer and API."""

from __future__ import annotations

DEPARTMENTS: list[str] = [
    "Operations",
    "Rolling Stock",
    "Civil / Track",
    "Signalling & Telecom",
    "Electrical",
    "Safety",
    "Finance",
    "Procurement",
    "HR",
    "Legal",
    "Projects",
    "IT",
]

UNASSIGNED_DEPARTMENT = "Unassigned"

PRIORITIES: list[str] = ["High", "Medium", "Low"]

DOCUMENT_TYPES: list[str] = [
    "Engineering Drawing",
    "Maintenance Job Card",
    "Incident Report",
    "Invoice",
    "Purchase Order",
    "Regulatory Directive",
    "Safety Circular",
    "Environmental Study",
    "HR Policy",
    "Legal Opinion",
    "Board Minutes",
    "Tender / Contract",
    "Training Material",
    "Technical Report",
    "Correspondence",
    "Other",
]

FILE_TYPES: list[str] = ["PDF", "DOCX", "TXT", "IMAGE"]

# Processing statuses -------------------------------------------------------
STATUS_QUEUED = "queued"
STATUS_PROCESSING = "processing"
STATUS_OCR = "ocr_processing"
STATUS_ANALYZING = "ai_analysis"
STATUS_COMPLETE = "analysis_complete"
STATUS_FAILED = "failed"

STATUSES: list[str] = [
    STATUS_QUEUED,
    STATUS_PROCESSING,
    STATUS_OCR,
    STATUS_ANALYZING,
    STATUS_COMPLETE,
    STATUS_FAILED,
]

STATUS_LABELS: dict[str, str] = {
    STATUS_QUEUED: "Queued",
    STATUS_PROCESSING: "Processing",
    STATUS_OCR: "OCR Processing",
    STATUS_ANALYZING: "AI Analysis",
    STATUS_COMPLETE: "AI Analysis Complete",
    STATUS_FAILED: "Failed",
}

# Aliases accepted by the API so the UI (and humans) can filter with labels.
STATUS_ALIASES: dict[str, str] = {
    "processing": STATUS_PROCESSING,
    "ocr processing": STATUS_OCR,
    "ocr_processing": STATUS_OCR,
    "ai analysis": STATUS_ANALYZING,
    "ai analysis complete": STATUS_COMPLETE,
    "analysis complete": STATUS_COMPLETE,
    "complete": STATUS_COMPLETE,
    "completed": STATUS_COMPLETE,
    "failed": STATUS_FAILED,
    "queued": STATUS_QUEUED,
}

# Department detection signals used by the rule-based analyser ---------------
DEPARTMENT_KEYWORDS: dict[str, list[str]] = {
    "Operations": [
        "train operation", "timetable", "headway", "station controller", "service",
        "ridership", "platform", "depot control", "train availability", "ots",
        "operations control centre", "occ", "punctuality", "revenue service",
    ],
    "Rolling Stock": [
        "rolling stock", "bogie", "traction motor", "trainset", "coach", "brake",
        "maximo", "job card", "overhaul", "wheel", "pantograph", "hvac unit",
        "propulsion", "car body", "preventive maintenance",
    ],
    "Civil / Track": [
        "viaduct", "track", "ballast", "alignment", "bridge", "civil works",
        "station building", "structural", "geometry", "tamping", "rail weld",
        "concrete", "girder",
    ],
    "Signalling & Telecom": [
        "signalling", "signaling", "cbtc", "interlocking", "telecom", "atp",
        "ato", "axle counter", "scada", "radio", "fibre", "fiber optic",
        "communication system",
    ],
    "Electrical": [
        "traction power", "ohe", "overhead equipment", "substation", "33kv",
        "electrical", "transformer", "ups", "power supply", "lighting",
        "earthing", "breaker",
    ],
    "Safety": [
        "safety", "incident", "accident", "hazard", "near miss", "cmrs",
        "commissioner of metro rail safety", "risk assessment", "emergency",
        "fire", "evacuation", "safety circular", "audit non-conformity",
        "ppe", "inspection",
    ],
    "Finance": [
        "invoice", "payment", "budget", "expenditure", "gst", "audit",
        "financial", "cash flow", "accounts", "reimbursement", "fare revenue",
        "cost centre", "bill",
    ],
    "Procurement": [
        "tender", "purchase order", "vendor", "supplier", "quotation", "rfp",
        "bid", "contract award", "spare parts", "procurement", "eoi",
        "emd", "delivery schedule",
    ],
    "HR": [
        "employee", "staff", "recruitment", "training", "leave policy",
        "payroll", "appraisal", "human resources", "attendance", "transfer order",
        "refresher training", "manpower",
    ],
    "Legal": [
        "legal", "litigation", "court", "arbitration", "agreement", "clause",
        "compliance notice", "affidavit", "counsel", "writ petition",
        "memorandum of understanding", "indemnity",
    ],
    "Projects": [
        "project", "corridor", "phase ii", "phase 2", "extension", "milestone",
        "dpr", "detailed project report", "construction", "commissioning",
        "new depot", "work package",
    ],
    "IT": [
        "software", "application", "database", "server", "network", "cyber",
        "erp", "sharepoint", "machine learning", "data analytics", "api",
        "system integration", "unified namespace", "iot", "dashboard",
        "algorithm", "neural network", "model training", "python",
    ],
}

DOCUMENT_TYPE_KEYWORDS: dict[str, list[str]] = {
    "Invoice": ["invoice", "tax invoice", "bill no", "amount payable", "gstin"],
    "Purchase Order": ["purchase order", "po number", "po no", "supply order"],
    "Maintenance Job Card": ["job card", "work order", "maximo", "maintenance schedule"],
    "Incident Report": ["incident report", "near miss", "accident report", "occurrence"],
    "Regulatory Directive": [
        "ministry of housing", "mohua", "commissioner of metro rail safety",
        "directive", "gazette", "statutory", "circular no", "regulation",
    ],
    "Safety Circular": ["safety circular", "safety bulletin", "safety advisory"],
    "Environmental Study": ["environmental impact", "eia", "pollution control", "green audit"],
    "HR Policy": [
        "hr policy", "leave policy", "code of conduct", "employee handbook",
        "human resources", "refresher training", "training calendar", "payroll",
        "hr circular", "appraisal",
    ],
    "Legal Opinion": ["legal opinion", "counsel opinion", "legal advice", "arbitration award"],
    "Board Minutes": ["minutes of the meeting", "board meeting", "agenda item", "resolved that"],
    "Tender / Contract": ["tender document", "nit", "contract agreement", "scope of work", "bid document"],
    "Engineering Drawing": ["drawing no", "revision", "scale 1:", "general arrangement", "as-built"],
    "Training Material": ["module", "unit", "syllabus", "lecture", "chapter", "learning outcome", "tutorial"],
    "Technical Report": [
        "technical report", "analysis report", "findings", "methodology",
        "test report", "technical note", "architecture note", "design note",
        "inspection report", "quarterly report", "recommendation",
    ],
    "Correspondence": [
        "dear sir", "with reference to", "yours faithfully", "subject:",
        "notice", "station notice", "for information",
    ],
}

HIGH_PRIORITY_SIGNALS = [
    "urgent", "immediate", "immediately", "critical", "safety critical",
    "non-compliance", "penalty", "deadline", "escalate", "emergency",
    "accident", "failure", "shutdown", "statutory", "mandatory",
    "within 24 hours", "within 48 hours", "top priority", "show cause",
]

MEDIUM_PRIORITY_SIGNALS = [
    "action required", "please ensure", "review", "submit", "schedule",
    "follow up", "reminder", "kindly arrange", "compliance", "due",
]

ACTION_SIGNALS = [
    "shall", "must", "is required to", "are required to", "please ensure",
    "action required", "to be completed", "submit", "ensure that", "arrange",
    "responsible for", "needs to", "should be", "is directed to", "comply",
    "is requested to", "are requested to", "requested to", "is due", "due within",
    "we advise", "advised to", "required to",
]

STOPWORDS = {
    "the", "and", "for", "are", "with", "that", "this", "from", "have", "has",
    "was", "were", "will", "shall", "been", "they", "their", "there", "which",
    "into", "such", "also", "any", "all", "not", "but", "can", "may", "our",
    "its", "per", "you", "your", "about", "these", "those", "than", "then",
    "when", "what", "who", "whom", "how", "each", "other", "some", "more",
    "most", "over", "under", "between", "during", "after", "before", "above",
    "below", "shall", "upon", "within", "including", "page", "figure", "table",
}


def normalise_department(value: str | None) -> str:
    if not value:
        return UNASSIGNED_DEPARTMENT
    cleaned = value.strip()
    for dept in DEPARTMENTS:
        if cleaned.lower() == dept.lower():
            return dept
    # tolerate a few common spellings
    lookup = {
        "civil": "Civil / Track",
        "track": "Civil / Track",
        "civil/track": "Civil / Track",
        "signalling": "Signalling & Telecom",
        "signaling": "Signalling & Telecom",
        "s&t": "Signalling & Telecom",
        "telecom": "Signalling & Telecom",
        "human resources": "HR",
        "information technology": "IT",
        "stores": "Procurement",
        "purchase": "Procurement",
        "accounts": "Finance",
    }
    return lookup.get(cleaned.lower(), UNASSIGNED_DEPARTMENT)


def normalise_priority(value: str | None) -> str:
    if not value:
        return "Medium"
    cleaned = value.strip().lower()
    for priority in PRIORITIES:
        if cleaned == priority.lower():
            return priority
    if cleaned in {"urgent", "critical", "p1"}:
        return "High"
    if cleaned in {"normal", "moderate", "p2"}:
        return "Medium"
    if cleaned in {"minor", "informational", "p3"}:
        return "Low"
    return "Medium"


def normalise_document_type(value: str | None) -> str:
    if not value:
        return "Other"
    cleaned = value.strip()
    for doc_type in DOCUMENT_TYPES:
        if cleaned.lower() == doc_type.lower():
            return doc_type
    return cleaned[:60] if cleaned else "Other"


def normalise_status(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip().lower()
    if cleaned in STATUSES:
        return cleaned
    return STATUS_ALIASES.get(cleaned)
