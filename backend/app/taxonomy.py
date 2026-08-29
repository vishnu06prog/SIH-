"""KMRL domain vocabulary shared by the analyser, routing engine, search layer and API."""

from __future__ import annotations

# ---------------------------------------------------------------- departments
DEPARTMENTS: list[str] = [
    "Operations",
    "Engineering",
    "Rolling Stock",
    "Electrical",
    "Signalling",
    "Procurement",
    "Finance",
    "Human Resources",
    "Safety",
    "Legal",
    "Environment",
    "Projects",
    "IT",
    "Administration",
    "Maintenance",
]

DEPARTMENT_CODES: dict[str, str] = {
    "Operations": "OPS",
    "Engineering": "ENG",
    "Rolling Stock": "RS",
    "Electrical": "ELE",
    "Signalling": "SIG",
    "Procurement": "PRC",
    "Finance": "FIN",
    "Human Resources": "HR",
    "Safety": "SAF",
    "Legal": "LEG",
    "Environment": "ENV",
    "Projects": "PRJ",
    "IT": "IT",
    "Administration": "ADM",
    "Maintenance": "MNT",
}

DEPARTMENT_DESCRIPTIONS: dict[str, str] = {
    "Operations": "Revenue service, timetabling, station and OCC operations.",
    "Engineering": "Civil, track, structures and permanent-way engineering.",
    "Rolling Stock": "Trainsets, bogies, propulsion, brakes and depot overhaul.",
    "Electrical": "Traction power, OHE, substations, UPS and station power.",
    "Signalling": "CBTC, interlocking, ATP/ATO, telecom and SCADA.",
    "Procurement": "Tenders, purchase orders, vendors and contract award.",
    "Finance": "Budgets, invoices, payments, taxation and audit.",
    "Human Resources": "Recruitment, payroll, training and employee relations.",
    "Safety": "Safety assurance, incidents, CMRS liaison and emergency response.",
    "Legal": "Contracts, litigation, arbitration and statutory opinions.",
    "Environment": "Environmental clearance, EIA, pollution control and green audit.",
    "Projects": "Corridor extensions, DPRs, construction and commissioning.",
    "IT": "Applications, ERP/Maximo, networks, data platform and cyber security.",
    "Administration": "Estate, general administration, records and facilities.",
    "Maintenance": "Station assets, escalators, lifts, HVAC and facility upkeep.",
}

UNASSIGNED_DEPARTMENT = "Unassigned"

# ---------------------------------------------------------------------- roles
ROLE_ADMIN = "ADMIN"
ROLE_EXECUTIVE = "EXECUTIVE"
ROLE_DEPARTMENT_HEAD = "DEPARTMENT_HEAD"
ROLE_MANAGER = "MANAGER"
ROLE_EMPLOYEE = "EMPLOYEE"

ROLES: list[str] = [
    ROLE_ADMIN,
    ROLE_EXECUTIVE,
    ROLE_DEPARTMENT_HEAD,
    ROLE_MANAGER,
    ROLE_EMPLOYEE,
]

ROLE_DESCRIPTIONS: dict[str, str] = {
    ROLE_ADMIN: "Full control: users, departments, routing rules, audit and settings.",
    ROLE_EXECUTIVE: "Organisation-wide intelligence, critical alerts and compliance view.",
    ROLE_DEPARTMENT_HEAD: "Owns their department's documents, approvals, routing and actions.",
    ROLE_MANAGER: "Works assigned actions, acknowledges alerts and searches the knowledge base.",
    ROLE_EMPLOYEE: "Uploads documents, searches, and works actions assigned to them.",
}

# Roles that may read every document irrespective of routing.
ORG_WIDE_ROLES = {ROLE_ADMIN, ROLE_EXECUTIVE}

# --------------------------------------------------------------- priorities
PRIORITIES: list[str] = ["Critical", "High", "Medium", "Low"]

PRIORITY_RANK: dict[str, int] = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}

CONFIDENTIALITY_LEVELS: list[str] = ["Public", "Internal", "Confidential", "Restricted"]

# ------------------------------------------------------------ document types
DOCUMENT_TYPES: list[str] = [
    "Safety Circular",
    "Regulatory Directive",
    "Incident Report",
    "Maintenance Report",
    "Engineering Document",
    "Procurement",
    "Invoice",
    "Purchase Order",
    "HR Policy",
    "Legal Document",
    "Environmental Report",
    "Board Meeting",
    "Project Document",
    "General Correspondence",
    "Other",
]

FILE_TYPES: list[str] = ["PDF", "DOCX", "TXT", "IMAGE"]

LANGUAGES: list[str] = ["English", "Malayalam", "Bilingual (EN/ML)", "Unknown"]

SOURCES: list[str] = [
    "Manual Upload",
    "Email",
    "SharePoint",
    "Maximo Export",
    "WhatsApp",
    "Cloud Link",
    "Scanner",
    "Vendor Portal",
    "Regulatory Body",
]

# ------------------------------------------------------- processing statuses
STATUS_UPLOADED = "UPLOADED"
STATUS_PROCESSING = "PROCESSING"
STATUS_OCR = "OCR"
STATUS_ANALYZING = "ANALYZING"
STATUS_INDEXING = "INDEXING"
STATUS_ROUTING = "ROUTING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"

STATUSES: list[str] = [
    STATUS_UPLOADED,
    STATUS_PROCESSING,
    STATUS_OCR,
    STATUS_ANALYZING,
    STATUS_INDEXING,
    STATUS_ROUTING,
    STATUS_COMPLETED,
    STATUS_FAILED,
]

STATUS_LABELS: dict[str, str] = {
    STATUS_UPLOADED: "Uploaded",
    STATUS_PROCESSING: "Processing",
    STATUS_OCR: "OCR",
    STATUS_ANALYZING: "Analyzing",
    STATUS_INDEXING: "Indexing",
    STATUS_ROUTING: "Routing",
    STATUS_COMPLETED: "Completed",
    STATUS_FAILED: "Failed",
}

# The ordered pipeline shown as a progress tracker in the UI.
PIPELINE_STAGES: list[str] = [
    STATUS_UPLOADED,
    STATUS_PROCESSING,
    STATUS_OCR,
    STATUS_ANALYZING,
    STATUS_INDEXING,
    STATUS_ROUTING,
    STATUS_COMPLETED,
]

STATUS_ALIASES: dict[str, str] = {
    "queued": STATUS_UPLOADED,
    "uploaded": STATUS_UPLOADED,
    "processing": STATUS_PROCESSING,
    "ocr": STATUS_OCR,
    "ocr processing": STATUS_OCR,
    "ocr_processing": STATUS_OCR,
    "analyzing": STATUS_ANALYZING,
    "analysing": STATUS_ANALYZING,
    "ai analysis": STATUS_ANALYZING,
    "indexing": STATUS_INDEXING,
    "routing": STATUS_ROUTING,
    "complete": STATUS_COMPLETED,
    "completed": STATUS_COMPLETED,
    "analysis complete": STATUS_COMPLETED,
    "failed": STATUS_FAILED,
}

# ---------------------------------------------------------- action lifecycle
ACTION_PENDING = "Pending"
ACTION_IN_PROGRESS = "In Progress"
ACTION_COMPLETED = "Completed"
ACTION_OVERDUE = "Overdue"
ACTION_CANCELLED = "Cancelled"

ACTION_STATUSES: list[str] = [
    ACTION_PENDING,
    ACTION_IN_PROGRESS,
    ACTION_COMPLETED,
    ACTION_OVERDUE,
    ACTION_CANCELLED,
]

OPEN_ACTION_STATUSES = {ACTION_PENDING, ACTION_IN_PROGRESS, ACTION_OVERDUE}

DEADLINE_TYPES: list[str] = [
    "Compliance",
    "Inspection",
    "Renewal",
    "Contract Expiry",
    "Response",
    "Payment",
    "Submission",
    "General",
]

RISK_SEVERITIES: list[str] = ["Critical", "High", "Medium", "Low"]

RISK_CATEGORIES: list[str] = [
    "Safety",
    "Regulatory",
    "Financial",
    "Operational",
    "Legal",
    "Environmental",
    "Reputational",
    "Technical",
]

ENTITY_TYPES: list[str] = [
    "Organisation",
    "Person",
    "Location",
    "Date",
    "Money",
    "Reference",
    "Asset",
    "Regulation",
    "Other",
]

# ------------------------------------------------- notification vocabulary
NOTIFY_CRITICAL_DOCUMENT = "CRITICAL_DOCUMENT"
NOTIFY_DOCUMENT_ROUTED = "DOCUMENT_ROUTED"
NOTIFY_ACTION_ASSIGNED = "ACTION_ASSIGNED"
NOTIFY_DEADLINE_APPROACHING = "DEADLINE_APPROACHING"
NOTIFY_DEADLINE_OVERDUE = "DEADLINE_OVERDUE"
NOTIFY_PROCESSING_COMPLETE = "PROCESSING_COMPLETE"
NOTIFY_DUPLICATE_DETECTED = "DUPLICATE_DETECTED"
NOTIFY_NEW_VERSION = "NEW_VERSION_DETECTED"

NOTIFICATION_TYPES: list[str] = [
    NOTIFY_CRITICAL_DOCUMENT,
    NOTIFY_DOCUMENT_ROUTED,
    NOTIFY_ACTION_ASSIGNED,
    NOTIFY_DEADLINE_APPROACHING,
    NOTIFY_DEADLINE_OVERDUE,
    NOTIFY_PROCESSING_COMPLETE,
    NOTIFY_DUPLICATE_DETECTED,
    NOTIFY_NEW_VERSION,
]

# ------------------------------------------------------------ audit actions
AUDIT_LOGIN = "USER_LOGIN"
AUDIT_UPLOAD = "DOCUMENT_UPLOADED"
AUDIT_VIEW = "DOCUMENT_VIEWED"
AUDIT_DOWNLOAD = "DOCUMENT_DOWNLOADED"
AUDIT_DELETE = "DOCUMENT_DELETED"
AUDIT_PROCESS = "DOCUMENT_PROCESSED"
AUDIT_REPROCESS = "DOCUMENT_REPROCESSED"
AUDIT_CLASSIFICATION_CHANGED = "CLASSIFICATION_CORRECTED"
AUDIT_SUMMARY_REGENERATED = "SUMMARY_REGENERATED"
AUDIT_ROUTING_CHANGED = "ROUTING_CHANGED"
AUDIT_ACTION_CHANGED = "ACTION_UPDATED"
AUDIT_ACTION_ASSIGNED = "ACTION_ASSIGNED"
AUDIT_REVIEWED = "DOCUMENT_REVIEWED"
AUDIT_USER_CHANGED = "USER_UPDATED"
AUDIT_USER_CREATED = "USER_CREATED"
AUDIT_SEARCH = "SEARCH_PERFORMED"
AUDIT_RAG_QUERY = "KNOWLEDGE_BASE_QUERY"
AUDIT_RULE_CHANGED = "ROUTING_RULE_CHANGED"

# ----------------------------- department detection signals (rule engine) ---
DEPARTMENT_KEYWORDS: dict[str, list[str]] = {
    "Operations": [
        "train operation", "timetable", "headway", "station controller", "ridership",
        "operations control centre", "occ", "punctuality", "revenue service",
        "train availability", "service disruption", "platform", "dwell time",
    ],
    "Engineering": [
        "viaduct", "track", "ballast", "alignment", "bridge", "civil works",
        "structural", "track geometry", "tamping", "rail weld", "girder",
        "design change", "as-built", "drawing no", "permanent way",
    ],
    "Rolling Stock": [
        "rolling stock", "bogie", "traction motor", "trainset", "coach", "brake",
        "emergency brake", "overhaul", "wheel", "pantograph", "propulsion",
        "car body", "train set", "brake pad", "depot maintenance",
    ],
    "Electrical": [
        "traction power", "ohe", "overhead equipment", "substation", "33kv",
        "transformer", "ups", "power supply", "earthing", "breaker",
        "electrical isolation", "auxiliary supply",
    ],
    "Signalling": [
        "signalling", "signaling", "cbtc", "interlocking", "telecom", "atp",
        "ato", "axle counter", "scada", "radio", "fibre optic", "fiber optic",
        "train detection", "point machine",
    ],
    "Procurement": [
        "tender", "purchase order", "vendor", "supplier", "quotation", "rfp",
        "bid", "contract award", "spare parts", "procurement", "eoi", "emd",
        "delivery schedule", "nit", "supply order",
    ],
    "Finance": [
        "invoice", "payment", "budget", "expenditure", "gst", "financial",
        "cash flow", "accounts payable", "reimbursement", "fare revenue",
        "cost centre", "tax invoice", "release payment", "gstin",
    ],
    "Human Resources": [
        "employee", "staff", "recruitment", "leave policy", "payroll",
        "appraisal", "human resources", "attendance", "transfer order",
        "refresher training", "manpower", "code of conduct", "training programme",
    ],
    "Safety": [
        "safety", "incident", "accident", "hazard", "near miss", "cmrs",
        "commissioner of metro rail safety", "risk assessment", "emergency",
        "fire", "evacuation", "safety circular", "ppe", "safety audit",
        "mock drill", "safety critical",
    ],
    "Legal": [
        "legal", "litigation", "court", "arbitration", "agreement", "clause",
        "affidavit", "counsel", "writ petition", "indemnity",
        "memorandum of understanding", "legal opinion", "breach of contract",
    ],
    "Environment": [
        "environmental", "eia", "environmental impact", "pollution control",
        "green audit", "effluent", "emission", "noise level", "consent to operate",
        "kerala state pollution control board", "waste management", "biodiversity",
    ],
    "Projects": [
        "project", "corridor", "phase ii", "phase 2", "extension", "milestone",
        "dpr", "detailed project report", "construction", "commissioning",
        "new depot", "work package", "project schedule",
    ],
    "IT": [
        "software", "application", "database", "server", "network", "cyber",
        "erp", "sharepoint", "maximo", "data analytics", "api",
        "system integration", "iot", "dashboard", "access control system",
    ],
    "Administration": [
        "office order", "circular", "estate", "facility management", "records",
        "stationery", "general administration", "housekeeping", "notice board",
        "administrative approval",
    ],
    "Maintenance": [
        "escalator", "lift", "elevator", "hvac", "preventive maintenance",
        "job card", "work order", "breakdown maintenance", "maintenance schedule",
        "platform screen door", "psd", "asset condition", "shutdown maintenance",
    ],
}

DOCUMENT_TYPE_KEYWORDS: dict[str, list[str]] = {
    "Safety Circular": [
        "safety circular", "safety bulletin", "safety advisory", "safety notice",
    ],
    "Regulatory Directive": [
        "commissioner of metro rail safety", "cmrs", "ministry of housing", "mohua",
        "directive", "gazette", "statutory", "regulation", "show cause",
        "metro railways act", "compliance report to be submitted",
    ],
    "Incident Report": [
        "incident report", "near miss", "accident report", "occurrence report",
        "root cause", "incident no",
    ],
    "Maintenance Report": [
        "maintenance report", "job card", "work order", "maximo", "breakdown",
        "preventive maintenance", "condition monitoring", "servicing record",
    ],
    "Engineering Document": [
        "drawing no", "general arrangement", "as-built", "design change notice",
        "specification", "revision", "technical specification", "scale 1:",
    ],
    "Procurement": [
        "tender document", "nit", "notice inviting tender", "scope of work",
        "bid document", "eoi", "quotation", "contract agreement",
    ],
    "Invoice": [
        "invoice", "tax invoice", "bill no", "amount payable", "gstin",
        "invoice no", "total payable",
    ],
    "Purchase Order": [
        "purchase order", "po number", "po no", "supply order", "order value",
    ],
    "HR Policy": [
        "hr policy", "leave policy", "code of conduct", "employee handbook",
        "hr circular", "training calendar", "appraisal policy", "staff circular",
    ],
    "Legal Document": [
        "legal opinion", "counsel opinion", "arbitration award", "writ petition",
        "legal notice", "memorandum of understanding", "indemnity clause",
    ],
    "Environmental Report": [
        "environmental impact", "eia", "pollution control", "green audit",
        "consent to operate", "environmental compliance", "emission monitoring",
    ],
    "Board Meeting": [
        "minutes of the meeting", "board meeting", "agenda item", "resolved that",
        "board of directors", "board resolution",
    ],
    "Project Document": [
        "detailed project report", "dpr", "project milestone", "work package",
        "project status report", "commissioning plan",
    ],
    "General Correspondence": [
        "dear sir", "with reference to", "yours faithfully", "subject:",
        "for information", "station notice", "office memorandum",
    ],
}

CRITICAL_PRIORITY_SIGNALS = [
    "safety critical", "immediate compliance", "show cause", "emergency",
    "accident", "fatality", "derailment", "statutory deadline", "shutdown",
    "within 24 hours", "stop operations", "grounding", "suspension of service",
    "non-compliance may", "penalty will be", "mandatory inspection",
]

HIGH_PRIORITY_SIGNALS = [
    "urgent", "immediate", "immediately", "critical", "non-compliance",
    "penalty", "deadline", "escalate", "failure", "statutory", "mandatory",
    "within 48 hours", "top priority", "must be completed", "cmrs",
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
    "advised to", "required to", "will conduct", "shall conduct", "to inspect",
]

RISK_SIGNALS = [
    "risk", "penalty", "non-compliance", "non compliance", "delay", "failure",
    "hazard", "breach", "litigation", "shortfall", "defect", "unsafe",
    "escalation", "shutdown", "liability", "overrun", "exposure", "may result in",
]

COMPLIANCE_SIGNALS = [
    "commissioner of metro rail safety", "cmrs", "ministry of housing",
    "mohua", "statutory", "regulation", "audit", "gazette", "compliance",
    "mandatory", "directive", "circular", "gst", "rti", "pollution control board",
]

STOPWORDS = {
    "the", "and", "for", "are", "with", "that", "this", "from", "have", "has",
    "was", "were", "will", "shall", "been", "they", "their", "there", "which",
    "into", "such", "also", "any", "all", "not", "but", "can", "may", "our",
    "its", "per", "you", "your", "about", "these", "those", "than", "then",
    "when", "what", "who", "whom", "how", "each", "other", "some", "more",
    "most", "over", "under", "between", "during", "after", "before", "above",
    "below", "upon", "within", "including", "page", "figure", "table", "shall",
}

# --------------------------------------------------------------- normalisers
_DEPARTMENT_ALIASES: dict[str, str] = {
    "civil": "Engineering",
    "civil / track": "Engineering",
    "civil/track": "Engineering",
    "track": "Engineering",
    "permanent way": "Engineering",
    "signalling & telecom": "Signalling",
    "signaling": "Signalling",
    "s&t": "Signalling",
    "telecom": "Signalling",
    "hr": "Human Resources",
    "human resource": "Human Resources",
    "personnel": "Human Resources",
    "information technology": "IT",
    "stores": "Procurement",
    "purchase": "Procurement",
    "purchasing": "Procurement",
    "accounts": "Finance",
    "finance & accounts": "Finance",
    "safety & security": "Safety",
    "environmental": "Environment",
    "admin": "Administration",
    "general administration": "Administration",
    "e&m": "Electrical",
    "electrical & mechanical": "Electrical",
    "o&m": "Maintenance",
    "facility maintenance": "Maintenance",
    "rolling-stock": "Rolling Stock",
}


def normalise_department(value: str | None) -> str:
    if not value:
        return UNASSIGNED_DEPARTMENT
    cleaned = value.strip()
    for dept in DEPARTMENTS:
        if cleaned.lower() == dept.lower():
            return dept
    return _DEPARTMENT_ALIASES.get(cleaned.lower(), UNASSIGNED_DEPARTMENT)


def normalise_priority(value: str | None) -> str:
    if not value:
        return "Medium"
    cleaned = value.strip().lower()
    for priority in PRIORITIES:
        if cleaned == priority.lower():
            return priority
    if cleaned in {"urgent", "p0", "p1", "severe", "blocker"}:
        return "Critical"
    if cleaned in {"important", "major"}:
        return "High"
    if cleaned in {"normal", "moderate", "p2"}:
        return "Medium"
    if cleaned in {"minor", "informational", "info", "p3", "p4"}:
        return "Low"
    return "Medium"


def normalise_document_type(value: str | None) -> str:
    if not value:
        return "Other"
    cleaned = value.strip()
    for doc_type in DOCUMENT_TYPES:
        if cleaned.lower() == doc_type.lower():
            return doc_type
    aliases = {
        "safety bulletin": "Safety Circular",
        "regulatory notice": "Regulatory Directive",
        "directive": "Regulatory Directive",
        "job card": "Maintenance Report",
        "maintenance job card": "Maintenance Report",
        "engineering drawing": "Engineering Document",
        "technical report": "Engineering Document",
        "tender": "Procurement",
        "tender / contract": "Procurement",
        "contract": "Procurement",
        "legal opinion": "Legal Document",
        "environmental study": "Environmental Report",
        "board minutes": "Board Meeting",
        "correspondence": "General Correspondence",
        "email": "General Correspondence",
        "training material": "HR Policy",
    }
    return aliases.get(cleaned.lower(), "Other")


def normalise_status(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip()
    if cleaned.upper() in STATUSES:
        return cleaned.upper()
    return STATUS_ALIASES.get(cleaned.lower())


def normalise_role(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip().upper().replace(" ", "_").replace("-", "_")
    return cleaned if cleaned in ROLES else None


def normalise_action_status(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip().lower().replace("_", " ")
    for status in ACTION_STATUSES:
        if cleaned == status.lower():
            return status
    return None


def normalise_confidentiality(value: str | None) -> str:
    if not value:
        return "Internal"
    cleaned = value.strip().lower()
    for level in CONFIDENTIALITY_LEVELS:
        if cleaned == level.lower():
            return level
    return "Internal"


def normalise_language(value: str | None) -> str:
    if not value:
        return "Unknown"
    cleaned = value.strip().lower()
    mapping = {
        "english": "English",
        "en": "English",
        "malayalam": "Malayalam",
        "ml": "Malayalam",
        "bilingual": "Bilingual (EN/ML)",
        "bilingual (en/ml)": "Bilingual (EN/ML)",
        "en/ml": "Bilingual (EN/ML)",
    }
    return mapping.get(cleaned, "Unknown")
