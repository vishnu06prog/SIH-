"""The smart routing engine.

Decides which departments must see a document, with a confidence, a reason and
an urgency for each. Three inputs are combined:

1. the analyser's own routing proposal (LLM or rule-based),
2. admin-authored routing rules, which always win and are marked as such,
3. the uploader's declared department, which is never silently dropped.

Department heads and admins can override the result; the override is stored
alongside the AI decision rather than replacing it, so the audit trail keeps
both.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Department, Document, RoutingResult, RoutingRule, User
from ..taxonomy import PRIORITY_RANK, UNASSIGNED_DEPARTMENT, normalise_priority
from .analysis import DocumentAnalysis, RoutedDepartment

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    department_id: str
    department_name: str
    confidence: float
    reason: str
    urgency: str
    is_primary: bool = False
    engine: str = "ai"


def department_map(db: Session) -> dict[str, Department]:
    """Departments keyed by lower-cased name, for name → row resolution."""
    rows = db.execute(select(Department)).scalars().all()
    return {row.name.lower(): row for row in rows}


def resolve_department(db: Session, name: str | None) -> Department | None:
    if not name or name == UNASSIGNED_DEPARTMENT:
        return None
    return department_map(db).get(name.strip().lower())


def _apply_rules(
    db: Session, document: Document, text: str, existing: dict[str, RoutingDecision]
) -> dict[str, RoutingDecision]:
    """Admin rules override the AI: an explicit policy beats a guess."""
    rules = db.execute(
        select(RoutingRule).where(RoutingRule.is_active.is_(True))
    ).scalars().all()
    haystack = f"{document.title}\n{document.filename}\n{text}".lower()

    for rule in rules:
        keywords = rule.keyword_list()
        keyword_hit = any(keyword in haystack for keyword in keywords) if keywords else False
        type_hit = bool(rule.document_type) and rule.document_type == document.document_type
        if not (keyword_hit or type_hit):
            continue
        department = rule.department
        if department is None:
            continue
        matched = [keyword for keyword in keywords if keyword in haystack][:3]
        why = (
            f"Routing rule “{rule.name}” matched "
            + (
                f"keyword(s) {', '.join(repr(k) for k in matched)}"
                if matched
                else f"document type '{rule.document_type}'"
            )
            + "."
        )
        existing[department.id] = RoutingDecision(
            department_id=department.id,
            department_name=department.name,
            confidence=1.0,
            reason=why,
            urgency=normalise_priority(rule.urgency or document.priority),
            is_primary=existing.get(department.id, None) is not None
            and existing[department.id].is_primary,
            engine="rule",
        )
    return existing


def compute_routing(
    db: Session,
    document: Document,
    analysis: DocumentAnalysis,
    text: str = "",
) -> list[RoutingDecision]:
    """Merge the AI proposal, admin rules and the declared department."""
    departments = department_map(db)
    decisions: dict[str, RoutingDecision] = {}

    proposals: list[RoutedDepartment] = list(analysis.routing)
    if not proposals:
        # Analyser gave no explicit routing — derive it from the classification.
        if analysis.department != UNASSIGNED_DEPARTMENT:
            proposals.append(
                RoutedDepartment(
                    department=analysis.department,
                    confidence=analysis.confidence or 0.6,
                    reason="Primary owning department identified during classification.",
                    urgency=analysis.priority,
                    is_primary=True,
                )
            )
        for name in analysis.secondary_departments:
            proposals.append(
                RoutedDepartment(
                    department=name,
                    confidence=0.5,
                    reason="Named as an affected department in the analysis.",
                    urgency=analysis.priority,
                )
            )

    for proposal in proposals:
        department = departments.get(proposal.department.strip().lower())
        if department is None:
            continue
        decisions[department.id] = RoutingDecision(
            department_id=department.id,
            department_name=department.name,
            confidence=round(min(max(proposal.confidence, 0.0), 1.0), 2),
            reason=proposal.reason or "Identified by the document analyser.",
            urgency=normalise_priority(proposal.urgency or analysis.priority),
            is_primary=proposal.is_primary,
            engine="ai",
        )

    decisions = _apply_rules(db, document, text, decisions)

    # The uploader's declared department is a human signal; keep it in the loop.
    if document.declared_department_id and document.declared_department_id not in decisions:
        declared = db.get(Department, document.declared_department_id)
        if declared is not None:
            decisions[declared.id] = RoutingDecision(
                department_id=declared.id,
                department_name=declared.name,
                confidence=0.55,
                reason="Selected by the uploader at submission.",
                urgency=normalise_priority(analysis.priority),
                engine="declared",
            )

    ordered = sorted(
        decisions.values(),
        key=lambda item: (
            PRIORITY_RANK.get(item.urgency, 3),
            -item.confidence,
            item.department_name,
        ),
    )
    if ordered and not any(item.is_primary for item in ordered):
        ordered[0].is_primary = True
    elif ordered:
        # Exactly one primary, and it is the highest-confidence primary.
        primaries = [item for item in ordered if item.is_primary]
        for item in primaries[1:]:
            item.is_primary = False
    return ordered


def persist_routing(
    db: Session, document: Document, decisions: list[RoutingDecision]
) -> list[RoutingResult]:
    """Replace the document's AI routing, preserving human overrides."""
    overrides = {
        row.department_id: row
        for row in db.execute(
            select(RoutingResult).where(
                RoutingResult.document_id == document.id,
                RoutingResult.is_override.is_(True),
            )
        ).scalars()
    }

    db.query(RoutingResult).filter(
        RoutingResult.document_id == document.id,
        RoutingResult.is_override.is_(False),
    ).delete(synchronize_session=False)
    db.flush()

    rows: list[RoutingResult] = []
    for decision in decisions:
        if decision.department_id in overrides:
            rows.append(overrides[decision.department_id])
            continue
        row = RoutingResult(
            document_id=document.id,
            department_id=decision.department_id,
            confidence=decision.confidence,
            reason=decision.reason,
            urgency=decision.urgency,
            is_primary=decision.is_primary,
            engine=decision.engine,
        )
        db.add(row)
        rows.append(row)

    primary = next((d for d in decisions if d.is_primary), None)
    if primary and not any(row.is_override and row.is_primary for row in overrides.values()):
        document.department_id = primary.department_id
    db.flush()
    return rows


def override_routing(
    db: Session,
    document: Document,
    department_ids: list[str],
    user: User,
    reason: str = "",
    urgency: str | None = None,
) -> list[RoutingResult]:
    """Replace routing with a human decision, flagged as an override."""
    db.query(RoutingResult).filter(RoutingResult.document_id == document.id).delete(
        synchronize_session=False
    )
    db.flush()

    rows: list[RoutingResult] = []
    for position, department_id in enumerate(department_ids):
        department = db.get(Department, department_id)
        if department is None:
            continue
        row = RoutingResult(
            document_id=document.id,
            department_id=department.id,
            confidence=1.0,
            reason=reason or f"Routing set manually by {user.full_name or user.email}.",
            urgency=normalise_priority(urgency or document.priority),
            is_primary=position == 0,
            is_override=True,
            overridden_by_id=user.id,
            engine="human",
        )
        db.add(row)
        rows.append(row)

    if rows:
        document.department_id = rows[0].department_id
    db.flush()
    return rows


def routed_department_ids(db: Session, document_id: str) -> list[str]:
    return [
        row
        for row in db.execute(
            select(RoutingResult.department_id).where(RoutingResult.document_id == document_id)
        ).scalars()
    ]
