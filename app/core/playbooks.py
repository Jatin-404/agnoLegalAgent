from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.schemas import LegalExtraction, PlaybookDefinition

PLAYBOOKS_DIR = Path(__file__).resolve().parents[2] / "playbooks"


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


@lru_cache
def load_playbooks() -> list[PlaybookDefinition]:
    playbooks: list[PlaybookDefinition] = []
    if not PLAYBOOKS_DIR.exists():
        return playbooks

    for path in sorted(PLAYBOOKS_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        playbooks.append(PlaybookDefinition.model_validate(payload))
    return playbooks


@lru_cache
def get_playbook_index() -> dict[str, PlaybookDefinition]:
    index: dict[str, PlaybookDefinition] = {}
    for playbook in load_playbooks():
        index[playbook.playbook_id] = playbook
        index[_normalize_label(playbook.contract_type)] = playbook
        for alias in playbook.aliases:
            index[_normalize_label(alias)] = playbook
    return index


def get_supported_document_types() -> list[str]:
    labels = {playbook.contract_type for playbook in load_playbooks()}
    return sorted(labels)


def get_default_playbook() -> PlaybookDefinition:
    index = get_playbook_index()
    return index.get("default", next(iter(load_playbooks())))


def load_playbook_for_type(document_type: str, jurisdiction: str | None = None) -> PlaybookDefinition:
    normalized = _normalize_label(document_type)
    playbook = get_playbook_index().get(normalized)
    if playbook is None:
        playbook = get_default_playbook()

    if jurisdiction:
        jurisdiction_normalized = jurisdiction.strip().lower()
        candidates = [
            candidate
            for candidate in load_playbooks()
            if _normalize_label(candidate.contract_type) == _normalize_label(playbook.contract_type)
            and candidate.jurisdiction.strip().lower() in {jurisdiction_normalized, "global"}
        ]
        exact = next((candidate for candidate in candidates if candidate.jurisdiction.strip().lower() == jurisdiction_normalized), None)
        if exact is not None:
            return exact
    return playbook


def compact_playbook_for_prompt(playbook: PlaybookDefinition) -> dict[str, Any]:
    return {
        "playbook_id": playbook.playbook_id,
        "playbook_version": playbook.playbook_version,
        "contract_type": playbook.contract_type,
        "jurisdiction": playbook.jurisdiction,
        "fields": playbook.fields,
        "clause_taxonomy": playbook.clause_taxonomy,
        "risk_rules": playbook.risk_rules,
        "cross_ref_patterns": playbook.cross_ref_patterns,
        "priority_terms": playbook.priority_terms,
    }


def _field_present(extraction: LegalExtraction, field_name: str) -> bool:
    normalized = _normalize_label(field_name)

    if normalized in {"parties", "party"}:
        return bool(extraction.parties)
    if normalized in {"shareholders", "shareholder", "shareholding"}:
        return bool(extraction.shareholders) or any(clause.clause_type == "shareholding" for clause in extraction.key_clauses)
    if normalized in {"financial_terms", "financials", "payment_terms"}:
        return bool(extraction.financial_terms)
    if normalized in {"exhibits", "schedules", "annexures"}:
        return bool(extraction.exhibits)
    if normalized in {"relationships"}:
        return bool(extraction.relationships)
    if normalized in {"governing_law", "dispute_resolution"}:
        return any(clause.clause_type == "disputes_governing_law" for clause in extraction.key_clauses)
    if normalized in {"confidentiality", "non_disclosure"}:
        return any("confidential" in f"{clause.title} {clause.summary}".lower() for clause in extraction.key_clauses)
    if normalized in {"term", "termination", "exit"}:
        return any(normalized.split("_")[0] in f"{clause.title} {clause.summary}".lower() for clause in extraction.key_clauses)
    if normalized in {"governance", "board", "voting"}:
        return any(clause.clause_type == "governance" for clause in extraction.key_clauses)
    if normalized in {"transfer_restrictions", "share_transfer"}:
        return any(clause.clause_type in {"transfer_restrictions", "tag_along", "drag_along"} for clause in extraction.key_clauses)
    return any(normalized in _normalize_label(clause.title) for clause in extraction.key_clauses)


def calculate_playbook_coverage(extraction: LegalExtraction, playbook: PlaybookDefinition) -> tuple[list[str], list[str]]:
    extracted: list[str] = []
    missing: list[str] = []
    for field in playbook.fields:
        if _field_present(extraction, field):
            extracted.append(field)
        else:
            missing.append(field)
    return extracted, missing


def evaluate_playbook_risks(extraction: LegalExtraction, playbook: PlaybookDefinition) -> list[str]:
    flags: list[str] = []
    for rule in playbook.risk_rules:
        rule_type = _normalize_label(str(rule.get("type") or ""))
        message = str(rule.get("message") or "Playbook risk rule triggered.")
        if rule_type == "missing_field":
            field_name = str(rule.get("field") or "")
            if field_name and not _field_present(extraction, field_name):
                flags.append(message)
        elif rule_type == "missing_clause_type":
            clause_type = str(rule.get("clause_type") or "")
            if clause_type and not any(clause.clause_type == clause_type for clause in extraction.key_clauses):
                flags.append(message)
        elif rule_type == "shareholder_concentration":
            threshold = float(rule.get("threshold") or 0)
            if any((shareholder.percentage or 0) > threshold for shareholder in extraction.shareholders):
                flags.append(message)
        elif rule_type == "requires_exhibits" and not extraction.exhibits:
            flags.append(message)
        elif rule_type == "requires_financial_terms" and not extraction.financial_terms:
            flags.append(message)
    return list(dict.fromkeys(flags))
