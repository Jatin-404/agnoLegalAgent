from __future__ import annotations

import asyncio
import re

from app.core.agents import get_document_classifier_agent
from app.core.config import settings
from app.core.extraction_utils import _coerce_content, _first_non_empty_line, _normalize_quotes, _strip_markdown_formatting
from app.core.playbooks import compact_playbook_for_prompt, get_default_playbook, load_playbook_for_type, load_playbooks
from app.core.schemas import DocumentClassification


def _heuristic_classify_document(text: str, jurisdiction_hint: str) -> DocumentClassification:
    normalized = _strip_markdown_formatting(_normalize_quotes(text)).lower()
    first_line = _first_non_empty_line(normalized)
    best_score = -1
    best_playbook = get_default_playbook()
    matched_keywords: list[str] = []

    for playbook in load_playbooks():
        score = 0
        matches: list[str] = []

        contract_label = playbook.contract_type.lower()
        if contract_label in first_line:
            score += 8
            matches.append(playbook.contract_type)
        if contract_label in normalized:
            score += 5
            matches.append(playbook.contract_type)

        for alias in playbook.aliases:
            alias_lower = alias.lower()
            if alias_lower in normalized:
                score += 6 if alias_lower in first_line else 3
                matches.append(alias)

        for priority_term in playbook.priority_terms:
            if priority_term.lower() in normalized:
                score += 1
                matches.append(priority_term)

        if playbook.jurisdiction.lower() == jurisdiction_hint.lower():
            score += 1

        if score > best_score:
            best_score = score
            best_playbook = playbook
            matched_keywords = matches[:8]

    if best_score <= 0:
        confidence = 0.45
        playbook = get_default_playbook()
    else:
        playbook = best_playbook
        confidence = min(0.97, 0.50 + min(best_score, 10) * 0.045)

    return DocumentClassification(
        document_type=playbook.contract_type,
        normalized_type=playbook.contract_type,
        jurisdiction=jurisdiction_hint or playbook.jurisdiction,
        confidence=confidence,
        matched_playbook_id=playbook.playbook_id,
        matched_keywords=list(dict.fromkeys(matched_keywords)),
        reasoning="heuristic_playbook_match",
    )


async def classify_document(text: str, jurisdiction_hint: str = "India") -> DocumentClassification:
    heuristic = _heuristic_classify_document(text, jurisdiction_hint)
    if heuristic.confidence >= settings.classifier_confidence_threshold or not settings.enable_classifier_agent:
        return heuristic

    candidate_playbooks = [
        {
            "playbook_id": playbook.playbook_id,
            "contract_type": playbook.contract_type,
            "jurisdiction": playbook.jurisdiction,
            "aliases": playbook.aliases[:5],
        }
        for playbook in load_playbooks()
    ]
    prompt = (
        f"Jurisdiction hint: {jurisdiction_hint}\n"
        f"Candidate document families: {candidate_playbooks}\n\n"
        "Document preview:\n"
        f"{text[:3500]}"
    )

    try:
        result = await asyncio.wait_for(
            get_document_classifier_agent().arun(input=prompt),
            timeout=max(1, settings.classifier_agent_call_timeout_seconds),
        )
        classified = _coerce_content(result.content, DocumentClassification)
        playbook = load_playbook_for_type(classified.document_type or classified.normalized_type, classified.jurisdiction or jurisdiction_hint)
        classified.normalized_type = playbook.contract_type
        classified.document_type = playbook.contract_type
        classified.jurisdiction = classified.jurisdiction or jurisdiction_hint or playbook.jurisdiction
        classified.matched_playbook_id = playbook.playbook_id
        if not classified.matched_keywords:
            classified.matched_keywords = heuristic.matched_keywords
        if classified.confidence < heuristic.confidence:
            return heuristic
        return classified
    except Exception:
        return heuristic


def get_playbook_for_classification(classification: DocumentClassification):
    return load_playbook_for_type(classification.document_type, classification.jurisdiction)
