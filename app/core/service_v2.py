from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from time import perf_counter
from typing import Any, Callable, Type, TypeVar

from agno.knowledge.document import Document
from langsmith import traceable
from pydantic import BaseModel

from app.core.agents import (
    get_v2_exhibit_agent,
    get_v2_financial_agent,
    get_v2_merge_agent,
    get_v2_relationship_agent,
    get_v2_section_agent,
    get_v2_taxonomy_agent,
)
from app.core.classifier import classify_document, get_playbook_for_classification
from app.core.config import settings
from app.core.document_loader import chunk_legal_document
from app.core.extraction_utils import (
    _clean_snippet,
    _coerce_content,
    _first_non_empty_line,
    _heuristic_clauses,
    _heuristic_entities,
    _heuristic_financial_terms,
    _heuristic_shareholdings,
    _normalize_quotes,
    _strip_markdown_formatting,
)
from app.core.graph_utils import build_document_graph, extract_cross_references
from app.core.observability import legal_tracing_context
from app.core.playbooks import calculate_playbook_coverage, compact_playbook_for_prompt, evaluate_playbook_risks
from app.core.schemas import (
    Clause,
    ClauseTaxonomy,
    ClauseTaxonomyResult,
    DocumentClassification,
    Exhibit,
    ExhibitsExtraction,
    FinancialTermV2,
    FinancialTermsExtraction,
    FlexibleEntity,
    LegalExtractionV2,
    PlaybookDefinition,
    RelationshipsExtraction,
    SectionExtractionV2,
    SemanticRelationship,
    Shareholder,
    SingleExtractionRequest,
    SingleExtractionResponseV2,
)

ModelT = TypeVar("ModelT", bound=BaseModel)
ItemT = TypeVar("ItemT")


def _safe_percentage_from_text(text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    return float(match.group(1)) if match else None


def _infer_contract_type(text: str) -> str:
    normalized = _strip_markdown_formatting(_normalize_quotes(text)).lower()
    patterns = [
        ("shareholders", "Shareholders Agreement"),
        ("lease", "Lease Agreement"),
        ("non-disclosure", "NDA"),
        ("nda", "NDA"),
        ("employment", "Employment Agreement"),
        ("services", "Service Agreement"),
        ("loan", "Loan Agreement"),
        ("policy", "Policy"),
    ]
    for keyword, label in patterns:
        if keyword in normalized:
            return label
    first_line = _first_non_empty_line(normalized).strip()
    return first_line.title()[:80] if first_line else "Legal Document"


def _detect_drafting_style(text: str) -> str:
    normalized = _strip_markdown_formatting(_normalize_quotes(text))
    lowered = normalized.lower()
    signals = []
    if "hereinafter referred to as" in lowered:
        signals.append("defined_terms")
    if "## " in text or "**1." in text or re.search(r"(?m)^#+\s+", text):
        signals.append("sectioned")
    if re.search(r"(?m)^\|.+\|$", text):
        signals.append("table_heavy")
    if "whereas" in lowered or "now therefore" in lowered:
        signals.append("formal_recitals")
    return "+".join(signals) if signals else "plain_legal"


def _map_to_v2_taxonomy(title: str, summary: str = "") -> ClauseTaxonomy:
    title_text = title.lower()
    text = f"{title} {summary}".lower()

    title_checks: list[tuple[list[str], ClauseTaxonomy]] = [
        (["competition restriction", "non compete", "non-compete"], "non_compete"),
        (["tag-along", "tag along"], "tag_along"),
        (["drag-along", "drag along"], "drag_along"),
        (["bad leaver", "good leaver", "abnormal exit"], "bad_leaver"),
        (["transfer", "right of first refusal", "share disposal", "assignability"], "transfer_restrictions"),
        (["exit", "termination"], "exit"),
        (["dispute", "governing law", "jurisdiction", "arbitration"], "disputes_governing_law"),
        (["board", "director", "governance", "proceedings", "voting", "reserved matters"], "governance"),
        (["shareholding", "ownership of the shares", "share capital", "equity shares"], "shareholding"),
        (["financial", "interest", "revenue", "payment", "costs", "capital", "valuation", "esop"], "financial"),
        (["breach", "default", "penalty", "damages", "waiver"], "penalties"),
    ]
    for keywords, taxonomy in title_checks:
        if any(keyword in title_text for keyword in keywords):
            return taxonomy

    body_checks: list[tuple[list[str], ClauseTaxonomy]] = [
        (["competition restriction", "non compete", "non-compete"], "non_compete"),
        (["tag-along", "tag along"], "tag_along"),
        (["drag-along", "drag along"], "drag_along"),
        (["bad leaver", "good leaver", "abnormal exit"], "bad_leaver"),
        (["transfer", "right of first refusal", "share disposal", "assignability"], "transfer_restrictions"),
        (["liquidation event", "trade sale", "ipo"], "exit"),
        (["dispute", "governing law", "jurisdiction", "arbitration"], "disputes_governing_law"),
        (["board", "director", "governance", "proceedings", "voting", "reserved matters"], "governance"),
        (["shareholding", "ownership of the shares", "share capital", "equity shares"], "shareholding"),
        (["financial", "interest", "revenue", "payment", "costs", "capital", "valuation", "esop"], "financial"),
        (["breach", "default", "penalty", "damages", "waiver"], "penalties"),
    ]
    for keywords, taxonomy in body_checks:
        if any(keyword in text for keyword in keywords):
            return taxonomy
    return "other"


def _to_v2_parties(text: str) -> list[FlexibleEntity]:
    return _heuristic_entities(text)


def _to_v2_shareholders(text: str, parties: list[FlexibleEntity]) -> list[Shareholder]:
    return _heuristic_shareholdings(text, parties)


def _to_v2_clauses(text: str) -> list[Clause]:
    clauses = []
    for index, clause in enumerate(_heuristic_clauses(text), start=1):
        clauses.append(
            Clause(
                clause_id=f"C{index}",
                clause_type=_map_to_v2_taxonomy(clause.title, clause.summary),
                title=clause.title,
                summary=clause.summary,
                source_text=clause.source_text,
                importance=clause.importance,
                confidence=clause.confidence,
            )
        )
    return clauses


def _to_v2_financial_terms(text: str) -> list[FinancialTermV2]:
    terms: list[FinancialTermV2] = []
    for term in _heuristic_financial_terms(text):
        amount = f"{term.currency} {int(term.amount):,}" if term.amount is not None and term.currency else None
        description = term.label.replace("_", " ")
        trigger = "when stated in the document"
        terms.append(
            FinancialTermV2(
                amount=amount,
                currency=term.currency or "INR",
                percentage=term.rate,
                description=description,
                trigger_condition=trigger,
                source_text=term.source_text,
                confidence=term.confidence,
            )
        )
    return terms


def _extract_exhibits_heuristically(text: str) -> list[Exhibit]:
    normalized = _strip_markdown_formatting(_normalize_quotes(text))
    blocks = re.findall(
        r"(?ims)^(?:#+\s+)?(?P<title>(?:Exhibit|Annexure|Schedule)\s+[A-Z0-9]+[^\n]*)\n(?P<body>.*?)(?=^(?:#+\s+)?(?:Exhibit|Annexure|Schedule)\s+[A-Z0-9]+|\Z)",
        normalized,
    )
    exhibits: list[Exhibit] = []
    for title, body in blocks:
        exhibit_id_match = re.match(r"(?i)(Exhibit|Annexure|Schedule)\s+([A-Z0-9]+)", title.strip())
        exhibit_id = f"{exhibit_id_match.group(1).title()} {exhibit_id_match.group(2)}" if exhibit_id_match else title.strip()
        linked_financials = _to_v2_financial_terms(body)
        exhibits.append(
            Exhibit(
                exhibit_id=exhibit_id,
                title=title.strip(),
                summary=_clean_snippet(body, 240),
                key_content=_clean_snippet(body, 400),
                linked_financials=linked_financials,
                source_text=_clean_snippet(f"{title}\n{body}", 500),
                confidence=0.7,
            )
        )
    return exhibits


def _infer_relationships_heuristically(
    parties: list[FlexibleEntity],
    shareholders: list[Shareholder],
    clauses: list[Clause],
    financial_terms: list[FinancialTermV2],
) -> list[SemanticRelationship]:
    relationships: list[SemanticRelationship] = []
    company = next((party.name for party in parties if party.entity_type == "organization"), None)

    for shareholder in shareholders:
        if company:
            relationships.append(
                SemanticRelationship(
                    source_entity=shareholder.name,
                    target_entity=company,
                    relationship_type="role",
                    condition="while shareholding remains effective",
                    impact=f"holds {shareholder.shares or 'unspecified'} shares",
                    source_text=shareholder.source_text,
                    confidence=0.78,
                )
            )

    for clause in clauses:
        if clause.clause_type in {"tag_along", "drag_along", "bad_leaver"} and shareholders:
            relationships.append(
                SemanticRelationship(
                    source_entity=shareholders[0].name,
                    target_entity=company or "Company",
                    relationship_type=clause.clause_type if clause.clause_type in {"tag_along", "drag_along", "bad_leaver"} else "other",
                    condition=clause.summary,
                    impact=clause.title,
                    source_text=clause.source_text,
                    confidence=0.68,
                )
            )

    for term in financial_terms:
        if term.linked_entity:
            target = company or (parties[0].name if parties else "Company")
            for entity in term.linked_entity:
                relationships.append(
                    SemanticRelationship(
                        source_entity=entity,
                        target_entity=target,
                        relationship_type="incentive_link",
                        condition=term.trigger_condition,
                        impact=term.description,
                        source_text=term.source_text,
                        confidence=0.66,
                    )
                )

    deduped: list[SemanticRelationship] = []
    seen: set[tuple[str, str, str, str]] = set()
    for rel in relationships:
        key = (rel.source_entity, rel.target_entity, rel.relationship_type, rel.impact)
        if key not in seen:
            seen.add(key)
            deduped.append(rel)
    return deduped


def _dedupe_by_key(items: list[ItemT], key_fn: Callable[[ItemT], Any]) -> list[ItemT]:
    deduped: list[ItemT] = []
    seen = set()
    for item in items:
        key = key_fn(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _calculate_completeness(
    *,
    text: str,
    parties: list[FlexibleEntity],
    shareholders: list[Shareholder],
    financial_terms: list[FinancialTermV2],
    exhibits: list[Exhibit],
    clauses: list[Clause],
    relationships: list[SemanticRelationship],
) -> int:
    score = 0
    normalized = _strip_markdown_formatting(_normalize_quotes(text)).lower()

    if parties:
        score += 20
    if clauses:
        score += 20
    if shareholders or "shareholder" not in normalized:
        score += 15
    if financial_terms or not re.search(r"\binr\b|rs\.|%|revenue|interest|capital|valuation|esop", normalized):
        score += 15
    if exhibits or "exhibit" not in normalized:
        score += 10
    if relationships:
        score += 10
    if any(clause.clause_type == "disputes_governing_law" for clause in clauses):
        score += 5
    if any(clause.clause_type in {"governance", "shareholding", "exit"} for clause in clauses):
        score += 5

    return max(0, min(100, score))


def _ensure_v2_short_summary(extraction: LegalExtractionV2, full_text: str) -> str:
    current = _strip_markdown_formatting((extraction.short_summary or "")).strip()
    if current:
        normalized = re.sub(r"\s+", " ", current).strip()
        looks_like_title = len(normalized.split()) <= 4 and normalized.upper() == normalized
        if not looks_like_title:
            return _clean_snippet(normalized, 260)

    parties = ", ".join(entity.name for entity in extraction.parties[:4] if entity.name)
    clause_titles = ", ".join(clause.title for clause in extraction.key_clauses[:3] if clause.title)
    if parties and clause_titles:
        return _clean_snippet(
            f"{extraction.contract_type} involving {parties}. Key sections include {clause_titles}.",
            260,
        )
    if parties:
        return _clean_snippet(f"{extraction.contract_type} involving {parties}.", 260)
    if clause_titles:
        return _clean_snippet(f"{extraction.contract_type} covering {clause_titles}.", 260)
    return _clean_snippet(_first_non_empty_line(_strip_markdown_formatting(full_text)) or extraction.contract_type, 260)


def _playbook_prompt_payload(classification: DocumentClassification, playbook: PlaybookDefinition) -> str:
    payload = {
        "classification": classification.model_dump(),
        "playbook": compact_playbook_for_prompt(playbook),
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)


def _default_routing_context(contract_type_hint: str, jurisdiction_hint: str = "India") -> tuple[DocumentClassification, PlaybookDefinition]:
    classification = DocumentClassification(
        document_type=contract_type_hint,
        normalized_type=contract_type_hint,
        jurisdiction=jurisdiction_hint,
        confidence=0.75,
        matched_playbook_id=None,
        matched_keywords=[],
        reasoning="default_test_context",
    )
    playbook = get_playbook_for_classification(classification)
    classification.matched_playbook_id = playbook.playbook_id
    return classification, playbook


def _apply_playbook_postprocessing(
    *,
    extraction: LegalExtractionV2,
    classification: DocumentClassification,
    playbook: PlaybookDefinition,
    full_text: str,
    truncated: bool,
) -> LegalExtractionV2:
    extraction.document_type = extraction.contract_type or classification.document_type
    extraction.contract_type = extraction.contract_type or classification.document_type
    extraction.jurisdiction = classification.jurisdiction or playbook.jurisdiction
    extraction.playbook_id = playbook.playbook_id
    extraction.playbook_version = playbook.playbook_version
    extraction.classifier_confidence = classification.confidence

    extracted_fields, missing_fields = calculate_playbook_coverage(extraction, playbook)
    extraction.playbook_fields_expected = playbook.fields
    extraction.playbook_fields_extracted = extracted_fields
    extraction.playbook_fields_missing = missing_fields

    coverage_ratio = (len(extracted_fields) / len(playbook.fields)) if playbook.fields else 1.0
    extraction.playbook_confidence = round(min(1.0, (classification.confidence * 0.55) + (coverage_ratio * 0.45)), 4)

    playbook_risks = evaluate_playbook_risks(extraction, playbook)
    extraction.red_flags = list(dict.fromkeys([*extraction.red_flags, *playbook_risks]))

    extraction.cross_references = extract_cross_references(
        full_text=full_text,
        clauses=extraction.key_clauses,
        exhibits=extraction.exhibits,
        playbook=playbook,
    )
    extraction.graph_nodes, extraction.graph_edges = build_document_graph(extraction)

    weighted_completeness = int((extraction.completeness_score * 0.7) + (coverage_ratio * 100 * 0.3))
    extraction.completeness_score = max(0, min(100, weighted_completeness))
    extraction.needs_human_review = bool(
        truncated
        or extraction.classifier_confidence < 0.58
        or extraction.playbook_confidence < 0.6
        or len(extraction.red_flags) > 0
        or len(extraction.playbook_fields_missing) > max(1, len(playbook.fields) // 2)
    )
    return extraction


def _heuristic_section_extraction(*, chunk: Document, contract_type_hint: str) -> SectionExtractionV2:
    parties = _to_v2_parties(chunk.content)
    shareholders = _to_v2_shareholders(chunk.content, parties)
    clauses = _to_v2_clauses(chunk.content)
    summary_seed = _first_non_empty_line(chunk.content) or chunk.content
    return SectionExtractionV2(
        contract_type=contract_type_hint,
        parties=parties,
        shareholders=shareholders,
        key_clauses=clauses,
        red_flags=[],
        drafting_style_detected=_detect_drafting_style(chunk.content),
        short_summary=_clean_snippet(summary_seed, 220),
    )


def _merge_section_result(base: SectionExtractionV2, candidate: SectionExtractionV2) -> SectionExtractionV2:
    parties = _dedupe_by_key(
        [*base.parties, *candidate.parties],
        lambda item: ((item.name or "").lower(), (item.defined_term or "").lower(), item.entity_type, (item.role or "").lower()),
    )
    shareholders = _dedupe_by_key(
        [*base.shareholders, *candidate.shareholders],
        lambda item: ((item.name or "").lower(), (item.defined_term or "").lower(), item.shares, item.percentage),
    )
    clauses = _dedupe_by_key(
        [*base.key_clauses, *candidate.key_clauses],
        lambda item: ((item.title or "").lower(), (item.source_text or "").lower()),
    )
    return SectionExtractionV2(
        contract_type=candidate.contract_type or base.contract_type,
        parties=parties,
        shareholders=shareholders,
        key_clauses=clauses,
        red_flags=list(dict.fromkeys([*base.red_flags, *candidate.red_flags])),
        drafting_style_detected=candidate.drafting_style_detected or base.drafting_style_detected,
        short_summary=candidate.short_summary or base.short_summary,
    )


def _merge_financial_collections(
    primary: list[FinancialTermV2],
    secondary: list[FinancialTermV2],
) -> list[FinancialTermV2]:
    return _dedupe_by_key(
        [*primary, *secondary],
        lambda item: ((item.description or "").lower(), item.amount, item.percentage, (item.trigger_condition or "").lower()),
    )


def _merge_exhibit_collections(primary: list[Exhibit], secondary: list[Exhibit]) -> list[Exhibit]:
    return _dedupe_by_key(
        [*primary, *secondary],
        lambda item: ((item.exhibit_id or "").lower(), (item.title or "").lower()),
    )


def _merge_relationship_collections(
    primary: list[SemanticRelationship],
    secondary: list[SemanticRelationship],
) -> list[SemanticRelationship]:
    return _dedupe_by_key(
        [*primary, *secondary],
        lambda item: (item.source_entity, item.target_entity, item.relationship_type, item.impact, item.condition),
    )


def _chunk_llm_priority(chunk: Document, playbook: PlaybookDefinition) -> tuple[int, int]:
    meta = dict(chunk.meta_data or {})
    text = chunk.content.lower()
    score = 0

    if meta.get("chunk_type") == "table":
        score += 5
    if meta.get("section_title") not in {None, "", "Preamble"}:
        score += 2

    keyword_hits = sum(
        keyword in text
        for keyword in (
            "shall",
            "share",
            "director",
            "board",
            "transfer",
            "termination",
            "governing law",
            "arbitration",
            "interest",
            "payment",
            "valuation",
            "confidential",
        )
    )
    score += min(6, keyword_hits)
    score += sum(2 for term in playbook.priority_terms if term.lower() in text)
    for pattern in playbook.cross_ref_patterns:
        try:
            if re.search(pattern, text):
                score += 2
        except re.error:
            continue
    score += min(3, max(0, len(chunk.content) // 2500))
    return score, -int(meta.get("chunk_id") or 0)


async def _run_agent_with_retry(*, agent: Any, payload: Any, schema: Type[ModelT]) -> ModelT:
    last_error: Exception | None = None
    attempts = max(1, settings.structured_retries)
    for _ in range(attempts):
        try:
            result = await asyncio.wait_for(
                agent.arun(input=payload),
                timeout=max(1, settings.v2_agent_call_timeout_seconds),
            )
            return _coerce_content(result.content, schema)
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError(f"Structured extraction failed for {schema.__name__}")


@traceable(run_type="chain", name="legal_v2_section_extraction")
async def _extract_section(
    *,
    chunk: Document,
    contract_type_hint: str,
    classification: DocumentClassification | None = None,
    playbook: PlaybookDefinition | None = None,
    use_model: bool = True,
) -> SectionExtractionV2:
    classification = classification or _default_routing_context(contract_type_hint)[0]
    playbook = playbook or get_playbook_for_classification(classification)
    baseline = _heuristic_section_extraction(chunk=chunk, contract_type_hint=contract_type_hint)
    if not use_model or not settings.v2_enable_section_agent:
        return baseline

    prompt = (
        f"Document routing context:\n{_playbook_prompt_payload(classification, playbook)}\n\n"
        f"Contract type hint: {contract_type_hint}\n"
        f"Section title: {(chunk.meta_data or {}).get('section_title', 'Unknown')}\n"
        f"Chunk type: {(chunk.meta_data or {}).get('chunk_type', 'text')}\n\n"
        f"{chunk.content}"
    )
    try:
        candidate = await _run_agent_with_retry(
            agent=get_v2_section_agent(),
            payload=prompt,
            schema=SectionExtractionV2,
        )
        return _merge_section_result(baseline, candidate)
    except Exception:
        return baseline


@traceable(run_type="chain", name="legal_v2_financial_pass")
async def _extract_financials(
    texts: list[str],
    classification: DocumentClassification,
    playbook: PlaybookDefinition,
) -> list[FinancialTermV2]:
    baseline = _dedupe_by_key(
        [item for text in texts for item in _to_v2_financial_terms(text)],
        lambda item: ((item.description or "").lower(), item.amount, item.percentage, (item.trigger_condition or "").lower()),
    )
    if not settings.v2_enable_financial_agent:
        return baseline

    semaphore = asyncio.Semaphore(max(1, settings.v2_chunk_concurrency))

    async def run_text(text: str) -> list[FinancialTermV2]:
        async with semaphore:
            try:
                extraction = await _run_agent_with_retry(
                    agent=get_v2_financial_agent(),
                    payload=(
                        f"Document routing context:\n{_playbook_prompt_payload(classification, playbook)}\n\n"
                        f"Text:\n{text}"
                    ),
                    schema=FinancialTermsExtraction,
                )
                return extraction.items
            except Exception:
                return []

    results = await asyncio.gather(*(run_text(text) for text in texts))
    agent_terms = [item for group in results for item in group]
    return _merge_financial_collections(baseline, agent_terms)


@traceable(run_type="chain", name="legal_v2_exhibit_pass")
async def _extract_exhibits(
    texts: list[str],
    classification: DocumentClassification,
    playbook: PlaybookDefinition,
) -> list[Exhibit]:
    baseline = _dedupe_by_key(
        [item for text in texts for item in _extract_exhibits_heuristically(text)],
        lambda item: ((item.exhibit_id or "").lower(), (item.title or "").lower()),
    )
    if not settings.v2_enable_exhibit_agent:
        return baseline

    semaphore = asyncio.Semaphore(max(1, settings.v2_chunk_concurrency))

    async def run_text(text: str) -> list[Exhibit]:
        async with semaphore:
            try:
                extraction = await _run_agent_with_retry(
                    agent=get_v2_exhibit_agent(),
                    payload=(
                        f"Document routing context:\n{_playbook_prompt_payload(classification, playbook)}\n\n"
                        f"Text:\n{text}"
                    ),
                    schema=ExhibitsExtraction,
                )
                return extraction.items
            except Exception:
                return []

    results = await asyncio.gather(*(run_text(text) for text in texts))
    agent_exhibits = [item for group in results for item in group]
    return _merge_exhibit_collections(baseline, agent_exhibits)


@traceable(run_type="chain", name="legal_v2_taxonomy_pass")
async def _refine_clause_taxonomy(clauses: list[Clause]) -> list[Clause]:
    refined: list[Clause] = []
    if not clauses:
        return refined

    for clause in clauses:
        clause = clause.model_copy(deep=True)
        clause.clause_type = _map_to_v2_taxonomy(clause.title, clause.summary)
        if not settings.v2_enable_taxonomy_agent:
            refined.append(clause)
            continue

        if clause.clause_type != "other":
            refined.append(clause)
            continue

        prompt = (
            f"Title: {clause.title}\n"
            f"Summary: {clause.summary}\n"
            f"Source text: {clause.source_text}"
        )
        try:
            result = await _run_agent_with_retry(
                agent=get_v2_taxonomy_agent(),
                payload=prompt,
                schema=ClauseTaxonomyResult,
            )
            clause.clause_type = result.clause_type
        except Exception:
            clause.clause_type = _map_to_v2_taxonomy(clause.title, clause.summary)
        refined.append(clause)
    return refined


def _deterministic_merge_v2(
    *,
    contract_type: str,
    section_results: list[SectionExtractionV2],
    financial_terms: list[FinancialTermV2],
    exhibits: list[Exhibit],
    full_text: str,
) -> LegalExtractionV2:
    parties = _dedupe_by_key(
        [party for result in section_results for party in result.parties],
        lambda item: ((item.name or "").lower(), (item.defined_term or "").lower()),
    )
    shareholders = _dedupe_by_key(
        [shareholder for result in section_results for shareholder in result.shareholders],
        lambda item: ((item.name or "").lower(), (item.defined_term or "").lower(), item.shares, item.percentage),
    )
    clauses = _dedupe_by_key(
        [clause for result in section_results for clause in result.key_clauses],
        lambda item: ((item.title or "").lower(), (item.source_text or "").lower()),
    )
    clauses = [clause.model_copy(deep=True) for clause in clauses]
    for index, clause in enumerate(clauses, start=1):
        clause.clause_id = clause.clause_id or f"C{index}"
        clause.clause_type = _map_to_v2_taxonomy(clause.title, clause.summary)

    red_flags = list(dict.fromkeys(flag for result in section_results for flag in result.red_flags))
    drafting_style = Counter(result.drafting_style_detected for result in section_results if result.drafting_style_detected).most_common(1)
    relationships = _infer_relationships_heuristically(parties, shareholders, clauses, financial_terms)
    completeness_score = _calculate_completeness(
        text=full_text,
        parties=parties,
        shareholders=shareholders,
        financial_terms=financial_terms,
        exhibits=exhibits,
        clauses=clauses,
        relationships=relationships,
    )
    summary = next((result.short_summary for result in section_results if result.short_summary), "") or _clean_snippet(_first_non_empty_line(full_text), 250)
    return LegalExtractionV2(
        contract_type=contract_type,
        parties=parties,
        shareholders=shareholders,
        financial_terms=financial_terms,
        exhibits=exhibits,
        key_clauses=clauses,
        relationships=relationships,
        red_flags=red_flags,
        drafting_style_detected=drafting_style[0][0] if drafting_style else _detect_drafting_style(full_text),
        completeness_score=completeness_score,
        short_summary=summary,
    )


@traceable(run_type="chain", name="legal_v2_merge")
async def _merge_extractions(
    *,
    contract_type: str,
    classification: DocumentClassification,
    playbook: PlaybookDefinition,
    section_results: list[SectionExtractionV2],
    financial_terms: list[FinancialTermV2],
    exhibits: list[Exhibit],
    full_text: str,
    truncated: bool,
) -> LegalExtractionV2:
    if not settings.v2_enable_merge_agent:
        return _deterministic_merge_v2(
            contract_type=contract_type,
            section_results=section_results,
            financial_terms=financial_terms,
            exhibits=exhibits,
            full_text=full_text,
        )

    payload = {
        "contract_type_hint": contract_type,
        "truncated": truncated,
        "routing_context": {
            "classification": classification.model_dump(),
            "playbook": compact_playbook_for_prompt(playbook),
        },
        "section_results": [result.model_dump() for result in section_results],
        "financial_terms": [item.model_dump() for item in financial_terms],
        "exhibits": [item.model_dump() for item in exhibits],
    }
    prompt = (
        "Merge these legal extraction artifacts into one final document extraction.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )
    try:
        merged = await _run_agent_with_retry(
            agent=get_v2_merge_agent(),
            payload=prompt,
            schema=LegalExtractionV2,
        )
        merged.financial_terms = financial_terms or merged.financial_terms
        merged.exhibits = exhibits or merged.exhibits
        if not merged.short_summary:
            merged.short_summary = _clean_snippet(_first_non_empty_line(full_text), 250)
        return merged
    except Exception:
        return _deterministic_merge_v2(
            contract_type=contract_type,
            section_results=section_results,
            financial_terms=financial_terms,
            exhibits=exhibits,
            full_text=full_text,
        )


@traceable(run_type="chain", name="legal_v2_relationship_inference")
async def _infer_relationships(
    merged: LegalExtractionV2,
    classification: DocumentClassification,
    playbook: PlaybookDefinition,
) -> list[SemanticRelationship]:
    baseline = _infer_relationships_heuristically(
        merged.parties,
        merged.shareholders,
        merged.key_clauses,
        merged.financial_terms,
    )
    if not settings.v2_enable_relationship_agent:
        return baseline

    try:
        extraction = await _run_agent_with_retry(
            agent=get_v2_relationship_agent(),
            payload=(
                f"Document routing context:\n{_playbook_prompt_payload(classification, playbook)}\n\n"
                f"Merged extraction:\n{merged.model_dump_json(indent=2)}"
            ),
            schema=RelationshipsExtraction,
        )
        relationships = extraction.items
        if relationships:
            return _merge_relationship_collections(baseline, relationships)
    except Exception:
        pass
    return baseline


def _build_specialized_pass_inputs(chunks: list[Document]) -> list[str]:
    combined = "\n\n".join(chunk.content for chunk in chunks)
    if len(combined) <= settings.max_text_chars:
        return [combined]
    return [chunk.content for chunk in chunks]


async def _extract_sections_in_parallel(
    chunks: list[Document],
    contract_type: str,
    classification: DocumentClassification | None = None,
    playbook: PlaybookDefinition | None = None,
) -> list[SectionExtractionV2]:
    classification, playbook = (
        (classification, playbook)
        if classification is not None and playbook is not None
        else _default_routing_context(contract_type)
    )
    semaphore = asyncio.Semaphore(max(1, settings.v2_chunk_concurrency))
    llm_indexes: set[int] = set()
    if settings.v2_enable_section_agent and settings.v2_max_model_chunks > 0:
        scored_indexes = sorted(
            range(len(chunks)),
            key=lambda index: _chunk_llm_priority(chunks[index], playbook),
            reverse=True,
        )
        llm_indexes = set(scored_indexes[: min(settings.v2_max_model_chunks, len(chunks))])

    async def run_chunk(index: int, chunk: Document) -> SectionExtractionV2:
        async with semaphore:
            return await _extract_section(
                chunk=chunk,
                contract_type_hint=contract_type,
                classification=classification,
                playbook=playbook,
                use_model=index in llm_indexes,
            )

    return await asyncio.gather(*(run_chunk(index, chunk) for index, chunk in enumerate(chunks)))


async def run_single_extraction_v2(
    request: SingleExtractionRequest,
    parser_mode: str = "text",
) -> SingleExtractionResponseV2:
    start = perf_counter()
    clean_text = request.document_text.strip()
    full_metadata = dict(request.document_metadata or {})
    classification = await classify_document(clean_text, request.jurisdiction_hint or "India")
    playbook = get_playbook_for_classification(classification)
    contract_type = classification.document_type or _infer_contract_type(clean_text)
    source_document = Document(
        name=request.document_name,
        content=clean_text,
        meta_data=full_metadata,
    )

    with legal_tracing_context(
        operation="v2_single_extraction",
        metadata={
            "document_name": request.document_name,
            "parser_mode": parser_mode,
            "contract_type_hint": contract_type,
            "playbook_id": playbook.playbook_id,
            "classifier_confidence": classification.confidence,
        },
    ):
        chunks, truncated = chunk_legal_document(
            source_document,
            chunk_size=settings.v2_chunk_size,
            max_chunks=settings.v2_max_chunks,
        )
        if not chunks:
            chunks = [source_document]

        full_text_inputs = _build_specialized_pass_inputs(chunks)
        section_results, financial_terms, exhibits = await asyncio.gather(
            _extract_sections_in_parallel(chunks, contract_type, classification, playbook),
            _extract_financials(full_text_inputs, classification, playbook),
            _extract_exhibits(full_text_inputs, classification, playbook),
        )
        merged = await _merge_extractions(
            contract_type=contract_type,
            classification=classification,
            playbook=playbook,
            section_results=section_results,
            financial_terms=financial_terms,
            exhibits=exhibits,
            full_text=clean_text,
            truncated=truncated,
        )
        merged.key_clauses = await _refine_clause_taxonomy(merged.key_clauses)
        merged.relationships = await _infer_relationships(merged, classification, playbook)
        merged.completeness_score = _calculate_completeness(
            text=clean_text,
            parties=merged.parties,
            shareholders=merged.shareholders,
            financial_terms=merged.financial_terms,
            exhibits=merged.exhibits,
            clauses=merged.key_clauses,
            relationships=merged.relationships,
        )
        merged = _apply_playbook_postprocessing(
            extraction=merged,
            classification=classification,
            playbook=playbook,
            full_text=clean_text,
            truncated=truncated,
        )
        merged.short_summary = _ensure_v2_short_summary(merged, clean_text)

    elapsed_ms = int((perf_counter() - start) * 1000)
    return SingleExtractionResponseV2(
        extraction=merged,
        truncated=truncated,
        model_used=settings.v2_model_id,
        processing_ms=elapsed_ms,
        parser_mode="docling" if parser_mode == "docling" else ("pypdf" if parser_mode == "pypdf" else "text"),
        total_chunks=len(chunks),
    )
