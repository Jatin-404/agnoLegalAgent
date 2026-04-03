from __future__ import annotations

import json
import re
from typing import Any, Literal, Optional, Type, TypeVar

from pydantic import BaseModel, Field

from app.core.schemas import FlexibleEntity, Shareholder

ModelT = TypeVar("ModelT", bound=BaseModel)


class HeuristicClause(BaseModel):
    clause_type: str
    title: str
    summary: str
    source_text: str
    importance: Literal["high", "medium", "low"] = "medium"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class HeuristicFinancialTerm(BaseModel):
    label: str
    amount: Optional[float] = None
    currency: Optional[str] = None
    rate: Optional[float] = None
    unit: Optional[str] = None
    source_text: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


def _coerce_content(content: Any, schema: Type[ModelT]) -> ModelT:
    if isinstance(content, schema):
        return content
    if isinstance(content, BaseModel):
        return schema.model_validate(content.model_dump())
    if isinstance(content, dict):
        return schema.model_validate(content)
    if isinstance(content, str):
        parsed = _loads_relaxed_json(content, schema)
        return schema.model_validate(parsed)
    raise ValueError(f"Unexpected model response type for {schema.__name__}: {type(content)!r}")


def _extract_json_candidate(raw: str) -> str:
    text = raw.strip()
    if not text:
        return text

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    if text.startswith("{") or text.startswith("["):
        return text

    obj_start = text.find("{")
    arr_start = text.find("[")
    starts = [i for i in (obj_start, arr_start) if i >= 0]
    if not starts:
        return text
    start = min(starts)
    return text[start:].strip()


def _extract_balanced_json_prefix(raw: str) -> str | None:
    candidate = _extract_json_candidate(raw)
    if not candidate or candidate[0] not in "[{":
        return None

    stack: list[str] = []
    in_string = False
    escaped = False
    last_complete_index: int | None = None
    matching = {"}": "{", "]": "["}

    for index, char in enumerate(candidate):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char in "[{":
            stack.append(char)
            continue

        if char in "}]":
            if not stack or stack[-1] != matching[char]:
                break
            stack.pop()
            if not stack:
                last_complete_index = index

    if last_complete_index is None:
        return None
    return candidate[: last_complete_index + 1]


def _strip_trailing_commas(raw: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", raw)


def _loads_relaxed_json(raw: str, schema: Type[ModelT]) -> Any:
    candidate = _extract_json_candidate(raw)
    decoder = json.JSONDecoder()

    attempts = [candidate]
    balanced = _extract_balanced_json_prefix(raw)
    if balanced and balanced not in attempts:
        attempts.append(balanced)
    cleaned = _strip_trailing_commas(candidate)
    if cleaned not in attempts:
        attempts.append(cleaned)
    if balanced:
        cleaned_balanced = _strip_trailing_commas(balanced)
        if cleaned_balanced not in attempts:
            attempts.append(cleaned_balanced)

    for attempt in attempts:
        if not attempt:
            continue
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            try:
                parsed, _ = decoder.raw_decode(attempt)
                return parsed
            except json.JSONDecodeError:
                continue

    preview = (raw or "").strip().replace("\n", " ")[:180]
    raise ValueError(
        f"Model returned unparseable JSON for {schema.__name__}. Raw preview: {preview!r}"
    )


def _first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _clean_snippet(text: str, max_len: int = 220) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_len:
        return normalized
    return normalized[:max_len].rstrip() + "..."


def _clean_entity_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name).strip(" ,.;:")
    cleaned = re.sub(r",\s*a company incorporated under.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",\s*having its registered office.*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" ,.;:")


def _normalize_quotes(text: str) -> str:
    return (
        text.replace("â€œ", '"')
        .replace("â€", '"')
        .replace("â€™", "'")
        .replace("â€˜", "'")
        .replace("â€“", "-")
    )


def _strip_markdown_formatting(text: str) -> str:
    stripped = text.replace("**", "").replace("__", "")
    return re.sub(r"`([^`]+)`", r"\1", stripped)


def _parse_indian_number(raw: str) -> Optional[float]:
    cleaned = raw.replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _heuristic_entities(text: str) -> list[FlexibleEntity]:
    normalized_text = _strip_markdown_formatting(_normalize_quotes(text))
    entities: list[FlexibleEntity] = []
    seen: set[tuple[Optional[str], Optional[str], str, Optional[str]]] = set()

    patterns = [
        (
            re.compile(
                r"(?P<name>[A-Z][A-Za-z0-9&.,()' /-]+?),\s+(?:a|an)\s+(?:private limited company|public limited company|company)\b.*?\(hereinafter referred to as (?:the )?\"?(?P<term>[^\")]+)\"?\)",
                flags=re.IGNORECASE | re.DOTALL,
            ),
            "organization",
        ),
        (
            re.compile(
                r"(?P<name>[A-Z][A-Za-z0-9&.,()' /-]+?),\s+a company incorporated.*?\(hereinafter referred to as (?:the )?\"?(?P<term>[^\")]+)\"?\)",
                flags=re.IGNORECASE | re.DOTALL,
            ),
            "organization",
        ),
        (
            re.compile(
                r"(?P<name>(?:Mr|Mrs|Ms)\.?\s+[A-Z][A-Za-z .]+?),.*?\(hereinafter referred to as (?:the )?\"(?P<term>[A-Z]{1,3})\"\)",
                flags=re.IGNORECASE | re.DOTALL,
            ),
            "person",
        ),
        (
            re.compile(
                r"(?:^|\n)\s*[a-z]\.\s*(?P<name>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,4}),.*?\(hereinafter referred to as (?:the )?\"?(?P<term>[A-Z]{1,5})\"?\)",
                flags=re.IGNORECASE | re.DOTALL,
            ),
            "person",
        ),
        (
            re.compile(
                r"(?:^|\n)\s*(?P<name>[A-Z][A-Za-z0-9&.,()' /-]+?(?:Private Limited|Private Ltd\.?|Pvt\. Ltd\.?|Limited|LLP|LLC|Inc\.?|Corporation|Company))\b.*?\(hereinafter referred to as (?:the )?\"?(?P<term>[^\")]+)\"?\)",
                flags=re.IGNORECASE | re.DOTALL,
            ),
            "organization",
        ),
        (
            re.compile(
                r"\"(?P<term>[A-Z])\"\s+means\s+(?P<name>(?:Mr|Mrs|Ms)\.?\s+[A-Z][A-Za-z .]+)",
                flags=re.IGNORECASE,
            ),
            "person",
        ),
        (
            re.compile(
                r"(?P<name>[A-Z][A-Za-z0-9&.,()' /-]+?),\s+a company incorporated under",
                flags=re.IGNORECASE,
            ),
            "organization",
        ),
    ]

    for pattern, entity_type in patterns:
        for match in pattern.finditer(normalized_text):
            name = match.groupdict().get("name")
            defined_term = match.groupdict().get("term")
            normalized_name = _clean_entity_name(name or "") or None
            normalized_term = defined_term.strip(" .") if defined_term else None
            if entity_type == "person":
                role = "Shareholder"
            elif normalized_term and normalized_term.lower() == "company":
                role = "Company"
            else:
                role = normalized_term
            key = (normalized_name, normalized_term, entity_type, role)
            if key in seen or not (normalized_name or normalized_term):
                continue
            seen.add(key)
            entities.append(
                FlexibleEntity(
                    name=normalized_name or normalized_term or "Unknown Entity",
                    defined_term=normalized_term,
                    entity_type=entity_type,
                    role=role,
                    source_text=_clean_snippet(match.group(0)),
                    confidence=0.7,
                )
            )

    signatory_pattern = re.compile(
        r"SIGNED AND DELIVERED(?: for and on behalf of [^\n]+)? by (?P<name>(?:Mr|Mrs|Ms)\.?\s+[A-Z][A-Za-z .]+)",
        flags=re.IGNORECASE,
    )
    signatory_names: list[str] = []
    for match in signatory_pattern.finditer(normalized_text):
        normalized_name = _clean_entity_name(match.group("name"))
        signatory_names.append(normalized_name)
        key = (normalized_name, None, "person", "Signatory")
        if key in seen:
            continue
        seen.add(key)
        entities.append(
            FlexibleEntity(
                name=normalized_name,
                defined_term=None,
                entity_type="person",
                role="Signatory",
                source_text=_clean_snippet(match.group(0)),
                confidence=0.62,
            )
        )

    single_letter_terms = [("A", 0), ("B", 1)]
    for term, idx in single_letter_terms:
        if len(signatory_names) <= idx:
            continue
        normalized_name = signatory_names[idx]
        key = (normalized_name, term, "person", "Shareholder")
        if key in seen:
            continue
        seen.add(key)
        entities.append(
            FlexibleEntity(
                name=normalized_name,
                defined_term=term,
                entity_type="person",
                role="Shareholder",
                source_text=f"{term} = {normalized_name}",
                confidence=0.6,
            )
        )

    shareholding_name_pattern = re.compile(
        r"(?m)^(?P<name>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,4})\s+\|\s+(?P<shares>[\d,]+)\s+\|\s+\d+%$"
    )
    for match in shareholding_name_pattern.finditer(normalized_text):
        normalized_name = _clean_entity_name(match.group("name"))
        key = (normalized_name, None, "person", "Shareholder")
        if key in seen:
            continue
        seen.add(key)
        entities.append(
            FlexibleEntity(
                name=normalized_name,
                defined_term=None,
                entity_type="person",
                role="Shareholder",
                source_text=_clean_snippet(match.group(0)),
                confidence=0.74,
            )
        )

    return entities


def _map_clause_type(title: str) -> str:
    cleaned_title = title.strip().lower()
    cleaned_title = re.sub(r"^\d+(?:\.\d+)*[.)]?\s+", "", cleaned_title)
    cleaned_title = re.sub(r"^exhibit\s+[a-z0-9]+\s*(?:[â€“:\-])?\s*", "", cleaned_title)
    normalized_title = re.sub(r"[^a-z0-9]+", "_", cleaned_title).strip("_")
    lowered = normalized_title
    mapping = {
        "term": "term",
        "scope_of_services": "scope_of_services",
        "payment_terms": "payment",
        "payment": "payment",
        "obligations": "obligations",
        "confidentiality": "confidentiality",
        "termination": "termination",
        "renewal": "renewal",
        "governing_law": "governing_law",
        "dispute_resolution": "dispute",
        "default": "default",
        "assignment": "assignment",
        "taxes": "taxes",
        "repair": "repair",
        "alterations": "alterations",
        "possession": "possession",
        "vesting": "vesting",
        "liability": "liability",
        "indemnity": "indemnity",
        "business": "business",
        "share_capital": "share_capital",
        "shareholding": "shareholding",
        "further_capital": "capital_raising",
        "board_of_directors": "board_governance",
        "voting": "voting_rights",
        "auditors": "auditors",
        "transfer_of_shares": "transfer_restrictions",
        "right_of_first_refusal": "right_of_first_refusal",
        "financial_arrangement": "financial_terms",
        "interest": "interest",
        "breach": "breach",
        "amendments": "amendments",
        "entire_agreement": "entire_agreement",
        "director_compliance": "director_compliance",
        "undertakings": "undertakings",
    }
    if lowered in mapping:
        return mapping[lowered]

    generic_cleanup = {"clause", "provisions", "provision", "section", "agreement"}
    parts = [part for part in normalized_title.split("_") if part and part not in generic_cleanup]
    if parts:
        return "_".join(parts[:5])
    return "other"


def _heuristic_clauses(text: str) -> list[HeuristicClause]:
    normalized_text = _strip_markdown_formatting(_normalize_quotes(text))
    clause_blocks = re.findall(
        r"(?:^|\n)(?:#{1,6}\s+|\*\*)?(\d{1,2})\.\s+([A-Z][A-Za-z'& /,_-]+?)(?:\*\*)?\s*\n(.*?)(?=(?:\n(?:#{1,6}\s+|\*\*)?\d{1,2}\.\s+[A-Z][A-Za-z'& /,_-]+(?:\*\*)?\s*\n)|\Z)",
        normalized_text,
        flags=re.DOTALL,
    )

    clauses: list[HeuristicClause] = []
    for _, raw_title, body in clause_blocks:
        title = raw_title.strip().title()
        clause_type = _map_clause_type(raw_title.strip())
        if title.lower() == "entire agreement":
            body = body.split("SIGNATURES")[0].strip()
        body_clean = _clean_snippet(body, 260)
        importance = "high" if clause_type in {"payment", "term", "termination", "dispute", "governing_law", "shareholding", "board_governance", "transfer_restrictions", "right_of_first_refusal", "financial_terms"} else "medium"
        clauses.append(
            HeuristicClause(
                clause_type=clause_type,
                title=title,
                summary=body_clean,
                source_text=body_clean,
                importance=importance,
                confidence=0.72,
            )
        )
    return clauses


def _heuristic_shareholdings(text: str, entities: list[Any]) -> list[Shareholder]:
    normalized_text = _strip_markdown_formatting(_normalize_quotes(text))
    entries: list[Shareholder] = []
    seen: set[tuple[Optional[str], Optional[str], Optional[int]]] = set()
    term_to_name = {getattr(entity, "defined_term", None): getattr(entity, "name", None) for entity in entities if getattr(entity, "defined_term", None)}

    for match in re.finditer(r"(?m)^(?:[-*]\s+)?(?P<term>[A-Z]{1,3}):\s*(?P<shares>[\d,]+)\s+shares\b", normalized_text):
        term = match.group("term")
        shares = _parse_indian_number(match.group("shares"))
        normalized_shares = int(shares) if shares is not None else None
        key = (term_to_name.get(term), term, normalized_shares)
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            Shareholder(
                name=term_to_name.get(term) or term,
                defined_term=term,
                shares=normalized_shares,
                source_text=_clean_snippet(match.group(0)),
                confidence=0.82,
            )
        )

    table_pattern = re.compile(
        r"(?m)^\s*\|\s*(?P<name>[A-Z][A-Za-z .-]+?)\s*\|\s*(?P<shares>[\d,]+)\s*\|\s*(?P<percent>\d+%)\s*\|?$|^\s*(?P<name2>[A-Z][A-Za-z .-]+?)\s+\|\s+(?P<shares2>[\d,]+)\s+\|\s+(?P<percent2>\d+%)$"
    )
    for match in table_pattern.finditer(normalized_text):
        raw_name = _clean_entity_name(match.group("name") or match.group("name2") or "")
        shares = _parse_indian_number(match.group("shares") or match.group("shares2") or "")
        normalized_shares = int(shares) if shares is not None else None
        defined_term = next((getattr(entity, "defined_term", None) for entity in entities if getattr(entity, "name", None) == raw_name), None)
        key = (raw_name, defined_term, normalized_shares)
        if key in seen:
            continue
        seen.add(key)
        percentage_match = re.search(r"(\d+(?:\.\d+)?)\s*%", match.group(0))
        percentage = float(percentage_match.group(1)) if percentage_match else None
        entries.append(
            Shareholder(
                name=raw_name,
                defined_term=defined_term,
                shares=normalized_shares,
                percentage=percentage,
                source_text=_clean_snippet(match.group(0)),
                confidence=0.86,
            )
        )
    return entries


def _heuristic_financial_terms(text: str) -> list[HeuristicFinancialTerm]:
    normalized_text = _strip_markdown_formatting(_normalize_quotes(text))
    terms: list[HeuristicFinancialTerm] = []

    amount_patterns = [
        ("authorised_share_capital", r"Authorised share capital:\s*Rs\.?\s*([\d,]+)"),
        ("loan_amount", r"HDFC Bank has provided Rs\.?\s*([\d,]+)"),
        ("investment_amount", r"B will invest:\s*Rs\.?\s*([\d,]+)"),
        ("remaining_secured_amount", r"Remaining:\s*Rs\.?\s*([\d,]+)"),
        ("profit_cap_monthly", r"Profit cap for A:\s*Rs\.?\s*([\d,]+)"),
        ("contract_value", r"total contract value shall be INR\s*([\d,]+)"),
        ("security_deposit", r"security deposit of INR\s*([\d,]+)"),
    ]
    for label, pattern in amount_patterns:
        match = re.search(pattern, normalized_text, flags=re.IGNORECASE)
        if not match:
            continue
        amount = _parse_indian_number(match.group(1))
        terms.append(
            HeuristicFinancialTerm(
                label=label,
                amount=amount,
                currency="INR",
                rate=None,
                unit=None,
                source_text=_clean_snippet(match.group(0)),
                confidence=0.8,
            )
        )

    rate_patterns = [
        ("interest_rate", r"(\d+(?:\.\d+)?)%\s+annual interest"),
        ("late_fee_rate", r"penalty of (\d+(?:\.\d+)?)%\s+per month"),
    ]
    for label, pattern in rate_patterns:
        match = re.search(pattern, normalized_text, flags=re.IGNORECASE)
        if not match:
            continue
        rate = float(match.group(1))
        unit = "percent_per_year" if label == "interest_rate" else "percent_per_month"
        terms.append(
            HeuristicFinancialTerm(
                label=label,
                amount=None,
                currency=None,
                rate=rate,
                unit=unit,
                source_text=_clean_snippet(match.group(0)),
                confidence=0.8,
            )
        )

    return terms
