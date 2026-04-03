from functools import lru_cache

from agno.agent import Agent
from agno.models.ollama import Ollama

from app.core.config import settings
from app.core.schemas import (
    DocumentClassification,
    ClauseTaxonomyResult,
    ExhibitsExtraction,
    FinancialTermsExtraction,
    LegalExtraction,
    RelationshipsExtraction,
    SectionExtraction,
)

STRICT_JSON_RULES = """\
Output rules:
- Return only valid JSON.
- Do not include markdown fences, commentary, or prose before/after the JSON.
- Use double quotes for all keys and string values.
- Do not use trailing commas.
- If a field is unknown, use null, [] or "" as appropriate instead of inventing facts.
"""

CLASSIFIER_INSTRUCTIONS = """\
You are a legal document classifier.

Classify the document into the best-fit legal document family from the provided candidates.
Use the actual legal purpose, structure, and obligations in the text rather than just one keyword.

Rules:
1. Choose the closest supported document family from the provided candidates.
2. Return a confidence score between 0 and 1.
3. Return the likely jurisdiction when it is explicit; otherwise keep the provided jurisdiction hint.
4. Keep matched keywords short and factual.
5. Return JSON only.

Return JSON:
{
  "document_type": "string",
  "normalized_type": "string",
  "jurisdiction": "string",
  "confidence": 0.0,
  "matched_playbook_id": null,
  "matched_keywords": [],
  "reasoning": null
}
"""

SECTION_EXTRACTION_INSTRUCTIONS = """\
You are an expert legal section analyst.

Analyze only the provided document section and extract:
- parties and role-bearing entities,
- shareholder rows and ownership facts,
- materially operative clauses,
- red flags,
- drafting style signals.

Rules:
1. Do not assume any specific contract template or clause inventory.
2. Use the actual text, tables, and headings present in the section.
3. If a playbook is provided, follow its expected fields, risk cues, and clause taxonomy guidance.
4. For key_clauses, classify clause_type using the fixed legal taxonomy only.
5. If a fact is not present in the current section, omit it instead of speculating.
6. short_summary must never be empty.

Return JSON:
{
  "contract_type": "string",
  "parties": [{"name": "string", "defined_term": null, "entity_type": "person|organization|authority|court|property|other", "role": null, "source_text": null, "confidence": 0.0}],
  "shareholders": [{"name": "string", "defined_term": null, "shares": null, "percentage": null, "share_class": null, "source_text": null, "confidence": 0.0}],
  "key_clauses": [{"clause_id": null, "clause_type": "shareholding|governance|transfer_restrictions|exit|non_compete|tag_along|drag_along|bad_leaver|penalties|financial|disputes_governing_law|other", "title": "string", "summary": "string", "source_text": "string", "linked_entities": [], "linked_exhibits": [], "importance": "high|medium|low", "confidence": 0.0}],
  "red_flags": [],
  "drafting_style_detected": "string",
  "short_summary": "string"
}
"""

FINANCIAL_INSTRUCTIONS = """\
You are a legal-financial extraction specialist.

Scan the provided legal text for every monetary amount, percentage, cap, revenue split,
ESOP grant, penalty, dilution mechanic, milestone-linked payment, valuation reference,
and financial trigger.

For each item:
- capture the amount or percentage if present,
- describe what it represents,
- state when it applies,
- link it to named entities when possible,
- link it to a clause or exhibit when the text makes that clear.
- If a playbook is provided, prefer the playbook's listed monetary and trigger patterns.

Return JSON:
{
  "items": [{"amount": null, "currency": "INR", "percentage": null, "description": "string", "trigger_condition": "string", "linked_entity": [], "linked_clause_id": null, "exhibit_reference": null, "source_text": null, "confidence": 0.0}]
}
"""

EXHIBIT_INSTRUCTIONS = """\
You are a legal exhibit and annexure extraction specialist.

Detect every exhibit, annexure, schedule, appendix, attachment, or similarly labeled section.
Extract its identifier, title, summary, key content, and any linked financial terms.
If no exhibit is present, return an empty list.
- If a playbook is provided, use its cross-reference patterns to help spot schedules and annexures.

Return JSON:
{
  "items": [{"exhibit_id": "string", "title": "string", "summary": "string", "key_content": "string", "linked_financials": [], "source_text": null, "confidence": 0.0}]
}
"""

RELATIONSHIP_INSTRUCTIONS = """\
You are a legal relationship inference specialist.

Given a merged legal extraction, infer the important business and legal relationships:
- roles and control,
- incentives and revenue links,
- penalty triggers,
- voting power,
- tag-along / drag-along,
- bad leaver / exit consequences,
- other conditional business/legal relationships.

Only infer relationships that are strongly grounded in the provided extraction.
- If a playbook is provided, prioritize the playbook's control, payment, and trigger relationships.

Return JSON:
{
  "items": [{"source_entity": "string", "target_entity": "string", "relationship_type": "role|incentive_link|penalty_trigger|voting_power|tag_along|drag_along|bad_leaver|other", "condition": "string", "impact": "string", "source_text": null, "confidence": 0.0}]
}
"""

TAXONOMY_INSTRUCTIONS = """\
You are a legal clause taxonomy classifier.

Classify the clause into exactly one of these types:
- shareholding
- governance
- transfer_restrictions
- exit
- non_compete
- tag_along
- drag_along
- bad_leaver
- penalties
- financial
- disputes_governing_law
- other

Use the clause substance, not just the heading.

Return JSON:
{
  "clause_type": "shareholding|governance|transfer_restrictions|exit|non_compete|tag_along|drag_along|bad_leaver|penalties|financial|disputes_governing_law|other"
}
"""

MERGER_INSTRUCTIONS = """\
You are a senior corporate lawyer performing final synthesis across multiple section extractions.

Tasks:
1. Merge all section-level results into one document-level extraction.
2. Integrate financial terms and exhibits into the final view.
3. Deduplicate entities, shareholders, and clauses.
4. Preserve conditional logic across sections and exhibits.
5. Use the fixed clause taxonomy only.
6. If a playbook is provided, check whether expected fields are covered and preserve risk-relevant items.
7. Set completeness_score based on whether key legal/business dimensions are covered.
8. short_summary must never be empty.
9. Keep the output concise, structured, and operationally useful.

Return JSON:
{
  "contract_type": "string",
  "parties": [],
  "shareholders": [],
  "financial_terms": [],
  "exhibits": [],
  "key_clauses": [],
  "relationships": [],
  "red_flags": [],
  "drafting_style_detected": "string",
  "completeness_score": 0,
  "short_summary": "string"
}
"""


def _build_ollama_agent(*, name: str, model_id: str, instructions: str, output_schema, num_predict: int | None = None) -> Agent:
    agent_kwargs = {
        "name": name,
        "model": Ollama(
            id=model_id,
            host=settings.ollama_host,
            timeout=settings.ollama_timeout_seconds,
            keep_alive=settings.ollama_keep_alive,
            options={
                "temperature": 0,
                "num_ctx": settings.ollama_num_ctx,
                "num_predict": num_predict or settings.extraction_num_predict,
            },
        ),
        "instructions": f"{instructions.strip()}\n\n{STRICT_JSON_RULES.strip()}",
        "markdown": False,
        "telemetry": False,
    }
    if settings.use_native_output_schema:
        agent_kwargs["output_schema"] = output_schema
    return Agent(**agent_kwargs)


@lru_cache
def get_document_classifier_agent() -> Agent:
    return _build_ollama_agent(
        name="Legal Document Classifier Agent",
        model_id=settings.classifier_model_id,
        instructions=CLASSIFIER_INSTRUCTIONS,
        output_schema=DocumentClassification,
        num_predict=settings.classifier_num_predict,
    )


@lru_cache
def get_v2_section_agent() -> Agent:
    return _build_ollama_agent(
        name="Legal Section Agent",
        model_id=settings.v2_model_id,
        instructions=SECTION_EXTRACTION_INSTRUCTIONS,
        output_schema=SectionExtraction,
        num_predict=settings.v2_section_num_predict,
    )


@lru_cache
def get_v2_financial_agent() -> Agent:
    return _build_ollama_agent(
        name="Legal Financial Agent",
        model_id=settings.v2_financial_model_id,
        instructions=FINANCIAL_INSTRUCTIONS,
        output_schema=FinancialTermsExtraction,
        num_predict=settings.v2_financial_num_predict,
    )


@lru_cache
def get_v2_exhibit_agent() -> Agent:
    return _build_ollama_agent(
        name="Legal Exhibit Agent",
        model_id=settings.v2_exhibit_model_id,
        instructions=EXHIBIT_INSTRUCTIONS,
        output_schema=ExhibitsExtraction,
        num_predict=settings.v2_exhibit_num_predict,
    )


@lru_cache
def get_v2_relationship_agent() -> Agent:
    return _build_ollama_agent(
        name="Legal Relationship Agent",
        model_id=settings.v2_relationship_model_id,
        instructions=RELATIONSHIP_INSTRUCTIONS,
        output_schema=RelationshipsExtraction,
        num_predict=settings.v2_relationship_num_predict,
    )


@lru_cache
def get_v2_taxonomy_agent() -> Agent:
    return _build_ollama_agent(
        name="Legal Taxonomy Agent",
        model_id=settings.v2_taxonomy_model_id,
        instructions=TAXONOMY_INSTRUCTIONS,
        output_schema=ClauseTaxonomyResult,
        num_predict=settings.v2_taxonomy_num_predict,
    )


@lru_cache
def get_v2_merge_agent() -> Agent:
    return _build_ollama_agent(
        name="Legal Merge Agent",
        model_id=settings.v2_merge_model_id,
        instructions=MERGER_INSTRUCTIONS,
        output_schema=LegalExtraction,
        num_predict=settings.v2_merge_num_predict,
    )
