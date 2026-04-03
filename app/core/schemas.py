from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SingleExtractionRequest(BaseModel):
    document_name: str
    document_text: str
    jurisdiction_hint: str = "India"
    document_metadata: Dict[str, Any] = Field(default_factory=dict)


ClauseTaxonomy = Literal[
    "shareholding",
    "governance",
    "transfer_restrictions",
    "exit",
    "non_compete",
    "tag_along",
    "drag_along",
    "bad_leaver",
    "penalties",
    "financial",
    "disputes_governing_law",
    "other",
]


RelationshipType = Literal[
    "role",
    "incentive_link",
    "penalty_trigger",
    "voting_power",
    "tag_along",
    "drag_along",
    "bad_leaver",
    "other",
]


class FlexibleEntity(BaseModel):
    name: str
    defined_term: Optional[str] = None
    entity_type: Literal["person", "organization", "authority", "court", "property", "other"] = "other"
    role: Optional[str] = None
    source_text: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class Shareholder(BaseModel):
    name: str
    defined_term: Optional[str] = None
    shares: Optional[int] = None
    percentage: Optional[float] = None
    share_class: Optional[str] = None
    source_text: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class Clause(BaseModel):
    clause_id: Optional[str] = None
    clause_type: ClauseTaxonomy = "other"
    title: str
    summary: str
    source_text: str
    linked_entities: List[str] = Field(default_factory=list)
    linked_exhibits: List[str] = Field(default_factory=list)
    importance: Literal["high", "medium", "low"] = "medium"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class FinancialTerm(BaseModel):
    amount: Optional[str] = None
    currency: str = "INR"
    percentage: Optional[float] = None
    description: str
    trigger_condition: str
    linked_entity: List[str] = Field(default_factory=list)
    linked_clause_id: Optional[str] = None
    exhibit_reference: Optional[str] = None
    source_text: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class Exhibit(BaseModel):
    exhibit_id: str
    title: str
    summary: str
    key_content: str
    linked_financials: List["FinancialTerm"] = Field(default_factory=list)
    source_text: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SemanticRelationship(BaseModel):
    source_entity: str
    target_entity: str
    relationship_type: RelationshipType
    condition: str
    impact: str
    source_text: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SectionExtraction(BaseModel):
    contract_type: str
    parties: List[FlexibleEntity] = Field(default_factory=list)
    shareholders: List[Shareholder] = Field(default_factory=list)
    key_clauses: List[Clause] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    drafting_style_detected: str = "unknown"
    short_summary: str = ""


class FinancialTermsExtraction(BaseModel):
    items: List[FinancialTerm] = Field(default_factory=list)


class ExhibitsExtraction(BaseModel):
    items: List[Exhibit] = Field(default_factory=list)


class RelationshipsExtraction(BaseModel):
    items: List[SemanticRelationship] = Field(default_factory=list)


class ClauseTaxonomyResult(BaseModel):
    clause_type: ClauseTaxonomy


class DocumentClassification(BaseModel):
    document_type: str
    normalized_type: str
    jurisdiction: str = "India"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    matched_playbook_id: Optional[str] = None
    matched_keywords: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = None


class PlaybookDefinition(BaseModel):
    playbook_id: str
    playbook_version: str = "1.0"
    contract_type: str
    jurisdiction: str = "Global"
    aliases: List[str] = Field(default_factory=list)
    fields: List[str] = Field(default_factory=list)
    clause_taxonomy: List[str] = Field(default_factory=list)
    risk_rules: List[Dict[str, Any]] = Field(default_factory=list)
    cross_ref_patterns: List[str] = Field(default_factory=list)
    priority_terms: List[str] = Field(default_factory=list)
    graph_entity_types: List[str] = Field(default_factory=list)


class CrossReference(BaseModel):
    source_label: str
    target_label: str
    reference_text: str
    reference_type: str = "clause_reference"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class GraphNode(BaseModel):
    node_id: str
    node_type: str
    label: str


class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    relation: str
    evidence: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class LegalExtraction(BaseModel):
    contract_type: str
    document_type: Optional[str] = None
    jurisdiction: str = "India"
    playbook_id: Optional[str] = None
    playbook_version: Optional[str] = None
    classifier_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    playbook_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    playbook_fields_expected: List[str] = Field(default_factory=list)
    playbook_fields_extracted: List[str] = Field(default_factory=list)
    playbook_fields_missing: List[str] = Field(default_factory=list)
    parties: List[FlexibleEntity] = Field(default_factory=list)
    shareholders: List[Shareholder] = Field(default_factory=list)
    financial_terms: List[FinancialTerm] = Field(default_factory=list)
    exhibits: List[Exhibit] = Field(default_factory=list)
    key_clauses: List[Clause] = Field(default_factory=list)
    relationships: List[SemanticRelationship] = Field(default_factory=list)
    cross_references: List[CrossReference] = Field(default_factory=list)
    graph_nodes: List[GraphNode] = Field(default_factory=list)
    graph_edges: List[GraphEdge] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    drafting_style_detected: str = "unknown"
    completeness_score: int = Field(..., ge=0, le=100)
    short_summary: str = ""
    needs_human_review: bool = False


class SingleExtractionResponse(BaseModel):
    extraction: LegalExtraction
    truncated: bool = False
    model_used: str
    processing_ms: int
    parser_mode: Literal["text", "pypdf", "docling"] = "text"
    total_chunks: int = 1


FinancialTermV2 = FinancialTerm
SectionExtractionV2 = SectionExtraction
LegalExtractionV2 = LegalExtraction
SingleExtractionResponseV2 = SingleExtractionResponse
