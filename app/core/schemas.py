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


class LegalExtraction(BaseModel):
    contract_type: str
    parties: List[FlexibleEntity] = Field(default_factory=list)
    shareholders: List[Shareholder] = Field(default_factory=list)
    financial_terms: List[FinancialTerm] = Field(default_factory=list)
    exhibits: List[Exhibit] = Field(default_factory=list)
    key_clauses: List[Clause] = Field(default_factory=list)
    relationships: List[SemanticRelationship] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    drafting_style_detected: str = "unknown"
    completeness_score: int = Field(..., ge=0, le=100)
    short_summary: str = ""


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
