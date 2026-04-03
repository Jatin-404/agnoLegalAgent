from __future__ import annotations

import re

from app.core.schemas import CrossReference, Exhibit, FlexibleEntity, GraphEdge, GraphNode, LegalExtraction, PlaybookDefinition, SemanticRelationship, Shareholder


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_") or "unknown"


def extract_cross_references(
    *,
    full_text: str,
    clauses,
    exhibits: list[Exhibit],
    playbook: PlaybookDefinition,
) -> list[CrossReference]:
    references: list[CrossReference] = []
    seen: set[tuple[str, str, str]] = set()
    sources = [(clause.title or clause.clause_id or "Document", f"{clause.summary}\n{clause.source_text}") for clause in clauses]
    sources.extend((exhibit.title or exhibit.exhibit_id, f"{exhibit.summary}\n{exhibit.key_content}") for exhibit in exhibits)

    patterns = playbook.cross_ref_patterns or [
        r"(?i)clause\s+\d+(?:\.\d+)*",
        r"(?i)section\s+\d+(?:\.\d+)*",
        r"(?i)exhibit\s+[A-Z0-9]+",
        r"(?i)schedule\s+[A-Z0-9]+",
    ]
    for source_label, source_text in sources:
        for pattern in patterns:
            try:
                for match in re.finditer(pattern, source_text):
                    target = re.sub(r"\s+", " ", match.group(0)).strip()
                    key = (source_label, target.lower(), pattern)
                    if key in seen:
                        continue
                    seen.add(key)
                    reference_type = "exhibit_reference" if target.lower().startswith(("exhibit", "schedule")) else "clause_reference"
                    references.append(
                        CrossReference(
                            source_label=source_label,
                            target_label=target,
                            reference_text=match.group(0),
                            reference_type=reference_type,
                            confidence=0.76,
                        )
                    )
            except re.error:
                continue

    if not references and full_text:
        for pattern in patterns:
            try:
                match = re.search(pattern, full_text)
            except re.error:
                match = None
            if match:
                references.append(
                    CrossReference(
                        source_label="Document",
                        target_label=match.group(0),
                        reference_text=match.group(0),
                        reference_type="clause_reference",
                        confidence=0.58,
                    )
                )
                break
    return references


def build_document_graph(extraction: LegalExtraction) -> tuple[list[GraphNode], list[GraphEdge]]:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    node_index: dict[str, str] = {}

    def ensure_node(label: str, node_type: str) -> str:
        key = f"{node_type}:{_slug(label)}"
        if key not in node_index:
            node_index[key] = key
            nodes.append(GraphNode(node_id=key, node_type=node_type, label=label))
        return key

    for party in extraction.parties:
        ensure_node(party.name, "party")
    for shareholder in extraction.shareholders:
        ensure_node(shareholder.name, "shareholder")
    for clause in extraction.key_clauses:
        ensure_node(clause.title or clause.clause_id or "Clause", "clause")
    for exhibit in extraction.exhibits:
        ensure_node(exhibit.title or exhibit.exhibit_id, "exhibit")

    for relationship in extraction.relationships:
        source_id = ensure_node(relationship.source_entity, "party")
        target_id = ensure_node(relationship.target_entity, "party")
        edges.append(
            GraphEdge(
                source_id=source_id,
                target_id=target_id,
                relation=relationship.relationship_type,
                evidence=relationship.source_text,
                confidence=relationship.confidence,
            )
        )

    for reference in extraction.cross_references:
        source_id = ensure_node(reference.source_label, "clause")
        target_type = "exhibit" if reference.reference_type == "exhibit_reference" else "clause"
        target_id = ensure_node(reference.target_label, target_type)
        edges.append(
            GraphEdge(
                source_id=source_id,
                target_id=target_id,
                relation=reference.reference_type,
                evidence=reference.reference_text,
                confidence=reference.confidence,
            )
        )

    deduped_edges: list[GraphEdge] = []
    seen_edges: set[tuple[str, str, str, str | None]] = set()
    for edge in edges:
        key = (edge.source_id, edge.target_id, edge.relation, edge.evidence)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        deduped_edges.append(edge)

    return nodes, deduped_edges
