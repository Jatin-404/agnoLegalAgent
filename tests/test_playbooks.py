import unittest

from app.core.classifier import classify_document
from app.core.graph_utils import build_document_graph
from app.core.playbooks import calculate_playbook_coverage, load_playbook_for_type, load_playbooks
from app.core.schemas import Clause, CrossReference, FlexibleEntity, LegalExtraction, Shareholder


class PlaybookTests(unittest.IsolatedAsyncioTestCase):
    async def test_classifier_routes_shareholders_agreement_to_matching_playbook(self) -> None:
        text = """
        SHAREHOLDERS AGREEMENT

        The shareholders of SpiceRoute Kitchens Private Limited agree to regulate equity ownership,
        board rights, transfer restrictions, and governing law.
        """
        classification = await classify_document(text, "India")
        self.assertEqual(classification.document_type, "Shareholders Agreement")
        self.assertEqual(classification.matched_playbook_id, "shareholders_agreement_india")
        self.assertGreater(classification.confidence, 0.8)

    def test_playbook_coverage_detects_missing_fields(self) -> None:
        playbook = load_playbook_for_type("Shareholders Agreement", "India")
        extraction = LegalExtraction(
            contract_type="Shareholders Agreement",
            parties=[FlexibleEntity(name="SpiceRoute Kitchens Private Limited", entity_type="organization")],
            shareholders=[Shareholder(name="Amit Sharma", shares=40000)],
            financial_terms=[],
            exhibits=[],
            key_clauses=[
                Clause(
                    title="Ownership Of The Shares",
                    clause_type="shareholding",
                    summary="Ownership table",
                    source_text="table"
                ),
                Clause(
                    title="Disputes And Governing Law",
                    clause_type="disputes_governing_law",
                    summary="Indian law applies",
                    source_text="Indian law applies"
                ),
            ],
            relationships=[],
            red_flags=[],
            drafting_style_detected="sectioned",
            completeness_score=80,
            short_summary="Shareholders Agreement extraction",
        )
        extracted, missing = calculate_playbook_coverage(extraction, playbook)
        self.assertIn("parties", extracted)
        self.assertIn("shareholders", extracted)
        self.assertIn("governing_law", extracted)
        self.assertIn("financial_terms", missing)

    def test_playbooks_are_available(self) -> None:
        playbooks = load_playbooks()
        self.assertGreaterEqual(len(playbooks), 5)

    def test_graph_builder_links_cross_references(self) -> None:
        extraction = LegalExtraction(
            contract_type="Service Agreement",
            parties=[FlexibleEntity(name="ClientCo"), FlexibleEntity(name="VendorCo")],
            shareholders=[],
            financial_terms=[],
            exhibits=[],
            key_clauses=[Clause(title="Payment Terms", summary="See Clause 5.2", source_text="See Clause 5.2")],
            relationships=[],
            cross_references=[CrossReference(source_label="Payment Terms", target_label="Clause 5.2", reference_text="Clause 5.2")],
            red_flags=[],
            drafting_style_detected="sectioned",
            completeness_score=75,
            short_summary="Service Agreement extraction",
        )
        nodes, edges = build_document_graph(extraction)
        self.assertTrue(any(node.label == "Payment Terms" for node in nodes))
        self.assertTrue(any(edge.relation == "clause_reference" for edge in edges))


if __name__ == "__main__":
    unittest.main()
