import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.core.schemas import LegalExtraction, SingleExtractionResponse


class RouteTests(unittest.TestCase):
    def test_extract_route_uses_main_pipeline(self) -> None:
        fake_response = SingleExtractionResponse(
            extraction=LegalExtraction(
                contract_type="Shareholders Agreement",
                parties=[],
                shareholders=[],
                financial_terms=[],
                exhibits=[],
                key_clauses=[],
                relationships=[],
                red_flags=[],
                drafting_style_detected="sectioned",
                completeness_score=75,
                short_summary="Shareholders Agreement extraction",
            ),
            truncated=False,
            model_used="qwen2.5:7b",
            processing_ms=1234,
            parser_mode="text",
            total_chunks=1,
        )

        with patch("app.api.routes.run_single_extraction_v2", new=AsyncMock(return_value=fake_response)):
            client = TestClient(app)
            response = client.post(
                "/extract",
                json={
                    "document_name": "sample.txt",
                    "document_text": "Shareholders Agreement",
                    "jurisdiction_hint": "India",
                    "document_metadata": {},
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model_used"], "qwen2.5:7b")
        self.assertEqual(payload["extraction"]["contract_type"], "Shareholders Agreement")

    def test_v1_route_is_removed(self) -> None:
        client = TestClient(app)
        response = client.post(
            "/v1/extract",
            json={
                "document_name": "sample.txt",
                "document_text": "Test",
                "jurisdiction_hint": "India",
                "document_metadata": {},
            },
        )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
