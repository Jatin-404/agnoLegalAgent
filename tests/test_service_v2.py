import unittest
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from agno.knowledge.document import Document

from app.core.config import settings
from app.core.schemas import SingleExtractionRequest
from app.core.service_v2 import _extract_section, _extract_sections_in_parallel, run_single_extraction_v2


class _FakeAgent:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0

    async def arun(self, input=None):  # noqa: ANN001
        self.calls += 1
        return SimpleNamespace(content=self.payload)


class _SlowAgent:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self.calls = 0

    async def arun(self, input=None):  # noqa: ANN001
        self.calls += 1
        await asyncio.sleep(self.delay_seconds)
        return SimpleNamespace(content='{"contract_type":"Shareholders Agreement","parties":[],"shareholders":[],"key_clauses":[],"red_flags":[],"drafting_style_detected":"sectioned","short_summary":"slow"}')


class ServiceV2Tests(unittest.IsolatedAsyncioTestCase):
    async def test_run_single_extraction_v2_heuristic_mode_extracts_core_fields(self) -> None:
        text = """
        # Shareholders Agreement

        SpiceRoute Kitchens Private Limited (hereinafter referred to as the Company)

        a. Amit Sharma, Gurgaon, Haryana (hereinafter referred to as AS)
        b. Rahul Verma, Noida, Uttar Pradesh (hereinafter referred to as RV)

        ## Ownership Of The Shares
        | Shareholder | Number of Shares | Percentage |
        |-------------|------------------|------------|
        | Amit Sharma | 40,000 | 40% |
        | Rahul Verma | 60,000 | 60% |
        """
        request = SingleExtractionRequest(
            document_name="sample.docx",
            document_text=text,
            jurisdiction_hint="India",
            document_metadata={"parser": "docling"},
        )

        with (
            patch.object(settings, "enable_classifier_agent", False),
            patch.object(settings, "v2_enable_section_agent", False),
            patch.object(settings, "v2_enable_financial_agent", False),
            patch.object(settings, "v2_enable_exhibit_agent", False),
            patch.object(settings, "v2_enable_merge_agent", False),
            patch.object(settings, "v2_enable_relationship_agent", False),
            patch.object(settings, "v2_enable_taxonomy_agent", False),
        ):
            response = await run_single_extraction_v2(request, parser_mode="docling")

        self.assertFalse(response.truncated)
        self.assertGreaterEqual(len(response.extraction.parties), 2)
        self.assertEqual(len(response.extraction.shareholders), 2)
        self.assertTrue(response.extraction.short_summary)
        self.assertEqual(response.extraction.playbook_id, "shareholders_agreement_india")
        self.assertGreater(response.extraction.classifier_confidence, 0.8)
        self.assertIn("parties", response.extraction.playbook_fields_extracted)

    async def test_extract_section_merges_model_output_with_heuristics(self) -> None:
        chunk = Document(
            name="sample.docx",
            content=(
                "## Ownership Of The Shares\n"
                "| Shareholder | Number of Shares | Percentage |\n"
                "|-------------|------------------|------------|\n"
                "| Amit Sharma | 40,000 | 40% |\n"
            ),
            meta_data={"section_title": "Ownership Of The Shares", "chunk_type": "table"},
        )
        fake_agent = _FakeAgent(
            (
                '{"contract_type":"Shareholders Agreement","parties":[],'
                '"shareholders":[],"key_clauses":[{"clause_id":"C9","clause_type":"shareholding",'
                '"title":"Ownership Of The Shares","summary":"Ownership is defined in a table.","source_text":"| Amit Sharma | 40,000 | 40% |",'
                '"linked_entities":[],"linked_exhibits":[],"importance":"high","confidence":0.91}],'
                '"red_flags":["cap table should be confirmed"],"drafting_style_detected":"table_heavy","short_summary":"Ownership section"}'
            )
        )

        with (
            patch.object(settings, "enable_classifier_agent", False),
            patch.object(settings, "v2_enable_section_agent", True),
            patch("app.core.service_v2.get_v2_section_agent", return_value=fake_agent),
        ):
            result = await _extract_section(chunk=chunk, contract_type_hint="Shareholders Agreement", use_model=True)

        self.assertEqual(len(result.shareholders), 1)
        self.assertEqual(result.shareholders[0].name, "Amit Sharma")
        self.assertIn("cap table should be confirmed", result.red_flags)
        self.assertEqual(result.key_clauses[0].clause_type, "shareholding")

    async def test_extract_sections_in_parallel_respects_model_budget(self) -> None:
        chunks = [
            Document(name="a.docx", content=f"## Clause {index}\nThe shareholders shall vote together.", meta_data={"chunk_id": index, "section_title": f"Clause {index}", "chunk_type": "section"})
            for index in range(1, 4)
        ]
        fake_agent = _FakeAgent(
            '{"contract_type":"Shareholders Agreement","parties":[],"shareholders":[],"key_clauses":[],"red_flags":[],"drafting_style_detected":"sectioned","short_summary":"chunk"}'
        )

        with (
            patch.object(settings, "enable_classifier_agent", False),
            patch.object(settings, "v2_enable_section_agent", True),
            patch.object(settings, "v2_max_model_chunks", 1),
            patch("app.core.service_v2.get_v2_section_agent", return_value=fake_agent),
        ):
            results = await _extract_sections_in_parallel(chunks, "Shareholders Agreement")

        self.assertEqual(len(results), 3)
        self.assertEqual(fake_agent.calls, 1)

    async def test_extract_section_timeout_falls_back_to_heuristics(self) -> None:
        chunk = Document(
            name="sample.docx",
            content="a. Amit Sharma, Gurgaon (hereinafter referred to as AS)\n| Amit Sharma | 40,000 | 40% |",
            meta_data={"chunk_id": 1, "section_title": "Ownership", "chunk_type": "table"},
        )
        slow_agent = _SlowAgent(delay_seconds=1.2)

        with (
            patch.object(settings, "enable_classifier_agent", False),
            patch.object(settings, "v2_enable_section_agent", True),
            patch.object(settings, "v2_agent_call_timeout_seconds", 1),
            patch.object(settings, "structured_retries", 1),
            patch("app.core.service_v2.get_v2_section_agent", return_value=slow_agent),
        ):
            result = await _extract_section(chunk=chunk, contract_type_hint="Shareholders Agreement", use_model=True)

        self.assertEqual(slow_agent.calls, 1)
        self.assertEqual(len(result.shareholders), 1)
        self.assertEqual(result.shareholders[0].name, "Amit Sharma")


if __name__ == "__main__":
    unittest.main()
