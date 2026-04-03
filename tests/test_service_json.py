import unittest

from app.core.schemas import ClauseTaxonomyResult, RelationshipsExtraction
from app.core.extraction_utils import _coerce_content


class RelaxedJsonParsingTests(unittest.TestCase):
    def test_coerce_content_parses_fenced_json(self) -> None:
        raw = '```json\n{"clause_type":"governance"}\n```'
        parsed = _coerce_content(raw, ClauseTaxonomyResult)
        self.assertEqual(parsed.clause_type, "governance")

    def test_coerce_content_parses_valid_prefix_with_trailing_text(self) -> None:
        raw = (
            '{"items":[{"source_entity":"Amit Sharma","target_entity":"SpiceRoute Kitchens Private Limited",'
            '"relationship_type":"role","condition":"while shareholding remains effective","impact":"holds shares",'
            '"source_text":"row","confidence":0.83}]}\nThe rest of this reply was accidental.'
        )
        parsed = _coerce_content(raw, RelationshipsExtraction)
        self.assertEqual(len(parsed.items), 1)
        self.assertEqual(parsed.items[0].source_entity, "Amit Sharma")

    def test_coerce_content_strips_trailing_commas(self) -> None:
        raw = '{"clause_type":"financial",}'
        parsed = _coerce_content(raw, ClauseTaxonomyResult)
        self.assertEqual(parsed.clause_type, "financial")


if __name__ == "__main__":
    unittest.main()
