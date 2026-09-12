import importlib.util
import sys
import unittest
from pathlib import Path


def load_module(name, filename):
    path = Path(__file__).parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RESEARCH = load_module("research_topic", "research_topic.py")
OUTLINE = load_module("build_outline", "build_outline.py")


class ResearchOutlineTests(unittest.TestCase):
    def test_research_packet_reports_missing_coverage(self):
        facts = []
        for index in range(1, 9):
            facts.append(
                {
                    "fact_id": f"fact_{index:03d}",
                    "claim": "A supported claim.",
                    "category": "study",
                    "source_urls": [f"https://example.edu/{index}"],
                    "evidence_summary": "Evidence summary.",
                    "caveats": "",
                }
            )
        packet = RESEARCH.validate_packet({"facts": facts})
        self.assertFalse(packet["coverage"]["outline_ready"])
        self.assertIn("history", packet["coverage"]["missing_recommended_categories"])

    def test_outline_rejects_unknown_fact(self):
        beats = []
        for index, name in enumerate(OUTLINE.BEAT_NAMES, start=1):
            beats.append(
                {
                    "beat_number": index,
                    "beat_name": name,
                    "purpose": "A sufficiently detailed purpose.",
                    "fact_ids": ["fact_999"] if index == 1 else [],
                    "narrative_moves": ["A sufficiently detailed narrative move."],
                }
            )
        with self.assertRaisesRegex(OUTLINE.OutlineError, "unknown facts"):
            OUTLINE.validate_outline({"beats": beats}, {"fact_001"})

    def test_outline_requires_fixed_order(self):
        beats = []
        for index, name in enumerate(reversed(OUTLINE.BEAT_NAMES), start=1):
            beats.append(
                {
                    "beat_number": index,
                    "beat_name": name,
                    "purpose": "A sufficiently detailed purpose.",
                    "fact_ids": [],
                    "narrative_moves": ["A sufficiently detailed narrative move."],
                }
            )
        with self.assertRaisesRegex(OUTLINE.OutlineError, "wrong beat_name"):
            OUTLINE.validate_outline({"beats": beats}, set())


if __name__ == "__main__":
    unittest.main()
