import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "fact_check_script.py"
SPEC = importlib.util.spec_from_file_location("fact_check_script", SCRIPT)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)


class FactCheckerTests(unittest.TestCase):
    def claim(self, verdict="verified", action="keep"):
        return {
            "claim_id": "claim_001",
            "claim": "A testable factual claim.",
            "category": "study",
            "verdict": verdict,
            "evidence": [
                {
                    "url": "https://example.edu/study",
                    "title": "Study",
                    "finding": "The source supports or rejects the claim.",
                }
            ],
            "explanation": "A sufficiently detailed explanation.",
            "recommended_action": action,
            "suggested_replacement": "",
        }

    def test_summary_is_computed_locally(self):
        report = CHECKER.validate_report({"claims": [self.claim()]})
        self.assertEqual(report["summary"]["verified"], 1)
        self.assertTrue(report["summary"]["ready_for_final"])

    def test_non_verified_claim_cannot_be_kept(self):
        with self.assertRaisesRegex(CHECKER.FactCheckError, "cannot keep"):
            CHECKER.validate_report(
                {"claims": [self.claim("unverified", "keep")]}
            )

    def test_claim_ids_must_be_sequential(self):
        claim = self.claim()
        claim["claim_id"] = "claim_009"
        with self.assertRaisesRegex(CHECKER.FactCheckError, "claim_001"):
            CHECKER.validate_report({"claims": [claim]})

    def test_schema_requires_evidence(self):
        evidence_schema = (
            CHECKER.factcheck_schema()["properties"]["claims"]["items"]["properties"][
                "evidence"
            ]
        )
        self.assertEqual(evidence_schema["minItems"], 1)

    def test_prompt_explicitly_protects_negation(self):
        prompt = CHECKER.build_prompt("The study does not establish fear.")
        self.assertIn("Never drop or", prompt)
        self.assertIn("does not establish fear", prompt)

    def framing_claim(self, text, position=1):
        claim = self.claim()
        claim["claim_id"] = f"claim_{position:03d}"
        claim["claim"] = text
        claim["category"] = "framing"
        return claim

    def test_framing_claim_is_accepted(self):
        report = CHECKER.validate_report(
            {"claims": [self.framing_claim("The obvious explanation is not enough.")]}
        )
        self.assertEqual(report["summary"]["framing_claims"], 1)
        self.assertTrue(report["summary"]["ready_for_final"])

    def test_framing_cannot_smuggle_factual_content(self):
        for text in (
            "Researchers showed that the boundary is cultural.",
            "In 2021 the divide was not about danger.",
            "The study did not settle the question.",
        ):
            with self.subTest(text=text):
                with self.assertRaisesRegex(CHECKER.FactCheckError, "framing"):
                    CHECKER.validate_report({"claims": [self.framing_claim(text)]})

    def test_framing_cannot_dominate_the_report(self):
        words = (
            "one", "two", "three", "four", "five",
            "six", "seven", "eight", "nine", "ten",
        )
        claims = [
            self.framing_claim(f"An authorial reframe called {word}.", position)
            for position, word in enumerate(words, start=1)
        ]
        with self.assertRaisesRegex(CHECKER.FactCheckError, "ceiling"):
            CHECKER.validate_report({"claims": claims})

    def test_framing_ceiling_ignores_small_reports(self):
        claims = [self.framing_claim("A short authorial reframe.", 1)]
        report = CHECKER.validate_report({"claims": claims})
        self.assertEqual(report["summary"]["framing_claims"], 1)


if __name__ == "__main__":
    unittest.main()
