import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "write_script.py"
SPEC = importlib.util.spec_from_file_location("write_script", SCRIPT)
assert SPEC and SPEC.loader
WRITER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = WRITER
SPEC.loader.exec_module(WRITER)


class WriterTests(unittest.TestCase):
    def test_revision_prompt_includes_only_flagged_claims(self):
        report = {
            "claims": [
                {
                    "claim_id": "claim_001",
                    "verdict": "verified",
                    "recommended_action": "keep",
                },
                {
                    "claim_id": "claim_002",
                    "verdict": "contradicted",
                    "recommended_action": "revise",
                    "suggested_replacement": "Corrected fact.",
                },
            ]
        }
        prompt = WRITER.revision_prompt("Draft text.", report)
        self.assertNotIn("claim_001", prompt)
        self.assertIn("claim_002", prompt)
        self.assertIn("Do not expand an excerpt", prompt)
        self.assertIn("Copy every unflagged sentence verbatim", prompt)
        self.assertIn("Keep the original joke, profanity", prompt)
        self.assertIn("do not turn the sentence into methodology language", prompt)

    def test_initial_prompt_forbids_new_facts(self):
        prompt = WRITER.initial_prompt({"beats": []}, {"sources": []}, 1900)
        self.assertIn("Do not add a researcher", prompt)
        self.assertIn("approximately 1900 words", prompt)

    def test_short_outline_uses_short_form_requirements(self):
        prompt = WRITER.initial_prompt(
            {
                "format": "short",
                "topic": "Why you cannot tickle yourself",
                "arc": ["hook", "mechanism"],
            },
            {"facts": []},
            185,
        )
        self.assertIn("60–65 second documentary short", prompt)
        self.assertIn("175–195 spoken words", prompt)
        self.assertIn("never fall below 168 words", prompt)
        self.assertIn("honest nuance or counterevidence", prompt)
        self.assertIn("first sentence must explicitly name the topic", prompt)
        self.assertIn("Have you ever wondered why", prompt)
        self.assertIn("uncensored curse words", prompt)
        self.assertIn("never a formal documentary narrator", prompt)
        self.assertIn("Why you cannot tickle yourself", prompt)
        self.assertNotIn("uncanny valley", prompt)
        self.assertNotIn("eight-beat curiosity-essay structure", prompt)

    def test_style_has_eight_beats(self):
        rules = WRITER.style_rules()
        for number in range(1, 9):
            self.assertIn(f"{number}.", rules)

    def test_shared_prose_rules_require_conversational_uncensored_voice(self):
        rules = WRITER.prose_rules()
        self.assertIn("friend at a bar", rules)
        self.assertIn("Cut every", rules)
        self.assertIn("never censor it", rules)
        self.assertIn("every one or two sentences", rules)


if __name__ == "__main__":
    unittest.main()
