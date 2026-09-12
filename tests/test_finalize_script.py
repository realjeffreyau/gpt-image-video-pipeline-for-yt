import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "finalize_script.py"
SPEC = importlib.util.spec_from_file_location("finalize_script", SCRIPT)
assert SPEC and SPEC.loader
FINALIZER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FINALIZER
SPEC.loader.exec_module(FINALIZER)


class FinalizerTests(unittest.TestCase):
    def make_report(self, draft: Path, text: str):
        return {
            "summary": {
                "ready_for_final": True,
                "contradicted": 0,
                "unverified": 0,
                "total_claims": 1,
            },
            "draft_file": str(draft),
            "draft_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "writer_model": "writer-model",
            "checker_model": "checker-model",
        }

    def test_clean_hash_matched_draft_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            draft = Path(directory) / "draft.txt"
            text = "A checked narration."
            draft.write_text(text, encoding="utf-8")
            accepted, digest = FINALIZER.verify_finalizable(
                draft, self.make_report(draft, text)
            )
            self.assertEqual(accepted, text)
            self.assertEqual(digest, hashlib.sha256(text.encode()).hexdigest())

    def test_changed_draft_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            draft = Path(directory) / "draft.txt"
            draft.write_text("Changed narration.", encoding="utf-8")
            report = self.make_report(draft, "Original narration.")
            with self.assertRaisesRegex(
                FINALIZER.FinalizationError, "changed after fact checking"
            ):
                FINALIZER.verify_finalizable(draft, report)

    def test_unready_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            draft = Path(directory) / "draft.txt"
            text = "A checked narration."
            draft.write_text(text, encoding="utf-8")
            report = self.make_report(draft, text)
            report["summary"]["ready_for_final"] = False
            with self.assertRaisesRegex(
                FINALIZER.FinalizationError, "not ready_for_final"
            ):
                FINALIZER.verify_finalizable(draft, report)


if __name__ == "__main__":
    unittest.main()
