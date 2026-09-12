import importlib.util
import sys
import unittest
import json
import tempfile
from unittest import mock
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "plan_shots.py"
SPEC = importlib.util.spec_from_file_location("plan_shots", SCRIPT)
assert SPEC and SPEC.loader
PLANNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLANNER
SPEC.loader.exec_module(PLANNER)


class ShotPlannerTests(unittest.TestCase):
    def test_sentence_units_preserve_incomplete_final_sentence(self):
        units = PLANNER.split_narration_units(
            "First sentence. Second question? An unfinished final thought"
        )
        self.assertEqual(
            units,
            ["First sentence.", "Second question?", "An unfinished final thought"],
        )

    def test_plan_must_cover_all_units_contiguously(self):
        plan = {
            "beats": [
                {
                    "unit_start": 0,
                    "unit_end": 0,
                    "visual_description": "A concrete visual description for frame one.",
                    "motion": "zoom_in",
                },
                {
                    "unit_start": 2,
                    "unit_end": 2,
                    "visual_description": "A concrete visual description for frame two.",
                    "motion": "pan_right",
                },
            ]
        }
        with self.assertRaisesRegex(PLANNER.ShotPlanningError, "expected 1"):
            PLANNER.validate_plan(plan, 3, 2)

    def test_materialization_preserves_text_and_style_prefix(self):
        units = ["First sentence.", "Second sentence.", "Third sentence."]
        beats = [
            {
                "unit_start": 0,
                "unit_end": 1,
                "visual_description": "Subject one in a specific environment.",
                "motion": "zoom_in",
            },
            {
                "unit_start": 2,
                "unit_end": 2,
                "visual_description": "Subject two in a different environment.",
                "motion": "pan_left",
            },
        ]
        output = PLANNER.materialize_shots(beats, units, "STYLE: ", "test")
        self.assertEqual(len(output["shots"]), 2)
        self.assertEqual(
            output["shots"][0]["narration_lines"], units[:2]
        )
        self.assertTrue(output["shots"][1]["image_prompt"].startswith("STYLE: "))
        self.assertEqual(output["shots"][1]["image_file"], "shot_002.png")

    def test_long_form_target_is_in_expected_range(self):
        self.assertEqual(PLANNER.target_shot_count(1900, 100), 54)

    def test_dry_run_does_not_require_codex_auth_or_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            input_dir = project / "input"
            input_dir.mkdir()
            (input_dir / "transcript.txt").write_text("A short sentence.", encoding="utf-8")
            (input_dir / "shot_style.json").write_text(
                json.dumps({"style_lock": "LOCK: "}), encoding="utf-8"
            )
            with mock.patch.object(
                PLANNER, "verify_chatgpt_auth", side_effect=AssertionError("auth should be skipped")
            ), mock.patch.object(
                PLANNER.shutil, "which", side_effect=AssertionError("which should be skipped")
            ), mock.patch.object(
                sys, "argv", ["plan_shots.py", str(project), "--dry-run"]
            ):
                self.assertEqual(PLANNER.main(), 0)


if __name__ == "__main__":
    unittest.main()
