import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "assemble_video.py"
SPEC = importlib.util.spec_from_file_location("assemble_video", SCRIPT)
assert SPEC and SPEC.loader
ASSEMBLER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ASSEMBLER
SPEC.loader.exec_module(ASSEMBLER)


class AssemblerTests(unittest.TestCase):
    def make_project(self, root: Path) -> tuple[dict, dict, Path]:
        input_dir = root / "input"
        images_dir = input_dir / "images"
        images_dir.mkdir(parents=True)
        (images_dir / "shot_001.png").touch()
        (images_dir / "shot_002.png").touch()
        (input_dir / "narration.wav").touch()

        shots = {
            "schema_version": "1.0",
            "project_id": "test",
            "shots": [
                {
                    "beat_id": "beat_001",
                    "image_file": "shot_001.png",
                    "narration_lines": ["Look at this huge spider."],
                    "image_prompt": "A spider",
                    "hold_seconds": 2.0,
                    "motion": {"type": "zoom_in", "strength": 0.08},
                },
                {
                    "beat_id": "beat_002",
                    "image_file": "shot_002.png",
                    "narration_lines": ["Absolutely disgusting, right?"],
                    "image_prompt": "A reaction",
                    "hold_seconds": 2.0,
                },
            ],
        }
        word_text = [
            "Look",
            "at",
            "this",
            "huge",
            "spider",
            "Absolutely",
            "disgusting",
            "right",
        ]
        words = [
            {
                "index": index,
                "word": word,
                "start": index * 0.5,
                "end": index * 0.5 + 0.4,
                "probability": 0.99,
            }
            for index, word in enumerate(word_text)
        ]
        timestamps = {
            "schema_version": "1.0",
            "language": "en",
            "audio_file": "narration.wav",
            "duration_seconds": 4.0,
            "words": words,
        }
        return shots, timestamps, input_dir

    def test_timeline_uses_first_word_of_next_beat_as_cut(self):
        with tempfile.TemporaryDirectory() as directory:
            shots, timestamps, input_dir = self.make_project(Path(directory))
            timeline, audio_file, score = ASSEMBLER.build_timeline(
                shots, timestamps, input_dir, minimum_match=0.75
            )
        self.assertEqual(audio_file, "narration.wav")
        self.assertEqual(score, 1.0)
        self.assertEqual(timeline[0].start, 0.0)
        self.assertEqual(timeline[0].end, 2.5)
        self.assertEqual(timeline[1].start, 2.5)
        self.assertEqual(timeline[1].end, 4.0)
        self.assertEqual(timeline[1].motion["type"], "pan_right")

    def test_punctuation_and_case_do_not_hurt_matching(self):
        self.assertEqual(
            ASSEMBLER.tokens("It's SPIDER-shaped—right?"),
            ["it's", "spider", "shaped", "right"],
        )

    def test_weak_narration_match_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            shots, timestamps, input_dir = self.make_project(Path(directory))
            for word in timestamps["words"]:
                word["word"] = "unrelated"
            with self.assertRaisesRegex(
                ASSEMBLER.AssemblyError, "match is too weak"
            ):
                ASSEMBLER.build_timeline(
                    shots, timestamps, input_dir, minimum_match=0.75
                )

    def test_filter_graph_contains_each_shot_and_concat(self):
        with tempfile.TemporaryDirectory() as directory:
            shots, timestamps, input_dir = self.make_project(Path(directory))
            timeline, _, _ = ASSEMBLER.build_timeline(
                shots, timestamps, input_dir, minimum_match=0.75
            )
            graph = ASSEMBLER.build_filter_graph(timeline, 1920, 1080, 30)
        self.assertIn("[0:v]", graph)
        self.assertIn("[1:v]", graph)
        self.assertIn("zoompan=", graph)
        self.assertIn("3-2*", graph)
        self.assertIn("concat=n=2:v=1:a=0[outv]", graph)

    def test_static_motion_bypasses_zoompan(self):
        with tempfile.TemporaryDirectory() as directory:
            shots, timestamps, input_dir = self.make_project(Path(directory))
            shots["shots"][0]["motion"] = {"type": "static"}
            timeline, _, _ = ASSEMBLER.build_timeline(
                shots, timestamps, input_dir, minimum_match=0.75
            )
            graph = ASSEMBLER.build_filter_graph(timeline, 1080, 1920, 30)
        self.assertNotIn("zoompan=", graph.split("[v0]")[0])
        self.assertIn("fps=30,trim=duration=2.500000", graph)
        self.assertEqual(timeline[0].motion["strength"], 0.0)

    def test_all_static_render_uses_intra_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shots, timestamps, input_dir = self.make_project(root)
            for shot in shots["shots"]:
                shot["motion"] = {"type": "static"}
            timeline, _, _ = ASSEMBLER.build_timeline(
                shots, timestamps, input_dir, minimum_match=0.75
            )
            command = ASSEMBLER.build_ffmpeg_command(
                timeline,
                input_dir / "narration.wav",
                root / "filter.txt",
                root / "out.mp4",
                width=1080,
                height=1920,
                fps=30,
                overwrite=True,
            )
        self.assertEqual(command[command.index("-g") + 1], "1")

    def test_dimensions_are_inferred_from_shot_aspect_ratio(self):
        self.assertEqual(ASSEMBLER.infer_dimensions({"aspect_ratio": "9:16"}), (1080, 1920))
        self.assertEqual(ASSEMBLER.infer_dimensions({"aspect_ratio": "16:9"}), (1920, 1080))
        self.assertEqual(ASSEMBLER.infer_dimensions({}), (1920, 1080))
        with self.assertRaisesRegex(ASSEMBLER.AssemblyError, "aspect_ratio"):
            ASSEMBLER.infer_dimensions({"aspect_ratio": "1:1"})

    def test_json_contract_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shots, timestamps, input_dir = self.make_project(root)
            shots_path = input_dir / "shots.json"
            timestamps_path = input_dir / "timestamps.json"
            shots_path.write_text(json.dumps(shots), encoding="utf-8")
            timestamps_path.write_text(json.dumps(timestamps), encoding="utf-8")
            self.assertEqual(ASSEMBLER.load_json(shots_path), shots)
            self.assertEqual(ASSEMBLER.load_json(timestamps_path), timestamps)


if __name__ == "__main__":
    unittest.main()
