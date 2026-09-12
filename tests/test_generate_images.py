import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_images.py"
SPEC = importlib.util.spec_from_file_location("generate_images", SCRIPT)
assert SPEC and SPEC.loader
GENERATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GENERATOR
SPEC.loader.exec_module(GENERATOR)


class ImageGeneratorTests(unittest.TestCase):
    def make_png(self, path: Path, width: int = 1024, height: int = 1024):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            b"\x89PNG\r\n\x1a\n"
            + struct.pack(">I", 13)
            + b"IHDR"
            + struct.pack(">II", width, height)
        )

    def test_auth_requires_chatgpt_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auth.json"
            path.write_text(json.dumps({"auth_mode": "api_key"}), encoding="utf-8")
            with self.assertRaisesRegex(
                GENERATOR.ImageGenerationError, "not 'chatgpt'"
            ):
                GENERATOR.verify_chatgpt_auth(path)

    def test_png_header_and_dimensions_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shot.png"
            self.make_png(path, 1536, 1024)
            self.assertEqual(GENERATOR.validate_png(path), (1536, 1024))

    def test_style_lock_must_be_exact_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            images = Path(directory) / "images"
            data = {
                "schema_version": "1.0",
                "style_lock": "LOCK: ",
                "shots": [
                    {
                        "beat_id": "beat_001",
                        "image_file": "shot_001.png",
                        "image_prompt": "Different prefix",
                    }
                ],
            }
            with self.assertRaisesRegex(
                GENERATOR.ImageGenerationError, "exact style_lock"
            ):
                GENERATOR.parse_shots(data, images)

    def test_prompt_hash_changes_with_visual_description(self):
        first = GENERATOR.prompt_hash("LOCK", "LOCK spider")
        second = GENERATOR.prompt_hash("LOCK", "LOCK crab")
        self.assertNotEqual(first, second)

    def test_codex_prompt_confines_output(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            shot = {
                "output_path": project / "input" / "images" / "shot_001.png",
                "image_prompt": "LOCK: spider",
            }
            prompt = GENERATOR.build_codex_prompt(shot, project)
            self.assertIn("input/images/shot_001.png", prompt)
            self.assertIn("Do not edit or create any other project file", prompt)

    def test_vertical_aspect_prompt_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "input" / "images"
            data = {
                "schema_version": "1.0",
                "aspect_ratio": "9:16",
                "style_lock": "LOCK: ",
                "shots": [
                    {
                        "beat_id": "beat_001",
                        "image_file": "shot_001.png",
                        "image_prompt": "LOCK: a robot",
                    }
                ],
            }
            _, shots = GENERATOR.parse_shots(data, images)
            self.assertEqual(shots[0]["aspect_ratio"], "9:16")
            prompt = GENERATOR.build_codex_prompt(shots[0], root.resolve())
            self.assertIn("Portrait 9:16", prompt)
            path = images / "shot_001.png"
            self.make_png(path, 1024, 1792)
            self.assertEqual(
                GENERATOR.validate_png(path, aspect_ratio="9:16"), (1024, 1792)
            )
            with self.assertRaisesRegex(GENERATOR.ImageGenerationError, "landscape"):
                GENERATOR.validate_png(path, aspect_ratio="16:9")

    def test_dry_run_does_not_require_codex_auth_or_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "input").mkdir()
            (project / "input" / "shots.json").write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "style_lock": "LOCK: ",
                        "shots": [
                            {
                                "beat_id": "beat_001",
                                "image_file": "shot_001.png",
                                "image_prompt": "LOCK: a simple robot",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                GENERATOR, "verify_chatgpt_auth", side_effect=AssertionError("auth should be skipped")
            ), mock.patch.object(
                GENERATOR.shutil, "which", side_effect=AssertionError("which should be skipped")
            ), mock.patch.object(
                sys, "argv", ["generate_images.py", str(project), "--dry-run"]
            ):
                self.assertEqual(GENERATOR.main(), 0)


if __name__ == "__main__":
    unittest.main()
