#!/usr/bin/env python3
"""Plan beat-level images for a finished narration script through Codex CLI."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


class ShotPlanningError(Exception):
    """A user-actionable shot-planning error."""


MOTIONS = ("zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down")


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise ShotPlanningError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ShotPlanningError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ShotPlanningError(f"{path} must contain a JSON object")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verify_chatgpt_auth(auth_path: Path) -> None:
    auth = load_json(auth_path)
    if auth.get("auth_mode") != "chatgpt":
        raise ShotPlanningError(
            "Codex must use ChatGPT authentication. Run `codex login` and choose "
            "ChatGPT before planning shots."
        )


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_narration_units(text: str) -> list[str]:
    normalized = normalize_whitespace(text)
    if not normalized:
        raise ShotPlanningError("transcript.txt is empty")
    units = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"“])', normalized)
    return [unit.strip() for unit in units if unit.strip()]


def target_shot_count(word_count: int, unit_count: int) -> int:
    desired = max(1, min(60, round(word_count / 35)))
    if word_count >= 45:
        desired = max(2, desired)
    return min(desired, unit_count)


def planner_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["beats"],
        "properties": {
            "beats": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "unit_start",
                        "unit_end",
                        "visual_description",
                        "motion",
                    ],
                    "properties": {
                        "unit_start": {"type": "integer", "minimum": 0},
                        "unit_end": {"type": "integer", "minimum": 0},
                        "visual_description": {"type": "string", "minLength": 20},
                        "motion": {"type": "string", "enum": list(MOTIONS)},
                    },
                },
            }
        },
    }


def build_prompt(units: list[str], target: int) -> str:
    numbered = "\n".join(
        f"[{index}] {unit}" for index, unit in enumerate(units)
    )
    return f"""You are the shot planner for a faceless long-form YouTube curiosity essay.

Group the numbered narration units below into visual beats. Return only the JSON
required by the supplied output schema.

Rules:
- Cover every unit exactly once, in order.
- Ranges must be contiguous: first unit_start is 0, the next unit_start is one
  more than the previous unit_end, and the final unit_end is {len(units) - 1}.
- Aim for {target} beats. Use fewer only when combining adjacent units creates a
  more natural visual beat.
- A beat should usually cover about 20-55 spoken words, not one image per line.
- Change images at conceptual turns, punchlines, historical reversals, named
  studies, examples, or reframes.
- visual_description describes one concrete, drawable frame. It must specify
  subject, action, environment, composition, and meaningful details.
- Write it as a noun-phrase description of what is visible, never as a sentence
  quoted or paraphrased from the narration. "A stick figure recoiling from a
  spider on a kitchen counter" is correct; restating the narrator's line is not.
- Refer to people as simple stick figures with round heads and thin limbs.
  Carry emotion through posture, gesture, and facial expression, since these
  figures have only dot eyes and a simple mouth.
- The frame must be readable with the sound off and with zero words rendered in
  the image. Never depend on lettering to convey meaning: no labels, captions,
  signs, headlines, book titles, screen text, chart legends, or written dates.
- Express abstract ideas through concrete objects, body language, scale, and
  side-by-side juxtaposition instead of labeled diagrams.
- Do not include style, medium, palette, lighting, aspect ratio, text, captions,
  or camera motion in visual_description; those are added deterministically.
- Choose subtle motion from the allowed enum. Vary it without rapid alternation.

Narration units:
{numbered}"""


def run_codex_planner(
    codex: str,
    project_dir: Path,
    prompt: str,
    schema_path: Path,
    response_path: Path,
    log_path: Path,
    *,
    timeout_seconds: int,
) -> None:
    command = [
        codex,
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--cd",
        str(project_dir),
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(response_path),
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        combined = ((exc.stdout or "") + "\n" + (exc.stderr or "")).strip()
        log_path.write_text(combined + "\n", encoding="utf-8")
        raise ShotPlanningError(
            f"Codex shot planner timed out after {timeout_seconds}s"
        ) from exc
    combined = (result.stdout + "\n" + result.stderr).strip()
    log_path.write_text(combined + ("\n" if combined else ""), encoding="utf-8")
    if result.returncode:
        tail = "\n".join(combined.splitlines()[-10:])
        raise ShotPlanningError(
            f"Codex shot planner exited {result.returncode}\n{tail}"
        )


def validate_plan(
    plan: dict[str, Any], unit_count: int, target: int
) -> list[dict[str, Any]]:
    beats = plan.get("beats")
    if not isinstance(beats, list) or not beats:
        raise ShotPlanningError("planner response has no beats")
    if len(beats) > 60:
        raise ShotPlanningError("planner returned more than 60 beats")
    minimum_reasonable = max(1, target - max(2, round(target * 0.25)))
    if len(beats) < minimum_reasonable:
        raise ShotPlanningError(
            f"planner returned only {len(beats)} beats; expected about {target}"
        )

    expected_start = 0
    for position, beat in enumerate(beats):
        if not isinstance(beat, dict):
            raise ShotPlanningError(f"planner beat {position} is not an object")
        start = beat.get("unit_start")
        end = beat.get("unit_end")
        if start != expected_start:
            raise ShotPlanningError(
                f"planner beat {position} starts at unit {start}; expected {expected_start}"
            )
        if isinstance(end, bool) or not isinstance(end, int) or end < start:
            raise ShotPlanningError(f"planner beat {position} has an invalid unit_end")
        if end >= unit_count:
            raise ShotPlanningError(f"planner beat {position} exceeds the transcript")
        description = beat.get("visual_description")
        if not isinstance(description, str) or len(description.strip()) < 20:
            raise ShotPlanningError(
                f"planner beat {position} visual_description is too short"
            )
        if beat.get("motion") not in MOTIONS:
            raise ShotPlanningError(f"planner beat {position} has invalid motion")
        expected_start = end + 1
    if expected_start != unit_count:
        raise ShotPlanningError(
            f"planner stopped at unit {expected_start - 1}; "
            f"transcript ends at {unit_count - 1}"
        )
    return beats


def materialize_shots(
    beats: list[dict[str, Any]],
    units: list[str],
    style_lock: str,
    project_id: str,
) -> dict[str, Any]:
    shots: list[dict[str, Any]] = []
    for position, beat in enumerate(beats, start=1):
        narration_lines = units[beat["unit_start"] : beat["unit_end"] + 1]
        word_count = len(re.findall(r"\b[\w’'-]+\b", " ".join(narration_lines)))
        description = normalize_whitespace(beat["visual_description"])
        shots.append(
            {
                "beat_id": f"beat_{position:03d}",
                "image_file": f"shot_{position:03d}.png",
                "narration_lines": narration_lines,
                "image_prompt": style_lock + description,
                "hold_seconds": round(max(2.5, word_count / 2.45), 2),
                "motion": {
                    "type": beat["motion"],
                    "strength": 0.06,
                    "focus_x": 0.5,
                    "focus_y": 0.5,
                },
            }
        )
    output = {
        "schema_version": "1.0",
        "project_id": project_id,
        "style_lock": style_lock,
        "shots": shots,
    }
    reconstructed = normalize_whitespace(
        " ".join(line for shot in shots for line in shot["narration_lines"])
    )
    source = normalize_whitespace(" ".join(units))
    if reconstructed != source:
        raise ShotPlanningError("internal error: materialized shots changed narration")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn a finished transcript into beat-level shots through Codex."
    )
    parser.add_argument("project", type=Path, help="project directory")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--auth-file",
        type=Path,
        default=Path.home() / ".codex" / "auth.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        print("error: --timeout must be greater than zero", file=sys.stderr)
        return 2
    try:
        project_dir = args.project.resolve()
        input_dir = project_dir / "input"
        transcript_path = input_dir / "transcript.txt"
        try:
            transcript = transcript_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ShotPlanningError(f"Missing required file: {transcript_path}") from exc
        style_data = load_json(input_dir / "shot_style.json")
        style_lock = style_data.get("style_lock")
        if not isinstance(style_lock, str) or not style_lock.strip():
            raise ShotPlanningError("shot_style.json style_lock must be non-empty")
        if not style_lock.endswith(" "):
            style_lock += " "

        units = split_narration_units(transcript)
        word_count = len(re.findall(r"\b[\w’'-]+\b", normalize_whitespace(transcript)))
        target = target_shot_count(word_count, len(units))
        print(
            f"{word_count} words, {len(units)} narration "
            f"units, target {target} shots"
        )
        if args.dry_run:
            for index, unit in enumerate(units):
                print(f"[{index}] {unit}")
            return 0

        # Keep dry-run a local preflight: it should not require credentials or
        # the external Codex binary just to inspect the planned units.
        verify_chatgpt_auth(args.auth_file.expanduser().resolve())
        codex = shutil.which("codex")
        if not codex:
            raise ShotPlanningError("codex CLI is not installed or not on PATH")

        output_path = input_dir / "shots.json"
        if output_path.exists() and not args.overwrite:
            raise ShotPlanningError(
                f"{output_path} already exists; use --overwrite to replace it"
            )
        work_dir = project_dir / "work"
        work_dir.mkdir(parents=True, exist_ok=True)
        schema_path = work_dir / "shot_plan.schema.json"
        response_path = work_dir / "shot_plan.response.json"
        log_path = work_dir / "shot_plan.codex.log"
        write_json_atomic(schema_path, planner_schema())
        run_codex_planner(
            codex,
            project_dir,
            build_prompt(units, target),
            schema_path,
            response_path,
            log_path,
            timeout_seconds=args.timeout,
        )
        plan = load_json(response_path)
        beats = validate_plan(plan, len(units), target)
        project_id = style_data.get("project_id", project_dir.name)
        if not isinstance(project_id, str) or not project_id:
            raise ShotPlanningError("shot_style.json project_id must be non-empty")
        shots = materialize_shots(beats, units, style_lock, project_id)
        write_json_atomic(output_path, shots)
        print(f"Created {output_path} with {len(shots['shots'])} shots")
        return 0
    except ShotPlanningError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
