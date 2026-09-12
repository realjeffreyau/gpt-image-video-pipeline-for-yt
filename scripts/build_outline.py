#!/usr/bin/env python3
"""Map a research packet onto the fixed eight-beat essay structure."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


class OutlineError(Exception):
    """A user-actionable outline-stage error."""


BEAT_NAMES = (
    "absurd_visual_hook",
    "plain_contradiction",
    "obvious_explanations_rebutted",
    "historical_reversal",
    "cultural_counterexample",
    "named_research_and_mechanism",
    "generalization",
    "viewer_reframe",
)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise OutlineError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise OutlineError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OutlineError(f"{path} must contain a JSON object")
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
        raise OutlineError("Codex auth_mode must be 'chatgpt'")


def outline_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["working_thesis", "beats"],
        "properties": {
            "working_thesis": {"type": "string", "minLength": 20},
            "beats": {
                "type": "array",
                "minItems": 8,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "beat_number",
                        "beat_name",
                        "purpose",
                        "fact_ids",
                        "narrative_moves",
                    ],
                    "properties": {
                        "beat_number": {"type": "integer", "minimum": 1, "maximum": 8},
                        "beat_name": {"type": "string", "enum": list(BEAT_NAMES)},
                        "purpose": {"type": "string", "minLength": 15},
                        "fact_ids": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "pattern": "^fact_[0-9]{3}$",
                            },
                        },
                        "narrative_moves": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string", "minLength": 10},
                        },
                    },
                },
            },
        },
    }


def build_prompt(research: dict[str, Any], target_words: int) -> str:
    return f"""You are outlining a long-form curiosity essay of about
{target_words} spoken words.

Map the supplied research packet onto exactly these eight beats, in this order:
{json.dumps(list(BEAT_NAMES), indent=2)}

Rules:
- Use only facts in the packet. Every factual narrative move must cite one or
  more corresponding fact_ids.
- Do not add names, studies, institutions, dates, numbers, anecdotes, or
  cultural practices.
- The hook should be concrete and visual without announcing the thesis.
- Beat 3 should test multiple intuitive explanations with quick rebuttals.
- Beat 6 should carry the causal explanation and correctly connect named work.
- The final beat should reframe the viewer's assumptions, not merely summarize.
- Respect every caveat in the research packet.
- Keep each fact_id attached to the beat where the writer may use it.

Return only the JSON required by the supplied schema.

<research_packet>
{json.dumps(research, ensure_ascii=False, indent=2)}
</research_packet>"""


def run_outliner(
    codex: str,
    model: str,
    project_dir: Path,
    prompt: str,
    schema_path: Path,
    raw_path: Path,
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
        "--model",
        model,
        "--cd",
        str(project_dir),
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(raw_path),
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
        raise OutlineError(f"outliner timed out after {timeout_seconds}s") from exc
    combined = (result.stdout + "\n" + result.stderr).strip()
    log_path.write_text(combined + ("\n" if combined else ""), encoding="utf-8")
    if result.returncode:
        tail = "\n".join(combined.splitlines()[-12:])
        raise OutlineError(f"outliner exited {result.returncode}\n{tail}")


def validate_outline(
    outline: dict[str, Any], known_fact_ids: set[str]
) -> dict[str, Any]:
    beats = outline.get("beats")
    if not isinstance(beats, list) or len(beats) != 8:
        raise OutlineError("outline must contain exactly eight beats")
    used: set[str] = set()
    for position, beat in enumerate(beats, start=1):
        if beat.get("beat_number") != position:
            raise OutlineError(f"beat {position} has the wrong beat_number")
        if beat.get("beat_name") != BEAT_NAMES[position - 1]:
            raise OutlineError(f"beat {position} has the wrong beat_name")
        fact_ids = beat.get("fact_ids")
        if not isinstance(fact_ids, list):
            raise OutlineError(f"beat {position} fact_ids must be an array")
        unknown = set(fact_ids) - known_fact_ids
        if unknown:
            raise OutlineError(
                f"beat {position} references unknown facts: {sorted(unknown)}"
            )
        used.update(fact_ids)
    outline["coverage"] = {
        "used_fact_ids": sorted(used),
        "unused_fact_ids": sorted(known_fact_ids - used),
    }
    return outline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the fixed eight-beat outline from research.json."
    )
    parser.add_argument("project", type=Path)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--overwrite", action="store_true")
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
        verify_chatgpt_auth(args.auth_file.expanduser().resolve())
        codex = shutil.which("codex")
        if not codex:
            raise OutlineError("codex CLI is not installed or not on PATH")
        project_dir = args.project.resolve()
        input_dir = project_dir / "input"
        topic = load_json(input_dir / "topic.json")
        research = load_json(input_dir / "research.json")
        if research.get("coverage", {}).get("outline_ready") is not True:
            raise OutlineError("research packet is not outline_ready")
        known_fact_ids = {
            fact["fact_id"] for fact in research.get("facts", []) if isinstance(fact, dict)
        }
        target_words = topic.get("target_words", 1900)
        if isinstance(target_words, bool) or not isinstance(target_words, int):
            raise OutlineError("topic target_words must be an integer")
        output_path = input_dir / "outline.json"
        if output_path.exists() and not args.overwrite:
            raise OutlineError(
                f"{output_path} already exists; use --overwrite to replace it"
            )
        work_dir = project_dir / "work" / "outline"
        work_dir.mkdir(parents=True, exist_ok=True)
        schema_path = work_dir / "outline.schema.json"
        raw_path = work_dir / "raw-response.json"
        log_path = work_dir / "codex.log"
        write_json_atomic(schema_path, outline_schema())
        print(
            f"ChatGPT auth verified; outliner={args.model}; "
            f"available facts={len(known_fact_ids)}"
        )
        run_outliner(
            codex,
            args.model,
            project_dir,
            build_prompt(research, target_words),
            schema_path,
            raw_path,
            log_path,
            timeout_seconds=args.timeout,
        )
        outline = validate_outline(load_json(raw_path), known_fact_ids)
        outline["schema_version"] = "1.0"
        outline["outliner_model"] = args.model
        write_json_atomic(output_path, outline)
        print(
            f"Created {output_path}; used "
            f"{len(outline['coverage']['used_fact_ids'])}/{len(known_fact_ids)} facts"
        )
        return 0
    except OutlineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
