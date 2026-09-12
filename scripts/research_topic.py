#!/usr/bin/env python3
"""Build a source-backed research packet for a video topic through Codex."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


class ResearchError(Exception):
    """A user-actionable research-stage error."""


CATEGORIES = (
    "mechanism",
    "study",
    "history",
    "culture",
    "counterexample",
    "number",
    "biography",
    "other",
)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise ResearchError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ResearchError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResearchError(f"{path} must contain a JSON object")
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
        raise ResearchError("Codex auth_mode must be 'chatgpt'")


def research_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["topic", "thesis_candidates", "facts"],
        "properties": {
            "topic": {"type": "string", "minLength": 3},
            "thesis_candidates": {
                "type": "array",
                "minItems": 2,
                "maxItems": 4,
                "items": {"type": "string", "minLength": 15},
            },
            "facts": {
                "type": "array",
                "minItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "fact_id",
                        "claim",
                        "category",
                        "source_urls",
                        "evidence_summary",
                        "caveats",
                    ],
                    "properties": {
                        "fact_id": {"type": "string", "pattern": "^fact_[0-9]{3}$"},
                        "claim": {"type": "string", "minLength": 10},
                        "category": {"type": "string", "enum": list(CATEGORIES)},
                        "source_urls": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "string",
                                "pattern": "^https?://",
                            },
                        },
                        "evidence_summary": {"type": "string", "minLength": 15},
                        "caveats": {"type": "string"},
                    },
                },
            },
        },
    }


def build_prompt(config: dict[str, Any]) -> str:
    topic = config.get("topic")
    audience = config.get("audience", "general adult audience")
    target_words = config.get("target_words", 1900)
    return f"""You are the research stage for a fact-sensitive YouTube curiosity essay.

Research this topic using web search: {topic}
Audience: {audience}
Planned narration length: about {target_words} words.

Build a packet broad enough to support:
- an absurd visual hook and plain contradiction,
- several obvious explanations with fast evidence-based rebuttals,
- a historical reversal,
- a cultural counterexample,
- the real mechanism through named researchers and named studies,
- generalization to other examples,
- a final reframe.

Rules:
- Search for every candidate fact and include only facts supported by URLs you
  actually checked.
- Prefer primary papers, PubMed records, universities, government agencies,
  museums, archives, and reputable scholarly references.
- Surprising historical anecdotes, named studies, dates, and numbers should
  have two independent URLs where possible.
- Split facts atomically. Record caveats and limits rather than smoothing them
  away.
- Do not write narration and do not force a thesis before reviewing evidence.
- Never invent a person, institution, study, quotation, date, number, or URL.
- Assign sequential IDs from fact_001.

Return only the JSON required by the supplied schema."""


def run_researcher(
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
        raise ResearchError(
            f"researcher timed out after {timeout_seconds}s"
        ) from exc
    combined = (result.stdout + "\n" + result.stderr).strip()
    log_path.write_text(combined + ("\n" if combined else ""), encoding="utf-8")
    if result.returncode:
        tail = "\n".join(combined.splitlines()[-12:])
        raise ResearchError(f"researcher exited {result.returncode}\n{tail}")


def validate_packet(packet: dict[str, Any]) -> dict[str, Any]:
    facts = packet.get("facts")
    if not isinstance(facts, list) or len(facts) < 8:
        raise ResearchError("research packet needs at least eight facts")
    categories: set[str] = set()
    for position, fact in enumerate(facts, start=1):
        if not isinstance(fact, dict):
            raise ResearchError(f"fact {position} is not an object")
        expected_id = f"fact_{position:03d}"
        if fact.get("fact_id") != expected_id:
            raise ResearchError(f"fact {position} ID must be {expected_id}")
        if fact.get("category") not in CATEGORIES:
            raise ResearchError(f"{expected_id} has invalid category")
        categories.add(fact["category"])
        urls = fact.get("source_urls")
        if not isinstance(urls, list) or not urls:
            raise ResearchError(f"{expected_id} has no source URLs")
        if len(set(urls)) != len(urls):
            raise ResearchError(f"{expected_id} repeats a source URL")
        if not all(isinstance(url, str) and re.match(r"^https?://", url) for url in urls):
            raise ResearchError(f"{expected_id} has an invalid source URL")
    recommended = {"study", "history", "culture"}
    missing = recommended - categories
    packet["coverage"] = {
        "fact_count": len(facts),
        "categories": sorted(categories),
        "missing_recommended_categories": sorted(missing),
        "outline_ready": not missing,
    }
    return packet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research a topic and persist a source-backed fact packet."
    )
    parser.add_argument("project", type=Path)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--timeout", type=int, default=1200)
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
            raise ResearchError("codex CLI is not installed or not on PATH")
        project_dir = args.project.resolve()
        input_dir = project_dir / "input"
        config = load_json(input_dir / "topic.json")
        topic = config.get("topic")
        if not isinstance(topic, str) or not topic.strip():
            raise ResearchError("topic.json topic must be a non-empty string")
        output_path = input_dir / "research.json"
        if output_path.exists() and not args.overwrite:
            raise ResearchError(
                f"{output_path} already exists; use --overwrite to replace it"
            )
        work_dir = project_dir / "work" / "research"
        work_dir.mkdir(parents=True, exist_ok=True)
        schema_path = work_dir / "research.schema.json"
        raw_path = work_dir / "raw-response.json"
        log_path = work_dir / "codex.log"
        write_json_atomic(schema_path, research_schema())
        print(f"ChatGPT auth verified; researcher={args.model}; topic={topic!r}")
        run_researcher(
            codex,
            args.model,
            project_dir,
            build_prompt(config),
            schema_path,
            raw_path,
            log_path,
            timeout_seconds=args.timeout,
        )
        packet = validate_packet(load_json(raw_path))
        packet["schema_version"] = "1.0"
        packet["researcher_model"] = args.model
        write_json_atomic(output_path, packet)
        coverage = packet["coverage"]
        print(
            f"Created {output_path} with {coverage['fact_count']} facts; "
            f"outline_ready={coverage['outline_ready']}"
        )
        return 0
    except ResearchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
