#!/usr/bin/env python3
"""Promote a checked draft only when its report is clean and hash-matched."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


class FinalizationError(Exception):
    """A user-actionable finalization error."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise FinalizationError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FinalizationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalizationError(f"{path} must contain a JSON object")
    return value


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    write_text_atomic(
        path, json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    )


def verify_finalizable(
    draft_path: Path, report: dict[str, Any]
) -> tuple[str, str]:
    draft_path = draft_path.resolve()
    summary = report.get("summary")
    if not isinstance(summary, dict) or summary.get("ready_for_final") is not True:
        raise FinalizationError("fact-check report is not ready_for_final")
    if summary.get("contradicted") != 0 or summary.get("unverified") != 0:
        raise FinalizationError("fact-check report still contains unresolved claims")
    if report.get("checker_model") == report.get("writer_model"):
        raise FinalizationError("writer and checker model identities are not separate")
    reported_path = report.get("draft_file")
    if not isinstance(reported_path, str) or Path(reported_path).resolve() != draft_path:
        raise FinalizationError("fact-check report refers to a different draft file")
    expected_hash = report.get("draft_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise FinalizationError("fact-check report has no valid draft_sha256")
    try:
        text = draft_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FinalizationError(f"Missing checked draft: {draft_path}") from exc
    actual_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if actual_hash != expected_hash:
        raise FinalizationError("draft changed after fact checking; run a new check")
    return text, actual_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote a clean, hash-matched checked draft to transcript.txt."
    )
    parser.add_argument("project", type=Path, help="project directory")
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--fact-report", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        project_dir = args.project.resolve()
        draft_path = args.draft.expanduser().resolve()
        report_path = args.fact_report.expanduser().resolve()
        report = load_json(report_path)
        text, digest = verify_finalizable(draft_path, report)
        output_path = project_dir / "input" / "transcript.txt"
        if output_path.exists() and not args.overwrite:
            raise FinalizationError(
                f"{output_path} already exists; use --overwrite to replace it"
            )
        write_text_atomic(output_path, text + "\n")
        metadata = {
            "schema_version": "1.0",
            "status": "fact_checked",
            "finalized_at": int(time.time()),
            "transcript_file": str(output_path),
            "draft_file": str(draft_path),
            "draft_sha256": digest,
            "fact_report": str(report_path),
            "factcheck_pass": report.get("pass_number"),
            "writer_model": report.get("writer_model"),
            "checker_model": report.get("checker_model"),
            "total_claims": report["summary"].get("total_claims"),
        }
        write_json_atomic(project_dir / "input" / "finalization.json", metadata)
        print(f"Finalized {output_path}")
        return 0
    except FinalizationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
