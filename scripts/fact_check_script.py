#!/usr/bin/env python3
"""Extract and independently verify narration claims through a checker model."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


class FactCheckError(Exception):
    """A user-actionable fact-checking error."""


VERDICTS = ("verified", "contradicted", "unverified")
CATEGORIES = (
    "person",
    "institution",
    "study",
    "date",
    "number",
    "historical",
    "cultural",
    "biological",
    "framing",
    "other",
)
ACTIONS = ("keep", "revise", "remove")

# "framing" covers the script's own argumentative connective tissue: the stated
# contradiction, the rebuttals, the generalization, and the closing reframe.
# Those are authorial positions rather than sourced assertions, so holding them
# to "a source directly establishes this" makes them permanently unverifiable
# and pushes the revision loop to hedge the script into a limitations section.
# They still require evidence and can still be contradicted; the bar is
# "consistent with the cited evidence" instead of "established by it".
#
# The guards below stop factual claims from being smuggled under the softer
# bar: a framing claim may not carry numbers, dates, or study/researcher
# vocabulary, and framing may not dominate the report.
MAX_FRAMING_SHARE = 0.25
# A share is only meaningful once there are enough claims to form one; a real
# pass over a full script produces 20-30.
MIN_CLAIMS_FOR_FRAMING_CEILING = 8
FRAMING_FORBIDDEN_PATTERN = re.compile(
    r"\d|\bstudy\b|\bstudies\b|\bresearchers?\b|\buniversity\b|\binstitute\b"
    r"|\bpercent\b|\bfound that\b|\bshowed that\b",
    re.IGNORECASE,
)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise FactCheckError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FactCheckError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FactCheckError(f"{path} must contain a JSON object")
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
        raise FactCheckError(
            "Codex must use ChatGPT authentication. Run `codex login` and choose "
            "ChatGPT before fact-checking."
        )


def factcheck_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["claims"],
        "properties": {
            "claims": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "claim_id",
                        "claim",
                        "category",
                        "verdict",
                        "evidence",
                        "explanation",
                        "recommended_action",
                        "suggested_replacement",
                    ],
                    "properties": {
                        "claim_id": {"type": "string", "pattern": "^claim_[0-9]{3}$"},
                        "claim": {"type": "string", "minLength": 5},
                        "category": {"type": "string", "enum": list(CATEGORIES)},
                        "verdict": {"type": "string", "enum": list(VERDICTS)},
                        "evidence": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["url", "title", "finding"],
                                "properties": {
                                    "url": {
                                        "type": "string",
                                        "pattern": "^https?://",
                                    },
                                    "title": {"type": "string", "minLength": 3},
                                    "finding": {"type": "string", "minLength": 5},
                                },
                            },
                        },
                        "explanation": {"type": "string", "minLength": 10},
                        "recommended_action": {
                            "type": "string",
                            "enum": list(ACTIONS),
                        },
                        "suggested_replacement": {"type": "string"},
                    },
                },
            }
        },
    }


def build_prompt(draft: str) -> str:
    return f"""You are an independent forensic fact-checker, not a scriptwriter.

Use web search for every externally verifiable factual claim in the draft below.
Extract claims atomically: names, affiliations, institutions, named studies,
dates, numbers, historical events, cultural practices, biological comparisons,
and claims about research findings. Do not skip a claim merely because it sounds
plausible.

For each claim:
- Preserve the draft's exact logical meaning when extracting it. Never drop or
  invert negation, uncertainty, qualifications, comparisons, or scope. For
  example, "does not establish fear" must not become "establishes fear."
- Search independently rather than relying on the draft or memory.
- Prefer primary research, university/institution pages, government sources,
  museum or archive records, and authoritative scholarly references.
- For named studies and surprising historical claims, seek two independent
  sources when available.
- Use "verified" only when the evidence supports the claim as worded.
- Use "contradicted" when reliable evidence shows it is wrong or materially
  misleading.
- Use "unverified" when evidence is absent, weak, circular, or insufficient.
- Never invent a citation. Include the exact public URL you actually checked.

Category "framing" is a narrow exception with a different evidential bar:
- Use it only for the script's own argumentative connective tissue: the stated
  contradiction, a rebuttal's conclusion, the generalization, and the closing
  reframe. These are the author's positions, not reports of external fact.
- A framing claim is "verified" when the cited evidence is consistent with it
  and does not contradict it. Do not mark it unverified merely because no
  single source states the author's thesis; that is expected of a thesis.
- Mark it "contradicted" when evidence actually cuts against it.
- Never use "framing" for anything containing a name, number, date, statistic,
  institution, or a report of what a study found. Those stay in their factual
  category under the strict bar, even inside an argumentative sentence.
- Framing must remain a small minority of claims. If most of the draft looks
  like framing, you are mis-categorizing sourced assertions.
- Before returning, compare every extracted claim against the source sentence
  and confirm that words such as not, no, may, can, some, and only retain their
  original force.
- Set recommended_action to revise or remove for every non-verified claim.
- suggested_replacement must be an evidence-supported correction, or an empty
  string when safe correction is not possible.

Do not rewrite the draft. Return only the JSON required by the supplied schema.
Assign claim IDs sequentially from claim_001.

<draft>
{draft}
</draft>"""


def run_checker(
    codex: str,
    checker_model: str,
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
        "--model",
        checker_model,
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
        raise FactCheckError(
            f"fact-checker timed out after {timeout_seconds}s"
        ) from exc
    combined = (result.stdout + "\n" + result.stderr).strip()
    log_path.write_text(combined + ("\n" if combined else ""), encoding="utf-8")
    if result.returncode:
        tail = "\n".join(combined.splitlines()[-12:])
        raise FactCheckError(f"fact-checker exited {result.returncode}\n{tail}")


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    claims = report.get("claims")
    if not isinstance(claims, list) or not claims:
        raise FactCheckError("fact-check report has no claims")
    counts = {verdict: 0 for verdict in VERDICTS}
    framing_count = 0
    seen_claims: set[str] = set()
    for position, claim in enumerate(claims, start=1):
        if not isinstance(claim, dict):
            raise FactCheckError(f"claim {position} is not an object")
        expected_id = f"claim_{position:03d}"
        if claim.get("claim_id") != expected_id:
            raise FactCheckError(
                f"claim {position} ID must be {expected_id}, "
                f"not {claim.get('claim_id')!r}"
            )
        claim_text = claim.get("claim")
        if not isinstance(claim_text, str) or len(claim_text.strip()) < 5:
            raise FactCheckError(f"{expected_id} has invalid claim text")
        normalized_claim = re.sub(r"\s+", " ", claim_text).strip().lower()
        if normalized_claim in seen_claims:
            raise FactCheckError(f"duplicate claim text at {expected_id}")
        seen_claims.add(normalized_claim)
        verdict = claim.get("verdict")
        if verdict not in VERDICTS:
            raise FactCheckError(f"{expected_id} has invalid verdict")
        counts[verdict] += 1
        category = claim.get("category")
        if category not in CATEGORIES:
            raise FactCheckError(f"{expected_id} has invalid category")
        if category == "framing":
            framing_count += 1
            smuggled = FRAMING_FORBIDDEN_PATTERN.search(claim_text)
            if smuggled:
                raise FactCheckError(
                    f"{expected_id} is categorized framing but contains "
                    f"{smuggled.group(0)!r}; factual content must use a factual "
                    "category and the strict evidential bar"
                )
        evidence = claim.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise FactCheckError(f"{expected_id} needs at least one evidence item")
        urls: set[str] = set()
        for item in evidence:
            if not isinstance(item, dict):
                raise FactCheckError(f"{expected_id} has malformed evidence")
            url = item.get("url")
            if not isinstance(url, str) or not re.match(r"^https?://", url):
                raise FactCheckError(f"{expected_id} has invalid evidence URL")
            if url in urls:
                raise FactCheckError(f"{expected_id} repeats an evidence URL")
            urls.add(url)
        action = claim.get("recommended_action")
        if action not in ACTIONS:
            raise FactCheckError(f"{expected_id} has invalid recommended_action")
        if verdict != "verified" and action == "keep":
            raise FactCheckError(f"{expected_id} cannot keep a non-verified claim")
        replacement = claim.get("suggested_replacement")
        if not isinstance(replacement, str):
            raise FactCheckError(f"{expected_id} suggested_replacement must be a string")
    if (
        len(claims) >= MIN_CLAIMS_FOR_FRAMING_CEILING
        and framing_count > len(claims) * MAX_FRAMING_SHARE
    ):
        raise FactCheckError(
            f"{framing_count} of {len(claims)} claims are categorized framing, "
            f"above the {MAX_FRAMING_SHARE:.0%} ceiling; sourced assertions are "
            "being mis-categorized to avoid the strict evidential bar"
        )
    report["summary"] = {
        "total_claims": len(claims),
        "verified": counts["verified"],
        "contradicted": counts["contradicted"],
        "unverified": counts["unverified"],
        "framing_claims": framing_count,
        "ready_for_final": counts["contradicted"] == 0 and counts["unverified"] == 0,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and verify every factual claim in a narration draft."
    )
    parser.add_argument("project", type=Path, help="project directory")
    parser.add_argument(
        "--draft",
        type=Path,
        help="draft path; defaults to input/draft.txt",
    )
    parser.add_argument("--pass-number", type=int, default=1)
    parser.add_argument("--checker-model", default="gpt-5.6-terra")
    parser.add_argument("--writer-model", default="gpt-5.6-sol")
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
    if args.pass_number <= 0 or args.timeout <= 0:
        print("error: pass number and timeout must be greater than zero", file=sys.stderr)
        return 2
    try:
        if args.checker_model == args.writer_model:
            raise FactCheckError(
                "checker-model and writer-model must be different model identities"
            )
        verify_chatgpt_auth(args.auth_file.expanduser().resolve())
        codex = shutil.which("codex")
        if not codex:
            raise FactCheckError("codex CLI is not installed or not on PATH")
        project_dir = args.project.resolve()
        draft_path = (
            args.draft.expanduser().resolve()
            if args.draft
            else project_dir / "input" / "draft.txt"
        )
        try:
            draft = draft_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise FactCheckError(f"Missing draft: {draft_path}") from exc
        if not draft:
            raise FactCheckError(f"Draft is empty: {draft_path}")

        pass_dir = project_dir / "work" / "factcheck" / f"pass_{args.pass_number:03d}"
        report_path = pass_dir / "report.json"
        if report_path.exists() and not args.overwrite:
            raise FactCheckError(
                f"{report_path} already exists; use --overwrite to replace it"
            )
        pass_dir.mkdir(parents=True, exist_ok=True)
        schema_path = pass_dir / "report.schema.json"
        raw_path = pass_dir / "raw-response.json"
        log_path = pass_dir / "codex.log"
        write_json_atomic(schema_path, factcheck_schema())
        print(
            f"ChatGPT auth verified; checker={args.checker_model}; "
            f"writer identity reserved as {args.writer_model}"
        )
        run_checker(
            codex,
            args.checker_model,
            project_dir,
            build_prompt(draft),
            schema_path,
            raw_path,
            log_path,
            timeout_seconds=args.timeout,
        )
        report = validate_report(load_json(raw_path))
        report["schema_version"] = "1.0"
        report["pass_number"] = args.pass_number
        report["checker_model"] = args.checker_model
        report["writer_model"] = args.writer_model
        report["draft_file"] = str(draft_path)
        report["draft_sha256"] = hashlib.sha256(
            draft.encode("utf-8")
        ).hexdigest()
        write_json_atomic(report_path, report)
        summary = report["summary"]
        print(
            f"Checked {summary['total_claims']} claims: "
            f"verified={summary['verified']}, "
            f"contradicted={summary['contradicted']}, "
            f"unverified={summary['unverified']}"
        )
        print(f"Created {report_path}")
        return 0
    except FactCheckError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
