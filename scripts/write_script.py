#!/usr/bin/env python3
"""Draft or revise narration with a writer model distinct from the checker."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class WritingError(Exception):
    """A user-actionable writing-stage error."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise WritingError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise WritingError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WritingError(f"{path} must contain a JSON object")
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
        raise WritingError(
            "Codex must use ChatGPT authentication. Run `codex login` and choose "
            "ChatGPT before writing."
        )


def prose_rules() -> str:
    return """Write narration only: no title, headings, stage directions, citations,
markdown, or commentary. Write like you are explaining something surprising to
a friend at a bar, not presenting it in a classroom. Use contractions, ordinary
words, short sentences, varied rhythm, and specific concrete imagery. Cut every
word that exists mainly to sound smart. Never use lecture-style openings such as
"This phenomenon occurs because" or "Research indicates that."

Keep attention moving: every one or two sentences needs a hook, twist, joke,
sarcastic aside, or "wait, what?" beat. Funny is better than formal, including
intentionally dumb jokes and mild exaggeration that does not alter the facts.
When profanity fits the configured tone, write the actual word; never censor it
with asterisks, dashes, or euphemisms. Avoid generic engagement bait and avoid
repeating the conclusion.
"""


def style_rules() -> str:
    return prose_rules() + """
Follow this eight-beat curiosity-essay structure:
1. Absurd concrete visual hook that does not announce the topic.
2. State the contradiction plainly.
3. Knock down obvious explanations one at a time with fast rebuttals.
4. Historical reversal anecdote.
5. Cultural counterexample showing the line is not universal.
6. Real mechanism through correctly named researchers and studies.
7. Generalize the mechanism to other examples.
8. Reframe the viewer's own assumptions."""


def short_style_rules() -> str:
    return prose_rules() + """
This is a fast 60–65 second documentary short, not a long-form essay. Aim for
175–195 spoken words and never fall below 168 words. Use a brisk, conversational,
amused, slightly frantic energy that remains clear and controlled rather than
shouted, dragged out, or parodic. Use a punchy hook → concrete contrast → plain
mechanism → primary evidence → honest nuance or counterevidence → funny reframe
arc. Let the supplied outline choose the topic, examples, named researchers,
study details, sequence, and caveats; follow every outline-specific requirement
without replacing it with a generic template.

The first sentence must explicitly name the topic as a direct curiosity hook,
normally in the form "Have you ever wondered why ...?" Explain it like a funny
friend, never a formal documentary narrator. Use two or three uncensored curse words
when they fit naturally and two or three jokes or sarcastic asides aimed
at young adults. Do not bleep profanity in text with symbols. Never let two flat,
purely explanatory sentences sit next to each other.

Use only facts in the supplied research packet. Preserve study boundaries and
qualifications, prefer "usually" or "almost impossible" to absolute claims when
the evidence is not universal, and avoid jargon, psychiatric examples, and
unsupported specificity. Keep the spoken prose natural for the configured voice
and do not add a title or heading.
"""


def rules_for_outline(outline: dict[str, Any]) -> str:
    return short_style_rules() if outline.get("format") == "short" else style_rules()


def initial_prompt(
    outline: dict[str, Any], research: dict[str, Any], target_words: int
) -> str:
    return f"""You are the narration writer. You are not the fact-checker.

{rules_for_outline(outline)}

Draft approximately {target_words} words. Use only factual material supplied in
the research packet. Do not add a researcher, institution, study, date, number,
quotation, historical event, or cultural practice that is absent from that
packet. If the outline asks for unsupported specificity, stay general rather
than inventing it.

<outline>
{json.dumps(outline, ensure_ascii=False, indent=2)}
</outline>

<research_packet>
{json.dumps(research, ensure_ascii=False, indent=2)}
</research_packet>

Return only the narration."""


def revision_prompt(draft: str, report: dict[str, Any]) -> str:
    flagged = [
        claim
        for claim in report.get("claims", [])
        if claim.get("verdict") != "verified"
        or claim.get("recommended_action") != "keep"
    ]
    source_words = len(re.findall(r"\b[\w’'-]+\b", draft))
    return f"""You are the narration writer revising a fact-checked draft. You are
not the fact-checker, and you must not perform or claim new research.

{prose_rules()}

Revise every flagged claim in the report. Use a suggested replacement only when
it is present and supported by the cited finding. Otherwise remove the claim or
rewrite the passage more generally. Do not introduce any new externally
verifiable facts, names, studies, institutions, dates, or numbers. Preserve the
draft's voice, pacing, paragraph structure, and unflagged material as much as
possible. The source is {source_words} words; keep the revision within roughly
10 percent of that length. Do not expand an excerpt into the full eight-beat
essay structure.

Make a surgical revision:
- Edit only sentences that contain a flagged claim.
- Copy every unflagged sentence verbatim, including its punctuation.
- Do not "improve," condense, expand, or restyle surrounding prose.
- Keep the original joke, profanity, contraction, rhythm, and conversational
  wording whenever the corrected fact permits it.
- Fix the specific number, name, date, scope, or causal claim that was flagged;
  do not turn the sentence into methodology language or a formal limitations
  paragraph merely because the fact needed narrowing.
- If a revision sounds more like a professor than the supplied draft, rewrite
  it again in plain conversational language without weakening the correction.
- Preserve negation and qualifications exactly unless that specific wording is
  flagged.
- After editing, compare the revision with the draft and undo every change that
  is not required by a flagged claim.

Return only the complete revised narration.

<draft>
{draft}
</draft>

<flagged_claims>
{json.dumps(flagged, ensure_ascii=False, indent=2)}
</flagged_claims>"""


def run_writer(
    codex: str,
    writer_model: str,
    project_dir: Path,
    prompt: str,
    output_path: Path,
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
        writer_model,
        "--cd",
        str(project_dir),
        "--output-last-message",
        str(output_path),
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
        raise WritingError(f"writer timed out after {timeout_seconds}s") from exc
    combined = (result.stdout + "\n" + result.stderr).strip()
    log_path.write_text(combined + ("\n" if combined else ""), encoding="utf-8")
    if result.returncode:
        tail = "\n".join(combined.splitlines()[-12:])
        raise WritingError(f"writer exited {result.returncode}\n{tail}")


def validate_narration(path: Path) -> tuple[str, int]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise WritingError(f"writer did not create output: {path}") from exc
    words = re.findall(r"\b[\w’'-]+\b", text)
    if len(words) < 20:
        raise WritingError(f"writer output is unexpectedly short: {len(words)} words")
    if re.search(r"^```|```$", text, flags=re.MULTILINE):
        raise WritingError("writer output contains a Markdown code fence")
    return text, len(words)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draft or revise narration through a dedicated writer model."
    )
    parser.add_argument("project", type=Path, help="project directory")
    parser.add_argument("--mode", choices=("draft", "revise"), required=True)
    parser.add_argument("--draft", type=Path, help="source draft for revise mode")
    parser.add_argument("--fact-report", type=Path, help="report for revise mode")
    parser.add_argument("--pass-number", type=int, default=1)
    parser.add_argument("--target-words", type=int, default=1900)
    parser.add_argument("--writer-model", default="gpt-5.6-sol")
    parser.add_argument("--checker-model", default="gpt-5.6-terra")
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
    if args.writer_model == args.checker_model:
        print("error: writer and checker models must be different", file=sys.stderr)
        return 2
    if args.pass_number <= 0 or args.target_words <= 0 or args.timeout <= 0:
        print("error: numeric options must be greater than zero", file=sys.stderr)
        return 2
    try:
        verify_chatgpt_auth(args.auth_file.expanduser().resolve())
        codex = shutil.which("codex")
        if not codex:
            raise WritingError("codex CLI is not installed or not on PATH")
        project_dir = args.project.resolve()
        work_dir = project_dir / "work" / "writing"
        work_dir.mkdir(parents=True, exist_ok=True)

        if args.mode == "draft":
            outline = load_json(project_dir / "input" / "outline.json")
            research = load_json(project_dir / "input" / "research.json")
            output_path = work_dir / "draft_001.txt"
            prompt = initial_prompt(outline, research, args.target_words)
            source_metadata = {
                "outline": str(project_dir / "input" / "outline.json"),
                "research": str(project_dir / "input" / "research.json"),
            }
        else:
            if not args.draft or not args.fact_report:
                raise WritingError(
                    "revise mode requires both --draft and --fact-report"
                )
            draft_path = args.draft.expanduser().resolve()
            report_path = args.fact_report.expanduser().resolve()
            try:
                draft = draft_path.read_text(encoding="utf-8").strip()
            except FileNotFoundError as exc:
                raise WritingError(f"Missing draft: {draft_path}") from exc
            report = load_json(report_path)
            if report.get("checker_model") == args.writer_model:
                raise WritingError(
                    "fact report checker_model matches the selected writer model"
                )
            output_path = work_dir / f"draft_{args.pass_number + 1:03d}.txt"
            prompt = revision_prompt(draft, report)
            source_metadata = {
                "draft": str(draft_path),
                "fact_report": str(report_path),
            }

        if output_path.exists() and not args.overwrite:
            raise WritingError(
                f"{output_path} already exists; use --overwrite to replace it"
            )
        log_path = output_path.with_suffix(".codex.log")
        print(
            f"ChatGPT auth verified; writer={args.writer_model}; "
            f"checker identity reserved as {args.checker_model}"
        )
        run_writer(
            codex,
            args.writer_model,
            project_dir,
            prompt,
            output_path,
            log_path,
            timeout_seconds=args.timeout,
        )
        text, word_count = validate_narration(output_path)
        if args.mode == "revise":
            source_word_count = len(re.findall(r"\b[\w’'-]+\b", draft))
            maximum = max(round(source_word_count * 1.25), source_word_count + 20)
            minimum = max(20, round(source_word_count * 0.55))
            if not minimum <= word_count <= maximum:
                raise WritingError(
                    f"revision changed length too much: source={source_word_count}, "
                    f"revision={word_count}, allowed={minimum}-{maximum}"
                )
        metadata = {
            "schema_version": "1.0",
            "mode": args.mode,
            "pass_number": args.pass_number,
            "writer_model": args.writer_model,
            "checker_model": args.checker_model,
            "word_count": word_count,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "created_at": int(time.time()),
            "sources": source_metadata,
            "output": str(output_path),
        }
        write_json_atomic(output_path.with_suffix(".json"), metadata)
        print(f"Created {output_path} ({word_count} words)")
        return 0
    except WritingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
