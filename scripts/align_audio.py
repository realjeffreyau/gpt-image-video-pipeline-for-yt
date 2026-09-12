#!/usr/bin/env python3
"""Create assembler-compatible word timestamps with local faster-whisper."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any


class AlignmentError(RuntimeError):
    pass


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def transcript_match_ratio(transcript: str, recognized_words: list[str]) -> float:
    return difflib.SequenceMatcher(a=tokens(transcript), b=tokens(" ".join(recognized_words))).ratio()


def audio_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as source:
            return source.getnframes() / source.getframerate()
    except (wave.Error, EOFError) as exc:
        raise AlignmentError("Alignment currently expects a WAV file (Gemini TTS writes one)") from exc


def word_timestamp_payload(
    raw_segments: list[dict[str, Any]],
    duration: float,
    *,
    audio_file: str | None = None,
) -> dict[str, Any]:
    words: list[dict[str, Any]] = []
    previous_end = 0.0
    for segment in raw_segments:
        for raw_word in segment.get("words", []):
            word = str(raw_word.get("word", "")).strip()
            if not word:
                continue
            start = max(0.0, float(raw_word["start"]), previous_end)
            end = max(start + 0.001, float(raw_word["end"]))
            words.append({"index": len(words), "word": word, "start": round(start, 3), "end": round(end, 3)})
            previous_end = end
    if not words:
        raise AlignmentError("Whisper produced no word-level timestamps")
    if words[-1]["end"] > duration + 2:
        raise AlignmentError("Whisper timestamps extend unexpectedly beyond the narration audio")
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "duration_seconds": round(duration, 3),
        "words": words,
    }
    if audio_file is not None:
        payload["audio_file"] = audio_file
    return payload


def transcribe(audio_path: Path, model_name: str, language: str, device: str, compute_type: str) -> list[dict[str, Any]]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise AlignmentError(
            "faster-whisper is not installed. Run: .venv/bin/python -m pip install -r requirements-alignment.txt"
        ) from exc
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        segments, _ = model.transcribe(str(audio_path), language=language, word_timestamps=True, vad_filter=True)
        materialized = []
        for segment in segments:
            materialized.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": [
                    {"word": item.word, "start": item.start, "end": item.end, "probability": item.probability}
                    for item in (segment.words or [])
                ],
            })
        return materialized
    except Exception as exc:  # The backend emits several implementation-specific exception types.
        raise AlignmentError(f"faster-whisper could not align the narration: {exc}") from exc


def run(project: Path, model_name: str, language: str, device: str, compute_type: str, minimum_match: float) -> None:
    transcript_path = project / "input" / "transcript.txt"
    audio_path = project / "input" / "narration.wav"
    transcript = transcript_path.read_text(encoding="utf-8").strip()
    if not transcript:
        raise AlignmentError(f"Transcript is empty: {transcript_path}")
    if not audio_path.exists():
        raise AlignmentError(f"Missing narration audio: {audio_path}")
    duration = audio_duration(audio_path)
    print(f"Aligning {audio_path.name} with faster-whisper {model_name}…")
    raw_segments = transcribe(audio_path, model_name, language, device, compute_type)
    payload = word_timestamp_payload(raw_segments, duration, audio_file=audio_path.name)
    recognized = [entry["word"] for entry in payload["words"]]
    ratio = transcript_match_ratio(transcript, recognized)
    work_dir = project / "work" / "alignment"
    write_json(work_dir / "raw_segments.json", {"segments": raw_segments})
    if ratio < minimum_match:
        raise AlignmentError(
            f"Alignment transcript match was only {ratio:.1%} (need {minimum_match:.1%}). "
            "Raw output was saved for review; do not assemble this timing file."
        )
    payload["source"] = {
        "backend": "faster-whisper",
        "model": model_name,
        "language": language,
        "transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
        "transcript_match_ratio": round(ratio, 4),
    }
    write_json(project / "input" / "timestamps.json", payload)
    write_json(work_dir / "alignment.json", payload["source"])
    print(f"Wrote input/timestamps.json: {len(payload['words'])} words, {ratio:.1%} transcript match.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--minimum-match", type=float, default=0.80)
    args = parser.parse_args()
    if not 0 < args.minimum_match <= 1:
        parser.error("--minimum-match must be between 0 and 1")
    try:
        run(args.project.resolve(), args.model, args.language, args.device, args.compute_type, args.minimum_match)
    except (AlignmentError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
