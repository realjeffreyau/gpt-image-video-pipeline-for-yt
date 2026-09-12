#!/usr/bin/env python3
"""Generate a narration WAV with Google AI Studio's Gemini TTS models.

The script deliberately uses the standard library rather than a Google SDK.  It
keeps the API key out of project JSON and makes each generated audio chunk
resumable under work/tts/.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import sys
import ssl
import tempfile
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any


class VoiceoverError(RuntimeError):
    pass


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VoiceoverError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise VoiceoverError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VoiceoverError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def archive_existing_take(project: Path, work_dir: Path, output_path: Path) -> Path:
    """Preserve a complete provider take before incompatible inputs replace it.

    The archive is deliberately created before any segment or state file is
    changed. This makes the TTS side effect idempotent from the workflow's
    point of view: a resumed or revised run can never silently destroy the raw
    audio that preceded it.
    """
    takes_dir = work_dir / "takes"
    takes_dir.mkdir(parents=True, exist_ok=True)
    index = 1
    while (takes_dir / f"take_{index:03d}_native").exists():
        index += 1
    archive_dir = takes_dir / f"take_{index:03d}_native"
    archive_dir.mkdir()

    copied = False
    if output_path.is_file():
        shutil.copy2(output_path, archive_dir / "narration.wav")
        copied = True
    for name in ("state.json", "voiceover.json"):
        source = work_dir / name
        if source.is_file():
            shutil.copy2(source, archive_dir / name)
            copied = True
    segments_dir = work_dir / "segments"
    if segments_dir.is_dir():
        shutil.copytree(segments_dir, archive_dir / "segments")
        copied = True
    if not copied:
        archive_dir.rmdir()
        raise VoiceoverError("no existing TTS artifacts were available to archive")

    metadata = {
        "schema_version": "1.0",
        "status": "archived_before_replacement",
        "project": project.name,
        "source_output": str(output_path.relative_to(project)),
    }
    write_json(archive_dir / "archive.json", metadata)
    return archive_dir


def load_project_env(path: Path) -> None:
    """Load only simple KEY=value lines; never print environment values."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key != "GEMINI_API_KEY" or key in os.environ:
            continue
        value = value.strip().strip("\"").strip("'")
        if value:
            os.environ[key] = value


def config_from_file(path: Path) -> dict[str, Any]:
    config = read_json(path)
    required = ("provider", "model", "voice_name", "style_instruction")
    missing = [key for key in required if not isinstance(config.get(key), str) or not config[key].strip()]
    if missing:
        raise VoiceoverError(f"voice_config.json is missing non-empty fields: {', '.join(missing)}")
    if config["provider"] != "google_gemini_tts":
        raise VoiceoverError("voice_config.json provider must be 'google_gemini_tts'")
    maximum = config.get("chunk_max_characters", 2200)
    if not isinstance(maximum, int) or not 300 <= maximum <= 10000:
        raise VoiceoverError("chunk_max_characters must be an integer between 300 and 10000")
    sample_rate = config.get("sample_rate_hz", 24000)
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise VoiceoverError("sample_rate_hz must be a positive integer")
    config["chunk_max_characters"] = maximum
    config["sample_rate_hz"] = sample_rate
    return config


def split_transcript(transcript: str, maximum: int) -> list[str]:
    """Split on paragraph/sentence boundaries without dropping spoken words."""
    normalized = re.sub(r"\s+", " ", transcript).strip()
    if not normalized:
        raise VoiceoverError("Transcript is empty")
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > maximum:
            if current:
                chunks.append(current)
                current = ""
            words = sentence.split()
            piece = ""
            for word in words:
                candidate = f"{piece} {word}".strip()
                if piece and len(candidate) > maximum:
                    chunks.append(piece)
                    piece = word
                else:
                    piece = candidate
            if piece:
                chunks.append(piece)
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > maximum:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def build_request_body(segment: str, config: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        "Read the transcript below exactly as written. Do not add, remove, summarize, "
        "or announce anything.\n\n"
        f"Performance direction: {config['style_instruction'].strip()}\n\n"
        "Transcript:\n"
        f"{segment}"
    )
    return {
        "model": config["model"],
        "input": prompt,
        "response_format": {"type": "audio"},
        "generation_config": {"speech_config": [{"voice": config["voice_name"]}]},
    }


def request_pcm(segment: str, config: dict[str, Any], api_key: str, timeout: int) -> bytes:
    endpoint = "https://generativelanguage.googleapis.com/v1beta/interactions"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(build_request_body(segment, config)).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    try:
        try:
            import certifi

            tls_context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            tls_context = ssl.create_default_context()
        with urllib.request.urlopen(request, timeout=timeout, context=tls_context) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        raise VoiceoverError(f"Gemini TTS request failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise VoiceoverError(f"Could not reach Gemini TTS: {exc.reason}") from exc
    encoded: str | None = None
    # output_audio is an SDK convenience property. REST responses place the
    # actual audio block under steps[].content[].
    output_audio = payload.get("output_audio")
    if isinstance(output_audio, dict) and isinstance(output_audio.get("data"), str):
        encoded = output_audio["data"]
    if encoded is None:
        for step in payload.get("steps", []):
            if not isinstance(step, dict) or step.get("type") != "model_output":
                continue
            for content in step.get("content", []):
                if isinstance(content, dict) and content.get("type") == "audio" and isinstance(content.get("data"), str):
                    encoded = content["data"]
                    break
            if encoded is not None:
                break
    if not encoded:
        status = payload.get("status", "unknown")
        raise VoiceoverError(f"Gemini TTS returned no usable inline audio data (status: {status})")
    try:
        audio = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as exc:
        raise VoiceoverError("Gemini TTS returned malformed base64 audio data") from exc
    if not audio:
        raise VoiceoverError("Gemini TTS returned an empty audio response")
    return audio


def write_pcm_wav(path: Path, pcm: bytes, sample_rate: int) -> float:
    if len(pcm) % 2:
        raise VoiceoverError("Gemini TTS returned invalid 16-bit PCM audio")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)
    return len(pcm) / 2 / sample_rate


def valid_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as source:
            return source.getnchannels() == 1 and source.getsampwidth() == 2 and source.getnframes() > 0
    except (FileNotFoundError, wave.Error, EOFError):
        return False


def concatenate_wavs(paths: list[Path], output_path: Path) -> float:
    params: tuple[int, int, int] | None = None
    frames: list[bytes] = []
    for path in paths:
        with wave.open(str(path), "rb") as source:
            current = (source.getnchannels(), source.getsampwidth(), source.getframerate())
            if params is None:
                params = current
            elif current != params:
                raise VoiceoverError(f"Audio format mismatch in {path}")
            frames.append(source.readframes(source.getnframes()))
    if params is None:
        raise VoiceoverError("No audio chunks were available")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as output:
        output.setnchannels(params[0])
        output.setsampwidth(params[1])
        output.setframerate(params[2])
        for chunk in frames:
            output.writeframes(chunk)
    return sum(len(chunk) for chunk in frames) / params[0] / params[1] / params[2]


def run(project: Path, dry_run: bool, overwrite: bool, timeout: int) -> None:
    input_dir = project / "input"
    transcript_path = input_dir / "transcript.txt"
    config_path = input_dir / "voice_config.json"
    transcript = transcript_path.read_text(encoding="utf-8").strip()
    if not transcript:
        raise VoiceoverError(f"Transcript is empty: {transcript_path}")
    config = config_from_file(config_path)
    chunks = split_transcript(transcript, config["chunk_max_characters"])
    transcript_hash = sha256_text(transcript)
    config_hash = sha256_text(json.dumps(config, sort_keys=True))
    work_dir = project / "work" / "tts"
    state_path = work_dir / "state.json"
    output_path = input_dir / "narration.wav"

    print(f"Gemini TTS: {len(chunks)} resumable chunks using {config['model']} / {config['voice_name']}.")
    if dry_run:
        print("Dry run only; no API key or network request is needed.")
        return
    # Accept the documented repository-level .env as well as a project-local
    # one. The project-local file wins if both are present.
    load_project_env(project / ".env")
    load_project_env(project.parent.parent / ".env")
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise VoiceoverError(
            "GEMINI_API_KEY is not set. Put it in the shell environment or in the repository .env "
            "(copy .env.example; that file is gitignored)."
        )
    old_state = read_json(state_path) if state_path.exists() else {}
    compatible = old_state.get("transcript_sha256") == transcript_hash and old_state.get("config_sha256") == config_hash
    if output_path.exists() and not compatible and not overwrite:
        raise VoiceoverError("narration.wav exists for different inputs; rerun with --overwrite after reviewing it")
    if output_path.exists() and not compatible and overwrite:
        archive_dir = archive_existing_take(project, work_dir, output_path)
        print(f"Archived previous native take at {archive_dir}.")
    state: dict[str, Any] = {
        "schema_version": "1.0",
        "provider": "google_gemini_tts",
        "transcript_sha256": transcript_hash,
        "config_sha256": config_hash,
        "model": config["model"],
        "voice_name": config["voice_name"],
        "segments": {},
    }
    old_segments = old_state.get("segments", {}) if compatible else {}
    wav_paths: list[Path] = []
    for index, chunk in enumerate(chunks, start=1):
        segment_id = f"segment_{index:03d}"
        wav_path = work_dir / "segments" / f"{segment_id}.wav"
        old = old_segments.get(segment_id, {})
        if old.get("text_sha256") == sha256_text(chunk) and valid_wav(wav_path):
            print(f"Reusing {segment_id}.")
        else:
            print(f"Generating {segment_id}/{len(chunks):03d}…")
            duration = write_pcm_wav(wav_path, request_pcm(chunk, config, api_key, timeout), config["sample_rate_hz"])
            print(f"  {duration:.1f}s generated")
        with wave.open(str(wav_path), "rb") as source:
            duration = source.getnframes() / source.getframerate()
        state["segments"][segment_id] = {
            "file": str(wav_path.relative_to(project)),
            "text_sha256": sha256_text(chunk),
            "duration_seconds": round(duration, 3),
        }
        write_json(state_path, state)
        wav_paths.append(wav_path)
    total_duration = concatenate_wavs(wav_paths, output_path)
    metadata = dict(state)
    metadata["output_file"] = str(output_path.relative_to(project))
    metadata["duration_seconds"] = round(total_duration, 3)
    write_json(work_dir / "voiceover.json", metadata)
    print(f"Wrote {output_path} ({total_duration:.1f}s). Next: run scripts/align_audio.py.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace narration.wav when inputs changed")
    parser.add_argument("--timeout", type=int, default=180, help="Per-request network timeout in seconds")
    args = parser.parse_args()
    try:
        run(args.project.resolve(), args.dry_run, args.overwrite, args.timeout)
    except VoiceoverError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
