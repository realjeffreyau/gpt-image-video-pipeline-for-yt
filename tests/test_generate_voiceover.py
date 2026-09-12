import sys
import wave
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import generate_voiceover as voice


def test_split_transcript_preserves_words_and_respects_limit():
    transcript = "One short sentence. Two short sentences. A very long final sentence with many separate words."
    chunks = voice.split_transcript(transcript, 35)
    assert " ".join(chunks).split() == transcript.split()
    assert all(len(chunk) <= 35 for chunk in chunks)


def test_request_body_is_audio_tts_with_requested_voice():
    config = {
        "style_instruction": "Calm and clear.",
        "voice_name": "Kore",
        "model": "gemini-3.1-flash-tts-preview",
    }
    payload = voice.build_request_body("Hello there.", config)
    assert payload["response_format"] == {"type": "audio"}
    assert payload["generation_config"]["speech_config"] == [{"voice": "Kore"}]
    assert payload["model"] == "gemini-3.1-flash-tts-preview"
    assert "Hello there." in payload["input"]


def test_concatenate_wavs(tmp_path):
    first, second, output = (tmp_path / "one.wav", tmp_path / "two.wav", tmp_path / "all.wav")
    for path, frames in ((first, b"\x01\x00" * 24), (second, b"\x02\x00" * 48)):
        with wave.open(str(path), "wb") as file:
            file.setnchannels(1)
            file.setsampwidth(2)
            file.setframerate(24)
            file.writeframes(frames)
    assert voice.concatenate_wavs([first, second], output) == 3
    with wave.open(str(output), "rb") as file:
        assert file.getnframes() == 72


def test_archive_existing_take_preserves_audio_state_and_segments(tmp_path):
    project = tmp_path / "project"
    work_dir = project / "work" / "tts"
    segments = work_dir / "segments"
    segments.mkdir(parents=True)
    output = project / "input" / "narration.wav"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"old narration")
    (segments / "segment_001.wav").write_bytes(b"old segment")
    (work_dir / "state.json").write_text('{"old": true}\n', encoding="utf-8")
    (work_dir / "voiceover.json").write_text(
        '{"duration_seconds": 91.5}\n', encoding="utf-8"
    )

    archive = voice.archive_existing_take(project, work_dir, output)

    assert archive.name == "take_001_native"
    assert (archive / "narration.wav").read_bytes() == b"old narration"
    assert (archive / "segments" / "segment_001.wav").read_bytes() == b"old segment"
    assert json.loads((archive / "state.json").read_text())["old"] is True
    assert json.loads((archive / "archive.json").read_text())["status"] == "archived_before_replacement"

    second = voice.archive_existing_take(project, work_dir, output)
    assert second.name == "take_002_native"
