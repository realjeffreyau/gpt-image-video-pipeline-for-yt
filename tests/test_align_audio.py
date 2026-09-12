import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import align_audio as alignment


def test_word_timestamp_payload_makes_assembler_schema_and_removes_overlap():
    payload = alignment.word_timestamp_payload(
        [{"words": [
            {"word": " Hello", "start": 0.0, "end": 0.4},
            {"word": "world", "start": 0.35, "end": 0.8},
        ]}],
        duration=1.0,
        audio_file="narration.wav",
    )
    assert payload["schema_version"] == "1.0"
    assert payload["audio_file"] == "narration.wav"
    assert payload["words"][1]["start"] == 0.4
    assert payload["words"][0]["index"] == 0


def test_alignment_match_ratio_detects_bad_transcript():
    assert alignment.transcript_match_ratio("A spider is here", ["A", "spider", "is", "here"]) == 1
    assert alignment.transcript_match_ratio("A spider is here", ["unrelated", "words"]) < 0.5


def test_word_timestamp_payload_requires_words():
    with pytest.raises(alignment.AlignmentError, match="no word"):
        alignment.word_timestamp_payload([{"words": []}], duration=2)
