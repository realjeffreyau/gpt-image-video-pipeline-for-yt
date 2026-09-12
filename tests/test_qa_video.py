import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import qa_video


def test_compare_reports_identical_and_changed_pixels():
    identical = qa_video.compare(bytes([10, 20, 30]), bytes([10, 20, 30]))
    changed = qa_video.compare(bytes([10, 20, 30]), bytes([20, 20, 40]))

    assert identical["mae"] == 0
    assert identical["changed_pct_gt1"] == 0
    assert changed["mae"] > 0
    assert changed["changed_pct_gt1"] > 0


def test_media_summary_extracts_required_fields():
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1080,
                "height": 1920,
                "avg_frame_rate": "30/1",
                "codec_name": "h264",
                "profile": "High",
                "pix_fmt": "yuv420p",
                "nb_frames": "1800",
                "duration": "60.0",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "profile": "LC",
                "sample_rate": "24000",
                "channels": 1,
                "duration": "60.0",
            },
        ],
        "format": {"duration": "60.0"},
    }

    summary = qa_video.media_summary(probe)

    assert summary["width"] == 1080
    assert summary["height"] == 1920
    assert summary["video_frames"] == 1800
    assert summary["audio_sample_rate_hz"] == 24000


def test_sample_dimensions_preserve_orientation_and_bound_memory():
    assert qa_video.sample_dimensions(1080, 1920) == (270, 480)
    assert qa_video.sample_dimensions(1920, 1080) == (480, 270)
    assert qa_video.sample_dimensions(320, 180) == (320, 180)


def test_static_report_reuses_supplied_video_hash(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"encoded video")

    def fake_decode(_ffmpeg, _video, frames, *, width, height):
        assert (width, height) == (270, 480)
        return {frame: b"same frame" for frame in frames}

    monkeypatch.setattr(qa_video, "decode_selected_frames", fake_decode)
    monkeypatch.setattr(
        qa_video,
        "sha256_file",
        lambda _path: pytest.fail("static report hashed the video twice"),
    )
    shot = SimpleNamespace(
        beat_id="beat_001",
        start=0.0,
        end=1.0,
        duration=1.0,
    )

    report = qa_video.build_static_report(
        "ffmpeg",
        video,
        [shot],
        width=1080,
        height=1920,
        fps=30,
        video_sha256="cached-digest",
    )

    assert report["video_sha256"] == "cached-digest"
    assert report["pass"] is True


def test_dynamic_report_is_diagnostic_and_surfaces_temporal_spikes(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"encoded video")

    def fake_decode(_ffmpeg, _video, frames, *, width, height):
        # A single abrupt excursion is a useful synthetic jitter signal. The
        # report must surface it without turning the overall QA gate red.
        payloads = []
        for index in range(len(frames)):
            level = 20 if index == 3 else index
            payloads.append(bytes([level]) * (width * height))
        return dict(zip(frames, payloads))

    monkeypatch.setattr(qa_video, "decode_selected_frames", fake_decode)
    monkeypatch.setattr(
        qa_video,
        "sha256_file",
        lambda _path: pytest.fail("dynamic report hashed the video twice"),
    )
    timeline = [
        SimpleNamespace(
            beat_id="static_beat",
            start=0.0,
            end=1.0,
            duration=1.0,
            motion={"type": "static"},
        ),
        SimpleNamespace(
            beat_id="moving_beat",
            start=1.0,
            end=3.0,
            duration=2.0,
            motion={"type": "pan_right"},
        ),
    ]

    report = qa_video.build_dynamic_report(
        "ffmpeg",
        video,
        timeline,
        width=320,
        height=180,
        fps=30,
        video_sha256="cached-digest",
        sample_fps=5,
    )

    assert report["status"] == "diagnostic"
    assert report["pass"] is True
    assert report["video_sha256"] == "cached-digest"
    assert report["dynamic_shot_count"] == 1
    assert report["review_shots"] == ["moving_beat"]
