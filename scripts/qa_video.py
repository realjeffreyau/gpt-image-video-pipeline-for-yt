#!/usr/bin/env python3
"""Run centralized media, sync, and zero-jitter QA for an assembled project."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ASSEMBLER_PATH = ROOT / "scripts" / "assemble_video.py"


class QAError(RuntimeError):
    pass


def load_assembler():
    spec = importlib.util.spec_from_file_location("assembler_for_shared_qa", ASSEMBLER_PATH)
    if spec is None or spec.loader is None:
        raise QAError(f"cannot import {ASSEMBLER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def run_checked(command: list[str], label: str) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise QAError(f"{label} failed with exit {result.returncode}: {detail}")
    return result


def probe_video(ffprobe: str, video: Path) -> dict[str, Any]:
    result = run_checked(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(video),
        ],
        "ffprobe",
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise QAError(f"ffprobe returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise QAError("ffprobe output must be a JSON object")
    return payload


def media_summary(probe: dict[str, Any]) -> dict[str, Any]:
    streams = probe.get("streams")
    if not isinstance(streams, list):
        raise QAError("ffprobe output has no streams array")
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    if not isinstance(video, dict) or not isinstance(audio, dict):
        raise QAError("final MP4 must contain video and audio streams")
    return {
        "width": video.get("width"),
        "height": video.get("height"),
        "fps": video.get("avg_frame_rate"),
        "video_codec": video.get("codec_name"),
        "video_profile": video.get("profile"),
        "pixel_format": video.get("pix_fmt"),
        "video_frames": int(video.get("nb_frames", 0)),
        "video_duration_seconds": float(video.get("duration", 0.0)),
        "audio_codec": audio.get("codec_name"),
        "audio_profile": audio.get("profile"),
        "audio_sample_rate_hz": int(audio.get("sample_rate", 0)),
        "audio_channels": int(audio.get("channels", 0)),
        "audio_duration_seconds": float(audio.get("duration", 0.0)),
        "container_duration_seconds": float(probe.get("format", {}).get("duration", 0.0)),
    }


def decode_selected_frames(
    ffmpeg: str,
    video: Path,
    frame_numbers: list[int],
    *,
    width: int,
    height: int,
) -> dict[int, bytes]:
    """Decode every requested frame in one ffmpeg process.

    Older project-local QA helpers started a new decoder for each frame. A
    single select expression is materially faster and produces the same raw
    gray pixels in ascending frame order.
    """
    unique = sorted(set(frame_numbers))
    if not unique:
        return {}
    expression = "+".join(f"eq(n\\,{number})" for number in unique)
    result = run_checked(
        [
            ffmpeg,
            "-xerror",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select={expression},scale={width}:{height}:flags=area",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        "selected-frame decode",
    )
    frame_size = width * height
    expected = frame_size * len(unique)
    if len(result.stdout) != expected:
        raise QAError(
            f"selected-frame decode returned {len(result.stdout)} bytes; expected {expected}"
        )
    return {
        number: result.stdout[index * frame_size : (index + 1) * frame_size]
        for index, number in enumerate(unique)
    }


def sample_dimensions(width: int, height: int, maximum_edge: int = 480) -> tuple[int, int]:
    """Return small, even, aspect-preserving dimensions for pixel QA."""
    if width <= 0 or height <= 0 or maximum_edge <= 0:
        raise QAError("sample dimensions must be positive")
    scale = min(1.0, maximum_edge / max(width, height))
    sampled_width = max(2, int(width * scale) // 2 * 2)
    sampled_height = max(2, int(height * scale) // 2 * 2)
    return sampled_width, sampled_height


def compare(left: bytes, right: bytes) -> dict[str, float | int | str]:
    differences = [abs(a - b) for a, b in zip(left, right)]
    return {
        "mae": round(sum(differences) / len(differences), 6),
        "max_abs": max(differences),
        "changed_pct_gt1": round(
            sum(value > 1 for value in differences) / len(differences) * 100, 6
        ),
        "sha_a": hashlib.sha256(left).hexdigest(),
        "sha_b": hashlib.sha256(right).hexdigest(),
    }


def build_static_report(
    ffmpeg: str,
    video: Path,
    timeline: list[Any],
    *,
    width: int,
    height: int,
    fps: int,
    video_sha256: str | None = None,
) -> dict[str, Any]:
    hold_samples: list[tuple[Any, list[float], list[int]]] = []
    cut_samples: list[tuple[float, int, int]] = []
    frames_needed: list[int] = []
    for shot in timeline:
        pad = min(0.6, shot.duration / 4)
        times = [shot.start + pad, (shot.start + shot.end) / 2, shot.end - pad]
        frames = [max(0, round(value * fps)) for value in times]
        hold_samples.append((shot, times, frames))
        frames_needed.extend(frames)
    for left, _right in zip(timeline, timeline[1:]):
        before = max(0, round((left.end - 0.1) * fps))
        after = round((left.end + 0.1) * fps)
        cut_samples.append((left.end, before, after))
        frames_needed.extend((before, after))

    sampled_width, sampled_height = sample_dimensions(width, height)
    decoded = decode_selected_frames(
        ffmpeg,
        video,
        frames_needed,
        width=sampled_width,
        height=sampled_height,
    )
    holds: list[dict[str, Any]] = []
    for shot, times, frames in hold_samples:
        comparisons = [compare(decoded[frames[0]], decoded[number]) for number in frames[1:]]
        holds.append(
            {
                "beat_id": shot.beat_id,
                "times_seconds": [round(value, 6) for value in times],
                "frames": frames,
                "comparisons_to_early": comparisons,
                "max_mae": max(item["mae"] for item in comparisons),
                "max_changed_pct_gt1": max(
                    item["changed_pct_gt1"] for item in comparisons
                ),
            }
        )
    cuts: list[dict[str, Any]] = []
    for boundary, before, after in cut_samples:
        cuts.append(
            {
                "boundary_seconds": round(boundary, 6),
                "before_frame": before,
                "after_frame": after,
                "comparison": compare(decoded[before], decoded[after]),
            }
        )
    hold_max_mae = max(item["max_mae"] for item in holds)
    hold_max_changed = max(item["max_changed_pct_gt1"] for item in holds)
    cut_min_mae = min(item["comparison"]["mae"] for item in cuts) if cuts else 1.0
    cut_min_changed = (
        min(item["comparison"]["changed_pct_gt1"] for item in cuts) if cuts else 100.0
    )
    passed = (
        hold_max_mae < 0.01
        and hold_max_changed < 0.1
        and cut_min_mae > 0.1
        and cut_min_changed > 5.0
    )
    return {
        "schema_version": "1.0",
        "video_file": str(video),
        "video_sha256": video_sha256 or sha256_file(video),
        "method": "one batched, full-stream-validated ffmpeg decode with aspect-preserving 480px grayscale samples at aligned early/mid/late holds and +/-0.1s around every cut",
        "width": width,
        "height": height,
        "sample_width": sampled_width,
        "sample_height": sampled_height,
        "fps": fps,
        "decoded_frame_count": len(decoded),
        "hold_comparison_count": len(holds) * 2,
        "cut_count": len(cuts),
        "holds": holds,
        "cuts": cuts,
        "hold_max_mae": hold_max_mae,
        "hold_max_changed_pct_gt1": hold_max_changed,
        "cut_min_mae": cut_min_mae,
        "cut_min_changed_pct_gt1": cut_min_changed,
        "status": "pass" if passed else "fail",
        "pass": passed,
    }


def _frame_mae(left: bytes, right: bytes) -> float:
    if not left or len(left) != len(right):
        raise QAError("cannot compare empty or differently sized frames")
    return sum(abs(a - b) for a, b in zip(left, right)) / len(left)


def _motion_sample_frames(shot: Any, fps: int, sample_fps: int) -> list[int]:
    step = max(1, round(fps / sample_fps))
    first = max(0, round(shot.start * fps) + 1)
    last = max(first, round(shot.end * fps) - 1)
    frames = list(range(first, last + 1, step))
    if len(frames) < 3:
        frames = sorted({first, (first + last) // 2, last})
    elif frames[-1] != last:
        frames.append(last)
    return frames


def build_dynamic_report(
    ffmpeg: str,
    video: Path,
    timeline: list[Any],
    *,
    width: int,
    height: int,
    fps: int,
    video_sha256: str | None = None,
    sample_fps: int = 5,
) -> dict[str, Any]:
    """Sample moving shots for temporal spikes without gating a render.

    Encoded motion is not expected to be pixel-identical frame to frame, so a
    hard pass/fail threshold would confuse normal codec noise with a visible
    hitch. This report deliberately remains diagnostic: it highlights abrupt
    MAE excursions for human review while the ``-xerror`` batched decode remains
    the authoritative media-integrity gate.
    """
    if fps <= 0 or sample_fps <= 0:
        raise QAError("motion QA frame rates must be positive")
    moving = [shot for shot in timeline if shot.motion["type"] != "static"]
    sampled_width, sampled_height = sample_dimensions(width, height, maximum_edge=160)
    if not moving:
        return {
            "schema_version": "1.0",
            "status": "not_applicable",
            "pass": True,
            "video_file": str(video),
            "video_sha256": video_sha256 or sha256_file(video),
            "sample_fps": sample_fps,
            "sample_width": sampled_width,
            "sample_height": sampled_height,
            "dynamic_shot_count": 0,
            "sample_frame_count": 0,
            "shots": [],
            "review_shots": [],
        }

    frame_plan = [
        (shot, _motion_sample_frames(shot, fps, sample_fps)) for shot in moving
    ]
    frame_numbers = [frame for _shot, frames in frame_plan for frame in frames]
    decoded = decode_selected_frames(
        ffmpeg,
        video,
        frame_numbers,
        width=sampled_width,
        height=sampled_height,
    )
    shot_reports: list[dict[str, Any]] = []
    review_shots: list[str] = []
    for shot, frames in frame_plan:
        differences = [
            _frame_mae(decoded[left], decoded[right])
            for left, right in zip(frames, frames[1:])
        ]
        baseline = max(statistics.median(differences), 0.01)
        spike_threshold = max(2.0, baseline * 4.0)
        low_neighbor = baseline * 2.0
        spikes = [
            index
            for index, value in enumerate(differences)
            if value > spike_threshold
            and (
                (index > 0 and differences[index - 1] <= low_neighbor)
                or (
                    index + 1 < len(differences)
                    and differences[index + 1] <= low_neighbor
                )
            )
        ]
        spike_events: list[list[int]] = []
        for index in spikes:
            if spike_events and index == spike_events[-1][-1] + 1:
                spike_events[-1].append(index)
            else:
                spike_events.append([index])
        isolated_events = [
            event
            for event in spike_events
            if len(event) <= 2
            and (event[0] == 0 or differences[event[0] - 1] <= low_neighbor)
            and (
                event[-1] == len(differences) - 1
                or differences[event[-1] + 1] <= low_neighbor
            )
        ]
        max_adjacent_delta = max(
            (abs(right - left) for left, right in zip(differences, differences[1:])),
            default=0.0,
        )
        # Repeated high differences are usually scene texture being revealed
        # by a steady pan. Only a small, isolated excursion with a materially
        # larger-than-baseline jump becomes a human-review signal.
        if (
            isolated_events
            and len(spike_events) <= 2
            and max_adjacent_delta > max(4.0, baseline * 8.0)
        ):
            review_shots.append(shot.beat_id)
        shot_reports.append(
            {
                "beat_id": shot.beat_id,
                "motion_type": shot.motion["type"],
                "sample_count": len(frames),
                "frames": frames,
                "mean_mae": round(statistics.mean(differences), 6),
                "median_mae": round(statistics.median(differences), 6),
                "max_mae": round(max(differences), 6),
                "max_adjacent_delta": round(max_adjacent_delta, 6),
                "spike_threshold": round(spike_threshold, 6),
                "spike_count": len(spikes),
                "spike_event_count": len(spike_events),
                "isolated_spike_event_count": len(isolated_events),
                "motion_detected_fraction": round(
                    sum(value > 0.01 for value in differences) / len(differences),
                    6,
                ),
            }
        )
    return {
        "schema_version": "1.0",
        "status": "diagnostic",
        "pass": True,
        "video_file": str(video),
        "video_sha256": video_sha256 or sha256_file(video),
        "method": "one batched, -xerror ffmpeg decode of 160px-aspect-preserving grayscale samples at 5fps; abrupt adjacent-frame MAE spikes are review signals, not automatic failures",
        "sample_fps": sample_fps,
        "sample_width": sampled_width,
        "sample_height": sampled_height,
        "dynamic_shot_count": len(moving),
        "sample_frame_count": len(set(frame_numbers)),
        "shots": shot_reports,
        "review_shots": review_shots,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--minimum-match", type=float, default=0.75)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        assembler = load_assembler()
    except QAError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        project = args.project.resolve()
        input_dir = project / "input"
        video = (
            args.video.resolve()
            if args.video
            else (assembler.FINAL_VIDEOS_DIR / f"{project.name}.mp4").resolve()
        )
        if not video.is_file():
            raise QAError(f"final video is missing: {video}")
        # Compute this once and reuse it in both QA reports. This avoids a
        # second full read of a potentially large MP4 while preserving the
        # exact report schema.
        video_sha256 = sha256_file(video)
        shots = assembler.load_json(input_dir / "shots.json")
        timestamps = assembler.load_json(input_dir / "timestamps.json")
        timeline, _audio_file, match = assembler.build_timeline(
            shots, timestamps, input_dir, minimum_match=args.minimum_match
        )
        ffmpeg, ffprobe = assembler.find_media_tools()
        probe = probe_video(ffprobe, video)
        summary = media_summary(probe)
        width, height = assembler.infer_dimensions(shots)
        if (summary["width"], summary["height"]) != (width, height):
            raise QAError(
                f"video is {summary['width']}x{summary['height']}; expected {width}x{height}"
            )
        if summary["fps"] != "30/1":
            raise QAError(f"video frame rate is {summary['fps']}; expected 30/1")
        if abs(summary["audio_duration_seconds"] - timeline[-1].end) > 0.05:
            raise QAError("audio duration does not match the aligned timeline")
        qa_dir = project / "work" / "qa"
        write_json_atomic(qa_dir / "ffprobe.json", probe)
        summary["narration_match"] = round(match, 6)

        all_static = all(shot.motion["type"] == "static" for shot in timeline)
        static_report = None
        motion_report = None
        if all_static:
            # The select filter must scan the complete stream to reach the last
            # requested frame. With -xerror this same pass is also the full
            # decode-integrity check, so a separate decode would be redundant.
            static_report = build_static_report(
                ffmpeg,
                video,
                timeline,
                width=width,
                height=height,
                fps=30,
                video_sha256=video_sha256,
            )
            write_json_atomic(qa_dir / "static_motion_qa.json", static_report)
            if not static_report["pass"]:
                raise QAError("decoded-frame static-motion QA failed")
        else:
            motion_report = build_dynamic_report(
                ffmpeg,
                video,
                timeline,
                width=width,
                height=height,
                fps=30,
                video_sha256=video_sha256,
            )
            write_json_atomic(qa_dir / "motion_qa.json", motion_report)
        summary["decode_status"] = "pass"
        summary["decode_passes"] = 1
        write_json_atomic(qa_dir / "ffprobe_summary.json", summary)
        result = {
            "schema_version": "1.0",
            "status": "pass",
            "video_file": str(video),
            "video_sha256": video_sha256,
            "narration_match": round(match, 6),
            "media": summary,
            "static_motion": static_report,
            "motion_smoothness": motion_report,
        }
        write_json_atomic(qa_dir / "video_qa.json", result)
        print(
            f"PASS {video}: {width}x{height}, {summary['container_duration_seconds']:.3f}s, "
            f"match={match:.1%}, batched_static_frames="
            f"{static_report['decoded_frame_count'] if static_report else 0}, "
            f"motion_review={len(motion_report['review_shots']) if motion_report else 0}"
        )
        return 0
    except (QAError, assembler.AssemblyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
