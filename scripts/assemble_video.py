#!/usr/bin/env python3
"""Assemble aligned narration and still images into a Ken Burns-style video."""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
import shutil
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# All final videos land in the repository's shared output folder.
FINAL_VIDEOS_DIR = Path(__file__).resolve().parents[1] / "final-videos"


class AssemblyError(Exception):
    """A user-actionable project or assembly error."""


@dataclass(frozen=True)
class Word:
    index: int
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class TimedShot:
    beat_id: str
    image_path: Path
    start: float
    end: float
    motion: dict[str, Any]

    @property
    def duration(self) -> float:
        return self.end - self.start


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise AssemblyError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AssemblyError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AssemblyError(f"{path} must contain a JSON object")
    return value


def tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", normalized, flags=re.UNICODE)


def require_number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssemblyError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise AssemblyError(f"{label} must be finite and at least {minimum}")
    return result


def parse_words(data: dict[str, Any]) -> tuple[list[Word], float, str]:
    if data.get("schema_version") != "1.0":
        raise AssemblyError("timestamps.json schema_version must be '1.0'")
    raw_words = data.get("words")
    if not isinstance(raw_words, list) or not raw_words:
        raise AssemblyError("timestamps.json words must be a non-empty array")

    words: list[Word] = []
    previous_end = 0.0
    for position, raw in enumerate(raw_words):
        if not isinstance(raw, dict):
            raise AssemblyError(f"timestamps word {position} must be an object")
        if raw.get("index") != position:
            raise AssemblyError(
                f"timestamps word indexes must be contiguous and zero-based; "
                f"expected {position}"
            )
        text = raw.get("word")
        if not isinstance(text, str) or not tokens(text):
            raise AssemblyError(f"timestamps word {position} has no usable text")
        start = require_number(raw.get("start"), f"word {position} start")
        end = require_number(raw.get("end"), f"word {position} end")
        if end <= start:
            raise AssemblyError(f"timestamps word {position} must end after it starts")
        if start + 0.001 < previous_end:
            raise AssemblyError(f"timestamps word {position} overlaps the previous word")
        words.append(Word(position, tokens(text)[0], start, end))
        previous_end = end

    duration = require_number(
        data.get("duration_seconds"), "timestamps duration_seconds", minimum=0.001
    )
    if duration + 0.05 < words[-1].end:
        raise AssemblyError("duration_seconds ends before the last timestamped word")
    audio_file = data.get("audio_file")
    if not isinstance(audio_file, str) or not audio_file.strip():
        raise AssemblyError("timestamps.json audio_file must be a non-empty string")
    return words, duration, audio_file


def parse_shots(
    data: dict[str, Any], images_dir: Path
) -> tuple[list[dict[str, Any]], list[str], list[int]]:
    if data.get("schema_version") != "1.0":
        raise AssemblyError("shots.json schema_version must be '1.0'")
    raw_shots = data.get("shots")
    if not isinstance(raw_shots, list) or not raw_shots:
        raise AssemblyError("shots.json shots must be a non-empty array")

    shots: list[dict[str, Any]] = []
    source_words: list[str] = []
    starts: list[int] = []
    seen_ids: set[str] = set()
    for position, raw in enumerate(raw_shots):
        if not isinstance(raw, dict):
            raise AssemblyError(f"shot {position} must be an object")
        beat_id = raw.get("beat_id")
        if not isinstance(beat_id, str) or not beat_id.strip():
            raise AssemblyError(f"shot {position} needs a non-empty beat_id")
        if beat_id in seen_ids:
            raise AssemblyError(f"duplicate beat_id: {beat_id}")
        seen_ids.add(beat_id)

        image_file = raw.get("image_file")
        if not isinstance(image_file, str) or not image_file.strip():
            raise AssemblyError(f"{beat_id} needs a non-empty image_file")
        image_path = (images_dir / image_file).resolve()
        try:
            image_path.relative_to(images_dir.resolve())
        except ValueError as exc:
            raise AssemblyError(f"{beat_id} image_file must stay inside images/") from exc
        if not image_path.is_file():
            raise AssemblyError(f"{beat_id} image is missing: {image_path}")

        lines = raw.get("narration_lines")
        if (
            not isinstance(lines, list)
            or not lines
            or not all(isinstance(line, str) and line.strip() for line in lines)
        ):
            raise AssemblyError(
                f"{beat_id} narration_lines must be a non-empty array of strings"
            )
        shot_words = tokens(" ".join(lines))
        if not shot_words:
            raise AssemblyError(f"{beat_id} narration_lines contain no usable words")
        starts.append(len(source_words))
        source_words.extend(shot_words)

        motion = raw.get("motion", {})
        if not isinstance(motion, dict):
            raise AssemblyError(f"{beat_id} motion must be an object when present")
        shots.append(
            {
                "beat_id": beat_id,
                "image_path": image_path,
                "motion": motion,
                "hold_seconds": raw.get("hold_seconds"),
            }
        )
    return shots, source_words, starts


def source_to_alignment_map(
    source_words: list[str], aligned_words: list[str]
) -> tuple[list[int], float]:
    matcher = difflib.SequenceMatcher(
        None, source_words, aligned_words, autojunk=False
    )
    mapping = [0] * len(source_words)
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        source_count = a1 - a0
        target_count = b1 - b0
        if tag == "equal":
            for offset in range(source_count):
                mapping[a0 + offset] = b0 + offset
        elif source_count:
            for offset in range(source_count):
                if target_count:
                    proportional = min(
                        target_count - 1, int(offset * target_count / source_count)
                    )
                    mapping[a0 + offset] = b0 + proportional
                else:
                    mapping[a0 + offset] = min(b0, len(aligned_words) - 1)
    return mapping, matcher.ratio()


def build_timeline(
    shots_data: dict[str, Any],
    timestamps_data: dict[str, Any],
    input_dir: Path,
    *,
    minimum_match: float,
) -> tuple[list[TimedShot], str, float]:
    words, audio_duration, audio_file = parse_words(timestamps_data)
    shots, source_words, source_starts = parse_shots(
        shots_data, input_dir / "images"
    )
    mapping, match_score = source_to_alignment_map(
        source_words, [word.text for word in words]
    )
    if match_score < minimum_match:
        raise AssemblyError(
            "Narration-to-timestamp match is too weak: "
            f"{match_score:.1%} (minimum {minimum_match:.1%}). "
            "Check that shots.json and timestamps.json describe the same narration."
        )

    boundaries = [0.0]
    for source_start in source_starts[1:]:
        target_index = mapping[source_start]
        boundaries.append(words[target_index].start)
    boundaries.append(audio_duration)

    result: list[TimedShot] = []
    for position, shot in enumerate(shots):
        start = boundaries[position]
        end = boundaries[position + 1]
        if end - start < 0.1:
            raise AssemblyError(
                f"{shot['beat_id']} resolves to only {end - start:.3f}s; "
                "check the narration split or word alignment"
            )
        motion = normalize_motion(shot["motion"], position)
        result.append(
            TimedShot(
                beat_id=shot["beat_id"],
                image_path=shot["image_path"],
                start=start,
                end=end,
                motion=motion,
            )
        )
    return result, audio_file, match_score


def normalize_motion(raw: dict[str, Any], position: int) -> dict[str, Any]:
    defaults = ("zoom_in", "pan_right", "zoom_out", "pan_left")
    motion_type = raw.get("type", defaults[position % len(defaults)])
    allowed = {
        "static",
        "zoom_in",
        "zoom_out",
        "pan_left",
        "pan_right",
        "pan_up",
        "pan_down",
    }
    if motion_type not in allowed:
        raise AssemblyError(
            f"Unsupported motion type {motion_type!r}; choose one of {sorted(allowed)}"
        )
    focus_x = require_number(raw.get("focus_x", 0.5), "motion focus_x")
    focus_y = require_number(raw.get("focus_y", 0.5), "motion focus_y")
    if focus_x > 1.0 or focus_y > 1.0:
        raise AssemblyError("motion focus_x and focus_y must be between 0.0 and 1.0")
    if motion_type == "static":
        return {
            "type": "static",
            "strength": 0.0,
            "focus_x": focus_x,
            "focus_y": focus_y,
        }
    strength = require_number(raw.get("strength", 0.06), "motion strength")
    if not 0.01 <= strength <= 0.25:
        raise AssemblyError("motion strength must be between 0.01 and 0.25")
    return {
        "type": motion_type,
        "strength": strength,
        "focus_x": focus_x,
        "focus_y": focus_y,
    }


def zoompan_expressions(
    motion: dict[str, Any], frames: int
) -> tuple[str, str, str]:
    kind = motion["type"]
    strength = motion["strength"]
    focus_x = motion["focus_x"]
    focus_y = motion["focus_y"]
    last = max(frames - 1, 1)
    linear_progress = f"min(on/{last},1)"
    # Smoothstep avoids the mechanical acceleration/deceleration that makes
    # short documentary beats feel jittery when a pan or zoom resets.
    progress = f"({linear_progress})*({linear_progress})*(3-2*({linear_progress}))"
    max_zoom = 1.0 + strength

    centered_x = (
        f"max(0,min(iw-iw/zoom,iw*{focus_x:.6f}-iw/zoom/2))"
    )
    centered_y = (
        f"max(0,min(ih-ih/zoom,ih*{focus_y:.6f}-ih/zoom/2))"
    )
    if kind == "zoom_in":
        z = f"min(1+{strength:.8f}*{progress},{max_zoom:.8f})"
        return z, centered_x, centered_y
    if kind == "zoom_out":
        z = f"max(1,{max_zoom:.8f}-{strength:.8f}*{progress})"
        return z, centered_x, centered_y

    z = f"{max_zoom:.8f}"
    if kind == "pan_right":
        return z, f"(iw-iw/zoom)*{progress}", centered_y
    if kind == "pan_left":
        return z, f"(iw-iw/zoom)*(1-{progress})", centered_y
    if kind == "pan_down":
        return z, centered_x, f"(ih-ih/zoom)*{progress}"
    return z, centered_x, f"(ih-ih/zoom)*(1-{progress})"


def build_filter_graph(
    timeline: list[TimedShot], width: int, height: int, fps: int
) -> str:
    filters: list[str] = []
    labels: list[str] = []
    for position, shot in enumerate(timeline):
        frames = max(1, round(shot.duration * fps))
        label = f"v{position}"
        base = (
            f"[{position}:v]"
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1"
        )
        if shot.motion["type"] == "static":
            filters.append(
                f"{base},fps={fps},trim=duration={shot.duration:.6f},"
                f"setpts=PTS-STARTPTS[{label}]"
            )
        else:
            z, x, y = zoompan_expressions(shot.motion, frames)
            filters.append(
                f"{base},"
                f"zoompan=z='{z}':x='{x}':y='{y}':d=1:"
                f"s={width}x{height}:fps={fps},"
                f"trim=duration={shot.duration:.6f},setpts=PTS-STARTPTS"
                f"[{label}]"
            )
        labels.append(f"[{label}]")
    if len(labels) == 1:
        filters.append(f"{labels[0]}null[outv]")
    else:
        filters.append(
            "".join(labels) + f"concat=n={len(labels)}:v=1:a=0[outv]"
        )
    return ";\n".join(filters) + "\n"


def build_ffmpeg_command(
    timeline: list[TimedShot],
    audio_path: Path,
    filter_path: Path,
    output_path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    overwrite: bool,
) -> list[str]:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning"]
    all_static = all(shot.motion["type"] == "static" for shot in timeline)
    command.append("-y" if overwrite else "-n")
    for shot in timeline:
        command.extend(
            [
                "-loop",
                "1",
                "-framerate",
                str(fps),
                "-t",
                f"{shot.duration:.6f}",
                "-i",
                str(shot.image_path),
            ]
        )
    command.extend(
        [
            "-i",
            str(audio_path),
            "-filter_complex_script",
            str(filter_path),
            "-map",
            "[outv]",
            "-map",
            f"{len(timeline)}:a:0",
            "-c:v",
            "libx264",
            *(["-g", "1"] if all_static else []),
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-t",
            f"{timeline[-1].end:.6f}",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    return command


def find_media_tools() -> tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        return ffmpeg, ffprobe
    try:
        from static_ffmpeg import run as static_ffmpeg_run

        static_ffmpeg, static_ffprobe = (
            static_ffmpeg_run.get_or_fetch_platform_executables_else_raise()
        )
        return str(static_ffmpeg), str(static_ffprobe)
    except (ImportError, OSError, RuntimeError) as exc:
        raise AssemblyError(
            "ffmpeg and ffprobe are not on PATH. Install them system-wide or "
            "run with the project virtual environment described in README.md."
        ) from exc


def probe_audio_duration(audio_path: Path, ffprobe: str) -> float:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise AssemblyError(
            f"ffprobe could not read {audio_path}: {result.stderr.strip()}"
        )
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise AssemblyError("ffprobe returned an invalid audio duration") from exc


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def infer_dimensions(shots_data: dict[str, Any]) -> tuple[int, int]:
    """Choose safe render dimensions from the shot-plan contract.

    Older plans did not declare an aspect ratio, so they retain the historical
    16:9 default. New portrait plans can no longer be accidentally rendered as
    landscape merely because the caller forgot two CLI flags.
    """
    aspect_ratio = shots_data.get("aspect_ratio", "16:9")
    if aspect_ratio == "9:16":
        return 1080, 1920
    if aspect_ratio == "16:9":
        return 1920, 1080
    raise AssemblyError(
        "shots.json aspect_ratio must be '9:16' or '16:9' when present"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble a project folder into an aligned slideshow video."
    )
    parser.add_argument("project", type=Path, help="project directory")
    parser.add_argument("--output", type=Path, help="override output MP4 path")
    parser.add_argument(
        "--width", type=positive_int, help="output width; inferred from shots.json by default"
    )
    parser.add_argument(
        "--height", type=positive_int, help="output height; inferred from shots.json by default"
    )
    parser.add_argument("--fps", type=positive_int, default=30)
    parser.add_argument("--minimum-match", type=float, default=0.75)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate and resolve timings without requiring ffmpeg",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.minimum_match <= 1.0:
        print("error: --minimum-match must be between 0 and 1", file=sys.stderr)
        return 2
    if (args.width is None) != (args.height is None):
        print("error: --width and --height must be supplied together", file=sys.stderr)
        return 2

    try:
        project = args.project.resolve()
        input_dir = project / "input"
        work_dir = project / "work"
        output_dir = project / "output"
        shots_data = load_json(input_dir / "shots.json")
        timestamps_data = load_json(input_dir / "timestamps.json")
        width, height = (
            (args.width, args.height)
            if args.width is not None and args.height is not None
            else infer_dimensions(shots_data)
        )
        timeline, audio_file, match_score = build_timeline(
            shots_data,
            timestamps_data,
            input_dir,
            minimum_match=args.minimum_match,
        )
        audio_path = (input_dir / audio_file).resolve()
        try:
            audio_path.relative_to(input_dir.resolve())
        except ValueError as exc:
            raise AssemblyError(
                "timestamps audio_file must stay inside the project's input/"
            ) from exc
        if not audio_path.is_file():
            raise AssemblyError(f"Narration audio is missing: {audio_path}")

        print(
            f"Validated {len(timeline)} shots; narration match {match_score:.1%}; "
            f"timeline {timeline[-1].end:.2f}s"
        )
        for shot in timeline:
            print(
                f"  {shot.beat_id}: {shot.start:8.3f}s–{shot.end:8.3f}s "
                f"({shot.duration:7.3f}s) {shot.motion['type']}"
            )
        if args.validate_only:
            return 0

        ffmpeg, ffprobe = find_media_tools()
        actual_audio_duration = probe_audio_duration(audio_path, ffprobe)
        expected_duration = timeline[-1].end
        if abs(actual_audio_duration - expected_duration) > 0.25:
            raise AssemblyError(
                "Audio duration differs from timestamps.json by "
                f"{abs(actual_audio_duration - expected_duration):.3f}s "
                f"(audio {actual_audio_duration:.3f}s, JSON {expected_duration:.3f}s)"
            )

        work_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        filter_path = work_dir / "filter_complex.txt"
        filter_path.write_text(
            build_filter_graph(
                timeline, width=width, height=height, fps=args.fps
            ),
            encoding="utf-8",
        )
        output_path = (
            args.output.resolve()
            if args.output
            else (FINAL_VIDEOS_DIR / f"{project.name}.mp4").resolve()
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not args.overwrite:
            raise AssemblyError(
                f"Output already exists: {output_path} (use --overwrite to replace it)"
            )

        command = build_ffmpeg_command(
            timeline,
            audio_path,
            filter_path,
            output_path,
            width=width,
            height=height,
            fps=args.fps,
            overwrite=args.overwrite,
        )
        command[0] = ffmpeg
        print(f"Rendering {output_path} ...")
        result = subprocess.run(command)
        if result.returncode:
            raise AssemblyError(f"ffmpeg exited with status {result.returncode}")
        print(f"Created {output_path}")
        return 0
    except AssemblyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
