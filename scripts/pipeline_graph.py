#!/usr/bin/env python3
"""Durable, resumable coordinator for the local video-production pipeline.

This intentionally borrows LangGraph's useful execution semantics without
requiring LangGraph (or another hosted model API) at runtime:

* explicit nodes and ordered edges;
* atomic checkpoints after every transition;
* content-addressed stage reuse and downstream invalidation;
* an evaluator/optimizer loop for writer -> fact-checker -> writer;
* artifact-bound human interrupts before external TTS and duration exceptions;
* a process lock so two coordinators cannot mutate one project concurrently.

The existing stage scripts remain the source of truth for each operation.
Commands are always executed as argument arrays with ``shell=False``.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Iterator, Sequence


SCHEMA_VERSION = "1.0"
GRAPH_VERSION = "1.2"
STAGES = (
    "research",
    "outline",
    "script",
    "shots",
    "images",
    "voiceover",
    "alignment",
    "assembly",
    "qa",
)
WAITING_EXIT = 3

WORKER_FILES = {
    "research": ("research_topic.py",),
    "outline": ("build_outline.py",),
    "script": ("write_script.py", "fact_check_script.py", "finalize_script.py"),
    "shots": ("plan_shots.py",),
    "images": ("generate_images.py",),
    "voiceover": ("generate_voiceover.py",),
    "alignment": ("align_audio.py",),
    "assembly": ("assemble_video.py",),
    "qa": ("qa_video.py",),
}


class PipelineError(RuntimeError):
    """A user-actionable pipeline coordination error."""


def load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        if not required:
            return {}
        raise PipelineError(f"Missing required file: {path}") from None
    except json.JSONDecodeError as exc:
        raise PipelineError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"Expected a JSON object in {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "missing": True}
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def transcript_digest(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise PipelineError(f"Missing transcript: {path}") from exc
    if not text:
        raise PipelineError(f"Transcript is empty: {path}")
    return sha256_text(text)


def audio_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as source:
            return source.getnframes() / source.getframerate()
    except (FileNotFoundError, wave.Error, EOFError) as exc:
        raise PipelineError(f"Could not read narration WAV: {path}") from exc


class Pipeline:
    def __init__(self, repository: Path, project: Path) -> None:
        self.repository = repository.resolve()
        self.project = project.resolve()
        if not self.project.is_dir():
            raise PipelineError(f"Project directory does not exist: {self.project}")
        try:
            self.project.relative_to((self.repository / "projects").resolve())
        except ValueError as exc:
            raise PipelineError("Project must be inside this repository's projects/") from exc
        self.input = self.project / "input"
        self.work = self.project / "work"
        self.state_path = self.work / "pipeline_state.json"
        self.final_video = self.repository / "final-videos" / f"{self.project.name}.mp4"
        self._hash_cache: dict[tuple[str, int, int, int], str] = {}
        self.topic = load_json(self.input / "topic.json")
        self.config = self._load_config()
        self.state = self._load_state()

    def _load_config(self) -> dict[str, Any]:
        overrides = load_json(self.input / "pipeline.json", required=False)
        profile = str(overrides.get("profile", self.topic.get("format", "long")))
        if profile not in {"short", "long"}:
            raise PipelineError("pipeline profile must be 'short' or 'long'")
        models = {
            "researcher": "gpt-5.6-terra",
            "outliner": "gpt-5.6-sol",
            "writer": "gpt-5.6-sol",
            "checker": "gpt-5.6-terra",
        }
        supplied_models = overrides.get("models", {})
        if supplied_models:
            if not isinstance(supplied_models, dict):
                raise PipelineError("pipeline models must be an object")
            models.update({key: str(value) for key, value in supplied_models.items()})
        if models["writer"] == models["checker"]:
            raise PipelineError("writer and checker models must be different")
        duration = self.topic.get("target_duration_seconds", {})
        if not isinstance(duration, dict):
            duration = {}
        duration = {**duration, **overrides.get("target_duration_seconds", {})}
        return {
            "profile": profile,
            "models": models,
            "target_words": int(overrides.get("target_words", self.topic.get("target_words", 1900))),
            "max_factcheck_passes": int(overrides.get("max_factcheck_passes", 8)),
            "duration_minimum": duration.get("minimum"),
            "duration_maximum": duration.get("maximum"),
            "fps": int(overrides.get("fps", 30)),
            "minimum_alignment_match": float(overrides.get("minimum_alignment_match", 0.80)),
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {
                "schema_version": SCHEMA_VERSION,
                "graph_version": GRAPH_VERSION,
                "project_id": self.project.name,
                "stages": {},
                "approvals": {},
                "history": [],
                "command_count": 0,
            }
        state = load_json(self.state_path)
        if state.get("project_id") != self.project.name:
            raise PipelineError("pipeline state belongs to a different project")
        if not isinstance(state.get("stages"), dict) or not isinstance(state.get("history"), list):
            raise PipelineError("pipeline state is malformed")
        state.setdefault("approvals", {})
        state.setdefault("command_count", 0)
        return state

    def save(self) -> None:
        self.state["updated_at"] = int(time.time())
        write_json_atomic(self.state_path, self.state)

    def digest(self, path: Path) -> str:
        """Hash an unchanged file once during this coordinator invocation."""
        path = path.resolve()
        stat = path.stat()
        key = (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns)
        cached = self._hash_cache.get(key)
        if cached is not None:
            return cached
        value = sha256_file(path)
        self._hash_cache[key] = value
        return value

    def fingerprint(self, path: Path) -> dict[str, Any]:
        path = path.resolve()
        if not path.is_file():
            return {"path": str(path), "missing": True}
        stat = path.stat()
        return {
            "path": str(path),
            "size": stat.st_size,
            "sha256": self.digest(path),
        }

    def transcript_digest(self, path: Path) -> str:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise PipelineError(f"Missing transcript: {path}") from exc
        if not text:
            raise PipelineError(f"Transcript is empty: {path}")
        return sha256_text(text)

    def worker_fingerprints(self, stage: str) -> list[dict[str, Any]]:
        # Use the normal fingerprint helper so lightweight test repositories
        # (and an incomplete checkout) produce a stable ``missing`` marker
        # instead of failing while merely inspecting a signature.
        return [
            self.fingerprint(self.repository / "scripts" / filename)
            for filename in WORKER_FILES[stage]
        ]

    def event(self, kind: str, **details: Any) -> None:
        self.state["history"].append({"at": int(time.time()), "event": kind, **details})
        self.save()

    @contextlib.contextmanager
    def exclusive_lock(self) -> Iterator[None]:
        self.work.mkdir(parents=True, exist_ok=True)
        lock_path = self.work / "pipeline.lock"
        with lock_path.open("a+", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PipelineError(
                    f"Another pipeline process is already running for {self.project.name}"
                ) from exc
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid={os.getpid()} started={int(time.time())}\n")
            handle.flush()
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def input_paths(self, stage: str) -> list[Path]:
        paths: dict[str, list[Path]] = {
            "research": [self.input / "topic.json"],
            "outline": [self.input / "topic.json", self.input / "research.json"],
            "script": [self.input / "topic.json", self.input / "research.json", self.input / "outline.json"],
            "shots": [self.input / "transcript.txt", self.input / "shot_style.json"],
            "images": [self.input / "shots.json", self.input / "shot_style.json"],
            "voiceover": [self.input / "transcript.txt", self.input / "voice_config.json"],
            "alignment": [self.input / "transcript.txt", self.input / "narration.wav"],
            "assembly": [self.input / "shots.json", self.input / "timestamps.json", self.input / "narration.wav"],
            "qa": [self.final_video, self.input / "shots.json", self.input / "timestamps.json"],
        }
        selected = list(paths[stage])
        if stage == "assembly" and (self.input / "shots.json").is_file():
            shots = load_json(self.input / "shots.json")
            for shot in shots.get("shots", []):
                if isinstance(shot, dict) and isinstance(shot.get("image_file"), str):
                    selected.append(self.input / "images" / shot["image_file"])
        return selected

    def signature(self, stage: str) -> str:
        relevant_config: dict[str, Any] = {}
        if stage == "research":
            relevant_config = {
                "researcher_model": self.config["models"]["researcher"],
            }
        elif stage == "outline":
            relevant_config = {
                "outliner_model": self.config["models"]["outliner"],
            }
        elif stage == "script":
            relevant_config = {
                "writer_model": self.config["models"]["writer"],
                "checker_model": self.config["models"]["checker"],
                "target_words": self.config["target_words"],
                "max_factcheck_passes": self.config["max_factcheck_passes"],
            }
        elif stage == "alignment":
            relevant_config = {"minimum_alignment_match": self.config["minimum_alignment_match"]}
        elif stage in {"assembly", "qa"}:
            relevant_config = {"fps": self.config["fps"]}
        return stable_hash(
            {
                "graph_version": GRAPH_VERSION,
                "stage": stage,
                "config": relevant_config,
                "worker_files": self.worker_fingerprints(stage),
                "inputs": [self.fingerprint(path) for path in self.input_paths(stage)],
            }
        )

    def output_paths(self, stage: str) -> list[Path]:
        outputs: dict[str, list[Path]] = {
            "research": [self.input / "research.json"],
            "outline": [self.input / "outline.json"],
            "script": [self.input / "transcript.txt", self.input / "finalization.json"],
            "shots": [self.input / "shots.json"],
            "images": [self.work / "image_generation_state.json"],
            "voiceover": [self.input / "narration.wav", self.work / "tts" / "voiceover.json"],
            "alignment": [self.input / "timestamps.json", self.work / "alignment" / "alignment.json"],
            "assembly": [self.final_video],
            "qa": [self.work / "qa" / "video_qa.json"],
        }
        selected = list(outputs[stage])
        if stage == "images" and (self.input / "shots.json").is_file():
            for shot in load_json(self.input / "shots.json").get("shots", []):
                if isinstance(shot, dict) and isinstance(shot.get("image_file"), str):
                    selected.append(self.input / "images" / shot["image_file"])
        return selected

    def artifact_valid(self, stage: str) -> bool:
        if not all(path.is_file() and path.stat().st_size > 0 for path in self.output_paths(stage)):
            return False
        try:
            if stage == "research":
                research = load_json(self.input / "research.json")
                coverage = research.get("coverage", {})
                return (
                    research.get("schema_version") == SCHEMA_VERSION
                    and isinstance(research.get("facts"), list)
                    and bool(research["facts"])
                    and isinstance(coverage, dict)
                    and coverage.get("outline_ready") is True
                )
            if stage == "outline":
                outline = load_json(self.input / "outline.json")
                units = outline.get("verses", outline.get("beats"))
                return (
                    outline.get("schema_version") == SCHEMA_VERSION
                    and isinstance(units, list)
                    and bool(units)
                    and isinstance(outline.get("fact_ids_used"), list)
                )
            if stage == "script":
                finalization = load_json(self.input / "finalization.json")
                return (
                    finalization.get("status") == "fact_checked"
                    and finalization.get("draft_sha256") == self.transcript_digest(self.input / "transcript.txt")
                    and finalization.get("writer_model") != finalization.get("checker_model")
                )
            if stage == "shots":
                shots = load_json(self.input / "shots.json")
                style_lock = shots.get("style_lock")
                raw_shots = shots.get("shots")
                if (
                    shots.get("schema_version") != SCHEMA_VERSION
                    or not isinstance(style_lock, str)
                    or not style_lock
                    or not isinstance(raw_shots, list)
                    or not raw_shots
                ):
                    return False
                beat_ids = [shot.get("beat_id") for shot in raw_shots if isinstance(shot, dict)]
                image_files = [shot.get("image_file") for shot in raw_shots if isinstance(shot, dict)]
                return (
                    len(beat_ids) == len(raw_shots)
                    and len(set(beat_ids)) == len(beat_ids)
                    and len(set(image_files)) == len(image_files)
                    and all(
                        isinstance(shot.get("image_prompt"), str)
                        and shot["image_prompt"].startswith(style_lock)
                        for shot in raw_shots
                    )
                )
            if stage == "images":
                image_state = load_json(self.work / "image_generation_state.json")
                shots = load_json(self.input / "shots.json")
                recorded = image_state.get("shots", {})
                return all(
                    isinstance(shot, dict)
                    and recorded.get(shot.get("beat_id"), {}).get("status") == "completed"
                    for shot in shots.get("shots", [])
                )
            if stage == "voiceover":
                metadata = load_json(self.work / "tts" / "voiceover.json")
                config = load_json(self.input / "voice_config.json")
                config_hash = sha256_text(json.dumps(config, sort_keys=True))
                return (
                    metadata.get("transcript_sha256") == self.transcript_digest(self.input / "transcript.txt")
                    and metadata.get("config_sha256") == config_hash
                    and audio_duration(self.input / "narration.wav") > 0
                )
            if stage == "alignment":
                timestamps = load_json(self.input / "timestamps.json")
                source = timestamps.get("source", {})
                return (
                    isinstance(source, dict)
                    and source.get("transcript_sha256") == self.transcript_digest(self.input / "transcript.txt")
                    and bool(timestamps.get("words"))
                )
            if stage == "qa":
                qa = load_json(self.work / "qa" / "video_qa.json")
                return qa.get("status") == "pass" and qa.get("video_sha256") == self.digest(self.final_video)
        except (PipelineError, KeyError, TypeError, ValueError):
            return False
        return True

    def stage_status(self, stage: str) -> str:
        record = self.state["stages"].get(stage)
        valid = self.artifact_valid(stage)
        if not record:
            return "untracked" if valid else "pending"
        if record.get("status") == "running":
            return "interrupted"
        if record.get("status") == "failed":
            return "failed"
        if record.get("status") != "complete" or not valid:
            return "stale"
        if record.get("input_signature") != self.signature(stage):
            return "stale"
        for output in record.get("outputs", []):
            path = Path(output.get("path", ""))
            if not path.is_file() or output.get("sha256") != self.digest(path):
                return "stale"
        return "complete"

    def record_complete(self, stage: str, *, adopted: bool = False) -> None:
        previous = self.state["stages"].get(stage, {})
        self.state["stages"][stage] = {
            **previous,
            "status": "complete",
            "input_signature": self.signature(stage),
            "outputs": [self.fingerprint(path) for path in self.output_paths(stage)],
            "finished_at": int(time.time()),
            "adopted_existing": adopted,
        }
        self.event("stage_complete", stage=stage, adopted_existing=adopted)

    def sync(self) -> None:
        """Adopt valid existing outputs so a legacy project can resume safely."""
        previous_version = self.state.get("graph_version")
        if previous_version != GRAPH_VERSION:
            self.state["stages"] = {}
            self.state["graph_version"] = GRAPH_VERSION
            self.event(
                "graph_migrated",
                previous_graph_version=previous_version,
                graph_version=GRAPH_VERSION,
            )
        for stage in STAGES:
            if self.stage_status(stage) == "untracked":
                self.record_complete(stage, adopted=True)

    def approval_subject(self, gate: str) -> tuple[str, str]:
        if gate == "tts":
            path = self.input / "transcript.txt"
            return self.transcript_digest(path), str(path)
        if gate == "duration":
            path = self.input / "narration.wav"
            if not path.is_file():
                raise PipelineError(f"Cannot approve duration before audio exists: {path}")
            return self.digest(path), str(path)
        raise PipelineError(f"Unknown approval gate: {gate}")

    def approve(self, gate: str, note: str | None) -> None:
        digest, subject = self.approval_subject(gate)
        self.state["approvals"][gate] = {
            "artifact_sha256": digest,
            "artifact": subject,
            "approved_at": int(time.time()),
            "note": note or "",
        }
        self.event("approval_granted", gate=gate, artifact_sha256=digest)

    def approved(self, gate: str) -> bool:
        try:
            digest, _ = self.approval_subject(gate)
        except PipelineError:
            return False
        approval = self.state["approvals"].get(gate, {})
        return approval.get("artifact_sha256") == digest

    def run_command(self, stage: str, command: Sequence[str], *, dry_run: bool) -> None:
        rendered = " ".join(json.dumps(part) for part in command)
        if dry_run:
            print(f"DRY {stage}: {rendered}")
            return
        attempt = int(self.state["stages"].get(stage, {}).get("attempts", 0)) + 1
        self.state["stages"][stage] = {
            "status": "running",
            "attempts": attempt,
            "started_at": int(time.time()),
            "input_signature": self.signature(stage),
            "command": list(command),
        }
        self.state["command_count"] += 1
        self.event("stage_started", stage=stage, attempt=attempt)
        print(f"RUN  {stage}: {rendered}", flush=True)
        started = time.monotonic()
        result = subprocess.run(list(command), cwd=self.repository, check=False)
        elapsed = round(time.monotonic() - started, 3)
        if result.returncode:
            self.state["stages"][stage].update(
                {"status": "failed", "exit_code": result.returncode, "elapsed_seconds": elapsed}
            )
            self.event("stage_failed", stage=stage, exit_code=result.returncode)
            raise PipelineError(f"{stage} failed with exit code {result.returncode}")
        if not self.artifact_valid(stage):
            self.state["stages"][stage].update(
                {"status": "failed", "exit_code": 0, "elapsed_seconds": elapsed}
            )
            self.event("stage_failed_validation", stage=stage)
            raise PipelineError(f"{stage} command exited successfully but its artifacts failed validation")
        self.record_complete(stage)
        self.state["stages"][stage]["elapsed_seconds"] = elapsed
        self.save()

    def prepare_script_run(self) -> str:
        """Start or resume only the script run for the current input signature.

        Draft and fact-check filenames are intentionally stable for the worker
        scripts. When their source packet changes, move the old active files to
        a recoverable history directory before starting at draft_001 again.
        """
        input_signature = self.signature("script")
        manifest_path = self.work / "script_loop.json"
        manifest = load_json(manifest_path, required=False)
        if manifest.get("input_signature") == input_signature:
            return input_signature

        active_dirs = [self.work / "writing", self.work / "factcheck"]
        existing = [path for path in active_dirs if path.exists()]
        archive_path: Path | None = None
        if existing:
            history_root = self.work / "script_history"
            history_root.mkdir(parents=True, exist_ok=True)
            label = str(manifest.get("input_signature", "legacy"))[:12]
            archive_path = history_root / f"{int(time.time())}-{label}"
            suffix = 2
            while archive_path.exists():
                archive_path = history_root / f"{int(time.time())}-{label}-{suffix}"
                suffix += 1
            archive_path.mkdir()
            for source in existing:
                shutil.move(str(source), str(archive_path / source.name))

        new_manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "input_signature": input_signature,
            "status": "running",
            "started_at": int(time.time()),
        }
        if archive_path is not None:
            new_manifest["archived_previous_run"] = str(archive_path.relative_to(self.project))
        write_json_atomic(manifest_path, new_manifest)
        self.event(
            "script_run_prepared",
            input_signature=input_signature,
            archived_previous_run=new_manifest.get("archived_previous_run"),
        )
        return input_signature

    def update_script_manifest(self, **updates: Any) -> None:
        manifest_path = self.work / "script_loop.json"
        manifest = load_json(manifest_path, required=False)
        manifest.update(updates)
        write_json_atomic(manifest_path, manifest)

    def script_loop(self, *, dry_run: bool) -> None:
        stage = "script"
        if self.stage_status(stage) == "complete":
            print("SKIP script: checkpoint and outputs are current", flush=True)
            return
        python = sys.executable
        scripts = self.repository / "scripts"
        models = self.config["models"]
        maximum = self.config["max_factcheck_passes"]
        if maximum <= 0:
            raise PipelineError("max_factcheck_passes must be greater than zero")

        if dry_run:
            command = [
                python, str(scripts / "write_script.py"), str(self.project),
                "--mode", "draft", "--target-words", str(self.config["target_words"]),
                "--writer-model", models["writer"], "--checker-model", models["checker"],
            ]
            self.run_command(stage, command, dry_run=True)
            print("DRY script: then alternate independent fact-check and revision until clean")
            return

        self.state["stages"][stage] = {
            "status": "running",
            "attempts": int(self.state["stages"].get(stage, {}).get("attempts", 0)) + 1,
            "started_at": int(time.time()),
            "input_signature": self.signature(stage),
        }
        self.event("stage_started", stage=stage, attempt=self.state["stages"][stage]["attempts"])
        self.prepare_script_run()
        for pass_number in range(1, maximum + 1):
            draft = self.work / "writing" / f"draft_{pass_number:03d}.txt"
            if not draft.is_file():
                if pass_number == 1:
                    command = [
                        python, str(scripts / "write_script.py"), str(self.project),
                        "--mode", "draft", "--target-words", str(self.config["target_words"]),
                        "--writer-model", models["writer"], "--checker-model", models["checker"],
                    ]
                else:
                    previous_draft = self.work / "writing" / f"draft_{pass_number - 1:03d}.txt"
                    previous_report = self.work / "factcheck" / f"pass_{pass_number - 1:03d}" / "report.json"
                    command = [
                        python, str(scripts / "write_script.py"), str(self.project),
                        "--mode", "revise", "--draft", str(previous_draft),
                        "--fact-report", str(previous_report), "--pass-number", str(pass_number - 1),
                        "--target-words", str(self.config["target_words"]),
                        "--writer-model", models["writer"], "--checker-model", models["checker"],
                    ]
                self._run_script_subcommand(command, f"writer_pass_{pass_number}")

            report = self.work / "factcheck" / f"pass_{pass_number:03d}" / "report.json"
            if not report.is_file():
                command = [
                    python, str(scripts / "fact_check_script.py"), str(self.project),
                    "--draft", str(draft), "--pass-number", str(pass_number),
                    "--checker-model", models["checker"], "--writer-model", models["writer"],
                ]
                self._run_script_subcommand(command, f"factcheck_pass_{pass_number}")
            report_data = load_json(report)
            if report_data.get("draft_sha256") != transcript_digest(draft):
                raise PipelineError(f"Fact-check pass {pass_number} does not match its draft hash")
            summary = report_data.get("summary", {})
            if summary.get("ready_for_final") is True:
                command = [
                    python, str(scripts / "finalize_script.py"), str(self.project),
                    "--draft", str(draft), "--fact-report", str(report),
                ]
                if (self.input / "transcript.txt").exists():
                    command.append("--overwrite")
                self._run_script_subcommand(command, "finalize")
                if not self.artifact_valid("script"):
                    raise PipelineError("Finalized script failed hash/model validation")
                self.update_script_manifest(
                    status="complete",
                    finalized_pass=pass_number,
                    transcript_sha256=self.transcript_digest(self.input / "transcript.txt"),
                )
                self.record_complete("script")
                return
            print(
                f"LOOP script: pass {pass_number} needs revision "
                f"(contradicted={summary.get('contradicted')}, unverified={summary.get('unverified')})"
            )
        self.state["stages"][stage]["status"] = "failed"
        self.event("factcheck_limit_reached", passes=maximum)
        raise PipelineError(f"Fact-check loop did not become clean after {maximum} passes")

    def _run_script_subcommand(self, command: Sequence[str], label: str) -> None:
        self.state["command_count"] += 1
        self.event("script_subcommand_started", label=label, command=list(command))
        result = subprocess.run(list(command), cwd=self.repository, check=False)
        if result.returncode:
            self.state["stages"]["script"]["status"] = "failed"
            self.event("script_subcommand_failed", label=label, exit_code=result.returncode)
            raise PipelineError(f"{label} failed with exit code {result.returncode}")
        self.event("script_subcommand_complete", label=label)

    def basic_command(self, stage: str) -> list[str]:
        python = sys.executable
        scripts = self.repository / "scripts"
        models = self.config["models"]
        commands = {
            "research": [python, str(scripts / "research_topic.py"), str(self.project), "--model", models["researcher"]],
            "outline": [python, str(scripts / "build_outline.py"), str(self.project), "--model", models["outliner"]],
            "shots": [python, str(scripts / "plan_shots.py"), str(self.project)],
            "images": [python, str(scripts / "generate_images.py"), str(self.project)],
            "voiceover": [python, str(scripts / "generate_voiceover.py"), str(self.project)],
            "alignment": [
                python, str(scripts / "align_audio.py"), str(self.project),
                "--minimum-match", str(self.config["minimum_alignment_match"]),
            ],
            "assembly": [
                python, str(scripts / "assemble_video.py"), str(self.project),
                "--fps", str(self.config["fps"]), "--overwrite",
            ],
            "qa": [python, str(scripts / "qa_video.py"), str(self.project)],
        }
        command = commands[stage]
        if stage in {"research", "outline"} and self.output_paths(stage)[0].exists():
            command.append("--overwrite")
        if stage == "shots" and (self.input / "shots.json").exists():
            image_dir = self.input / "images"
            if image_dir.is_dir() and any(image_dir.glob("*.png")):
                raise PipelineError(
                    "shots.json is stale but generated images exist; review prompt/style changes before replacing the shot plan"
                )
            command.append("--overwrite")
        if stage == "voiceover" and (self.input / "narration.wav").exists():
            command.append("--overwrite")
        return command

    def duration_outside_target(self) -> tuple[bool, float]:
        duration = audio_duration(self.input / "narration.wav")
        minimum = self.config["duration_minimum"]
        maximum = self.config["duration_maximum"]
        outside = (minimum is not None and duration < float(minimum)) or (
            maximum is not None and duration > float(maximum)
        )
        return outside, duration

    def run(self, through: str, *, dry_run: bool) -> int:
        if through not in STAGES:
            raise PipelineError(f"Unknown --through stage: {through}")
        if not dry_run:
            self.sync()
        for stage in STAGES[: STAGES.index(through) + 1]:
            if stage == "script":
                script_was_complete = self.stage_status(stage) == "complete"
                self.script_loop(dry_run=dry_run)
                if dry_run and not script_was_complete:
                    return 0
                continue
            status = self.stage_status(stage)
            if status == "complete":
                print(f"SKIP {stage}: checkpoint and outputs are current", flush=True)
                continue
            if stage == "voiceover" and not self.approved("tts"):
                digest, _ = self.approval_subject("tts")
                self.event("interrupt", gate="tts", artifact_sha256=digest)
                print("WAIT tts: external transmission is not approved for this exact transcript")
                print(f"Approve it with: {sys.executable} scripts/pipeline_graph.py approve {self.project} --gate tts")
                return WAITING_EXIT
            if stage == "alignment":
                outside, duration = self.duration_outside_target()
                if outside and not self.approved("duration"):
                    self.event("interrupt", gate="duration", duration_seconds=round(duration, 3))
                    print(
                        f"WAIT duration: native take is {duration:.2f}s, outside the configured "
                        f"{self.config['duration_minimum']}–{self.config['duration_maximum']}s target"
                    )
                    print(f"Approve this exact take with: {sys.executable} scripts/pipeline_graph.py approve {self.project} --gate duration")
                    return WAITING_EXIT
            command = self.basic_command(stage)
            self.run_command(stage, command, dry_run=dry_run)
            if dry_run:
                return 0
        return 0

    def status_rows(self) -> list[dict[str, Any]]:
        rows = []
        for stage in STAGES:
            record = self.state["stages"].get(stage, {})
            rows.append(
                {
                    "stage": stage,
                    "status": self.stage_status(stage),
                    "attempts": record.get("attempts", 0),
                    "elapsed_seconds": record.get("elapsed_seconds"),
                }
            )
        return rows


def resolve_project(repository: Path, value: Path) -> Path:
    candidate = value.expanduser()
    if not candidate.is_absolute():
        direct = (Path.cwd() / candidate).resolve()
        candidate = direct if direct.is_dir() else repository / "projects" / candidate
    return candidate.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("status", "sync", "run", "history"):
        child = subparsers.add_parser(action)
        child.add_argument("project", type=Path)
        if action == "status":
            child.add_argument("--json", action="store_true")
        if action == "run":
            child.add_argument("--through", choices=STAGES, default="qa")
            child.add_argument("--dry-run", action="store_true")
    approve = subparsers.add_parser("approve")
    approve.add_argument("project", type=Path)
    approve.add_argument("--gate", required=True, choices=("tts", "duration"))
    approve.add_argument("--note")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    try:
        pipeline = Pipeline(repository, resolve_project(repository, args.project))
        if args.action == "status":
            rows = pipeline.status_rows()
            if args.json:
                print(json.dumps({"project": pipeline.project.name, "stages": rows}, indent=2))
            else:
                for row in rows:
                    elapsed = "" if row["elapsed_seconds"] is None else f" {row['elapsed_seconds']:.1f}s"
                    print(f"{row['stage']:<10} {row['status']:<11} attempts={row['attempts']}{elapsed}")
                print(f"commands executed by coordinator: {pipeline.state['command_count']}")
            return 0
        if args.action == "history":
            print(json.dumps(pipeline.state["history"], indent=2))
            return 0
        with pipeline.exclusive_lock():
            if args.action == "sync":
                pipeline.sync()
                print(f"Synchronized valid existing artifacts for {pipeline.project.name}")
                return 0
            if args.action == "approve":
                pipeline.approve(args.gate, args.note)
                print(f"Approved {args.gate} for the current immutable artifact hash")
                return 0
            if args.action == "run":
                return pipeline.run(args.through, dry_run=args.dry_run)
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
