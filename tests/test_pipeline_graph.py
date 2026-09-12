import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pipeline_graph


def make_pipeline(tmp_path: Path) -> pipeline_graph.Pipeline:
    repository = tmp_path / "youtube-pipeline"
    project = repository / "projects" / "voice-test"
    input_dir = project / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "topic.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "project_id": "voice-test",
                "format": "short",
                "target_words": 180,
                "target_duration_seconds": {"minimum": 60, "maximum": 65},
            }
        ),
        encoding="utf-8",
    )
    return pipeline_graph.Pipeline(repository, project)


def test_tts_approval_is_bound_to_exact_transcript(tmp_path):
    pipeline = make_pipeline(tmp_path)
    transcript = pipeline.input / "transcript.txt"
    transcript.write_text("First approved narration.\n", encoding="utf-8")

    pipeline.approve("tts", "approved for provider transmission")

    assert pipeline.approved("tts") is True
    transcript.write_text("Changed narration.\n", encoding="utf-8")
    assert pipeline.approved("tts") is False


def test_duration_approval_is_bound_to_exact_audio_bytes(tmp_path):
    pipeline = make_pipeline(tmp_path)
    narration = pipeline.input / "narration.wav"
    narration.write_bytes(b"take one")

    pipeline.approve("duration", None)

    assert pipeline.approved("duration") is True
    narration.write_bytes(b"take two")
    assert pipeline.approved("duration") is False


def test_assembly_signature_includes_every_planned_image(tmp_path):
    pipeline = make_pipeline(tmp_path)
    images = pipeline.input / "images"
    images.mkdir()
    (images / "shot_001.png").write_bytes(b"image one")
    (pipeline.input / "shots.json").write_text(
        json.dumps({"shots": [{"image_file": "shot_001.png"}]}), encoding="utf-8"
    )
    (pipeline.input / "timestamps.json").write_text("{}", encoding="utf-8")
    (pipeline.input / "narration.wav").write_bytes(b"audio")
    before = pipeline.signature("assembly")

    (images / "shot_001.png").write_bytes(b"image one revised")

    assert pipeline.signature("assembly") != before


def test_model_change_invalidates_only_the_relevant_language_stage(tmp_path):
    pipeline = make_pipeline(tmp_path)
    research_before = pipeline.signature("research")
    script_before = pipeline.signature("script")

    pipeline.config["models"]["writer"] = "different-writer"

    assert pipeline.signature("research") == research_before
    assert pipeline.signature("script") != script_before


def test_stale_script_run_is_archived_before_reuse(tmp_path):
    pipeline = make_pipeline(tmp_path)
    writing = pipeline.work / "writing"
    factcheck = pipeline.work / "factcheck" / "pass_001"
    writing.mkdir(parents=True)
    factcheck.mkdir(parents=True)
    (writing / "draft_001.txt").write_text("old draft", encoding="utf-8")
    (writing / "draft_001.json").write_text("{}", encoding="utf-8")
    (factcheck / "report.json").write_text("{}", encoding="utf-8")
    (pipeline.work / "script_loop.json").write_text(
        json.dumps({"schema_version": "1.0", "input_signature": "old-signature"}),
        encoding="utf-8",
    )

    current_signature = pipeline.signature("script")
    pipeline.prepare_script_run()

    archives = list((pipeline.work / "script_history").glob("*"))
    assert len(archives) == 1
    assert (archives[0] / "writing" / "draft_001.txt").read_text(encoding="utf-8") == "old draft"
    assert not (writing / "draft_001.txt").exists()
    manifest = json.loads((pipeline.work / "script_loop.json").read_text(encoding="utf-8"))
    assert manifest["input_signature"] == current_signature


def test_script_run_with_current_signature_is_reused(tmp_path):
    pipeline = make_pipeline(tmp_path)
    current_signature = pipeline.signature("script")
    manifest_path = pipeline.work / "script_loop.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"schema_version": "1.0", "input_signature": current_signature}),
        encoding="utf-8",
    )
    draft = pipeline.work / "writing" / "draft_001.txt"
    draft.parent.mkdir(parents=True)
    draft.write_text("current draft", encoding="utf-8")

    pipeline.prepare_script_run()

    assert draft.exists()
    assert not (pipeline.work / "script_history").exists()


def test_status_hashes_each_path_once_per_pipeline_instance(tmp_path, monkeypatch):
    pipeline = make_pipeline(tmp_path)
    research = pipeline.input / "research.json"
    research.write_text(
        '{"schema_version":"1.0","facts":["one"],"coverage":{"outline_ready":true}}',
        encoding="utf-8",
    )
    pipeline.record_complete("research", adopted=True)
    pipeline._hash_cache.clear()
    calls = {}
    original = pipeline_graph.sha256_file

    def counted(path):
        calls[path] = calls.get(path, 0) + 1
        return original(path)

    monkeypatch.setattr(pipeline_graph, "sha256_file", counted)
    pipeline.status_rows()

    assert calls
    assert max(calls.values()) == 1


def test_sync_adopts_valid_existing_outputs_without_running_commands(tmp_path):
    pipeline = make_pipeline(tmp_path)
    (pipeline.input / "research.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "facts": [{"fact_id": "fact_001"}],
                "coverage": {"outline_ready": True},
            }
        ),
        encoding="utf-8",
    )
    (pipeline.input / "outline.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "verses": [{"verse_id": "verse_001"}],
                "fact_ids_used": ["fact_001"],
            }
        ),
        encoding="utf-8",
    )

    pipeline.sync()

    assert pipeline.stage_status("research") == "complete"
    assert pipeline.stage_status("outline") == "complete"
    assert pipeline.state["command_count"] == 0


def test_sync_migrates_old_graph_version_before_adopting_outputs(tmp_path):
    pipeline = make_pipeline(tmp_path)
    research = pipeline.input / "research.json"
    research.write_text(
        '{"schema_version":"1.0","facts":[{"fact_id":"fact_001"}],'
        '"coverage":{"outline_ready":true}}',
        encoding="utf-8",
    )
    pipeline.state["graph_version"] = "1.1"
    pipeline.state["stages"] = {
        "research": {"status": "complete", "input_signature": "old"}
    }

    pipeline.sync()

    assert pipeline.state["graph_version"] == pipeline_graph.GRAPH_VERSION
    assert pipeline.stage_status("research") == "complete"
    assert pipeline.state["command_count"] == 0


def test_content_change_marks_checkpoint_stale(tmp_path):
    pipeline = make_pipeline(tmp_path)
    research = pipeline.input / "research.json"
    research.write_text(
        '{"schema_version":"1.0","facts":["one"],"coverage":{"outline_ready":true}}',
        encoding="utf-8",
    )
    pipeline.record_complete("research", adopted=True)
    assert pipeline.stage_status("research") == "complete"

    (pipeline.input / "topic.json").write_text(
        '{"schema_version":"1.0","project_id":"voice-test","format":"short","topic":"changed"}',
        encoding="utf-8",
    )

    assert pipeline.stage_status("research") == "stale"


def test_completion_keeps_attempt_and_command_metadata(tmp_path):
    pipeline = make_pipeline(tmp_path)
    (pipeline.input / "research.json").write_text(
        '{"schema_version":"1.0","facts":["one"],"coverage":{"outline_ready":true}}',
        encoding="utf-8",
    )
    pipeline.state["stages"]["research"] = {
        "status": "running",
        "attempts": 2,
        "command": ["python", "research_topic.py"],
    }

    pipeline.record_complete("research")

    record = pipeline.state["stages"]["research"]
    assert record["attempts"] == 2
    assert record["command"] == ["python", "research_topic.py"]


def test_basic_commands_are_argument_lists_without_shell_wrappers(tmp_path):
    pipeline = make_pipeline(tmp_path)

    for stage in ("research", "outline", "shots", "images", "voiceover", "alignment", "assembly", "qa"):
        command = pipeline.basic_command(stage)
        assert isinstance(command, list)
        assert command[0] == sys.executable
        assert all(isinstance(part, str) for part in command)


def test_project_lock_rejects_a_duplicate_coordinator(tmp_path):
    first = make_pipeline(tmp_path)
    second = pipeline_graph.Pipeline(first.repository, first.project)

    with first.exclusive_lock():
        with pytest.raises(pipeline_graph.PipelineError, match="already running"):
            with second.exclusive_lock():
                pass
