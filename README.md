# YouTube Image and Video Pipeline

A local, resumable Python pipeline for producing source-backed YouTube videos from a topic through research, writing, fact-checking, images, narration, assembly, and media QA.

[![CI](https://github.com/realjeffreyau/gpt-image-video-pipeline-for-yt/actions/workflows/ci.yml/badge.svg)](https://github.com/realjeffreyau/gpt-image-video-pipeline-for-yt/actions/workflows/ci.yml)
[![CodeQL](https://github.com/realjeffreyau/gpt-image-video-pipeline-for-yt/actions/workflows/codeql.yml/badge.svg)](https://github.com/realjeffreyau/gpt-image-video-pipeline-for-yt/actions/workflows/codeql.yml)

The pipeline is designed for a creator's own computer. It keeps project state, generated media, model prompts, approvals, and QA reports together so a stopped run can resume without silently reusing stale work.

## What it does

- Builds a source-backed research packet through the Codex CLI.
- Maps research onto an eight-beat outline and a conversational narration draft.
- Runs an independent fact-checking pass before a draft can be finalized.
- Generates one style-locked image per narration beat through ChatGPT-authenticated Codex.
- Generates resumable Gemini TTS audio in chunks.
- Aligns narration locally with faster-whisper.
- Renders landscape or portrait video with deterministic Ken Burns motion.
- Runs decoded-frame media QA, static-hold checks, and advisory motion diagnostics.
- Tracks stage input signatures, worker fingerprints, output hashes, approvals, and history in a durable local checkpoint.

## Pipeline

~~~text
research -> outline -> writer <-> fact-checker -> shots -> images
  -> [approve exact transcript] -> voiceover
  -> [approve duration exception when needed] -> alignment -> assembly -> QA
~~~

The coordinator is inspired by useful state-machine ideas from LangGraph, but it does not require LangGraph or a hosted orchestration service.

## Requirements

- macOS or Linux.
- Python 3.10 or newer.
- `ffmpeg` and `ffprobe` on `PATH` for rendering and final QA. The runtime dependency can provide platform binaries when system binaries are unavailable.
- The Codex CLI, signed in with ChatGPT OAuth, for research, outline, writing, fact-checking, shot planning, and image generation.
- A Google AI Studio API key only for the optional Gemini narration stage.
- Optional: the local faster-whisper dependency for word-level audio alignment.

The repository does not run a hosted server, expose an API, create an account system, or upload project data to a pipeline service.

## Install from GitHub

~~~bash
git clone https://github.com/realjeffreyau/gpt-image-video-pipeline-for-yt.git
cd gpt-image-video-pipeline-for-yt
./setup.sh --dev
~~~

`setup.sh` creates `.venv/`, installs the runtime requirements, and optionally installs the test tools. Add `--alignment` when you want faster-whisper installed at setup time:

~~~bash
./setup.sh --dev --alignment
~~~

The script never asks for or stores a provider key.

## Quick start

Copy the portable configuration seed into a new project directory:

~~~bash
cp -R examples/short-form projects/my-video
~~~

Sign in to Codex with ChatGPT OAuth. The pipeline checks the local auth mode before any model stage:

~~~bash
codex login
~~~

Inspect the new project without launching a provider or media process:

~~~bash
.venv/bin/python scripts/pipeline_graph.py status projects/my-video
.venv/bin/python scripts/pipeline_graph.py run projects/my-video --dry-run
~~~

Run the complete workflow when the project inputs and provider access are ready:

~~~bash
.venv/bin/python scripts/pipeline_graph.py run projects/my-video
~~~

The coordinator returns exit code `3` at a human approval gate and prints the exact command needed to continue. Approvals are bound to the current transcript or audio hash.

## Provider setup

### Codex CLI

The Codex stages use the local ChatGPT OAuth session at `~/.codex/auth.json`. They reject an auth file whose `auth_mode` is not `chatgpt`. No OpenAI API key is required by this project.

Model names are configurable in `input/pipeline.json` and the individual script flags. The example uses the repository defaults; change them to models available in your Codex account.

### Gemini TTS

Copy the example environment file into the repository root and add your Google AI Studio key:

~~~bash
cp .env.example .env
~~~

The voiceover script reads only `GEMINI_API_KEY`, preserves an already-exported environment value, and never writes the key into project JSON. `.env` is ignored by Git and must remain private.

Run a no-network preflight first:

~~~bash
.venv/bin/python scripts/generate_voiceover.py projects/my-video --dry-run
~~~

### Local alignment

Install the optional dependency when alignment is needed:

~~~bash
.venv/bin/python -m pip install -r requirements-alignment.txt
.venv/bin/python scripts/align_audio.py projects/my-video --model small.en
~~~

The first faster-whisper run may download the selected model. The audio and transcript stay local to this stage.

## Project contract

Each project lives under `projects/<project-name>/`:

~~~text
projects/<project-name>/
├── input/       # topic, transcript, shot plan, audio, images, timestamps
├── work/        # checkpoints, provider responses, logs, QA reports
└── output/      # temporary project-local output when a worker needs it
~~~

The final MP4 defaults to `final-videos/<project-name>.mp4` inside the cloned repository. Pass `--output` to `assemble_video.py` for a deliberate one-off destination.

Project names become output filenames. Use lowercase, hyphen-separated names with two to four words and avoid dates, versions, `final`, or special characters.

### Core JSON inputs

`input/topic.json` defines the topic, audience, format, and target length. `input/shot_style.json` defines the exact style-lock prefix used by every image prompt. `input/voice_config.json` controls Gemini TTS. The scripts validate their own JSON contracts before launching external work.

`shots.json` contains one image and narration grouping per beat. `timestamps.json` contains contiguous, zero-based word timestamps. The assembler derives cut boundaries from aligned words rather than trusting advisory hold durations.

### Standalone commands

The workers can be run independently when a staged workflow is more useful:

~~~bash
.venv/bin/python scripts/research_topic.py projects/my-video
.venv/bin/python scripts/build_outline.py projects/my-video
.venv/bin/python scripts/write_script.py projects/my-video --mode draft --target-words 185
.venv/bin/python scripts/fact_check_script.py projects/my-video
.venv/bin/python scripts/finalize_script.py projects/my-video --draft <draft-path> --fact-report <report-path>
.venv/bin/python scripts/plan_shots.py projects/my-video
.venv/bin/python scripts/generate_images.py projects/my-video
.venv/bin/python scripts/generate_voiceover.py projects/my-video
.venv/bin/python scripts/align_audio.py projects/my-video
.venv/bin/python scripts/assemble_video.py projects/my-video
.venv/bin/python scripts/qa_video.py projects/my-video
~~~

Use `--dry-run` on image planning, image generation, and voiceover where available to validate local inputs without provider calls.

## Safety and privacy boundaries

- Never commit `.env`, provider keys, Codex auth files, raw provider logs, generated audio, generated images, or finished videos.
- Treat topic text, research text, narration, and image descriptions as untrusted content. The image worker explicitly confines visual descriptions and asks the image tool to write one expected PNG only.
- Fact-checking enforces evidence and blocks contradicted or unverified claims, but it cannot replace human review of sources, wording, copyright, or suitability for publication.
- Generated images, voices, and final edits still require human review. Automated QA checks media integrity and timing; it does not certify creative quality.
- Provider terms, model availability, source licenses, music rights, voice rights, and publication decisions remain the operator's responsibility.

## Tests and checks

Run the local test suite with:

~~~bash
.venv/bin/python -m pytest -q
~~~

The GitHub workflows also run the test suite, Python compilation checks, dependency auditing, Bandit checks, CodeQL, and dependency review. Workflow permissions are kept read-only except for the security-events permission required by CodeQL.

## Repository security

The repository includes:

- least-privilege GitHub Actions permissions;
- CodeQL analysis for Python;
- dependency review on pull requests;
- Dependabot updates for Python and GitHub Actions;
- secret and path exclusions in `.gitignore`;
- a private vulnerability-reporting process in `SECURITY.md`.

Repository administrators should also enable GitHub secret scanning, push protection, Dependabot alerts, automated security fixes, and branch protection for `main` with the CI and CodeQL checks required before merging.

## Documentation

- `docs/workflow-efficiency-audit.md` records the coordinator and media-QA efficiency decisions.
- `examples/short-form/` is a clean configuration seed with no generated media or credentials.
- `CONTRIBUTING.md` describes the review and validation expectations.

## License

MIT. See `LICENSE`.
