# Workflow efficiency audit

Audit date: 2026-08-16

Representative project: `projects/recorded-voice-weird` (9 portrait images,
71.28-second final video, 2,138 encoded frames).

## Findings

| Area | Before | Current behavior | Result |
|---|---|---|---|
| Production control | Individual commands and manually maintained manifests | One durable coordinator with atomic stage checkpoints | Resume from the first stale node; no repeated completed stages |
| Duplicate work | Two terminals could launch the same project | Nonblocking OS lock per project | A second coordinator fails before starting side effects |
| Freshness | Existence checks and several stage-local hashes | Direct-input signatures plus output hashes for every graph node | Upstream edits invalidate only affected downstream work |
| Worker freshness | Stage signatures omitted the worker source files | Each stage signature includes the SHA-256 fingerprint of its worker script(s) | Editing a worker cannot silently reuse its old checkpoint |
| Script revision | Manually relayed writer/checker commands | Sequential evaluator/optimizer loop with separate model identities | No checking a moving draft; exact clean draft is finalized |
| Stale script drafts | Stable `draft_001`/fact-check paths could be reused after input changes | Current script signature is recorded; stale active drafts move to `work/script_history/` | A new source packet starts from a clean draft while old work remains recoverable |
| TTS approval | Human approval described outside execution state | Approval bound to the exact transcript SHA-256 | Script edits automatically revoke approval |
| Duration exception | Informal/manual | Approval bound to exact WAV SHA-256 | A replacement take must be reviewed again |
| TTS replacement | Previous native WAV could be overwritten | Previous WAV, segments, and metadata archived before replacement | Expensive provider output remains recoverable |
| Portrait assembly | Width/height depended on caller flags | Dimensions inferred from `shots.json` aspect ratio | Prevents accidental landscape render and rerender |
| Static QA | One ffmpeg process per sampled frame in older helpers | One full-stream-validated ffmpeg sample decode | About 45 media processes become 2 total (ffprobe + ffmpeg) |
| QA frame memory | 43 full 1080x1920 gray frames, about 85 MiB | 43 aspect-correct 270x480 gray frames, about 5.3 MiB | 16x smaller sample payload |
| Dynamic-motion QA | Moving projects received decode-only validation | One batched 160px grayscale sample pass reports isolated temporal spikes for review | Smoothness signals are visible without risking false render failures |
| Repeated hashing | Status/signature/output checks could reread the same files in one invocation | In-memory `(path, inode, size, mtime)` hash memoization | The representative status pass hashed 35 times / 206 MB instead of 62 times / 605 MB |
| Dry-run preflight | Image and shot dry-runs checked OAuth and the Codex binary first | Local input validation and preview happen before external preflight | Planning/preview works offline and avoids unnecessary auth/CLI checks |
| Shell safety | Direct subprocess argument lists | Preserved; no shell strings | No `shell=True`, `os.system`, or polling sleeps |

## Measured QA benchmark

Both runs used the same final MP4 and the same 43 aligned early/mid/late and
cut-adjacent frame indices.

| Metric | Initial batched implementation | Final implementation | Change |
|---|---:|---:|---:|
| Wall time | 13.03 s | 8.03 s | 38.4% lower |
| User CPU | 47.68 s | 23.64 s | 50.4% lower |
| System CPU | 1.06 s | 1.08 s | effectively unchanged |
| Video decode passes | 2 | 1 | 50% fewer |
| Sample payload | about 85 MiB | about 5.3 MiB | 16x smaller |

Acceptance evidence from the final run:

- full-stream decode: pass (`-xerror`);
- dimensions: 1080x1920;
- frame rate: 30/1;
- audio/timeline duration: 71.28 seconds;
- narration match: 98.9474%;
- static-hold maximum MAE: 0.0;
- static-hold changed pixels above 1 level: 0.0%;
- minimum cut MAE: 58.969645;
- final status: pass.

The post-change coordinator no-op benchmark ran three times at 0.112–0.117 s,
with nine stage skips and zero stage subprocesses on every run. The existing
MP4 was not reassembled: its SHA-256 remains
`ed2b0d141e6fe083499e5b251dde5329344c3b81aa43f414c1daa3abf0c4753c`.

The moving `full-smoke` reference also passed its new diagnostic: 24 dynamic
shots, 1,870 sampled frames, and zero isolated spikes requiring review. This is
intentionally advisory; the existing full-stream decode remains the hard QA
gate because codec texture can resemble a one-frame hitch.

## Why image generation remains serial

One `codex exec` process per image is the slowest shell-heavy stage, but blind
parallelism is not automatically an improvement. Each call owns one output,
one prompt hash, one log, and one resumable checkpoint, while provider-side
image work is the dominant cost. Serial execution also avoids concurrent Codex
sessions competing for local resources and makes a bad style visible before a
whole batch is consumed. Valid images are skipped on resume.

If measured image throughput later shows that local concurrency helps, add a
small opt-in worker count (start with two), preserve per-shot atomic state, and
benchmark it on a disposable shot list before changing the default.

## Operational commands

```bash
.venv/bin/python scripts/pipeline_graph.py sync projects/<project-name>
.venv/bin/python scripts/pipeline_graph.py status projects/<project-name>
.venv/bin/python scripts/pipeline_graph.py run projects/<project-name>
```

The final command safely resumes after a failure or approval pause. `status`
does not launch provider, model, Whisper, or ffmpeg work.
