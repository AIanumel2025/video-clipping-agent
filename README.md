# Incremental Video Transcript Alignment & Clip Extraction

An agent that takes a video and its transcript and produces correctly-timed
clips. On a re-run with an edited transcript, it diffs old vs. new, works
out what changed, and only re-processes the segments that need
it — reusing cached alignment, embeddings, and clip files everywhere else.

The agent solves the problem of re-cutting an entire video from scratch every
time a transcript gets a single wording fix doesn't scale. This agent
detects what changes and reprocesses only the affected segments.

## My Framework

Two pipelines, both callable as a single function from `agent/pipeline.py`:

- **`run_baseline_pipeline()`** (Stage 1) — ingest → locate flagged
  segments → align (whisperx forced alignment) → attribute word timings
  back onto segments → cut clips (ffmpeg) → export `manifest.json` +
  `state.json`. Every run realigns everything from scratch — this is the
  "dumb but correct" first pass.
- **`run_incremental_pipeline()`** (Stage 2) — diff the incoming
  transcript against the last committed run → embed changed segments to
  tell a rewording from an unrelated rewrite → decide an action per
  segment → execute those decisions (propagating timing shifts downstream)
  → log the reasoning → reuse, re-cut, or retire clips accordingly →
  commit the result so the *next* run diffs against this one.

```python
from agent.pipeline import run_baseline_pipeline, run_incremental_pipeline

# First run
run_baseline_pipeline("https://youtube.com/watch?v=YOUR_VIDEO_ID")

# Later runs, after editing transcript_flagged_new.json
run_incremental_pipeline("https://youtube.com/watch?v=YOUR_VIDEO_ID")
```

## Installation

```bash
git clone https://github.com/AIanumel2025/video-transcript-clip-extraction
cd video-transcript-clip-extraction
pip install -r requirements.txt        # to run the pipeline
pip install -r requirements-dev.txt    # to also run the test suite
```

`ffmpeg` and `ffprobe` are required and are **not** pip-installable —
`apt-get install ffmpeg` (for Debian/Ubuntu), `brew install ffmpeg` (for macOS),
or already present if you're running in Colab.

## Input

Video can be a local file path or a YouTube URL (downloaded via `yt-dlp`
on first use, cached locally after that). By default the pipeline looks
in `./input/` for:

- **`transcript.json`** — the raw transcript
- **`transcript_flagged.json`** — the same segments, hand-curated with
  `make_clip` flags and a `clip_title` on the ones worth cutting
- **`transcript_flagged_new.json`** — for incremental runs, the edited
  version to diff against the last commit

Each segment needs at minimum:

```json
{
  "video_id": "your_video_id",
  "segments": [
    {
      "id": "seg_001",
      "speaker": "A",
      "text": "This is what was said.",
      "make_clip": true,
      "clip_title": "A short, human-readable title"
    }
  ]
}
```

`clip_title` is optional on non-clip segments; a segment flagged
`make_clip: true` without one gets a warning, not a hard failure.

Paths are configurable via environment variables — nothing is hardcoded
to a personal machine or Drive folder:

| Variable | Purpose | Default |
|---|---|---|
| `VIDEO_AGENT_INPUT_DIR` | Where the transcript files and video live | `./input` |
| `VIDEO_AGENT_OUTPUT_DIR` | Where `manifest.json`, `state.json`, clips, and `decisions.log` get written | `./output` |
| `VIDEO_AGENT_USE_DRIVE` | Set to `1` to mount Google Drive instead (Colab only) | unset (off) |
| `VIDEO_AGENT_DRIVE_ROOT` | Drive path to use when the above is set | `/content/drive/MyDrive` |

## The decision engine

Every segment in an incremental run gets one action, logged with a
plain-language reason in `decisions.log`:

| Action | Meaning |
|---|---|
| `REUSE` | Text unchanged, nothing upstream changed either — old timing trusted as-is |
| `VERIFY_SHIFT` | Text unchanged, but occurs after an earlier edit/insert/delete — timing needs re-checking |
| `PATCH` | Minor reword (high embedding similarity) — realign locally, not the whole file |
| `FULL_REALIGN` | Substantial rewrite (low similarity) — too different to trust a local patch |
| `NEW` | Newly inserted segment — no prior alignment exists |
| `REMOVE` | Segment no longer exists in the new transcript |

Each segment decision then drives a clip-level decision:

| Clip action | Meaning |
|---|---|
| `REUSE_CLIP` | Text and timing both unchanged — clip file kept untouched |
| `RECUT` | Timing moved beyond a small float-rounding tolerance — re-cut with the new boundaries |
| `CUT_NEW` | Newly flagged for a clip — no prior file exists |
| `RETIRE_CLIP` | Source segment deleted or un-flagged — old file moved to `clips/_retired/` |

## What's real vs. mocked — read this before trusting the numbers

- **Real:** Stage 1's forced alignment (whisperx against real audio),
  ffmpeg cutting, the diff engine (identity-based `SequenceMatcher`
  matching, not positional), and embedding similarity
  (`sentence-transformers`, disk-cached).
- **Mocked:** in Stage 2, the duration estimate for `PATCH`,
  `FULL_REALIGN`, and `NEW` segments is a words-per-second rate derived
  from Stage 1's real alignment — not an actual windowed re-alignment
  call. Every reason string this produces is labeled `MOCKED` explicitly
  in `decisions.log` so this is never silently mistaken for a verified
  timestamp. Swapping in a real windowed whisperx call here is the
  single largest piece of unfinished work in this project.

## Repo structure

```
agent/
  config.py       path resolution, device detection
  ingest.py       transcript loading/validation, video resolution
  locate.py       select segments flagged for clipping
  align.py        audio extraction, ASR caching, forced alignment
  attribute.py    word-to-segment re-attribution after alignment
  verify.py       interactive spot-check playback (Colab/Jupyter only)
  cutter.py       ffmpeg clip cutting
  manifest.py     manifest.json writer
  state.py        hashing, state.json persistence, sanity checks
  diff.py         Stage 2: classify segments vs. the last commit
  embeddings.py   disk-cached sentence embeddings
  decide.py       the decision engine
  execute.py      executes decisions, propagates timing deltas
  logger.py       decisions.log writer
  clip_reuse.py   decides which clip files are still valid
  commit.py       executes clip decisions, persists the result
  pipeline.py     orchestrates both pipelines end to end
scripts/
  generate_fixtures.py   synthetic test fixture for the smoke test
tests/
  ...             see Testing below
```

Generated artifacts (`input/`, `output/`, caches, clip files) are not
committed — see `.gitignore`. No `.mp4` files live in this repo by
design; a walkthrough video will be linked here once recorded.

## Testing

Two tiers, both run with plain `pytest` from the repo root:

- **Tier 1** (68 tests) — every pure-logic module tested in isolation
  with hand-built fixtures. No models, no audio, no network — runs in
  about 2 seconds anywhere, including CI.
- **Tier 2** (1 test) — a real end-to-end run of the full pipeline
  against a tiny synthetic fixture generated fresh by
  `scripts/generate_fixtures.py` (no committed binary files). Exercises
  real `ffmpeg` cutting and the real decision engine; only the two
  whisperx-dependent functions and the embedding model's loading are
  stood in for, using known injected values instead of a real model call.

```bash
pytest                  # everything
pytest -m "not smoke"   # Tier 1 only, for quick iteration
```

## Known limitations

- Stage 2's `PATCH`/`FULL_REALIGN`/`NEW` timing is mocked (see above) —
  not yet backed by real windowed re-alignment.
- Dependency versions in `requirements.txt` are not pinned to exact
  numbers yet — pin them from `pip freeze` in a known-working environment
  for full reproducibility.