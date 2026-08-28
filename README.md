# Incremental Video Transcript Alignment & Clip Extraction

An agent that takes a video and its transcript and produces correctly-timed
clips. On a re-run with an edited transcript, it diffs old vs. new, works
out what actually changed, and only re-processes the segments that need
it — reusing cached alignment, embeddings, and clip files everywhere else.

The problem this solves: re-cutting an entire video from scratch every
time a transcript gets a single wording fix doesn't scale. This agent
detects what changed and reprocesses only the affected segments.

## How it's organized

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
git clone https://github.com/AIanumel2025/video-clipping-agent
cd video-clipping-agent
pip install -r requirements.txt        # to run the pipeline
pip install -r requirements-dev.txt    # to also run the test suite
```

`ffmpeg` and `ffprobe` are required and are **not** pip-installable —
`apt-get install ffmpeg` (Debian/Ubuntu), `brew install ffmpeg` (macOS),
or already present if you're running this in Colab.

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

## Verified against real content

Both pipelines have been run end to end against a real ~60-minute
interview (140 segments, 16 flagged clips), on real hardware (Colab,
GPU) — not just the test suite's synthetic fixture.

**Run 1 (baseline):** produced 16 correctly-titled clips, matching what
the original, pre-refactor notebook had already validated — same
segments, same titles, same count.

**Run 2 (incremental),** against a real edited transcript (one segment
reworded, one deleted, one inserted): correctly classified all three
changes (`{'unchanged': 138, 'edited': 1, 'deleted': 1, 'inserted': 1}`),
computed a real embedding similarity (0.880) that correctly routed the
rework to `PATCH`, propagated a compounding -13.89s delta across 124
downstream `VERIFY_SHIFT` segments, correctly left the 2 untouched clips
alone and re-cut the 14 that had actually moved — confirmed against the
running cumulative delta, and by spot-listening to the affected clips
(`agent/verify.py`'s `spot_check_clips()`).

Notebooks for both runs live in `notebooks/` (see below).

## Repo structure

```
agent/
  config.py       path resolution, device detection
  ingest.py       transcript loading/validation, video resolution
  locate.py       select segments flagged for clipping
  align.py        audio extraction, ASR caching, forced alignment
  attribute.py    word-to-segment re-attribution after alignment
  verify.py       interactive spot-check and clip verification (Colab/Jupyter only)
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
notebooks/
  01_development.ipynb    the original build, phase by phase
  02_verification.ipynb   real-model verification run (see above)
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
  not yet backed by real windowed re-alignment. Confirmed still mocked
  during the real verification run above; the estimate held up
  reasonably well by ear, but it's still an estimate, not alignment.
- YouTube downloads via `yt-dlp` can get blocked by bot-detection on
  cloud/datacenter IPs, including Colab — hit live during verification.
  `resolve_video()` accepts a local or Drive file path as a reliable
  fallback; there's no cookie-based workaround built in yet.
- Video walkthrough not yet recorded/published.

## Path to total autonomy (a work in progress)

Everything in this repo assumes a human has already decided which
segments are clip-worthy — that's what `make_clip` and `clip_title` in
the flagged transcript represent. The diff/decision engine, caching, and
incremental logic are all completely content-agnostic by design: nothing
in `diff.py`, `decide.py`, `execute.py`, or `clip_reuse.py` reads *what*
was said, only what changed. That's deliberate scope, not an oversight.

Getting to a version that ingests any long-form video and transcript and
decides for itself what's worth clipping would need:

1. **A transcript segmentation adapter** — for input that isn't already
   in this repo's schema (raw prose, SRT/VTT, YouTube auto-captions).
   Could reuse the existing embedding infrastructure: a similarity dip
   between adjacent sentences is a reasonable topic-boundary signal.
2. **A curator module** — an LLM pass that reads the transcript and
   proposes which segments are worth clipping. Nothing in the current
   backend does content judgment; this is genuinely new capability, not
   a config change.
3. **Explicit, configurable selection criteria** — "clip-worthy" isn't
   fixed. The 16 segments flagged in this project's real run were
   selected against one specific brief; a different use case needs
   different judgment, as an actual input, not something implicit in
   one prompt.
4. **A decision on clip boundaries** — whether the curator only picks
   from existing segment boundaries, or can define its own spans that
   get mapped back onto what alignment already works with.
5. **A human-review checkpoint before committing** — propose, don't
   auto-commit. In keeping with how this project treats mocked/estimated
   output everywhere else (see "What's real vs. mocked" above), an
   autonomous selection step should be reviewable before it drives real
   clip cutting, not fully unattended.
6. **Incremental caching for selection itself** — the same content-hash
   caching already used for alignment and embeddings, extended one stage
   earlier, so re-running the curator on an edited transcript doesn't
   re-ask the model about segments that haven't changed.
7. **Chunking for long transcripts** — a multi-hour video easily produces
   hundreds of segments, too many for one LLM call reliably; needs
   windowing, similar in spirit to how alignment already handles long
   audio.

Items 1–3 are really one product decision — what counts as clip-worthy,
for whom, at what granularity — more than three separate engineering
tasks, and are worth settling deliberately before building the rest.