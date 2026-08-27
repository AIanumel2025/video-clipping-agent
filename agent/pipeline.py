"""
Orchestrates both of this project's pipelines end to end.

run_baseline_pipeline: ingest -> locate -> align -> attribute -> cut ->
export (Stage 1). Nothing "smart" -- every run realigns every segment
from scratch.

run_incremental_pipeline: diff -> embed -> decide -> execute -> log ->
clip-reuse -> commit (Stage 2). Reads the state.json a prior run wrote,
works out what actually changed, and only redoes the parts that need it.

Spot-check playback (verify.py) is deliberately left out of both -- it's
a human-in-the-loop review step, not something a one-line pipeline call
should do unattended.
"""

import json
from pathlib import Path
from typing import Optional

from .config import get_input_dir, get_output_dir
from .ingest import (
    load_transcript, resolve_video, resolve_run_dir, probe_duration,
    validate_paired_transcripts, summarize, resolve_old_transcript_path,
)
from .locate import summarize_clip_segments
from .align import extract_audio, transcribe_audio_cached, estimate_segment_windows, force_align
from .attribute import flatten_words, attribute_words_to_segments
from .cutter import cut_all_clips
from .manifest import write_manifest
from .state import write_state, verify_state_matches_transcript
from .diff import classify_segments, summarize_classifications
from .embeddings import embed_segments
from .decide import decide_actions, summarize_decisions
from .execute import estimate_words_per_second, execute_decisions
from .logger import append_decisions_log
from .clip_reuse import load_old_manifest, decide_clip_actions, summarize_clip_decisions
from .commit import commit_run


def run_baseline_pipeline(video_ref: str, input_dir: Optional[Path] = None,
                           output_dir: Optional[Path] = None,
                           transcript_filename: str = "transcript.json",
                           flagged_filename: str = "transcript_flagged.json") -> dict:
    input_dir = input_dir or get_input_dir()
    output_dir = output_dir or get_output_dir()

    transcript = load_transcript(input_dir / transcript_filename)
    summarize(transcript)

    flagged_transcript = load_transcript(input_dir / flagged_filename)
    validate_paired_transcripts(transcript, flagged_transcript)
    summarize_clip_segments(flagged_transcript)

    run_dir = resolve_run_dir(transcript, output_dir)
    video_path = resolve_video(video_ref, workdir=run_dir)
    video_duration = probe_duration(video_path)
    print(f"Loaded video: {video_path.name} ({video_duration:.1f}s = {video_duration/60:.2f} min)")

    audio_path = extract_audio(video_path, run_dir)
    asr_words = transcribe_audio_cached(audio_path, run_dir / "cache" / "alignments")
    windows = estimate_segment_windows(flagged_transcript["segments"], asr_words)
    aligned_result = force_align(windows, audio_path)

    all_words = flatten_words(aligned_result)
    resolved = attribute_words_to_segments(flagged_transcript["segments"], all_words)

    clips_dir = run_dir / "clips"
    clip_records = cut_all_clips(resolved, video_path, clips_dir)

    manifest = write_manifest(transcript["video_id"], video_path, clip_records, run_dir / "manifest.json")
    state = write_state(transcript["video_id"], resolved, clip_records, run_dir / "state.json")

    return {
        "run_dir": run_dir, "resolved": resolved, "clip_records": clip_records,
        "manifest": manifest, "state": state,
    }


def run_incremental_pipeline(video_ref: str, input_dir: Optional[Path] = None,
                              output_dir: Optional[Path] = None,
                              transcript_filename: str = "transcript.json",
                              fallback_flagged_filename: str = "transcript_flagged.json",
                              new_flagged_filename: str = "transcript_flagged_new.json") -> dict:
    """Diff the incoming transcript against the most recently committed
    run (or run 1, on the first incremental pass), decide what changed,
    execute those decisions (mocked timing -- no real windowed
    re-alignment yet), reuse or re-cut clips accordingly, and commit --
    including current_transcript.json, so the run after this one diffs
    against this commit, not against run 1 forever.
    """
    input_dir = input_dir or get_input_dir()
    output_dir = output_dir or get_output_dir()

    transcript = load_transcript(input_dir / transcript_filename)
    run_dir = resolve_run_dir(transcript, output_dir)
    video_path = resolve_video(video_ref, workdir=run_dir)

    old_path = resolve_old_transcript_path(input_dir, run_dir, fallback_flagged_filename)
    old_transcript = load_transcript(old_path)
    old_segments = old_transcript["segments"]

    state_path = run_dir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"No state.json at {state_path} -- run the baseline pipeline at least once first.")
    old_state = json.loads(state_path.read_text())
    verify_state_matches_transcript(old_segments, old_state)

    new_path = input_dir / new_flagged_filename
    new_transcript = load_transcript(new_path)
    new_segments = new_transcript["segments"]
    print(f"\nOld: {len(old_segments)} segments   New: {len(new_segments)} segments\n")

    classifications = classify_segments(old_segments, new_segments)
    summarize_classifications(classifications)

    embed_cache_dir = run_dir / "cache" / "embeddings"
    embed_segments(old_segments, embed_cache_dir)
    embed_segments(new_segments, embed_cache_dir)

    decisions = decide_actions(classifications, old_state, embed_cache_dir)
    summary = summarize_decisions(decisions)

    avg_sec_per_word = estimate_words_per_second(old_state, old_segments)
    execution = execute_decisions(decisions, old_state, avg_sec_per_word)
    resolved_v2 = execution["resolved_v2"]
    cumulative_delta = execution["cumulative_delta"]

    log_path = run_dir / "decisions.log"
    append_decisions_log(resolved_v2, old_path, new_path, summary, cumulative_delta, log_path)

    old_manifest = load_old_manifest(run_dir / "manifest.json")
    clip_decisions = decide_clip_actions(resolved_v2, new_segments, old_manifest)
    summarize_clip_decisions(clip_decisions)

    result = commit_run(
        clip_decisions, resolved_v2, classifications,
        old_manifest, old_state, new_transcript,
        transcript["video_id"], video_path, run_dir, new_path,
    )

    return {
        "run_dir": run_dir, "classifications": classifications, "decisions": decisions,
        "resolved_v2": resolved_v2, "clip_decisions": clip_decisions,
        "manifest": result["manifest"], "state": result["state"],
    }
