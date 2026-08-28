"""
Command-line entry point for the video-clipping-agent pipelines.

Exists specifically to make this package runnable outside a notebook --
e.g. from a Docker container, where there's no interactive Python session
to call run_baseline_pipeline() / run_incremental_pipeline() directly.

Input/output directories are still resolved via config.py's existing
env vars (VIDEO_AGENT_INPUT_DIR / VIDEO_AGENT_OUTPUT_DIR), not CLI flags
-- reuses the path-resolution logic already built rather than duplicating
it here.

Usage:
    python -m agent baseline "https://youtube.com/watch?v=..." \\
        --transcript-filename transcript.json \\
        --flagged-filename transcript_flagged.json

    python -m agent incremental "https://youtube.com/watch?v=..." \\
        --transcript-filename transcript.json \\
        --fallback-flagged-filename transcript_flagged.json \\
        --new-flagged-filename transcript_flagged_new.json
"""

import argparse
import json
import sys

from .pipeline import run_baseline_pipeline, run_incremental_pipeline


def _print_result(result: dict) -> None:
    """The pipelines already print progress as they run; this just adds
    a machine-checkable summary line at the end, useful for anything
    scripting around this container (e.g. checking exit status + this
    line rather than scraping the full log)."""
    clips = result.get("clip_records")
    if clips is None:
        clips = result.get("clip_decisions") or []
    summary = {"run_dir": str(result.get("run_dir", "")), "clips": len(clips)}
    print(f"\nDone: {json.dumps(summary)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agent",
        description="Incremental video transcript alignment and clip extraction.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="Run the baseline pipeline (Stage 1).")
    baseline.add_argument("video_ref", help="Local video path or YouTube URL")
    baseline.add_argument("--transcript-filename", default="transcript.json")
    baseline.add_argument("--flagged-filename", default="transcript_flagged.json")

    incremental = subparsers.add_parser("incremental", help="Run the incremental pipeline (Stage 2).")
    incremental.add_argument("video_ref", help="Local video path or YouTube URL")
    incremental.add_argument("--transcript-filename", default="transcript.json")
    incremental.add_argument("--fallback-flagged-filename", default="transcript_flagged.json")
    incremental.add_argument("--new-flagged-filename", default="transcript_flagged_new.json")

    args = parser.parse_args(argv)

    try:
        if args.command == "baseline":
            result = run_baseline_pipeline(
                args.video_ref,
                transcript_filename=args.transcript_filename,
                flagged_filename=args.flagged_filename,
            )
        else:
            result = run_incremental_pipeline(
                args.video_ref,
                transcript_filename=args.transcript_filename,
                fallback_flagged_filename=args.fallback_flagged_filename,
                new_flagged_filename=args.new_flagged_filename,
            )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _print_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
