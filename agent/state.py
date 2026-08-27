"""
Phase 5 (part 2): write state.json -- per-segment text fingerprints and
run history, so a future run has something to diff against instead of
treating every transcript as brand new.

Nothing reads this back yet. This phase only writes forward; Stage 2 is
what reads it.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def verify_state_matches_transcript(segments: List[dict], state: dict) -> List[str]:
    """Sanity check that state.json's stored hashes actually match the
    transcript file it's meant to describe, before trusting it as a diff
    baseline. Warns (and returns the mismatched ids) rather than raising --
    a mismatch is informative but not automatically fatal to continuing.
    """
    mismatches = [
        s["id"] for s in segments
        if state["segments"].get(s["id"], {}).get("text_hash") != text_hash(s["text"])
    ]
    if mismatches:
        print(f"WARNING: {len(mismatches)} segment(s) don't match state.json's stored hash: {mismatches}")
    else:
        print(f"Sanity check passed: state.json's {len(state['segments'])} stored hashes "
              f"match the transcript file it was built from.")
    return mismatches


def build_segment_state(resolved: List[dict]) -> dict:
    return {
        r["id"]: {
            "text_hash": text_hash(r["text"]),
            "start": round(r["start"], 3) if r["start"] is not None else None,
            "end": round(r["end"], 3) if r["end"] is not None else None,
            "confidence": r["confidence"],
            "make_clip": r["make_clip"],
        }
        for r in resolved
    }


def load_run_history(state_path: Path) -> list:
    if not state_path.exists():
        return []
    return json.loads(state_path.read_text()).get("run_history", [])


def compute_transcript_hash(segments: List[dict]) -> str:
    """Whole-transcript fingerprint -- same method every time (raw
    concatenation, not JSON-encoded), so this hash is comparable across
    every run and every stage that computes it. Works for either resolved
    segments (Stage 1) or raw transcript segments (Stage 2), since both
    carry a "text" field.
    """
    return text_hash("".join(s["text"] for s in segments))


def write_state_file(video_id: str, segment_state: dict, transcript_hash: str,
                      run_history_entry: dict, state_path: Path) -> dict:
    """Low-level writer: appends one run_history entry and persists
    state.json. Takes an already-built segment_state rather than building
    it internally, so different stages (Stage 1's real alignment
    confidence levels, Stage 2's mocked/shifted ones) can supply their own
    without duplicating the disk-write and history-append mechanics.
    """
    run_history = load_run_history(state_path)
    run_history.append(run_history_entry)

    state = {
        "video_id": video_id,
        "transcript_hash": transcript_hash,
        "segments": segment_state,
        "run_history": run_history,
    }
    state_path.write_text(json.dumps(state, indent=2))
    print(f"Wrote {state_path} ({len(segment_state)} segment hashes, run_history now has {len(run_history)} entries)")
    print(f"Transcript hash: {transcript_hash}")
    return state


def write_state(video_id: str, resolved: List[dict], clip_records: List[dict], state_path: Path) -> dict:
    segment_state = build_segment_state(resolved)
    transcript_hash = compute_transcript_hash(resolved)
    run_history_entry = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "n_segments": len(resolved),
        "n_clips": len(clip_records),
        "transcript_hash": transcript_hash,
    }
    return write_state_file(video_id, segment_state, transcript_hash, run_history_entry, state_path)
