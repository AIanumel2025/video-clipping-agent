"""
Stage 2, Phase 5: commit -- execute the clip decisions and persist the
result, so the next run diffs against what run 2 actually produced, not
against a stale prior state.

Renders new video for RECUT/CUT_NEW, leaves REUSE_CLIP untouched on disk,
retires clips whose source segment is gone (deleted or un-flagged), then
overwrites manifest.json and state.json.

Reuses manifest.py's write_manifest() directly rather than a second,
differently-named implementation -- the original had its own inline
json.dump with a "run_timestamp" field where Stage 1 calls the same
concept "generated_at". Same file format, same writer.
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .cutter import cut_clip
from .clip_reuse import index_old_clips_by_segment
from .manifest import write_manifest
from .state import compute_transcript_hash, write_state_file, text_hash

CONFIDENCE_FOR_ACTION = {
    "VERIFY_SHIFT": "shifted_unverified",
    "PATCH": "mocked",
    "FULL_REALIGN": "mocked",
    "NEW": "mocked",
}


def backup_manifest(manifest_path: Path, run_dir: Path) -> None:
    """Keep the previous run's manifest around under its own name before
    overwriting manifest.json -- handy for diffing v1 vs v2 later."""
    backup_path = run_dir / "manifest_v1.json"
    if not backup_path.exists() and manifest_path.exists():
        shutil.copy(manifest_path, backup_path)
        print(f"Backed up prior manifest to {backup_path.name}")


def _next_clip_index(clips_dir: Path) -> int:
    existing_indices = [
        int(p.stem.split("_")[1]) for p in clips_dir.glob("clip_*.mp4")
        if p.stem.split("_")[1].isdigit()
    ]
    return max(existing_indices, default=0) + 1


def commit_clips(clip_decisions: List[dict], resolved_v2: List[dict], new_segments: List[dict],
                  old_manifest: dict, video_path: Path, clips_dir: Path) -> dict:
    old_clips_by_segment = index_old_clips_by_segment(old_manifest)
    new_by_id = {s["id"]: s for s in new_segments}
    resolved_by_id = {r["id"]: r for r in resolved_v2}

    run_timestamp = datetime.now(timezone.utc).isoformat()
    next_clip_index = _next_clip_index(clips_dir)

    manifest_v2_clips = []
    counts = {"REUSED": 0, "RECUT": 0, "NEW": 0, "RETIRED": 0}

    for d in clip_decisions:
        old_clip = old_clips_by_segment.get(d.get("old_id"))

        if d["action"] == "RETIRE_CLIP":
            if old_clip:
                old_clip_path = clips_dir / old_clip["filename"]
                if old_clip_path.exists():
                    retired_dir = clips_dir / "_retired"
                    retired_dir.mkdir(exist_ok=True)
                    shutil.move(str(old_clip_path), str(retired_dir / old_clip["filename"]))
            counts["RETIRED"] += 1
            print(f"RETIRE_CLIP  {d['id']}  -> {old_clip['filename'] if old_clip else '(no prior file)'} moved to _retired/")
            continue

        if d["action"] == "REUSE_CLIP":
            manifest_v2_clips.append({
                **old_clip,
                "reuse_status": "REUSED",
                "reason": d["reason"],
                "checked_at": run_timestamp,
            })
            counts["REUSED"] += 1
            print(f"REUSE_CLIP   {d['id']}  -> kept {old_clip['filename']} untouched")
            continue

        # RECUT or CUT_NEW both require actually rendering a clip
        r = resolved_by_id.get(d["id"])
        if r is None or r["start"] is None or r["end"] is None:
            print(f"SKIPPED      {d['id']}  -- flagged {d['action']} but still has no resolved timestamp, can't cut yet")
            continue

        if old_clip:
            filename = old_clip["filename"]
        else:
            filename = f"clip_{next_clip_index:03d}.mp4"
            next_clip_index += 1

        out_path = clips_dir / filename
        cut_clip(video_path, r["start"], r["end"], out_path)

        seg = new_by_id.get(d["id"], {})
        clip_record = {
            "clip_id": filename.replace(".mp4", ""),
            "segment_id": d["id"],
            "filename": filename,
            "clip_title": seg.get("clip_title") or (old_clip.get("clip_title") if old_clip else None),
            "start": round(r["start"], 3),
            "end": round(r["end"], 3),
            "duration": round(r["end"] - r["start"], 3),
            "confidence": "mocked",  # timing is estimated, not from a real re-alignment yet
            "text": r["text"],
            "reuse_status": "RECUT" if old_clip else "NEW",
            "reason": d["reason"],
            "generated_at": run_timestamp,
        }
        manifest_v2_clips.append(clip_record)

        if old_clip:
            counts["RECUT"] += 1
            action_label = "RECUT"
        else:
            counts["NEW"] += 1
            action_label = "CUT_NEW"
        print(f"{action_label:<12} {d['id']}  -> {'re-cut' if old_clip else 'cut new'} {filename}  "
              f"[{clip_record['start']}s - {clip_record['end']}s]")

    print(f"\nRun 2 clip summary: {counts['REUSED']} reused, {counts['RECUT']} recut, "
          f"{counts['NEW']} new, {counts['RETIRED']} retired")
    return {"clips": manifest_v2_clips, "run_timestamp": run_timestamp, "counts": counts}


def build_segment_state_v2(new_segments: List[dict], resolved_v2: List[dict], old_state: dict) -> dict:
    """Stage 2's segment_state, distinct from Stage 1's build_segment_state
    (state.py): REUSE carries forward the old (real) confidence; every
    other action gets an honest label reflecting that its timing is either
    mocked (not from real re-alignment yet) or arithmetically shifted (not
    independently re-verified) -- collapsing all of these into one vague
    label would overstate how trustworthy run 2's un-reused timings are.
    """
    resolved_by_id = {r["id"]: r for r in resolved_v2}
    segment_state = {}
    for s in new_segments:
        r = resolved_by_id.get(s["id"])
        if r is None:
            continue
        if r["action"] == "REUSE":
            confidence = old_state["segments"].get(s["id"], {}).get("confidence", "n/a")
        else:
            confidence = CONFIDENCE_FOR_ACTION.get(r["action"], "n/a")
        segment_state[s["id"]] = {
            "text_hash": text_hash(s["text"]),
            "start": round(r["start"], 3) if r["start"] is not None else None,
            "end": round(r["end"], 3) if r["end"] is not None else None,
            "confidence": confidence,
            "make_clip": s.get("make_clip", False),
        }
    return segment_state


def commit_transcript_snapshot(new_transcript: dict, run_dir: Path) -> Path:
    """Write current_transcript.json -- the committed snapshot a future
    incremental run reads as its 'old' side (see
    ingest.resolve_old_transcript_path). Without this, that function would
    keep falling back to the original transcript_flagged.json forever,
    diffing every future run against run 1 instead of the most recently
    committed version -- invisible until you actually try to run this
    more than once.
    """
    snapshot_path = run_dir / "current_transcript.json"
    snapshot_path.write_text(json.dumps(new_transcript, indent=2))
    print(f"Committed {snapshot_path.name} as the baseline for the next incremental run")
    return snapshot_path


def commit_run(clip_decisions: List[dict], resolved_v2: List[dict], classifications: List[dict],
                old_manifest: dict, old_state: dict, new_transcript: dict,
                video_id: str, video_path: Path, run_dir: Path, new_path: Path) -> dict:
    new_segments = new_transcript["segments"]
    manifest_path = run_dir / "manifest.json"
    state_path = run_dir / "state.json"
    clips_dir = run_dir / "clips"

    backup_manifest(manifest_path, run_dir)

    commit_result = commit_clips(clip_decisions, resolved_v2, new_segments, old_manifest, video_path, clips_dir)
    manifest_v2 = write_manifest(video_id, video_path, commit_result["clips"], manifest_path)

    diff_recount: dict = {}
    for c in classifications:
        diff_recount[c["kind"]] = diff_recount.get(c["kind"], 0) + 1

    segment_state = build_segment_state_v2(new_segments, resolved_v2, old_state)
    transcript_hash = compute_transcript_hash(new_segments)
    counts = commit_result["counts"]
    run_history_entry = {
        "run_timestamp": commit_result["run_timestamp"],
        "transcript_path": str(new_path),
        "diff_summary": diff_recount,
        "clip_summary": {"reused": counts["REUSED"], "recut": counts["RECUT"],
                          "new": counts["NEW"], "retired": counts["RETIRED"]},
    }
    state_v2 = write_state_file(video_id, segment_state, transcript_hash, run_history_entry, state_path)

    commit_transcript_snapshot(new_transcript, run_dir)

    return {"manifest": manifest_v2, "state": state_v2}
