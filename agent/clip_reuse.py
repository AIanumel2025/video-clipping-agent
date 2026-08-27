"""
Stage 2, Phase 4: decide which previously-cut clip files are still valid,
and which need re-cutting or retiring.

A clip is only safe to reuse if BOTH its source text is unchanged AND its
resolved timestamp hasn't moved at all -- either one changing (a reword,
or just a downstream time shift from an earlier edit) means the old clip
file no longer matches reality and must be re-cut.

The old manifest is reloaded from disk here, not read from an in-memory
variable -- same principle as reloading state.json in Stage 2 Phase 1: a
real second run wouldn't have last run's Python variables sitting around.

Looks up the old manifest by old_id (now carried through resolved_v2
explicitly by execute.py), not by the new segment id -- the two are only
guaranteed equal by coincidence in this test data, and the diff engine
was deliberately built identity-based rather than assuming ids are
stable across versions.
"""

import json
from pathlib import Path
from typing import List

TIMESTAMP_TOLERANCE = 0.05  # seconds -- allows for float rounding, not real drift


def load_old_manifest(manifest_path: Path) -> dict:
    return json.loads(manifest_path.read_text())


def index_old_clips_by_segment(old_manifest: dict) -> dict:
    """Exposed separately (not just inlined in decide_clip_actions) so
    Phase 5's commit step can look up the same old-clip records by old_id
    without rebuilding this index or depending on decide_clip_actions'
    internal state."""
    return {c["segment_id"]: c for c in old_manifest["clips"]}


def decide_clip_actions(resolved_v2: List[dict], new_segments: List[dict], old_manifest: dict,
                         timestamp_tolerance: float = TIMESTAMP_TOLERANCE) -> List[dict]:
    old_clips_by_segment = index_old_clips_by_segment(old_manifest)
    new_by_id = {s["id"]: s for s in new_segments}

    clip_decisions = []

    for r in resolved_v2:
        old_id = r.get("old_id")

        if r["action"] == "REMOVE":
            old_clip = old_clips_by_segment.get(old_id)
            if old_clip:
                clip_decisions.append({"id": r["id"], "old_id": old_id, "action": "RETIRE_CLIP",
                                        "reason": f"source segment deleted -- {old_clip['filename']} no longer has a source"})
            continue

        seg = new_by_id.get(r["id"])
        old_clip = old_clips_by_segment.get(old_id)

        if seg is None or not seg.get("make_clip"):
            # Covers both "never was a clip candidate" and "was flagged
            # before, isn't now" -- the latter needs the same retirement
            # as an outright deletion, since the old clip file is equally
            # orphaned either way.
            if old_clip:
                clip_decisions.append({"id": r["id"], "old_id": old_id, "action": "RETIRE_CLIP",
                                        "reason": f"no longer flagged for a clip -- {old_clip['filename']} no longer has a source"})
            continue

        if old_clip is None:
            clip_decisions.append({"id": r["id"], "old_id": old_id, "action": "CUT_NEW",
                                    "reason": "newly flagged as a clip -- no prior clip file exists"})
            continue

        if r["start"] is None or r["end"] is None:
            clip_decisions.append({"id": r["id"], "old_id": old_id, "action": "CUT_NEW",
                                    "reason": f"flagged for a clip but no resolved timestamp yet "
                                              f"(needs fresh alignment before it can be cut) -- "
                                              f"{old_clip['filename']} can't be validated against it"})
            continue

        start_matches = abs(old_clip["start"] - r["start"]) < timestamp_tolerance
        end_matches = abs(old_clip["end"] - r["end"]) < timestamp_tolerance

        if start_matches and end_matches:
            clip_decisions.append({"id": r["id"], "old_id": old_id, "action": "REUSE_CLIP",
                                    "reason": f"text and timestamp both unchanged -- {old_clip['filename']} is still valid"})
        else:
            clip_decisions.append({"id": r["id"], "old_id": old_id, "action": "RECUT",
                                    "reason": f"timestamp moved ({old_clip['start']:.2f}-{old_clip['end']:.2f}s -> "
                                              f"{r['start']:.2f}-{r['end']:.2f}s) -- {old_clip['filename']} is stale"})

    return clip_decisions


def summarize_clip_decisions(clip_decisions: List[dict]) -> dict:
    summary: dict = {}
    for d in clip_decisions:
        summary[d["action"]] = summary.get(d["action"], 0) + 1
    print("Clip decision summary:", summary, "\n")

    for d in clip_decisions:
        print(f"Decision: {d['action']:<12} {d['id']}")
        print(f"Reason:   {d['reason']}\n")

    return summary
