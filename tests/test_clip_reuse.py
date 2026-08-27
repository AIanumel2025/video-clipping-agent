"""Tests for clip_reuse.py -- deciding which clips are still valid.
Covers the two gaps fixed while building this: un-flagged (not deleted)
retirement, and old_id-based lookup rather than assuming new_id==old_id.
"""

from agent.clip_reuse import decide_clip_actions, index_old_clips_by_segment


def resolved(id_, old_id, action="REUSE", text="hello", start=1.0, end=2.0):
    return {"id": id_, "old_id": old_id, "action": action, "text": text, "start": start, "end": end}


def old_manifest_with(clips):
    return {"clips": clips}


def clip(segment_id, filename, start=1.0, end=2.0, clip_title="Title"):
    return {"segment_id": segment_id, "filename": filename, "start": start, "end": end, "clip_title": clip_title}


def test_reuse_clip_when_text_and_timing_unchanged():
    resolved_v2 = [resolved("s1", "s1", action="REUSE", start=1.0, end=2.0)]
    new_segments = [{"id": "s1", "make_clip": True}]
    old_manifest = old_manifest_with([clip("s1", "clip_001.mp4", 1.0, 2.0)])
    decisions = decide_clip_actions(resolved_v2, new_segments, old_manifest)
    assert decisions[0]["action"] == "REUSE_CLIP"


def test_recut_when_timestamp_moved_beyond_tolerance():
    resolved_v2 = [resolved("s1", "s1", action="VERIFY_SHIFT", start=1.5, end=2.5)]
    new_segments = [{"id": "s1", "make_clip": True}]
    old_manifest = old_manifest_with([clip("s1", "clip_001.mp4", 1.0, 2.0)])
    decisions = decide_clip_actions(resolved_v2, new_segments, old_manifest)
    assert decisions[0]["action"] == "RECUT"


def test_within_tolerance_still_reuses():
    resolved_v2 = [resolved("s1", "s1", action="VERIFY_SHIFT", start=1.02, end=2.02)]
    new_segments = [{"id": "s1", "make_clip": True}]
    old_manifest = old_manifest_with([clip("s1", "clip_001.mp4", 1.0, 2.0)])
    decisions = decide_clip_actions(resolved_v2, new_segments, old_manifest)
    assert decisions[0]["action"] == "REUSE_CLIP"


def test_cut_new_when_newly_flagged():
    resolved_v2 = [resolved("s2", None, action="NEW", start=5.0, end=6.0)]
    new_segments = [{"id": "s2", "make_clip": True}]
    old_manifest = old_manifest_with([])
    decisions = decide_clip_actions(resolved_v2, new_segments, old_manifest)
    assert decisions[0]["action"] == "CUT_NEW"


def test_retire_clip_when_segment_deleted():
    resolved_v2 = [resolved("s1", "s1", action="REMOVE", start=None, end=None)]
    new_segments = []
    old_manifest = old_manifest_with([clip("s1", "clip_001.mp4")])
    decisions = decide_clip_actions(resolved_v2, new_segments, old_manifest)
    assert decisions[0]["action"] == "RETIRE_CLIP"


def test_retire_clip_when_unflagged_not_deleted():
    """The gap this module was specifically fixed for: a segment that
    still exists but is no longer make_clip -- its old clip is just as
    orphaned as if the segment had been deleted outright."""
    resolved_v2 = [resolved("s1", "s1", action="REUSE", start=1.0, end=2.0)]
    new_segments = [{"id": "s1", "make_clip": False}]
    old_manifest = old_manifest_with([clip("s1", "clip_001.mp4", 1.0, 2.0)])
    decisions = decide_clip_actions(resolved_v2, new_segments, old_manifest)
    assert decisions[0]["action"] == "RETIRE_CLIP"


def test_lookup_uses_old_id_not_new_id():
    """Guards the fix from the commit-phase integration bug: the old
    manifest lookup must key on old_id even when new_id differs from it."""
    resolved_v2 = [resolved("new_s1", "orig_s1", action="VERIFY_SHIFT", start=1.5, end=2.5)]
    new_segments = [{"id": "new_s1", "make_clip": True}]
    old_manifest = old_manifest_with([clip("orig_s1", "clip_001.mp4", 1.0, 2.0)])
    decisions = decide_clip_actions(resolved_v2, new_segments, old_manifest)
    assert decisions[0]["action"] == "RECUT"  # found the old clip via old_id, timing moved


def test_no_resolved_timestamp_yet_does_not_crash():
    resolved_v2 = [resolved("s1", "s1", action="PATCH", start=None, end=None)]
    new_segments = [{"id": "s1", "make_clip": True}]
    old_manifest = old_manifest_with([clip("s1", "clip_001.mp4", 1.0, 2.0)])
    decisions = decide_clip_actions(resolved_v2, new_segments, old_manifest)
    assert decisions[0]["action"] == "CUT_NEW"
    assert "no resolved timestamp yet" in decisions[0]["reason"]


def test_index_old_clips_by_segment():
    old_manifest = old_manifest_with([clip("s1", "clip_001.mp4"), clip("s2", "clip_002.mp4")])
    index = index_old_clips_by_segment(old_manifest)
    assert set(index.keys()) == {"s1", "s2"}
    assert index["s1"]["filename"] == "clip_001.mp4"
