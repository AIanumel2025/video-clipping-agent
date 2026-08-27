"""Tests for state.py -- state.json persistence, hashing, and the
sanity-check that catches a stale/mismatched transcript before trusting
it as a diff baseline."""

import json

from agent.state import (
    text_hash, verify_state_matches_transcript, build_segment_state,
    load_run_history, compute_transcript_hash, write_state_file, write_state,
)


def test_text_hash_is_deterministic():
    assert text_hash("hello world") == text_hash("hello world")


def test_text_hash_differs_for_different_text():
    assert text_hash("hello world") != text_hash("hello there")


def test_verify_state_matches_transcript_passes_when_hashes_match():
    segments = [{"id": "s1", "text": "hello world"}]
    state = {"segments": {"s1": {"text_hash": text_hash("hello world")}}}
    assert verify_state_matches_transcript(segments, state) == []


def test_verify_state_matches_transcript_flags_a_stale_hash():
    segments = [{"id": "s1", "text": "hello world, edited"}]
    state = {"segments": {"s1": {"text_hash": text_hash("hello world")}}}  # stale
    assert verify_state_matches_transcript(segments, state) == ["s1"]


def test_verify_state_matches_transcript_flags_a_segment_missing_from_state():
    segments = [{"id": "s1", "text": "hello world"}, {"id": "s2", "text": "new one"}]
    state = {"segments": {"s1": {"text_hash": text_hash("hello world")}}}  # s2 never recorded
    assert verify_state_matches_transcript(segments, state) == ["s2"]


def test_build_segment_state_carries_through_resolved_fields():
    resolved = [{"id": "s1", "text": "hello", "start": 1.234567, "end": 2.5,
                 "confidence": "aligned", "make_clip": True}]
    segment_state = build_segment_state(resolved)
    assert segment_state["s1"]["text_hash"] == text_hash("hello")
    assert segment_state["s1"]["start"] == 1.235  # rounded to 3 places
    assert segment_state["s1"]["confidence"] == "aligned"
    assert segment_state["s1"]["make_clip"] is True


def test_build_segment_state_keeps_unresolved_timing_as_none():
    resolved = [{"id": "s1", "text": "hello", "start": None, "end": None,
                 "confidence": "unresolved", "make_clip": False}]
    segment_state = build_segment_state(resolved)
    assert segment_state["s1"]["start"] is None
    assert segment_state["s1"]["end"] is None


def test_load_run_history_returns_empty_list_when_no_file_yet(tmp_path):
    assert load_run_history(tmp_path / "state.json") == []


def test_load_run_history_reads_existing_entries(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"run_history": [{"run_at": "yesterday"}]}))
    assert load_run_history(state_path) == [{"run_at": "yesterday"}]


def test_compute_transcript_hash_is_consistent_across_different_callers():
    """Regression test: Stage 1's write_state() and the Stage 2 commit
    step used to hash the whole transcript two different, incompatible
    ways (raw concatenation vs json.dumps of a list) -- the same content
    produced two different hashes depending which stage wrote it. Both
    now go through this one function.
    """
    resolved_shaped = [{"text": "hello"}, {"text": "world"}]
    raw_shaped = [{"id": "s1", "text": "hello"}, {"id": "s2", "text": "world"}]
    assert compute_transcript_hash(resolved_shaped) == compute_transcript_hash(raw_shaped)
    assert compute_transcript_hash(resolved_shaped) == text_hash("helloworld")


def test_write_state_file_persists_and_appends_run_history(tmp_path):
    state_path = tmp_path / "state.json"
    segment_state = {"s1": {"text_hash": "abc", "start": 1.0, "end": 2.0,
                             "confidence": "aligned", "make_clip": True}}
    write_state_file("vid1", segment_state, "hash1", {"run_at": "t1"}, state_path)
    state = json.loads(state_path.read_text())
    assert state["video_id"] == "vid1"
    assert state["transcript_hash"] == "hash1"
    assert len(state["run_history"]) == 1

    # a second write should append, not overwrite, run_history
    write_state_file("vid1", segment_state, "hash2", {"run_at": "t2"}, state_path)
    state2 = json.loads(state_path.read_text())
    assert len(state2["run_history"]) == 2
    assert state2["transcript_hash"] == "hash2"


def test_write_state_wraps_build_segment_state_and_compute_transcript_hash(tmp_path):
    state_path = tmp_path / "state.json"
    resolved = [{"id": "s1", "text": "hello", "start": 1.0, "end": 2.0,
                 "confidence": "aligned", "make_clip": True}]
    write_state("vid1", resolved, clip_records=[], state_path=state_path)
    state = json.loads(state_path.read_text())
    assert state["segments"]["s1"]["text_hash"] == text_hash("hello")
    assert state["transcript_hash"] == compute_transcript_hash(resolved)
