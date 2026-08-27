"""Tests for diff.py -- segment classification via hash-based
SequenceMatcher. Pure functions, no I/O, no models."""

from agent.diff import classify_segments


def seg(id_, text):
    return {"id": id_, "text": text, "speaker": "A", "make_clip": False}


def test_unchanged_segment_classified_correctly():
    old = [seg("s1", "hello world")]
    new = [seg("s1", "hello world")]
    result = classify_segments(old, new)
    assert len(result) == 1
    assert result[0]["kind"] == "unchanged"
    assert result[0]["old_id"] == "s1"
    assert result[0]["new_id"] == "s1"


def test_edited_segment_classified_as_replace():
    old = [seg("s1", "hello world")]
    new = [seg("s1", "hello there world")]
    result = classify_segments(old, new)
    assert len(result) == 1
    assert result[0]["kind"] == "edited"
    assert result[0]["old_text"] == "hello world"
    assert result[0]["new_text"] == "hello there world"


def test_inserted_segment_has_no_old_id():
    old = [seg("s1", "hello world")]
    new = [seg("s1", "hello world"), seg("s2", "brand new sentence")]
    result = classify_segments(old, new)
    inserted = next(c for c in result if c["kind"] == "inserted")
    assert inserted["old_id"] is None
    assert inserted["new_id"] == "s2"


def test_deleted_segment_has_no_new_id():
    old = [seg("s1", "hello world"), seg("s2", "goodbye now")]
    new = [seg("s1", "hello world")]
    result = classify_segments(old, new)
    deleted = next(c for c in result if c["kind"] == "deleted")
    assert deleted["new_id"] is None
    assert deleted["old_id"] == "s2"


def test_insertion_does_not_reclassify_later_unchanged_segments():
    """Matching is by content, not position -- an insertion earlier in
    the transcript shouldn't make a later, genuinely unchanged segment
    look edited just because its index shifted."""
    old = [seg("s1", "first"), seg("s2", "second")]
    new = [seg("s0", "brand new first"), seg("s1", "first"), seg("s2", "second")]
    result = classify_segments(old, new)
    kinds_by_new_id = {c["new_id"]: c["kind"] for c in result if c["new_id"]}
    assert kinds_by_new_id["s1"] == "unchanged"
    assert kinds_by_new_id["s2"] == "unchanged"


def test_empty_transcripts():
    assert classify_segments([], []) == []
