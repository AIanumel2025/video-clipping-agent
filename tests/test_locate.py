"""Tests for locate.py -- pure segment selection, no I/O."""

from agent.locate import select_clip_segments


def test_selects_only_flagged_segments():
    transcript = {"segments": [
        {"id": "s1", "text": "a", "make_clip": True, "clip_title": "A"},
        {"id": "s2", "text": "b", "make_clip": False},
    ]}
    selected = select_clip_segments(transcript)
    assert [s["id"] for s in selected] == ["s1"]


def test_warns_but_does_not_fail_on_missing_clip_title(capsys):
    transcript = {"segments": [{"id": "s1", "text": "a", "make_clip": True}]}
    selected = select_clip_segments(transcript)
    assert len(selected) == 1  # not a hard failure
    captured = capsys.readouterr()
    assert "Warning" in captured.out and "s1" in captured.out


def test_no_flagged_segments():
    transcript = {"segments": [{"id": "s1", "text": "a", "make_clip": False}]}
    assert select_clip_segments(transcript) == []
