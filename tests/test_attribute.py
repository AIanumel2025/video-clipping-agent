"""Tests for attribute.py -- positional word re-attribution after
alignment. Pure functions, no model."""

from agent.attribute import flatten_words, attribute_words_to_segments


def test_flatten_words_preserves_order_across_resegmented_groups():
    aligned_result = {"segments": [
        {"words": [{"word": "hello", "start": 0.0, "end": 0.3}]},
        {"words": [{"word": "world", "start": 0.3, "end": 0.6}, {"word": "again", "start": 0.6, "end": 0.9}]},
    ]}
    words = flatten_words(aligned_result)
    assert [w["word"] for w in words] == ["hello", "world", "again"]


def test_attribute_words_matches_our_own_segment_boundaries(capsys):
    segments = [
        {"id": "s1", "text": "hello world", "speaker": "A", "make_clip": True, "clip_title": "T"},
        {"id": "s2", "text": "goodbye now", "speaker": "A"},
    ]
    all_words = [
        {"word": "hello", "start": 0.0, "end": 0.3},
        {"word": "world", "start": 0.3, "end": 0.6},
        {"word": "goodbye", "start": 0.6, "end": 1.0},
        {"word": "now", "start": 1.0, "end": 1.2},
    ]
    resolved = attribute_words_to_segments(segments, all_words)
    assert resolved[0]["start"] == 0.0 and resolved[0]["end"] == 0.6
    assert resolved[0]["confidence"] == "aligned"
    assert resolved[1]["start"] == 0.6 and resolved[1]["end"] == 1.2

    captured = capsys.readouterr()
    assert "match exactly" in captured.out  # word-count sanity check passed


def test_attribute_words_warns_on_word_count_mismatch(capsys):
    segments = [{"id": "s1", "text": "one two three", "speaker": "A"}]
    all_words = [{"word": "one", "start": 0.0, "end": 0.1}]  # only 1 of 3 expected words
    attribute_words_to_segments(segments, all_words)
    captured = capsys.readouterr()
    assert "WARNING" in captured.out


def test_attribute_words_marks_missing_timing_as_unresolved():
    segments = [{"id": "s1", "text": "one", "speaker": "A"}]
    all_words = [{"word": "one", "start": None, "end": None}]
    resolved = attribute_words_to_segments(segments, all_words)
    assert resolved[0]["confidence"] == "unresolved"
    assert resolved[0]["start"] is None


def test_attribute_words_marks_partial_timing_as_interpolated():
    segments = [{"id": "s1", "text": "one two", "speaker": "A"}]
    all_words = [
        {"word": "one", "start": 0.0, "end": 0.2},
        {"word": "two", "start": None, "end": None},  # one of two words timed
    ]
    resolved = attribute_words_to_segments(segments, all_words)
    assert resolved[0]["confidence"] == "interpolated"
    assert resolved[0]["start"] == 0.0
    assert resolved[0]["end"] == 0.2
