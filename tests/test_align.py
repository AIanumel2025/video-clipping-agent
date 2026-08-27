"""Tests for align.py's pure functions -- window estimation, no model
calls. transcribe_audio_cached and force_align need real audio/models
and aren't covered here; they're Tier 2 / Colab territory.
"""

from agent.align import estimate_rough_boundaries, estimate_segment_windows


def test_estimate_rough_boundaries_splits_proportionally_by_word_count():
    segments = [
        {"id": "s1", "text": "one two"},              # 2 words
        {"id": "s2", "text": "three four five six"},  # 4 words
    ]
    # total 6 words over 60s -> 10s/word
    rough = estimate_rough_boundaries(segments, total_duration=60.0)
    assert rough[0]["start"] == 0.0
    assert rough[0]["end"] == 20.0   # 2 words * 10s/word
    assert rough[1]["start"] == 20.0
    assert rough[1]["end"] == 60.0


def test_estimate_rough_boundaries_applies_lead_in_offset():
    segments = [{"id": "s1", "text": "one two"}]
    rough = estimate_rough_boundaries(segments, total_duration=10.0, lead_in_sec=5.0)
    assert rough[0]["start"] == 5.0
    assert rough[0]["end"] == 15.0


def test_estimate_segment_windows_uses_asr_timing_for_matched_words():
    segments = [{"id": "s1", "text": "hello world"}]
    asr_words = [
        {"word": "hello", "start": 1.0, "end": 1.3},
        {"word": "world", "start": 1.3, "end": 1.6},
    ]
    windows = estimate_segment_windows(segments, asr_words)
    assert windows[0]["start"] == 1.0
    assert windows[0]["end"] == 1.6


def test_estimate_segment_windows_falls_back_when_no_match_found(capsys):
    segments = [{"id": "s1", "text": "completely unrelated words"}]
    asr_words = [{"word": "totally", "start": 5.0, "end": 5.2}, {"word": "different", "start": 5.2, "end": 5.5}]
    windows = estimate_segment_windows(segments, asr_words)
    assert windows[0]["end"] - windows[0]["start"] == 1.5  # the 1.5s fallback guess
    captured = capsys.readouterr()
    assert "WEAK MATCH" in captured.out
