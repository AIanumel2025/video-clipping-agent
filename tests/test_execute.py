"""Tests for execute.py -- pure execution of already-decided actions.
No I/O, no models: takes decisions and old_state, computes mocked timing.
"""

from agent.execute import execute_decisions, estimate_words_per_second


def decision(action, kind, old_id=None, new_id=None, old_text=None, new_text=None, reason="test"):
    return {"action": action, "kind": kind, "old_id": old_id, "new_id": new_id,
            "old_text": old_text, "new_text": new_text, "reason": reason}


def test_reuse_keeps_old_timing_unshifted():
    decisions = [decision("REUSE", "unchanged", old_id="s1", new_id="s1", new_text="hello")]
    old_state = {"segments": {"s1": {"start": 1.0, "end": 2.0}}}
    result = execute_decisions(decisions, old_state, avg_sec_per_word=0.4)
    r = result["resolved_v2"][0]
    assert r["start"] == 1.0 and r["end"] == 2.0
    assert result["cumulative_delta"] == 0.0


def test_patch_with_longer_text_pushes_cumulative_delta_forward():
    decisions = [decision("PATCH", "edited", old_id="s1", new_id="s1",
                           old_text="two words", new_text="now four words here")]
    old_state = {"segments": {"s1": {"start": 0.0, "end": 1.0}}}
    result = execute_decisions(decisions, old_state, avg_sec_per_word=0.5)
    # word_delta = 4 - 2 = 2 words * 0.5s/word = +1.0s
    assert result["cumulative_delta"] == 1.0
    r = result["resolved_v2"][0]
    assert r["end"] == 2.0  # 1.0 (old end) + 0.0 (delta before) + 1.0 (this edit's change)


def test_verify_shift_after_a_patch_inherits_the_cumulative_delta():
    decisions = [
        decision("PATCH", "edited", old_id="s1", new_id="s1", old_text="a b", new_text="a b c d"),
        decision("VERIFY_SHIFT", "unchanged", old_id="s2", new_id="s2", new_text="unchanged text"),
    ]
    old_state = {"segments": {
        "s1": {"start": 0.0, "end": 1.0},
        "s2": {"start": 1.0, "end": 2.0},
    }}
    result = execute_decisions(decisions, old_state, avg_sec_per_word=0.5)
    # s1: +2 words * 0.5 = +1.0s delta, carried into s2's shift
    r2 = result["resolved_v2"][1]
    assert r2["start"] == 2.0  # 1.0 (old start) + 1.0 (cumulative delta)
    assert r2["end"] == 3.0


def test_remove_decreases_cumulative_delta_by_removed_duration():
    decisions = [decision("REMOVE", "deleted", old_id="s1", old_text="gone")]
    old_state = {"segments": {"s1": {"start": 5.0, "end": 7.5}}}
    result = execute_decisions(decisions, old_state, avg_sec_per_word=0.5)
    assert result["cumulative_delta"] == -2.5
    r = result["resolved_v2"][0]
    assert r["start"] is None and r["end"] is None


def test_new_segment_starts_after_previous_with_gap():
    decisions = [
        decision("REUSE", "unchanged", old_id="s1", new_id="s1", new_text="first"),
        decision("NEW", "inserted", new_id="s2", new_text="two new words"),
    ]
    old_state = {"segments": {"s1": {"start": 0.0, "end": 3.0}}}
    result = execute_decisions(decisions, old_state, avg_sec_per_word=0.5)
    r_new = result["resolved_v2"][1]
    assert r_new["start"] == 3.3  # prev end (3.0) + 0.3s gap
    assert r_new["end"] == 4.8   # start + (3 words * 0.5s/word)


def test_patch_with_no_prior_timestamp_cannot_compute_shifted_timing():
    decisions = [decision("PATCH", "edited", old_id="s1", new_id="s1", old_text="a", new_text="a b")]
    old_state = {"segments": {"s1": {"start": None, "end": None}}}
    result = execute_decisions(decisions, old_state, avg_sec_per_word=0.5)
    r = result["resolved_v2"][0]
    assert r["start"] is None and r["end"] is None
    assert "never resolved a prior timestamp" in r["reason"]


def test_estimate_words_per_second_uses_only_resolved_segments_for_duration():
    old_state = {"segments": {
        "s1": {"start": 0.0, "end": 2.0},
        "s2": {"start": None, "end": None},  # unresolved -- excluded from duration
    }}
    old_segments = [
        {"id": "s1", "text": "one two three four"},  # 4 words
        {"id": "s2", "text": "five six seven"},        # 3 words -- still counted here
    ]
    rate = estimate_words_per_second(old_state, old_segments)
    assert rate == 2.0 / 7
