"""Tests for decide.py -- the single source of truth for per-segment
actions. Embeddings are primed directly into the disk cache (see
conftest.seed_similarity_pair) so these never load a real model.
"""

from agent.decide import decide_actions, SIMILARITY_THRESHOLD


def cls(kind, old_id=None, new_id=None, old_text=None, new_text=None):
    return {"kind": kind, "old_id": old_id, "new_id": new_id, "old_text": old_text, "new_text": new_text}


def state_with(segments):
    """segments: dict of id -> (start, end) tuple, or None for unresolved."""
    return {"segments": {
        seg_id: {"start": span[0], "end": span[1]} if span else {"start": None, "end": None}
        for seg_id, span in segments.items()
    }}


def test_unchanged_before_any_change_is_reuse(embed_cache_dir):
    classifications = [cls("unchanged", "s1", "s1")]
    decisions = decide_actions(classifications, state_with({"s1": (0.0, 2.0)}), embed_cache_dir)
    assert decisions[0]["action"] == "REUSE"


def test_unchanged_after_a_change_is_verify_shift(embed_cache_dir):
    classifications = [
        cls("inserted", new_id="s0", new_text="brand new"),
        cls("unchanged", "s1", "s1"),
    ]
    decisions = decide_actions(classifications, state_with({"s1": (0.0, 2.0)}), embed_cache_dir)
    assert decisions[1]["action"] == "VERIFY_SHIFT"


def test_unchanged_with_no_prior_timestamp_is_verify_shift_even_first(embed_cache_dir):
    """A segment with no resolved timestamp in Stage 1 has nothing to
    reuse, regardless of whether anything upstream changed."""
    classifications = [cls("unchanged", "s1", "s1")]
    decisions = decide_actions(classifications, state_with({"s1": None}), embed_cache_dir)
    assert decisions[0]["action"] == "VERIFY_SHIFT"
    assert "never resolved" in decisions[0]["reason"]


def test_zero_word_delta_edit_still_flips_downstream_of_change(embed_cache_dir, seed_similarity_pair):
    """The specific edge case decide.py was built to handle correctly: a
    reworded segment with the SAME word count as before must still mark
    everything after it VERIFY_SHIFT -- a naive numeric 'did the mocked
    duration actually change' check would say no and wrongly reuse.
    """
    seed_similarity_pair("its fine", "it's fine", similarity=0.99)
    classifications = [
        cls("edited", "s1", "s1", old_text="its fine", new_text="it's fine"),
        cls("unchanged", "s2", "s2"),
    ]
    old_state = state_with({"s1": (0.0, 1.0), "s2": (1.0, 2.0)})
    decisions = decide_actions(classifications, old_state, embed_cache_dir)
    assert decisions[0]["action"] == "PATCH"
    assert decisions[1]["action"] == "VERIFY_SHIFT"


def test_high_similarity_edit_is_patch(embed_cache_dir, seed_similarity_pair):
    seed_similarity_pair("old wording here", "new wording here", similarity=0.92)
    classifications = [cls("edited", "s1", "s1", old_text="old wording here", new_text="new wording here")]
    decisions = decide_actions(classifications, state_with({"s1": (0.0, 1.0)}), embed_cache_dir)
    assert decisions[0]["action"] == "PATCH"


def test_low_similarity_edit_is_full_realign(embed_cache_dir, seed_similarity_pair):
    seed_similarity_pair("the weather is nice", "quarterly revenue grew", similarity=0.1)
    classifications = [cls("edited", "s1", "s1", old_text="the weather is nice", new_text="quarterly revenue grew")]
    decisions = decide_actions(classifications, state_with({"s1": (0.0, 1.0)}), embed_cache_dir)
    assert decisions[0]["action"] == "FULL_REALIGN"


def test_similarity_exactly_at_threshold_is_patch(embed_cache_dir, seed_similarity_pair):
    """>= threshold, not >, per decide.py's own comparison."""
    seed_similarity_pair("a", "b", similarity=SIMILARITY_THRESHOLD)
    classifications = [cls("edited", "s1", "s1", old_text="a", new_text="b")]
    decisions = decide_actions(classifications, state_with({"s1": (0.0, 1.0)}), embed_cache_dir)
    assert decisions[0]["action"] == "PATCH"


def test_inserted_is_always_new(embed_cache_dir):
    classifications = [cls("inserted", new_id="s1", new_text="brand new")]
    decisions = decide_actions(classifications, state_with({}), embed_cache_dir)
    assert decisions[0]["action"] == "NEW"


def test_deleted_is_always_remove(embed_cache_dir):
    classifications = [cls("deleted", old_id="s1", old_text="gone now")]
    decisions = decide_actions(classifications, state_with({"s1": (0.0, 1.0)}), embed_cache_dir)
    assert decisions[0]["action"] == "REMOVE"
