"""
Stage 2, Phase 3, Step 1: decide what should happen to each segment.

This is the single source of truth for action (REUSE / PATCH /
FULL_REALIGN / NEW / VERIFY_SHIFT / REMOVE) -- execute.py (Step 2) reads
these decisions and carries them out; it does not re-derive them. That
used to not be true (an earlier draft of Step 2 independently re-decided
REUSE vs VERIFY_SHIFT with a different, disagreeing rule, which could
make decisions.log's own summary contradict its own itemized entries) --
fixed here by making this module's output the only place `action` gets
decided.

REUSE vs VERIFY_SHIFT for unchanged segments uses a one-way flag, not a
running numeric delta: the first change encountered anywhere earlier in
the transcript marks every unchanged segment after it VERIFY_SHIFT, for
the rest of the file, even if a later edit's mocked duration change
happens to be exactly zero. Deliberate, not an oversight -- trusting an
exact-zero *mocked* duration change as proof "nothing actually shifted"
would be over-trusting an approximation the project is honest about
being a stand-in for real re-alignment. Conservative-by-default here
costs some unnecessary VERIFY_SHIFT work, not correctness.

Whether an unchanged segment even has a prior timestamp to reuse is
folded into the decision here too (not left for execute.py to check) --
it's part of deciding REUSE vs VERIFY_SHIFT, not a detail of executing
a decision already made.
"""

from pathlib import Path
from typing import List

from .embeddings import get_embedding, cosine_similarity

SIMILARITY_THRESHOLD = 0.85  # tunable: above this, treat an edit as a minor reword


def decide_actions(classifications: List[dict], old_state: dict, embed_cache_dir: Path,
                    similarity_threshold: float = SIMILARITY_THRESHOLD) -> List[dict]:
    decisions = []
    downstream_of_change = False  # flips true the first time we cross any change point

    for c in classifications:
        old_id, new_id = c.get("old_id"), c.get("new_id")
        old_text, new_text = c.get("old_text"), c.get("new_text")

        if c["kind"] == "unchanged":
            old = old_state["segments"].get(old_id, {})
            has_prior_timestamp = old.get("start") is not None and old.get("end") is not None

            if not has_prior_timestamp:
                action = "VERIFY_SHIFT"
                reason = ("unchanged text, but Stage 1 never resolved a timestamp for it -- "
                          "needs fresh alignment, not a shift")
            elif not downstream_of_change:
                action = "REUSE"
                reason = "unchanged text, no upstream edits before it -- old timestamp is trustworthy as-is"
            else:
                action = "VERIFY_SHIFT"
                reason = ("text unchanged, but occurs after an earlier edit/insertion/deletion -- "
                          "absolute timing may have shifted, needs offset re-verification")

        elif c["kind"] == "edited":
            sim = cosine_similarity(
                get_embedding(old_text, embed_cache_dir),
                get_embedding(new_text, embed_cache_dir),
            )
            if sim >= similarity_threshold:
                action = "PATCH"
                reason = (f"minor reword (similarity {sim:.3f} >= {similarity_threshold}) -- "
                          f"realign this segment locally, not the whole file")
            else:
                action = "FULL_REALIGN"
                reason = (f"substantial rewrite (similarity {sim:.3f} < {similarity_threshold}) -- "
                          f"too different to trust a local patch")
            downstream_of_change = True

        elif c["kind"] == "inserted":
            action = "NEW"
            reason = "no prior alignment exists for this segment -- must align fresh"
            downstream_of_change = True

        elif c["kind"] == "deleted":
            action = "REMOVE"
            reason = "segment no longer exists in the new transcript"
            downstream_of_change = True

        decisions.append({
            "old_id": old_id, "new_id": new_id,
            "old_text": old_text, "new_text": new_text,
            "kind": c["kind"], "action": action, "reason": reason,
        })

    return decisions


def summarize_decisions(decisions: List[dict]) -> dict:
    summary: dict = {}
    for d in decisions:
        summary[d["action"]] = summary.get(d["action"], 0) + 1
    print("Decision summary:", summary, "\n")

    for d in decisions:
        if d["action"] == "REUSE":
            continue  # too many to print, least interesting
        display_id = d["new_id"] or d["old_id"]
        print(f"Decision: {d['action']:<12} {display_id}")
        print(f"Reason:   {d['reason']}\n")

    return summary
