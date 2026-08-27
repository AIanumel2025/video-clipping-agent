"""
Stage 2, Phase 1: classify every segment as unchanged, edited, inserted,
or deleted, by diffing the incoming ("new") transcript against the last
one recorded (the "old" side -- see ingest.resolve_old_transcript_path).

Matches on TEXT CONTENT (via hash), not position -- this correctly
handles insertions/deletions shifting everything after them, since
SequenceMatcher finds the longest matching runs by value, not by index.

This phase only classifies. Deciding what to actually DO about each
classification (reuse, patch, full realign, ...) is a separate step.

Within a SequenceMatcher "replace" block, segments are paired
positionally as edits when the old and new blocks are the same length;
any length mismatch is treated as pure delete/insert. Telling a genuine
rewording apart from an unrelated segment that happens to land in the
same replace block is exactly what embedding similarity (the next
phase) refines -- this phase is deliberately coarse.

Pure functions, no I/O -- takes already-loaded segment lists, so this
is testable with canned fixtures and no file access at all.
"""

import difflib
from typing import List

from .state import text_hash


def classify_segments(old_segments: List[dict], new_segments: List[dict]) -> List[dict]:
    old_ids = [s["id"] for s in old_segments]
    new_ids = [s["id"] for s in new_segments]
    old_hashes = [text_hash(s["text"]) for s in old_segments]
    new_hashes = [text_hash(s["text"]) for s in new_segments]

    matcher = difflib.SequenceMatcher(a=old_hashes, b=new_hashes, autojunk=False)

    classifications = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for oi, ni in zip(range(i1, i2), range(j1, j2)):
                classifications.append({
                    "kind": "unchanged", "old_id": old_ids[oi], "new_id": new_ids[ni],
                    "old_text": old_segments[oi]["text"], "new_text": new_segments[ni]["text"],
                })
        elif tag == "replace":
            old_block, new_block = list(range(i1, i2)), list(range(j1, j2))
            for oi, ni in zip(old_block, new_block):
                classifications.append({
                    "kind": "edited", "old_id": old_ids[oi], "new_id": new_ids[ni],
                    "old_text": old_segments[oi]["text"], "new_text": new_segments[ni]["text"],
                })
            for oi in old_block[len(new_block):]:
                classifications.append({
                    "kind": "deleted", "old_id": old_ids[oi], "new_id": None,
                    "old_text": old_segments[oi]["text"], "new_text": None,
                })
            for ni in new_block[len(old_block):]:
                classifications.append({
                    "kind": "inserted", "old_id": None, "new_id": new_ids[ni],
                    "old_text": None, "new_text": new_segments[ni]["text"],
                })
        elif tag == "delete":
            for oi in range(i1, i2):
                classifications.append({
                    "kind": "deleted", "old_id": old_ids[oi], "new_id": None,
                    "old_text": old_segments[oi]["text"], "new_text": None,
                })
        elif tag == "insert":
            for ni in range(j1, j2):
                classifications.append({
                    "kind": "inserted", "old_id": None, "new_id": new_ids[ni],
                    "old_text": None, "new_text": new_segments[ni]["text"],
                })

    return classifications


def summarize_classifications(classifications: List[dict]) -> dict:
    summary: dict = {}
    for c in classifications:
        summary[c["kind"]] = summary.get(c["kind"], 0) + 1
    print("Diff summary:", summary, "\n")

    for c in classifications:
        if c["kind"] == "unchanged":
            continue
        if c["kind"] == "edited":
            print(f"[EDITED]   {c['old_id']} -> {c['new_id']}\n   old: {c['old_text'][:80]}\n   new: {c['new_text'][:80]}\n")
        elif c["kind"] == "inserted":
            print(f"[INSERTED] {c['new_id']}\n   new: {c['new_text'][:80]}\n")
        elif c["kind"] == "deleted":
            print(f"[DELETED]  {c['old_id']}\n   old: {c['old_text'][:80]}\n")

    return summary
