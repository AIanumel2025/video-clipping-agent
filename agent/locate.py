"""
Phase 2: locate which transcript segments are clip-worthy.

This is a selection step, not a timing step -- it answers "which segments
should become clips," using the make_clip flag a producer/editor has
already set. It runs before alignment, and is distinct from resolving a
segment's *timing* once alignment has happened (that job lives in
attribute.py -- the build plan originally sketched it as locate.py, but
that name is taken by this module now, matching the notebook's own
"Locating segments" phase).

Deliberately has no file I/O: it operates on an already-loaded transcript
dict, however that dict got loaded. Whether the caller passes the same
transcript object from Phase 1, or a second, separately-loaded "flagged"
transcript, is a decision made by the calling code, not by this module.
"""

from typing import List


def select_clip_segments(transcript: dict) -> List[dict]:
    """Return the segments flagged make_clip=true. Warns (does not fail)
    about any that are missing a clip_title, since that's an editorial
    gap rather than a schema break.
    """
    segments = transcript["segments"]
    clip_segments = [s for s in segments if s.get("make_clip")]

    untitled = [s["id"] for s in clip_segments if not s.get("clip_title")]
    if untitled:
        print(f"Warning: {len(untitled)} clip-flagged segment(s) have no clip_title: {untitled}")

    return clip_segments


def summarize_clip_segments(transcript: dict) -> None:
    segments = transcript["segments"]
    clip_segments = select_clip_segments(transcript)

    if not clip_segments:
        print(f"0 of {len(segments)} segments are flagged make_clip=true.")
        return

    for s in clip_segments:
        title = s.get("clip_title", "(no clip_title set)")
        print(f"{s['id']:8} \"{title}\"")
        print(f"         {s['text']}\n")
