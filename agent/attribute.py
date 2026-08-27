"""
Post-alignment: resolve each of OUR OWN segments' start/end time from
whisperx's word-level output.

Named attribute.py rather than locate.py (the build plan's original
name for this) -- locate.py now means segment *selection* (Phase 2),
matching the notebook's own "Locating segments" phase. This module does
timing resolution instead, which is a different job that happens to run
right after alignment.

whisperx.align() internally re-splits on sentence-ending punctuation for
alignment robustness, so its returned segments don't map 1:1 back onto
our original segment list. This module ignores whisperx's own segment
grouping entirely, flattens its word timings into one ordered list, and
re-attributes words back onto our segments positionally, by word count --
same idea whether the underlying aligner is whisperx or PocketSphinx.
"""

from typing import List


def flatten_words(aligned_result: dict) -> List[dict]:
    """Every aligned word across all of whisperx's (re-split) segments,
    in one ordered timeline. Word order survives resegmentation even
    though sentence grouping doesn't."""
    all_words = []
    for seg in aligned_result["segments"]:
        for w in seg.get("words", []):
            all_words.append(w)
    print(f"Total aligned words: {len(all_words)}")
    return all_words


def attribute_words_to_segments(segments: List[dict], all_words: List[dict]) -> List[dict]:
    """Re-attribute words onto OUR original segments, in order, by
    consuming len(segment_text.split()) words per segment.

    Sanity-checks word count first: if our transcript's word count (via
    .split()) doesn't match len(all_words) exactly, this warns rather
    than failing outright, since re-attribution will silently drift for
    every segment after the first divergence. Common cause: bracketed
    non-verbal annotations ([Laughs], [pause]) tokenized differently by
    whisperx than by .split().
    """
    our_word_count = sum(len(seg["text"].split()) for seg in segments)
    print(f"Our transcript's word count (via .split()): {our_word_count}")
    if our_word_count != len(all_words):
        print(f"WARNING: mismatch of {abs(our_word_count - len(all_words))} words -- "
              f"re-attribution below will drift for segments after the first divergence.")
    else:
        print("Word counts match exactly -- positional re-attribution is safe.")

    word_cursor = 0
    resolved = []
    for seg in segments:
        n_words = len(seg["text"].split())
        seg_words = all_words[word_cursor: word_cursor + n_words]
        word_cursor += n_words

        timed = [w for w in seg_words if w.get("start") is not None and w.get("end") is not None]
        if timed:
            start = min(w["start"] for w in timed)
            end = max(w["end"] for w in timed)
            confidence = "aligned" if len(timed) == len(seg_words) else "interpolated"
        else:
            start = end = None
            confidence = "unresolved"

        resolved.append({
            "id": seg["id"], "text": seg["text"], "speaker": seg.get("speaker"),
            "make_clip": seg.get("make_clip", False), "clip_title": seg.get("clip_title"),
            "start": start, "end": end, "confidence": confidence,
        })

    n_unresolved = sum(1 for r in resolved if r["confidence"] == "unresolved")
    print(f"Resolved {len(resolved)} segments ({n_unresolved} unresolved)")
    return resolved
