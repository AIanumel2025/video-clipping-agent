"""
Stage 0: segment a raw transcript into this project's schema, before
Phase 2 (locate.py) has anything to select from.

Handles four real shapes, checked in order:
  1. Speaker-labeled lines ("Name: text", one turn per line) -- what
     interview/meeting transcripts (Otter, Rev, Zoom) typically look like.
  2. Timestamped caption fragments ("0:00 some words", one small chunk
     per line, no speakers) -- YouTube's own "Show transcript" export,
     confirmed against a real single-speaker video's raw output.
  3. Timecode-RANGE lines on their own ("00:00:44:18 - 00:00:45:22"),
     with the text on the following line -- broadcast/SRT-style export
     with sequence numbers already stripped, confirmed against a real
     NASA subject-matter-expert video transcript. Everything before the
     first timecode line (a title, a "Narration: ..." header) is
     dropped, not guessed at.
  4. Plain prose, no structure detected at all -- falls back to
     sentence-splitting the whole thing, speaker set to "Unknown".

For both timestamp-based paths (2 and 3), the actual timing gets
discarded -- real, precise timing comes from forced alignment in
Phase 3 against the actual audio, not from coarse caption boundaries.

Every segment comes out with make_clip: false -- nothing has been
curated yet. Deciding what's clip-worthy is the curator module's job,
not this one's; this module's only contract is producing valid,
correctly-schemed segments for locate.py to eventually select from.
"""

import re
from typing import List, Optional

SPEAKER_LINE_PATTERN = re.compile(r'^([A-Z][A-Za-z .]{1,40}):\s*(.+)$')
TIMESTAMP_LINE_PATTERN = re.compile(r'^(?:\d+:)?\d{1,2}:\d{2}\s+(.+)$')
TIMECODE_RANGE_LINE_PATTERN = re.compile(r'^\d{2}:\d{2}:\d{2}:\d{2}\s*-\s*\d{2}:\d{2}:\d{2}:\d{2}$')


def segment_raw_transcript(raw_text: str, video_id: str) -> dict:
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    speaker_matches = [SPEAKER_LINE_PATTERN.match(ln) for ln in lines]
    timestamp_matches = [TIMESTAMP_LINE_PATTERN.match(ln) for ln in lines]
    timecode_range_hits = [bool(TIMECODE_RANGE_LINE_PATTERN.match(ln)) for ln in lines]

    if sum(1 for m in speaker_matches if m) >= 2:
        segments = _segment_by_speaker_lines(lines, speaker_matches)
        mode = "speaker-labeled lines"
    elif any(timecode_range_hits):
        segments = _segment_by_timecode_range_lines(lines, timecode_range_hits)
        mode = ("timecode-range lines (SRT/broadcast-style) -- "
                "timecodes and any header text before the first one discarded")
    elif lines and sum(1 for m in timestamp_matches if m) > len(lines) / 2:
        segments = _segment_by_timestamped_lines(timestamp_matches)
        mode = ("timestamped caption fragments (YouTube-style) -- "
                "timestamps discarded, real timing comes from forced alignment in Phase 3")
    else:
        segments = _segment_by_sentences(raw_text)
        mode = "no speaker labels or timestamps detected -- sentence-split, speaker set to 'Unknown'"

    print(f"Segmented into {len(segments)} segments ({mode})")
    return {"video_id": video_id, "segments": segments}


def _segment_by_speaker_lines(lines: List[str], matches: List[Optional[re.Match]]) -> List[dict]:
    segments: List[dict] = []
    for line, match in zip(lines, matches):
        if match:
            speaker, text = match.group(1).strip(), match.group(2).strip()
            segments.append({
                "id": f"seg_{len(segments) + 1:03d}",
                "speaker": speaker,
                "text": text,
                "make_clip": False,
            })
        elif segments:
            # Continuation line, no new speaker label -- the same turn
            # wrapping across lines, not a new segment.
            segments[-1]["text"] += " " + line
        # else: a stray line before any speaker label appears at all;
        # dropped rather than guessed at.
    return segments


def _segment_by_timecode_range_lines(lines: List[str], is_timecode_line: List[bool]) -> List[dict]:
    # Timecode-range lines mark caption-chunk boundaries; the text for
    # each chunk is whatever comes after it, up to the next timecode
    # line. Anything before the FIRST timecode line (a title, a
    # "Narration: ..." header) is header content, not spoken text --
    # dropped rather than guessed at, same rule as every other path here.
    text_fragments = []
    seen_first_timecode = False
    for line, is_timecode in zip(lines, is_timecode_line):
        if is_timecode:
            seen_first_timecode = True
            continue
        if seen_first_timecode:
            text_fragments.append(line)
    full_text = " ".join(text_fragments)
    return _segment_by_sentences(full_text)


def _segment_by_timestamped_lines(matches: List[Optional[re.Match]]) -> List[dict]:
    # Reassemble into one continuous block, then re-split on real
    # sentence boundaries -- each line is a caption-chunk fragment, not
    # a sentence. Non-matching lines (a title, a wrapper sentence some
    # tool prepended) are dropped rather than guessed at, same rule as
    # the speaker-line path's stray preamble.
    fragments = [m.group(1) for m in matches if m]
    full_text = " ".join(fragments)
    return _segment_by_sentences(full_text)


def _segment_by_sentences(raw_text: str, speaker: str = "Unknown") -> List[dict]:
    # Deliberately simple for a first version -- a plain sentence-
    # boundary split, not embedding-based topic detection. Upgradeable
    # later without touching anything downstream, since this function's
    # only contract is producing valid segments.
    sentences = re.split(r'(?<=[.!?])\s+', raw_text.strip())
    return [
        {"id": f"seg_{i + 1:03d}", "speaker": speaker, "text": s.strip(), "make_clip": False}
        for i, s in enumerate(sentences) if s.strip()
    ]


