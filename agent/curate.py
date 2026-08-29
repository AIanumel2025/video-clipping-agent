"""
Stage 0.5: the curator. Reads segmented transcript text (from segment.py
or an already-curated file) and decides which spans are clip-worthy,
producing the same transcript_flagged.json shape locate.py already
consumes -- one segment per clip, make_clip + clip_title set, no
changes needed anywhere downstream.

Segments from segment.py are single-sentence and usually too short to
be a real clip on their own, so the curator's real job is identifying a
*span* of consecutive segments and merging them into one clip candidate
-- not just flagging existing segments true/false. That merge is what
keeps the existing, proven backend's 1:1 segment-to-clip assumption
intact; nothing downstream needs to know spans were ever involved.

The actual model call (call_curator_model) is deferred-imported and
isolated, same pattern as align.py's whisperx and embeddings.py's
sentence-transformers -- prompt construction and response parsing are
pure functions, testable with a canned response and no API key at all.
"""

import json
from typing import List, Optional

DEFAULT_CRITERIA = (
    "Self-contained: makes sense without additional context. "
    "Quotable or insight-dense: a strong idea, story, or claim, not filler. "
    "Roughly 15-90 seconds long when spoken aloud. "
    "Prefer complete thoughts over fragments."
)

DEFAULT_MODEL = "claude-sonnet-5"


def build_curation_prompt(segments: List[dict], criteria: str = DEFAULT_CRITERIA) -> str:
    numbered = "\n".join(f"[{s['id']}] {s['text']}" for s in segments)
    return f"""You are selecting clip-worthy moments from a video transcript.

Selection criteria:
{criteria}

Transcript segments, in order:
{numbered}

For each moment worth clipping, identify the exact range of segment IDs
it spans (a clip can be one segment or several consecutive ones) and a
short, specific title for it.

Respond with ONLY a JSON object in this exact shape, no other text:
{{"clips": [{{"start_segment_id": "seg_003", "end_segment_id": "seg_005", "clip_title": "A short specific title"}}]}}

If nothing in the transcript meets the criteria, respond with {{"clips": []}}.
"""


def parse_curator_response(response_json: dict, segments: List[dict]) -> List[dict]:
    """Merge each proposed span into one clip segment, leave everything
    else untouched with make_clip=false. Bad spans from the model
    (unknown ids, reversed ranges, overlaps) are skipped with a warning
    rather than trusted -- the same "don't crash on a bad decision,
    report it" discipline as the rest of this project.
    """
    segments_by_id = {s["id"]: s for s in segments}
    order = [s["id"] for s in segments]

    absorbed_ids = set()
    merged_segments = {}

    for span in response_json.get("clips", []):
        start_id, end_id = span.get("start_segment_id"), span.get("end_segment_id")
        if start_id not in segments_by_id or end_id not in segments_by_id:
            print(f"WARNING: curator proposed a span referencing unknown segment id(s) "
                  f"({start_id!r} - {end_id!r}) -- skipped")
            continue

        start_idx, end_idx = order.index(start_id), order.index(end_id)
        if start_idx > end_idx:
            print(f"WARNING: curator proposed a span with start after end "
                  f"({start_id} - {end_id}) -- skipped")
            continue

        span_ids = order[start_idx:end_idx + 1]
        if absorbed_ids.intersection(span_ids) or start_id in merged_segments:
            print(f"WARNING: curator proposed a span overlapping an earlier one at "
                  f"{start_id} -- skipped")
            continue

        texts = [segments_by_id[sid]["text"] for sid in span_ids]
        speakers = {segments_by_id[sid]["speaker"] for sid in span_ids}
        merged_segments[start_id] = {
            "id": start_id,
            "speaker": speakers.pop() if len(speakers) == 1 else "Unknown",
            "text": " ".join(texts),
            "make_clip": True,
            "clip_title": span.get("clip_title") or f"(untitled: {start_id})",
        }
        absorbed_ids.update(span_ids)

    final_segments = []
    for sid in order:
        if sid in merged_segments:
            final_segments.append(merged_segments[sid])
        elif sid not in absorbed_ids:
            seg = dict(segments_by_id[sid])
            seg["make_clip"] = False
            final_segments.append(seg)
        # else: absorbed into an earlier merged clip -- dropped, its
        # text already lives inside that clip's merged segment.
    return final_segments


def call_curator_model(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """The real API call -- deferred import, so this module can be
    imported (and build_curation_prompt/parse_curator_response tested)
    without the anthropic package installed or an API key present.
    """
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def curate_transcript(transcript: dict, criteria: str = DEFAULT_CRITERIA,
                       model: str = DEFAULT_MODEL) -> dict:
    segments = transcript["segments"]
    prompt = build_curation_prompt(segments, criteria)
    raw_response = call_curator_model(prompt, model=model)
    response_json = json.loads(raw_response)
    final_segments = parse_curator_response(response_json, segments)

    n_clips = sum(1 for s in final_segments if s["make_clip"])
    print(f"Curator flagged {n_clips} clip(s) from {len(segments)} segments")

    return {"video_id": transcript["video_id"], "segments": final_segments}
