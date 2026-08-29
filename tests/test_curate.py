"""Tests for curate.py -- prompt construction and response parsing are
pure functions, testable with a canned response and no API key or
model call at all. call_curator_model itself isn't unit-tested here,
same reasoning as align.py's whisperx calls and embeddings.py's
sentence-transformers loading -- it's Tier 2 / real-run territory.
"""

from agent.curate import build_curation_prompt, parse_curator_response, DEFAULT_CRITERIA


def seg(id_, text, speaker="Unknown"):
    return {"id": id_, "speaker": speaker, "text": text, "make_clip": False}


def test_prompt_includes_every_segment_id_and_text():
    segments = [seg("seg_001", "First sentence."), seg("seg_002", "Second sentence.")]
    prompt = build_curation_prompt(segments)
    assert "[seg_001] First sentence." in prompt
    assert "[seg_002] Second sentence." in prompt


def test_prompt_includes_custom_criteria_when_given():
    segments = [seg("seg_001", "Hello.")]
    prompt = build_curation_prompt(segments, criteria="Only pick segments about cats.")
    assert "Only pick segments about cats." in prompt


def test_prompt_uses_default_criteria_when_not_given():
    segments = [seg("seg_001", "Hello.")]
    prompt = build_curation_prompt(segments)
    assert DEFAULT_CRITERIA in prompt


def test_single_segment_span_becomes_one_clip():
    segments = [seg("seg_001", "First."), seg("seg_002", "Second."), seg("seg_003", "Third.")]
    response = {"clips": [{"start_segment_id": "seg_002", "end_segment_id": "seg_002",
                            "clip_title": "The Second One"}]}
    result = parse_curator_response(response, segments)
    assert len(result) == 3
    clip = next(s for s in result if s["id"] == "seg_002")
    assert clip["make_clip"] is True
    assert clip["text"] == "Second."
    assert clip["clip_title"] == "The Second One"
    assert all(not s["make_clip"] for s in result if s["id"] != "seg_002")


def test_multi_segment_span_merges_into_one_clip_at_start_id():
    segments = [seg("seg_001", "Intro."), seg("seg_002", "The real"),
                seg("seg_003", "point here."), seg("seg_004", "Outro.")]
    response = {"clips": [{"start_segment_id": "seg_002", "end_segment_id": "seg_003",
                            "clip_title": "The Real Point"}]}
    result = parse_curator_response(response, segments)
    # seg_002 and seg_003 collapse into one merged clip at seg_002's id;
    # seg_003 no longer appears as its own separate entry.
    ids = [s["id"] for s in result]
    assert ids == ["seg_001", "seg_002", "seg_004"]
    clip = next(s for s in result if s["id"] == "seg_002")
    assert clip["text"] == "The real point here."
    assert clip["make_clip"] is True


def test_unselected_segments_stay_unmerged_and_unflagged():
    segments = [seg("seg_001", "Not picked."), seg("seg_002", "Picked.")]
    response = {"clips": [{"start_segment_id": "seg_002", "end_segment_id": "seg_002",
                            "clip_title": "Picked One"}]}
    result = parse_curator_response(response, segments)
    untouched = next(s for s in result if s["id"] == "seg_001")
    assert untouched["make_clip"] is False
    assert untouched["text"] == "Not picked."


def test_unknown_segment_id_is_skipped_not_crashed_on():
    segments = [seg("seg_001", "Real segment.")]
    response = {"clips": [{"start_segment_id": "seg_999", "end_segment_id": "seg_999",
                            "clip_title": "Doesn't exist"}]}
    result = parse_curator_response(response, segments)
    assert all(not s["make_clip"] for s in result)


def test_reversed_span_is_skipped_not_crashed_on():
    segments = [seg("seg_001", "A."), seg("seg_002", "B."), seg("seg_003", "C.")]
    response = {"clips": [{"start_segment_id": "seg_003", "end_segment_id": "seg_001",
                            "clip_title": "Backwards"}]}
    result = parse_curator_response(response, segments)
    assert all(not s["make_clip"] for s in result)


def test_overlapping_spans_second_one_is_skipped():
    segments = [seg("seg_001", "A."), seg("seg_002", "B."), seg("seg_003", "C.")]
    response = {"clips": [
        {"start_segment_id": "seg_001", "end_segment_id": "seg_002", "clip_title": "First"},
        {"start_segment_id": "seg_002", "end_segment_id": "seg_003", "clip_title": "Overlapping"},
    ]}
    result = parse_curator_response(response, segments)
    clips = [s for s in result if s["make_clip"]]
    assert len(clips) == 1
    assert clips[0]["clip_title"] == "First"


def test_consistent_speaker_across_span_is_preserved():
    segments = [seg("seg_001", "Part one.", speaker="Alice"),
                seg("seg_002", "Part two.", speaker="Alice")]
    response = {"clips": [{"start_segment_id": "seg_001", "end_segment_id": "seg_002",
                            "clip_title": "Alice's Point"}]}
    result = parse_curator_response(response, segments)
    clip = next(s for s in result if s["make_clip"])
    assert clip["speaker"] == "Alice"


def test_mixed_speakers_across_span_fall_back_to_unknown():
    segments = [seg("seg_001", "Alice speaks.", speaker="Alice"),
                seg("seg_002", "Bob replies.", speaker="Bob")]
    response = {"clips": [{"start_segment_id": "seg_001", "end_segment_id": "seg_002",
                            "clip_title": "Exchange"}]}
    result = parse_curator_response(response, segments)
    clip = next(s for s in result if s["make_clip"])
    assert clip["speaker"] == "Unknown"


def test_missing_clip_title_gets_a_fallback_not_a_crash():
    segments = [seg("seg_001", "Hello.")]
    response = {"clips": [{"start_segment_id": "seg_001", "end_segment_id": "seg_001"}]}
    result = parse_curator_response(response, segments)
    clip = next(s for s in result if s["make_clip"])
    assert "seg_001" in clip["clip_title"]


def test_no_clips_proposed_leaves_everything_unflagged():
    segments = [seg("seg_001", "Nothing special."), seg("seg_002", "Also nothing.")]
    result = parse_curator_response({"clips": []}, segments)
    assert all(not s["make_clip"] for s in result)
    assert len(result) == 2


def test_output_matches_ingest_schema():
    """Same contract test as segment.py's -- the curator's output must
    satisfy the schema load_transcript() already validates."""
    from agent.ingest import REQUIRED_SEGMENT_FIELDS
    segments = [seg("seg_001", "Hello."), seg("seg_002", "World.")]
    response = {"clips": [{"start_segment_id": "seg_001", "end_segment_id": "seg_001",
                            "clip_title": "Greeting"}]}
    result = parse_curator_response(response, segments)
    for s in result:
        assert REQUIRED_SEGMENT_FIELDS.issubset(s.keys())
