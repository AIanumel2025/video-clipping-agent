"""Tests for segment.py -- raw transcript segmentation. Pure functions,
no I/O, no models."""

from agent.segment import segment_raw_transcript


def test_speaker_labeled_lines_produce_correct_segments():
    raw = "Brewster: Record and post everywhere!\nIan: Perfect."
    result = segment_raw_transcript(raw, video_id="test_video")
    assert result["video_id"] == "test_video"
    segs = result["segments"]
    assert len(segs) == 2
    assert segs[0] == {"id": "seg_001", "speaker": "Brewster",
                        "text": "Record and post everywhere!", "make_clip": False}
    assert segs[1]["speaker"] == "Ian"
    assert segs[1]["text"] == "Perfect."


def test_continuation_lines_merge_into_previous_segment():
    raw = ("Brewster: This is a long answer\n"
           "that wraps across two lines.\n"
           "Ian: A short reply.")
    result = segment_raw_transcript(raw, video_id="test")
    segs = result["segments"]
    assert len(segs) == 2
    assert segs[0]["text"] == "This is a long answer that wraps across two lines."
    assert segs[1]["text"] == "A short reply."


def test_no_speaker_labels_falls_back_to_sentence_splitting():
    raw = "This is the first sentence. This is the second one! And a third?"
    result = segment_raw_transcript(raw, video_id="test")
    segs = result["segments"]
    assert len(segs) == 3
    assert all(s["speaker"] == "Unknown" for s in segs)
    assert segs[0]["text"] == "This is the first sentence."
    assert segs[1]["text"] == "This is the second one!"
    assert segs[2]["text"] == "And a third?"


def test_every_segment_starts_unflagged():
    raw = "Alice: Hello there.\nBob: Hi Alice."
    result = segment_raw_transcript(raw, video_id="test")
    assert all(s["make_clip"] is False for s in result["segments"])


def test_stray_line_before_any_speaker_label_is_dropped_not_guessed():
    raw = ("some preamble with no speaker label\n"
           "Alice: Now this has a speaker.\n"
           "Bob: And so does this.")
    result = segment_raw_transcript(raw, video_id="test")
    segs = result["segments"]
    assert len(segs) == 2
    assert segs[0]["speaker"] == "Alice"
    assert segs[1]["speaker"] == "Bob"


def test_single_header_style_label_does_not_hijack_a_different_format():
    """Based on the real bug this caught: a "Narration: Name" header
    line matches the speaker-line pattern on its own, but one match
    shouldn't be enough to route an entire timecode-range transcript
    down the wrong path."""
    raw = (
        "Narration: Dr. Someone\n\n"
        "Transcript:\n\n"
        "00:00:01:00 - 00:00:02:00\n\n"
        "Hello there.\n\n"
        "00:00:02:00 - 00:00:03:00\n\n"
        "General Kenobi.\n"
    )
    result = segment_raw_transcript(raw, video_id="test")
    segs = result["segments"]
    assert not any("Narration" in s["text"] or "Transcript:" in s["text"] for s in segs)
    assert not any(__import__('re').search(r'\d{2}:\d{2}:\d{2}:\d{2}', s["text"]) for s in segs)
    assert segs[0]["text"] == "Hello there."
    assert segs[1]["text"] == "General Kenobi."


def test_output_matches_ingest_schema():
    """The actual point of this module: its output must satisfy the
    same schema load_transcript() already validates -- proof this
    slots into the existing pipeline, not just an assertion of it."""
    from agent.ingest import REQUIRED_SEGMENT_FIELDS
    raw = "Alice: Testing schema compatibility."
    result = segment_raw_transcript(raw, video_id="test")
    for seg in result["segments"]:
        assert REQUIRED_SEGMENT_FIELDS.issubset(seg.keys())


def test_youtube_style_timestamps_are_detected_and_stripped():
    """Based on a real uploaded YouTube transcript export -- caption
    fragments prefixed with MM:SS, no speaker labels at all."""
    raw = (
        "0:00 Hello everybody. Welcome to another\n"
        "0:01 video. In this video, we're going to be\n"
        "0:03 talking about how to actually change\n"
    )
    result = segment_raw_transcript(raw, video_id="test")
    segs = result["segments"]
    assert len(segs) == 3
    assert segs[0]["text"] == "Hello everybody."
    assert segs[1]["text"] == "Welcome to another video."
    assert segs[2]["text"] == "In this video, we're going to be talking about how to actually change"
    assert all(s["speaker"] == "Unknown" for s in segs)
    # No stray timestamp digits leaked into any segment's text
    import re
    assert not any(re.search(r'\b\d{1,2}:\d{2}\b', s["text"]) for s in segs)


def test_non_timestamped_header_line_is_dropped_not_glued_in():
    """Based on the same real file -- a wrapper sentence some tool
    prepended, with no timestamp of its own, ahead of the real
    timestamped content."""
    raw = (
        "The transcript for the video \"Some Title\" is provided below:\n"
        "0:00 Hello everybody. Welcome to another\n"
        "0:01 video about testing.\n"
    )
    result = segment_raw_transcript(raw, video_id="test")
    segs = result["segments"]
    assert not any("is provided below" in s["text"] for s in segs)
    assert segs[0]["text"] == "Hello everybody."


def test_hour_prefixed_timestamps_are_also_matched():
    raw = (
        "1:02:03 This line comes from over an hour into\n"
        "1:02:05 the video and should still be detected.\n"
    )
    result = segment_raw_transcript(raw, video_id="test")
    segs = result["segments"]
    assert len(segs) == 1
    assert "1:02:03" not in segs[0]["text"]
    assert "1:02:05" not in segs[0]["text"]


def test_timecode_range_lines_are_detected_header_dropped_text_reassembled():
    """Based on a real NASA SME video transcript export -- broadcast
    timecode (HH:MM:SS:FF) as a bare range on its own line, text on
    the following line, with a header block before any content starts."""
    raw = (
        "Dr. Someone SME Transcript\n\n"
        "Narration: Dr. Someone\n\n"
        "Transcript:\n\n"
        "00:00:44:18 - 00:00:45:22\n\n"
        "The Earth is\n\n"
        "00:00:45:22 - 00:00:46:07\n\n"
        "the most\n\n"
        "00:00:46:07 - 00:00:47:16\n\n"
        "important planet that we study,\n\n"
        "00:00:47:16 - 00:00:49:21\n\n"
        "at least in my opinion.\n"
    )
    result = segment_raw_transcript(raw, video_id="test")
    segs = result["segments"]
    assert len(segs) == 1
    assert segs[0]["text"] == "The Earth is the most important planet that we study, at least in my opinion."
    assert segs[0]["speaker"] == "Unknown"
