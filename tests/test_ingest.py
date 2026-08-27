"""Tests for ingest.py -- transcript loading/validation, path resolution."""

import json

import pytest

from agent.ingest import (
    load_transcript, TranscriptValidationError, validate_paired_transcripts,
    resolve_old_transcript_path, resolve_run_dir,
)


def write_json(path, data):
    path.write_text(json.dumps(data))
    return path


def test_load_transcript_accepts_well_formed_file(tmp_path):
    path = write_json(tmp_path / "t.json", {
        "video_id": "vid1",
        "segments": [{"id": "s1", "speaker": "A", "text": "hi", "make_clip": False}],
    })
    transcript = load_transcript(path)
    assert transcript["video_id"] == "vid1"


def test_load_transcript_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_transcript(tmp_path / "does_not_exist.json")


def test_load_transcript_missing_segments_key_raises_clear_error(tmp_path):
    path = write_json(tmp_path / "t.json", {"video_id": "vid1"})
    with pytest.raises(TranscriptValidationError, match="missing top-level 'segments'"):
        load_transcript(path)


def test_load_transcript_reports_every_malformed_segment_specifically(tmp_path):
    path = write_json(tmp_path / "t.json", {
        "video_id": "vid1",
        "segments": [
            {"id": "s1", "speaker": "A", "text": "ok", "make_clip": False},
            {"id": "s2", "speaker": "A"},  # missing text, make_clip
        ],
    })
    with pytest.raises(TranscriptValidationError) as exc_info:
        load_transcript(path)
    message = str(exc_info.value)
    assert "s2" in message
    assert "'make_clip'" in message and "'text'" in message
    assert "id=s1" not in message  # the well-formed segment shouldn't be flagged


def test_validate_paired_transcripts_passes_when_video_ids_match():
    validate_paired_transcripts({"video_id": "vid1"}, {"video_id": "vid1"})  # should not raise


def test_validate_paired_transcripts_raises_on_mismatch():
    with pytest.raises(TranscriptValidationError, match="video_id mismatch"):
        validate_paired_transcripts({"video_id": "vid1"}, {"video_id": "vid2"})


def test_resolve_old_transcript_path_prefers_committed_snapshot(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_json(input_dir / "transcript_flagged.json", {"video_id": "vid1", "segments": []})
    write_json(run_dir / "current_transcript.json", {"video_id": "vid1", "segments": []})

    result = resolve_old_transcript_path(input_dir, run_dir)
    assert result == run_dir / "current_transcript.json"


def test_resolve_old_transcript_path_falls_back_to_original_on_first_run(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_json(input_dir / "transcript_flagged.json", {"video_id": "vid1", "segments": []})
    # no current_transcript.json yet -- this is the first incremental run

    result = resolve_old_transcript_path(input_dir, run_dir)
    assert result == input_dir / "transcript_flagged.json"


def test_resolve_old_transcript_path_raises_when_neither_exists(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_old_transcript_path(input_dir, run_dir)


def test_resolve_run_dir_uses_video_id(tmp_path):
    run_dir = resolve_run_dir({"video_id": "brewster_kahle"}, tmp_path)
    assert run_dir == tmp_path / "brewster_kahle"
    assert run_dir.exists()


def test_resolve_run_dir_requires_video_id(tmp_path):
    with pytest.raises(TranscriptValidationError):
        resolve_run_dir({}, tmp_path)
