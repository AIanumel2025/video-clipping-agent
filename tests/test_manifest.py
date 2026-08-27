"""Tests for manifest.py -- manifest.json writing."""

import json

from agent.manifest import write_manifest


def test_write_manifest_persists_expected_shape(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    clip_records = [{"clip_id": "clip_001", "segment_id": "s1", "filename": "clip_001.mp4"}]
    manifest = write_manifest("vid1", tmp_path / "video.mp4", clip_records, manifest_path)

    assert manifest["video_id"] == "vid1"
    assert manifest["clips"] == clip_records
    assert "generated_at" in manifest

    on_disk = json.loads(manifest_path.read_text())
    assert on_disk == manifest


def test_write_manifest_handles_zero_clips(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest = write_manifest("vid1", tmp_path / "video.mp4", [], manifest_path)
    assert manifest["clips"] == []
