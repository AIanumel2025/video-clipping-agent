"""
Tier 2: fixture-based smoke test.

Exercises the real pipeline wiring end to end -- ffmpeg cutting, JSON
I/O, and the full diff/decide/execute/clip-reuse/commit logic -- for
both a baseline run and a follow-up incremental run, against a tiny
synthetic fixture (see scripts/generate_fixtures.py).

Two things are deliberately NOT real here, since they need models this
suite is designed to run without:
  - align.py's transcribe_audio_cached() and force_align() (need
    whisperx) -- word-level timing is injected directly from the
    fixture's known ground truth instead, then fed through the REAL
    flatten_words() / attribute_words_to_segments().
  - embeddings.py's model loading (needs sentence-transformers) -- the
    one "edited" segment's embeddings are seeded directly into the
    cache (agent.embeddings.seed_cache), same trick the Tier 1
    decide.py tests use.

Everything else -- ffmpeg cutting, manifest/state/log writing, and the
diff/decide/execute/clip-reuse/commit modules -- runs unmodified.

The run-2 edit is deliberately spread across 6 segments with unchanged
"anchor" segments between each change (see the comment below), because
a bare delete+insert landing in the same gap gets classified by
SequenceMatcher as a same-position "replace" -- i.e. paired together as
an edit, not recognized as separate delete+insert. Spacing them out with
anchors is what makes this fixture actually exercise all five decision
actions distinctly instead of accidentally collapsing two of them.
"""

import json
import math
import shutil

import pytest

from scripts.generate_fixtures import generate_fixtures
from agent.ingest import load_transcript, resolve_run_dir, validate_paired_transcripts, resolve_old_transcript_path
from agent.locate import select_clip_segments
from agent.attribute import flatten_words, attribute_words_to_segments
from agent.cutter import cut_all_clips
from agent.manifest import write_manifest
from agent.state import write_state
from agent.diff import classify_segments
from agent.embeddings import seed_cache
from agent.decide import decide_actions
from agent.execute import execute_decisions, estimate_words_per_second
from agent.clip_reuse import decide_clip_actions, load_old_manifest
from agent.commit import commit_run

pytestmark = pytest.mark.smoke


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _fake_words_for_segment(text: str, start: float, end: float) -> list:
    """Stand-in for real forced alignment: evenly spaces this segment's
    words across its known ground-truth span."""
    words = text.split()
    per_word = (end - start) / len(words)
    return [
        {"word": w, "start": round(start + i * per_word, 3), "end": round(start + (i + 1) * per_word, 3)}
        for i, w in enumerate(words)
    ]


def _fake_aligned_result(segments: list, ground_truth: dict) -> dict:
    return {"segments": [
        {"words": _fake_words_for_segment(seg["text"], ground_truth[seg["id"]]["start"], ground_truth[seg["id"]]["end"])}
        for seg in segments
    ]}


def _seed_similarity(cache_dir, text_a: str, text_b: str, similarity: float) -> None:
    vec_a = [1.0, 0.0]
    vec_b = [similarity, math.sqrt(max(0.0, 1.0 - similarity ** 2))]
    seed_cache(text_a, vec_a, cache_dir)
    seed_cache(text_b, vec_b, cache_dir)


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg/ffprobe not installed")
def test_full_pipeline_wiring(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    fixtures = generate_fixtures(input_dir)

    # ---------- Run 1 (baseline) ----------
    transcript = load_transcript(fixtures["transcript_path"])
    flagged_transcript = load_transcript(fixtures["flagged_path"])
    validate_paired_transcripts(transcript, flagged_transcript)
    assert len(select_clip_segments(flagged_transcript)) == 4  # all but seg_2

    run_dir = resolve_run_dir(transcript, output_dir)
    video_path = fixtures["video_path"]

    aligned_result = _fake_aligned_result(flagged_transcript["segments"], fixtures["ground_truth"])
    all_words = flatten_words(aligned_result)
    resolved = attribute_words_to_segments(flagged_transcript["segments"], all_words)
    assert all(r["confidence"] == "aligned" for r in resolved)
    for r in resolved:
        gt = fixtures["ground_truth"][r["id"]]
        assert r["start"] == pytest.approx(gt["start"], abs=0.01)
        assert r["end"] == pytest.approx(gt["end"], abs=0.01)

    clips_dir = run_dir / "clips"
    clip_records = cut_all_clips(resolved, video_path, clips_dir)
    assert len(clip_records) == 4
    for record in clip_records:
        clip_path = clips_dir / record["filename"]
        assert clip_path.exists() and clip_path.stat().st_size > 0

    write_manifest(transcript["video_id"], video_path, clip_records, run_dir / "manifest.json")
    write_state(transcript["video_id"], resolved, clip_records, run_dir / "state.json")
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "state.json").exists()

    # ---------- Run 2 (incremental) ----------
    # seg_1: unchanged anchor.
    # seg_2: edited (+2 words) -- not itself a clip, but its mocked
    #        duration change should propagate as a shift to everything after.
    # seg_3: unchanged anchor, downstream of seg_2's edit -> VERIFY_SHIFT -> RECUT.
    # seg_4: deleted -> REMOVE -> RETIRE_CLIP.
    # seg_5: unchanged anchor, downstream of BOTH seg_2's edit and seg_4's
    #        deletion -> VERIFY_SHIFT -> RECUT (compound delta, not just seg_2's).
    # seg_6: inserted (after the seg_5 anchor, its own separate gap) -> NEW -> CUT_NEW.
    old_segs = flagged_transcript["segments"]
    new_segments = [
        old_segs[0],  # seg_1
        {**old_segs[1], "text": "this is now the second segment here"},  # seg_2, 5->7 words
        old_segs[2],  # seg_3
        # seg_4 omitted -- deleted
        old_segs[4],  # seg_5
        {"id": "seg_6", "speaker": "A", "text": "brand new final segment here",
         "make_clip": True, "clip_title": "Sixth"},
    ]
    new_transcript = {"video_id": transcript["video_id"], "segments": new_segments}
    new_path = input_dir / "transcript_flagged_new.json"
    new_path.write_text(json.dumps(new_transcript, indent=2))

    old_path = resolve_old_transcript_path(input_dir, run_dir)
    assert old_path == fixtures["flagged_path"]  # first incremental run, no commit yet
    old_transcript = load_transcript(old_path)
    old_segments = old_transcript["segments"]
    old_state = json.loads((run_dir / "state.json").read_text())

    classifications = classify_segments(old_segments, new_segments)
    kinds = {(c["new_id"] or c["old_id"]): c["kind"] for c in classifications}
    assert kinds == {
        "seg_1": "unchanged", "seg_2": "edited", "seg_3": "unchanged",
        "seg_4": "deleted", "seg_5": "unchanged", "seg_6": "inserted",
    }

    embed_cache_dir = run_dir / "cache" / "embeddings"
    _seed_similarity(embed_cache_dir, "this is the second segment",
                      "this is now the second segment here", similarity=0.9)

    decisions = decide_actions(classifications, old_state, embed_cache_dir)
    decisions_by_id = {(d["new_id"] or d["old_id"]): d["action"] for d in decisions}
    assert decisions_by_id == {
        "seg_1": "REUSE", "seg_2": "PATCH", "seg_3": "VERIFY_SHIFT",
        "seg_4": "REMOVE", "seg_5": "VERIFY_SHIFT", "seg_6": "NEW",
    }

    avg_sec_per_word = estimate_words_per_second(old_state, old_segments)
    execution = execute_decisions(decisions, old_state, avg_sec_per_word)
    resolved_v2 = execution["resolved_v2"]
    assert execution["cumulative_delta"] != 0.0  # a real net shift happened

    old_manifest = load_old_manifest(run_dir / "manifest.json")
    clip_decisions = decide_clip_actions(resolved_v2, new_segments, old_manifest)
    clip_decisions_by_id = {d["id"]: d["action"] for d in clip_decisions}
    assert clip_decisions_by_id == {
        "seg_1": "REUSE_CLIP", "seg_3": "RECUT", "seg_4": "RETIRE_CLIP",
        "seg_5": "RECUT", "seg_6": "CUT_NEW",
    }
    # seg_2 was never flagged for a clip -- correctly produces no clip decision at all
    assert "seg_2" not in clip_decisions_by_id

    commit_run(
        clip_decisions, resolved_v2, classifications,
        old_manifest, old_state, new_transcript,
        transcript["video_id"], video_path, run_dir, new_path,
    )

    assert (run_dir / "manifest_v1.json").exists()  # backed up before overwrite
    assert (run_dir / "current_transcript.json").exists()
    committed = json.loads((run_dir / "current_transcript.json").read_text())
    assert [s["id"] for s in committed["segments"]] == ["seg_1", "seg_2", "seg_3", "seg_5", "seg_6"]

    retired_path = clips_dir / "_retired" / "clip_003.mp4"  # seg_4 was the 3rd flagged segment in run 1
    assert retired_path.exists()

    manifest_v2 = json.loads((run_dir / "manifest.json").read_text())
    filenames_by_segment = {c["segment_id"]: c["filename"] for c in manifest_v2["clips"]}
    assert set(filenames_by_segment.keys()) == {"seg_1", "seg_3", "seg_5", "seg_6"}
    for segment_id, filename in filenames_by_segment.items():
        clip_path = clips_dir / filename
        assert clip_path.exists() and clip_path.stat().st_size > 0

    state_v2 = json.loads((run_dir / "state.json").read_text())
    assert len(state_v2["run_history"]) == 2  # run 1 + run 2, both recorded
    assert "seg_4" not in state_v2["segments"]  # deleted segments don't persist forward
