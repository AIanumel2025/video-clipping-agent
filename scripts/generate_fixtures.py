"""
Generate a tiny synthetic video + transcript pair for the Tier 2 fixture
smoke test -- no espeak-ng, no network, just ffmpeg's built-in lavfi test
sources. Fast enough to regenerate on every test run, so nothing here
gets committed to the repo as a binary file -- same "no MP4s in the
repo" principle the real project already follows for actual clips.

Ground truth is exact by construction: each segment gets a fixed-duration
slice of the synthesized video, so there's nothing to "recognize" here.
This fixture is for exercising pipeline WIRING (ffmpeg cutting, JSON I/O,
the diff/decide/execute/commit logic), not alignment accuracy -- real
alignment accuracy was already validated separately, against real audio,
in the actual Colab runs.
"""

import json
import subprocess
from pathlib import Path

# (id, speaker, text, make_clip, clip_title)
SEGMENT_TEXTS = [
    ("seg_1", "A", "this is the first segment", True, "First"),
    ("seg_2", "B", "this is the second segment", False, None),
    ("seg_3", "A", "this is the third segment here", True, "Third"),
    ("seg_4", "B", "this is the fourth segment now", True, "Fourth"),
    ("seg_5", "A", "this is the fifth segment", True, "Fifth"),
]
SEGMENT_DURATION = 2.0  # seconds, per segment


def generate_video(video_path: Path, total_duration: float) -> None:
    """A blue test-pattern video with a tone -- valid, cuttable content;
    doesn't need to be real speech since alignment is injected from known
    ground truth, not actually run.
    """
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", f"color=c=blue:s=320x240:d={total_duration}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={total_duration}",
         "-c:v", "libx264", "-c:a", "aac", "-shortest", str(video_path)],
        check=True, capture_output=True,
    )


def generate_fixtures(output_dir: Path) -> dict:
    """Writes interview.mp4, transcript.json, transcript_flagged.json,
    and ground_truth.json into output_dir. Returns their paths plus the
    ground truth dict, for direct use without re-reading from disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = []
    ground_truth = {}
    for i, (seg_id, speaker, text, make_clip, clip_title) in enumerate(SEGMENT_TEXTS):
        start = i * SEGMENT_DURATION
        end = start + SEGMENT_DURATION
        seg = {"id": seg_id, "speaker": speaker, "text": text, "make_clip": make_clip}
        if clip_title:
            seg["clip_title"] = clip_title
        segments.append(seg)
        ground_truth[seg_id] = {"start": start, "end": end}

    total_duration = len(SEGMENT_TEXTS) * SEGMENT_DURATION
    video_path = output_dir / "interview.mp4"
    generate_video(video_path, total_duration)

    transcript = {"video_id": "fixture_smoke_test", "segments": segments}
    transcript_path = output_dir / "transcript.json"
    flagged_path = output_dir / "transcript_flagged.json"
    transcript_path.write_text(json.dumps(transcript, indent=2))
    flagged_path.write_text(json.dumps(transcript, indent=2))

    ground_truth_path = output_dir / "ground_truth.json"
    ground_truth_path.write_text(json.dumps(ground_truth, indent=2))

    return {
        "video_path": video_path, "transcript_path": transcript_path,
        "flagged_path": flagged_path, "ground_truth_path": ground_truth_path,
        "transcript": transcript, "ground_truth": ground_truth,
    }


if __name__ == "__main__":
    import sys
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures")
    result = generate_fixtures(target)
    print(f"Wrote fixtures to {target}: {result['video_path'].name}, "
          f"{result['transcript_path'].name}, {result['ground_truth_path'].name}")
