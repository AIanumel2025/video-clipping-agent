"""
Phase 4: cut the video into individual clip files.

Two details matter here, both purely engineering, not alignment-related:

1. Padding -- resolved timestamps mark exactly where aligned words begin
   and end, which is a little too exact for a clip to feel natural. A
   clip is padded (0.15s before, 0.35s after -- more generous on the
   tail so trailing consonants don't get chopped) purely for perceptual
   smoothness. This is cosmetic breathing room, not a correction to the
   alignment itself. Unlike LEAD_IN_SEC (Phase 3), these two numbers are
   the same for every video -- there's no video-specific reason for the
   trade-off to differ -- so they stay as sensible defaults rather than
   something each run needs to configure.

2. Seeking -- ffmpeg can jump to a timestamp two ways: before decoding
   starts (fast, but can only land on the nearest keyframe, sometimes a
   second or more off), or after opening the file and decoding forward
   to the exact frame (slower, frame-accurate). -ss placed after -i (as
   below) is the accurate method, and the output is re-encoded rather
   than stream-copied, since stream-copying can only cut on keyframe
   boundaries too.
"""

import subprocess
from pathlib import Path
from typing import List

CLIP_PAD_START = 0.15
CLIP_PAD_END = 0.35


def cut_clip(source_video: Path, start: float, end: float, out_path: Path,
             pad_start: float = CLIP_PAD_START, pad_end: float = CLIP_PAD_END) -> None:
    padded_start = max(0.0, start - pad_start)
    padded_end = end + pad_end
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(source_video),
         "-ss", str(padded_start), "-to", str(padded_end),
         "-c:v", "libx264", "-c:a", "aac", "-avoid_negative_ts", "make_zero",
         str(out_path)],
        check=True, capture_output=True,
    )


def cut_all_clips(resolved: List[dict], video_path: Path, clips_dir: Path,
                   pad_start: float = CLIP_PAD_START, pad_end: float = CLIP_PAD_END) -> List[dict]:
    """Cut every make_clip=true, resolved segment into its own file, and
    return the in-memory manifest records for them (clip -> segment ->
    title -> timing -> confidence). Writing manifest.json to disk is a
    separate step (manifest.py), not this function's job.
    """
    clips_dir.mkdir(parents=True, exist_ok=True)

    clip_records = []
    clip_index = 0
    skipped_unresolved = []

    for r in resolved:
        if not r["make_clip"]:
            continue
        if r["start"] is None or r["end"] is None:
            skipped_unresolved.append(r["id"])
            print(f"SKIPPED (unresolved): {r['id']} -- \"{r['text'][:50]}\" -- "
                  f"flagged for a clip but alignment never resolved a timestamp for it")
            continue

        clip_index += 1
        filename = f"clip_{clip_index:03d}.mp4"
        out_path = clips_dir / filename
        cut_clip(video_path, r["start"], r["end"], out_path, pad_start, pad_end)

        title = r["clip_title"] or f"(untitled: {r['id']})"
        clip_records.append({
            "clip_id": filename.replace(".mp4", ""),
            "filename": filename,
            "segment_id": r["id"],
            "clip_title": title,
            "start": round(r["start"], 3),
            "end": round(r["end"], 3),
            "duration": round(r["end"] - r["start"], 3),
            "confidence": r["confidence"],
            "text": r["text"],
        })
        print(f"Cut {filename}  ({clip_records[-1]['duration']:.2f}s)  \"{title}\"")

    print(f"\nCut {len(clip_records)} clips into {clips_dir}"
          + (f" ({len(skipped_unresolved)} skipped as unresolved)" if skipped_unresolved else ""))
    return clip_records
