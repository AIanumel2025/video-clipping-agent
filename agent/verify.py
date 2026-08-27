"""
Spot-check by ear. Two distinct checks, both interactive/notebook-only
(IPython.display), not part of the scriptable pipeline path:

  spot_check_segments -- cuts fresh snippets from raw audio around
  resolved segments, before clips exist. Used right after alignment.

  spot_check_clips -- plays back the actual, already-generated clip
  files against manifest.json. Works against any manifest.json and
  clips directory, from any run of either pipeline, for any video --
  nothing here is tied to a specific transcript or source.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


def spot_check_segments(resolved_by_id: Dict[str, dict], audio_path: Path, run_dir: Path,
                         check_ids: List[str], pad: float = 0.3) -> None:
    from IPython.display import Audio, display

    checks_dir = run_dir / "spot_checks"
    checks_dir.mkdir(exist_ok=True)

    print(f"Checking {len(check_ids)} segments")
    for check_id in check_ids:
        r = resolved_by_id[check_id]
        start, end = r["start"], r["end"]
        if start is None or end is None:
            print(f"\n{check_id}: unresolved, skipping spot check")
            continue

        clip_start = max(0, start - pad)
        clip_duration = (end + pad) - clip_start
        snippet_path = checks_dir / f"{check_id}.wav"

        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(clip_start), "-i", str(audio_path),
             "-t", str(clip_duration), str(snippet_path)],
            check=True, capture_output=True,
        )

        print(f"\n{check_id}  [{start:.2f}s - {end:.2f}s]  duration={clip_duration:.2f}s  confidence={r['confidence']}")
        print(f"Expected text: \"{r['text']}\"")
        display(Audio(str(snippet_path)))


def spot_check_clips(manifest_path: Path, clips_dir: Path,
                      segment_ids: Optional[List[str]] = None, n: int = 5) -> None:
    """Play back real, already-cut clip files for a manual sanity check,
    reading straight from a manifest.json -- works for any run, any
    video, any transcript, since it only depends on the two artifacts
    every pipeline run produces.

    If segment_ids isn't given, samples the first n clips listed in the
    manifest, so this is usable with zero arguments for a quick check.
    """
    from IPython.display import Audio, display

    manifest = json.loads(Path(manifest_path).read_text())
    clips_by_segment = {c["segment_id"]: c for c in manifest["clips"]}

    if segment_ids is None:
        segment_ids = list(clips_by_segment.keys())[:n]

    print(f"Spot-checking {len(segment_ids)} of {len(manifest['clips'])} clips from {Path(manifest_path).name}\n")

    for segment_id in segment_ids:
        record = clips_by_segment.get(segment_id)
        if record is None:
            print(f"{segment_id}: not in this manifest, skipping\n")
            continue

        clip_path = Path(clips_dir) / record["filename"]
        if not clip_path.exists():
            print(f"{segment_id}: manifest references {record['filename']}, but the file is missing\n")
            continue

        print(f"{segment_id}  [{record['start']:.2f}s - {record['end']:.2f}s]  "
              f"duration={record['duration']:.2f}s  confidence={record.get('confidence', 'n/a')}")
        print(f"Clip title:    \"{record.get('clip_title', '(untitled)')}\"")
        print(f"Expected text: \"{record.get('text', '(not stored in this manifest)')}\"")
        display(Audio(str(clip_path)))
        print()

