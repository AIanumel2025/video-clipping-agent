"""
Spot-check resolved alignment by ear -- cuts short audio snippets around
chosen segments and plays them inline for comparison against expected
text. Inherently an interactive, notebook-only step (uses
IPython.display), not part of the scriptable pipeline path.
"""

import subprocess
from pathlib import Path
from typing import Dict, List


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
