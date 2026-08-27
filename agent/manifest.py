"""
Phase 5 (part 1): write manifest.json -- what clips exist, and where each
one came from.

Built from clip_records (cutter.py's output) rather than re-derived from
`resolved` -- a clip's provenance should describe the clip that actually
got cut, not be recomputed from data that may have moved on since.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List


def write_manifest(video_id: str, source_video: Path, clip_records: List[dict], manifest_path: Path) -> dict:
    manifest = {
        "video_id": video_id,
        "source_video": str(source_video),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "clips": clip_records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path} ({len(clip_records)} clips)")
    return manifest
