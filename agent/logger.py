"""
Stage 2, Phase 3, Step 3: write decisions.log -- a human-readable
reasoning trail, one entry per run, appended (not overwritten) so the
full history of past incremental runs stays readable in one place.

Because execute.py no longer re-decides actions (see decide.py), the
Summary line here and the itemized entries below it are guaranteed to
agree -- both trace back to the same `action` per segment, decided once.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List


def append_decisions_log(resolved_v2: List[dict], old_path: Path, new_path: Path,
                          summary: dict, cumulative_delta: float, log_path: Path) -> None:
    with open(log_path, "a") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"Run at: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Old transcript: {old_path.name}\n")
        f.write(f"New transcript: {new_path.name}\n")
        f.write(f"Summary: {summary}\n")
        f.write(f"Final cumulative delta: {cumulative_delta:+.2f}s\n")
        f.write(f"{'='*70}\n\n")

        for r in resolved_v2:
            span = f"[{r['start']:.2f}s - {r['end']:.2f}s]" if r["start"] is not None else ""
            f.write(f"Decision: {r['action']:<12} {r['id']}  {span}\n")
            f.write(f"Reason:   {r['reason']}\n\n")

    print(f"Appended {len(resolved_v2)} decision entries to {log_path}")
