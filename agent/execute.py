"""
Stage 2, Phase 3, Step 2: execute the decisions from Step 1 -- do the
small realignment, compute the shift, apply it.

Reads `action` from each decision (decide.py) as authoritative. This
module does not decide REUSE vs VERIFY_SHIFT or PATCH vs FULL_REALIGN --
it only works out the resulting (mocked) timing for whichever action was
already decided, and propagates the cumulative timing delta downstream.

Duration changes for PATCH/FULL_REALIGN/NEW are MOCKED: estimated from a
words-per-second rate derived from Stage 1's real alignment, not from an
actual windowed whisperx re-alignment (no new audio exists yet for a
purely-synthetic v2 transcript). Swap in a real windowed alignment call
once real audio exists -- everything downstream (delta propagation,
cutting, manifest) doesn't care how the number was produced, only that
each decision carries one.
"""

from typing import List


def estimate_words_per_second(old_state: dict, old_segments: List[dict]) -> float:
    """Grounded mock rate, derived from Stage 1's real alignment output --
    not an invented constant."""
    total_duration = sum(
        v["end"] - v["start"] for v in old_state["segments"].values()
        if v["start"] is not None and v["end"] is not None
    )
    total_words = sum(len(s["text"].split()) for s in old_segments)
    rate = total_duration / total_words
    print(f"Mock rate: {rate:.3f} sec/word (derived from real Stage 1 alignment, not invented)\n")
    return rate


def execute_decisions(decisions: List[dict], old_state: dict, avg_sec_per_word: float) -> dict:
    resolved_v2 = []
    cumulative_delta = 0.0

    for d in decisions:
        action = d["action"]

        if action in ("REUSE", "VERIFY_SHIFT") and d["kind"] == "unchanged":
            old = old_state["segments"].get(d["old_id"], {})
            new_start, new_end = old.get("start"), old.get("end")
            if new_start is not None and new_end is not None:
                new_start += cumulative_delta
                new_end += cumulative_delta
            resolved_v2.append({"id": d["new_id"], "old_id": d["old_id"], "text": d["new_text"],
                                 "start": new_start, "end": new_end,
                                 "action": action, "reason": d["reason"]})

        elif action in ("PATCH", "FULL_REALIGN"):
            old = old_state["segments"].get(d["old_id"], {})
            word_delta = len(d["new_text"].split()) - len(d["old_text"].split())
            mock_duration_change = round(word_delta * avg_sec_per_word, 2)
            reason = f"{d['reason']}, mock duration change {mock_duration_change:+.2f}s (MOCKED -- pending real windowed re-alignment)"

            if old.get("start") is None or old.get("end") is None:
                new_start, new_end = None, None
                reason = f"{d['reason']}, but Stage 1 never resolved a prior timestamp -- needs fresh alignment"
            else:
                new_start = old["start"] + cumulative_delta
                new_end = old["end"] + cumulative_delta + mock_duration_change
                cumulative_delta += mock_duration_change

            resolved_v2.append({"id": d["new_id"], "old_id": d["old_id"], "text": d["new_text"],
                                 "start": new_start, "end": new_end,
                                 "action": action, "reason": reason})

        elif action == "NEW":
            mock_duration = round(len(d["new_text"].split()) * avg_sec_per_word, 2)
            prev_end = resolved_v2[-1]["end"] if resolved_v2 and resolved_v2[-1]["end"] is not None else 0.0
            new_start = prev_end + 0.3
            new_end = new_start + mock_duration
            cumulative_delta += (new_end - prev_end)
            resolved_v2.append({"id": d["new_id"], "old_id": d["old_id"], "text": d["new_text"],
                                 "start": new_start, "end": new_end, "action": "NEW",
                                 "reason": f"{d['reason']}, mock duration {mock_duration:.2f}s (MOCKED -- pending real alignment)"})

        elif action == "REMOVE":
            old = old_state["segments"].get(d["old_id"], {})
            if old.get("start") is None or old.get("end") is None:
                reason = "segment deleted (had no resolved timestamp in Stage 1 anyway)"
            else:
                removed_duration = old["end"] - old["start"]
                cumulative_delta -= removed_duration
                reason = f"segment deleted, removes {removed_duration:.2f}s from the timeline"
            resolved_v2.append({"id": d["old_id"], "old_id": d["old_id"], "text": d["old_text"], "start": None, "end": None,
                                 "action": "REMOVE", "reason": reason})

    print(f"Final cumulative delta by end of transcript: {cumulative_delta:+.2f}s\n")
    return {"resolved_v2": resolved_v2, "cumulative_delta": cumulative_delta}
