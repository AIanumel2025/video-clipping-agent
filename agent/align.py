"""
Phase 3: determine timestamps via forced alignment.

Two stages, in order:
  1. A blind ASR pass over the raw audio (whisperx transcribe), completely
     independent of our own transcript text. Cached by audio file hash --
     since the underlying audio doesn't change between transcript edits,
     this step only needs to run once ever per video, regardless of how
     many times the transcript itself gets revised. This is a different
     cache from the per-segment "alignment cache" the build plan
     describes; it's a lower-level input feeding into it, so it's kept
     under its own name (transcribe_audio_cached) to avoid conflating
     the two.
  2. Our own known transcript text is diff-matched against that blind ASR
     output (word-level, via difflib.SequenceMatcher) to get a window per
     segment, which then seeds whisperx's CTC forced-alignment pass --
     snapping our exact, already-correct text onto the audio, rather than
     trusting whatever the blind ASR pass thought was said.

A second, simpler windowing approach also exists here: estimate_rough_
boundaries(), a proportional word-count-based guess with a configurable
lead-in offset. It isn't part of the default chain below -- the
diff-matched windows from estimate_segment_windows() are what actually
feed force_align() -- but it's kept as a fallback for cases where the
ASR-diff match isn't available or reliable, since downstream code not
shown in this stage still depends on it. (The comment that used to sit
above the whisperx.align() call, describing rough_boundaries as its
input, was stale -- the code there has passed new_boundaries for a
while. Fixed here, not removed.)

Heavy dependencies (torch, whisperx) are imported inside the functions
that need them, not at module level -- so this file can be imported (and
its non-model logic unit-tested) without those packages installed.
"""

import hashlib
import json
import re
import subprocess
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional

from .config import get_device


def file_hash(path: Path, block_size: int = 8192) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()[:16]


def extract_audio(video_path: Path, run_dir: Path) -> Path:
    """Extract mono 16kHz audio from the video -- whisperx aligns against
    a waveform, not the mp4 container directly."""
    audio_path = run_dir / "interview_audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-ac", "1", "-ar", "16000", str(audio_path)],
        check=True, capture_output=True,
    )
    return audio_path


def transcribe_audio_cached(audio_path: Path, cache_dir: Path, model_size: str = "small") -> List[dict]:
    """Blind whisperx transcription + alignment of the raw audio (not
    using our own transcript text at all), cached by audio content hash.
    Returns a flat list of {"word", "start", "end"} dicts.
    """
    audio_hash = file_hash(audio_path)
    # Scoped by model_size too, not just audio hash -- different whisperx
    # model sizes can produce meaningfully different transcriptions, so a
    # cache keyed on content alone would silently serve a "small" model's
    # output even after switching to "large".
    model_dir = cache_dir / model_size
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_path = model_dir / f"asr_words_{audio_hash}.json"

    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        print(f"Loaded {len(cached)} cached ASR words for audio hash {audio_hash} -- whisperx skipped entirely")
        return cached

    import whisperx

    device = get_device()
    print(f"No cache for audio hash {audio_hash} -- running whisperx (device={device})")

    asr_model = whisperx.load_model(model_size, device, compute_type="int8" if device == "cpu" else "float16")
    asr_result = asr_model.transcribe(str(audio_path), batch_size=16)

    align_model, align_metadata = whisperx.load_align_model(language_code="en", device=device)
    audio = whisperx.load_audio(str(audio_path))
    aligned_asr = whisperx.align(
        asr_result["segments"], align_model, align_metadata, audio, device,
        return_char_alignments=False,
    )

    asr_words = []
    for seg in aligned_asr["segments"]:
        for w in seg.get("words", []):
            if w.get("start") is not None and w.get("end") is not None:
                asr_words.append({
                    "word": re.sub(r"[^a-z0-9]", "", w["word"].lower()),
                    "start": w["start"],
                    "end": w["end"],
                })

    cache_path.write_text(json.dumps(asr_words))
    print(f"Computed {len(asr_words)} ASR words, cached to {cache_path.name}")
    return asr_words


def _normalize(text: str) -> List[str]:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).split()


def estimate_rough_boundaries(segments: List[dict], total_duration: float, lead_in_sec: float = 0.0) -> List[dict]:
    """Rough, proportional starting guess for each segment's start/end,
    based on how many words it has relative to the whole transcript.
    NOT a final timestamp -- a fallback window for when the ASR-diff
    match isn't available or reliable.

    total_duration should be the video's speech-bearing duration (i.e.
    video length minus any lead-in). lead_in_sec shifts every boundary
    by a fixed offset to skip a title card or cold open before speech
    starts -- this is video-specific and can't be inferred from the
    transcript alone, so pass whatever's right for the video at hand
    rather than relying on a baked-in default.
    """
    word_counts = [len(s["text"].split()) for s in segments]
    total_words = sum(word_counts)

    rough = []
    cursor = 0.0
    for seg, wc in zip(segments, word_counts):
        frac = wc / total_words
        duration = frac * total_duration
        rough.append({
            "id": seg["id"],
            "text": seg["text"],
            "start": round(cursor + lead_in_sec, 2),
            "end": round(cursor + duration + lead_in_sec, 2),
        })
        cursor += duration
    return rough


def estimate_segment_windows(segments: List[dict], asr_words: List[dict]) -> List[dict]:
    """Diff-match our own known transcript text against the blind ASR
    word list to get a start/end window per segment. This seeds
    force_align()'s CTC refinement -- it is not the final timing.
    """
    target_words_flat: List[str] = []
    segment_ranges = []
    for seg in segments:
        words = _normalize(seg["text"])
        start_idx = len(target_words_flat)
        target_words_flat.extend(words)
        segment_ranges.append((seg, start_idx, len(target_words_flat)))

    asr_word_list = [w["word"] for w in asr_words]
    sm = SequenceMatcher(None, target_words_flat, asr_word_list, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    print(f"Global diff: {len(blocks)} matching blocks, "
          f"{sum(b.size for b in blocks)} of {len(target_words_flat)} words matched overall")

    windows = []
    last_good_end = 0.0
    n_weak = 0
    for seg, start_idx, end_idx in segment_ranges:
        overlapping = [b for b in blocks if b.a < end_idx and b.a + b.size > start_idx]

        if overlapping:
            asr_indices: List[int] = []
            for b in overlapping:
                lo = max(b.a, start_idx) - b.a + b.b
                hi = min(b.a + b.size, end_idx) - b.a + b.b
                asr_indices.extend(range(lo, hi))
            start = asr_words[min(asr_indices)]["start"]
            end = asr_words[max(asr_indices)]["end"]
            last_good_end = end
        else:
            start = last_good_end
            end = start + 1.5
            n_weak += 1
            print(f"WEAK MATCH: {seg['id']} -- {seg['text'][:50]}")

        windows.append({"id": seg["id"], "start": start, "end": end, "text": seg["text"]})

    print(f"Matched {len(windows) - n_weak} of {len(windows)} segments confidently ({n_weak} weak)")
    return windows


def force_align(windows: List[dict], audio_path: Path, device: Optional[str] = None) -> dict:
    """Snap our own known text onto the audio via whisperx's CTC forced
    alignment, using `windows` (from estimate_segment_windows) as the
    seed segments -- refined into precise word-level timestamps.
    """
    import whisperx

    device = device or get_device()
    align_model, align_metadata = whisperx.load_align_model(language_code="en", device=device)
    audio = whisperx.load_audio(str(audio_path))
    return whisperx.align(
        windows, align_model, align_metadata, audio, device,
        return_char_alignments=False,
    )
