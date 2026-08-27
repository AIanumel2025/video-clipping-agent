"""
Stage 1: ingest a transcript.json and a video into the pipeline.

Generalizes the original Colab cell, which located files by globbing for
`brewster_kahle_transcript.json` and `*Brewster Kahle*Interview*.mp4` under
a personal Drive folder. Neither pattern means anything for a third
party's own video, so both are replaced with explicit paths (or a
YouTube URL for the video, since the source video isn't shipped in this
repo).
"""

import json
import subprocess
from pathlib import Path
from typing import Union

REQUIRED_SEGMENT_FIELDS = {"id", "speaker", "text", "make_clip"}


class TranscriptValidationError(ValueError):
    """Raised when transcript.json doesn't match the schema the pipeline
    expects, with a specific per-segment report rather than a bare KeyError
    surfacing three stages downstream."""


def load_transcript(path: Union[str, Path]) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Transcript not found: {path}")

    transcript = json.loads(path.read_text())

    if "segments" not in transcript:
        raise TranscriptValidationError(f"{path}: missing top-level 'segments' key.")

    problems = []
    for i, seg in enumerate(transcript["segments"]):
        missing = REQUIRED_SEGMENT_FIELDS - seg.keys()
        if missing:
            problems.append(
                f"  segment {i} (id={seg.get('id', '?')}): missing {sorted(missing)}"
            )

    if problems:
        raise TranscriptValidationError(
            f"{path}: {len(problems)} segment(s) failed schema check:\n"
            + "\n".join(problems)
        )

    return transcript


def validate_paired_transcripts(transcript: dict, flagged_transcript: dict) -> None:
    """Guard against a base transcript.json and a transcript_flagged.json
    that don't actually describe the same video -- e.g. a stale flagged
    file left over from a previous video. The two are maintained as
    separate files by design (see locate.py), so nothing else in the
    pipeline guarantees they're paired correctly; this is the natural
    place to check, since it's where both get used together.
    """
    base_id = transcript.get("video_id")
    flagged_id = flagged_transcript.get("video_id")
    if base_id != flagged_id:
        raise TranscriptValidationError(
            f"video_id mismatch: transcript.json says '{base_id}', "
            f"transcript_flagged.json says '{flagged_id}'. These should describe the same video."
        )


def resolve_video(video_ref: str, workdir: Path) -> Path:
    """Resolve a video reference to a local mp4 path.

    video_ref may be:
      - a local filesystem path to an existing video file, or
      - a YouTube URL, downloaded via yt-dlp into workdir.

    This is what makes the repo testable by a third party without you
    distributing the raw interview mp4: they can point this at your
    unlisted walkthrough link (or their own video) instead of needing
    Drive access.
    """
    if video_ref.startswith("http://") or video_ref.startswith("https://"):
        return _download_youtube_video(video_ref, workdir)

    path = Path(video_ref)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    return path


def _download_youtube_video(url: str, workdir: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)

    existing = sorted(workdir.glob("*.mp4"))
    if existing:
        print(f"Video already downloaded at {existing[0].name} -- skipping re-download")
        return existing[0]

    out_template = str(workdir / "%(id)s.%(ext)s")
    result = subprocess.run(
        ["yt-dlp", "-f", "mp4", "-o", out_template, url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (exit code {result.returncode}) downloading {url}.\n"
            f"If this mentions sign-in or a 429, it's very likely YouTube "
            f"bot-blocking requests from this machine's IP (common on cloud/"
            f"datacenter IPs, including Colab) -- not a bug in this code.\n"
            f"--- yt-dlp output ---\n{result.stderr}"
        )
    matches = sorted(workdir.glob("*.mp4"))
    if not matches:
        raise FileNotFoundError(
            f"yt-dlp reported success downloading {url} but no .mp4 landed in {workdir}"
        )
    return matches[0]


def probe_duration(video_path: Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return float(probe.stdout.strip())


def resolve_run_dir(transcript: dict, output_dir: Path) -> Path:
    """Scoped per-video output folder, derived from the transcript's own
    video_id instead of a hardcoded project name (previously
    'brewster_kahle'), so a different transcript never collides with or
    gets confused for another run's artifacts.
    """
    video_id = transcript.get("video_id")
    if not video_id:
        raise TranscriptValidationError(
            "transcript.json is missing 'video_id', needed to scope this run's output folder."
        )
    run_dir = output_dir / video_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_old_transcript_path(input_dir: Path, run_dir: Path,
                                 fallback_filename: str = "transcript_flagged.json") -> Path:
    """The 'old' side of a diff is the last committed transcript snapshot
    (current_transcript.json, written at the end of a successful
    incremental run) if one exists, or the original flagged transcript
    otherwise -- i.e. this is the first incremental run since Stage 1.
    """
    committed_path = run_dir / "current_transcript.json"
    if committed_path.exists():
        print(f"Using {committed_path.name} (last committed transcript) as the 'old' side")
        return committed_path

    fallback_path = input_dir / fallback_filename
    if not fallback_path.exists():
        raise FileNotFoundError(
            f"No committed snapshot at {committed_path} and no {fallback_filename} at "
            f"{fallback_path} to diff against."
        )
    print(f"No committed snapshot found -- using original {fallback_path.name} as the 'old' side (first run)")
    return fallback_path


def summarize(transcript: dict, n_preview: int = 5) -> None:
    segments = transcript["segments"]
    print(f"video_id: {transcript['video_id']}")
    print(f"{len(segments)} segments total, "
          f"{sum(s['make_clip'] for s in segments)} flagged make_clip=true\n")
    for s in segments[:n_preview]:
        flag = "CLIP" if s["make_clip"] else "    "
        print(f"[{flag}] {s['id']:8} {s['speaker']:10} {s['text'][:70]}")
    if len(segments) > n_preview:
        print("...")
