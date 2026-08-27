"""
Central path/environment resolution for the video-agent pipeline.

Third-party runnability was previously blocked by a hardcoded personal
Google Drive path (`/content/drive/MyDrive/Projects/LEC AI Project`) baked
directly into the ingest cell. This module replaces that with a priority
chain, so the same code runs for the original author (Colab + Drive), a
different Colab user, and someone running the extracted package outside
Colab entirely.

Resolution order for INPUT_DIR / OUTPUT_DIR:
  1. Explicit environment variable
     (VIDEO_AGENT_INPUT_DIR / VIDEO_AGENT_OUTPUT_DIR)
  2. Google Drive — only if running in Colab AND VIDEO_AGENT_USE_DRIVE=1 is
     set. Mounting is opt-in, never automatic or silent.
  3. A local ./input and ./output directory relative to wherever the code
     is run from.
"""

import os
from pathlib import Path


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_dir(env_var: str, default_local: str) -> Path:
    explicit = os.environ.get(env_var)
    if explicit:
        return Path(explicit)

    if in_colab() and os.environ.get("VIDEO_AGENT_USE_DRIVE") == "1":
        from google.colab import drive
        drive.mount("/content/drive")
        drive_root = os.environ.get(
            "VIDEO_AGENT_DRIVE_ROOT", "/content/drive/MyDrive"
        )
        return Path(drive_root)

    return Path(default_local)


def get_input_dir() -> Path:
    return _resolve_dir("VIDEO_AGENT_INPUT_DIR", "./input")


def get_output_dir() -> Path:
    out = _resolve_dir("VIDEO_AGENT_OUTPUT_DIR", "./output")
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_device() -> str:
    """Shared GPU/CPU detection for any model-backed step (alignment now,
    embeddings later) so it's resolved once, not recomputed per cell."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
