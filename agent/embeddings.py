"""
Stage 2, Phase 2: embed each segment's text into a vector, to validate
and refine the "edited" pairs the diff phase (Phase 1) already found.

Cached to disk, keyed by the same text_hash already used in state.json --
so an unchanged segment's embedding is never recomputed on a future run,
and if every segment in a given run is a cache hit, the model never even
gets loaded. This resolves a known discrepancy: earlier drafts of this
project used an in-memory-only cache (a plain dict), while the README
described it as persisted -- this makes the code match that description,
following the same disk-cache pattern align.py already uses for ASR
transcription.

The cache is scoped by model name, not just text_hash: embeddings from
different models live in different vector spaces and aren't comparable,
so reusing a cached vector from a different model as if it were current
would silently corrupt every downstream similarity score. Switching
models later is then just a matter of pointing at an empty subfolder,
not a footgun.
"""

import json
from pathlib import Path
from typing import List

import numpy as np

from .state import text_hash

DEFAULT_MODEL = "all-MiniLM-L6-v2"

_model = None
_model_name = None


def _get_model(model_name: str = DEFAULT_MODEL):
    global _model, _model_name
    if _model is None or _model_name != model_name:
        from sentence_transformers import SentenceTransformer
        print(f"Loading {model_name} (first cache miss this run)...")
        _model = SentenceTransformer(model_name)
        _model_name = model_name
    return _model


def _cache_path(text: str, cache_dir: Path, model_name: str) -> Path:
    model_dir = cache_dir / model_name.replace("/", "__")
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / f"embed_{text_hash(text)}.json"


def get_embedding(text: str, cache_dir: Path, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Get a segment's embedding, from disk cache if available, otherwise
    computed (loading the model only on this first genuine miss) and
    cached for next time.
    """
    cache_path = _cache_path(text, cache_dir, model_name)
    if cache_path.exists():
        return np.array(json.loads(cache_path.read_text()))

    vector = _get_model(model_name).encode(text, normalize_embeddings=True)
    cache_path.write_text(json.dumps(vector.tolist()))
    return vector


def seed_cache(text: str, vector, cache_dir: Path, model_name: str = DEFAULT_MODEL) -> None:
    """Write a vector into the cache as if it had been computed by the
    model -- lets tests prime known embeddings and control similarity
    scores precisely, without ever loading sentence-transformers.
    """
    cache_path = _cache_path(text, cache_dir, model_name)
    cache_path.write_text(json.dumps(list(vector)))


def embed_segments(segments: List[dict], cache_dir: Path, model_name: str = DEFAULT_MODEL) -> dict:
    """Ensure every segment in the list has a cached embedding. Reports
    how many were genuinely computed this run vs. already on disk (from
    this run's duplicates, or a previous run entirely) -- that gap is
    the whole point of caching by content instead of by run.
    """
    seen_hashes = set()
    computed = 0
    for s in segments:
        h = text_hash(s["text"])
        cache_path = _cache_path(s["text"], cache_dir, model_name)
        if not cache_path.exists() and h not in seen_hashes:
            computed += 1
        seen_hashes.add(h)
        get_embedding(s["text"], cache_dir, model_name)

    print(f"Embedded {len(seen_hashes)} unique segment(s) from {len(segments)} total "
          f"({computed} newly computed, {len(seen_hashes) - computed} from cache)")
    return {"total": len(segments), "unique": len(seen_hashes), "computed": computed}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # safe shortcut since embeddings are normalized above
