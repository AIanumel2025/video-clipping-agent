"""
Shared pytest fixtures for the agent test suite.

Tier 1: these tests never touch a real model or real audio. align.py's
and embeddings.py's heavy imports (torch, whisperx, sentence-transformers)
are deferred inside the functions that need them, so importing any of
these modules doesn't require those packages installed at all. Where a
test needs an embedding, it primes the disk cache directly (see
seed_similarity_pair below) instead of loading the real model.
"""

import math

import pytest

from agent.embeddings import seed_cache


@pytest.fixture
def run_dir(tmp_path):
    """A throwaway run directory, same shape resolve_run_dir() would hand
    back, without touching the real filesystem layout."""
    d = tmp_path / "run"
    d.mkdir()
    return d


@pytest.fixture
def embed_cache_dir(run_dir):
    d = run_dir / "cache" / "embeddings"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def seed_similarity_pair(embed_cache_dir):
    """Factory fixture: seed_similarity_pair(text_a, text_b, similarity=0.9)
    primes the cache so cosine_similarity(get_embedding(a), get_embedding(b))
    comes back as exactly that value, without loading sentence-transformers.
    Uses simple 2D unit vectors -- the real model produces 384-dim ones,
    but cosine_similarity only cares that both vectors are unit length,
    not their dimensionality.
    """
    def _seed(text_a: str, text_b: str, similarity: float) -> None:
        vec_a = [1.0, 0.0]
        vec_b = [similarity, math.sqrt(max(0.0, 1.0 - similarity ** 2))]
        seed_cache(text_a, vec_a, embed_cache_dir)
        seed_cache(text_b, vec_b, embed_cache_dir)

    return _seed
