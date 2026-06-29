"""Embedding providers: text in, vectors out.

The pipeline depends only on the small `Embedder` contract below, so we can
swap providers (offline now, a real model later) without touching anything
else. That one design choice is what makes the engine scalable.

HashEmbedder is our offline starter: deterministic, no API key, no network.
Each word is hashed into a fixed bucket; texts that share words land near
each other. Enough to run the whole pipeline -- but lexical, not semantic.

VoyageEmbedder is the real, semantic embedder (Voyage AI free tier). It
understands meaning, so a question worded differently from the source text
still matches. It plugs in through the same Embedder contract.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """The contract every embedder must follow."""

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        ...


class HashEmbedder:
    """Offline, deterministic embedder. No API key needed.

    Each word maps to a bucket; we count bucket hits to build the vector.
    Texts with overlapping words get similar vectors. Stable and fast,
    but understands words, not meaning.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in text.lower().split():
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed_one(t) for t in texts])


class VoyageEmbedder:
    """Real semantic embeddings via Voyage AI (free tier).

    Needs `pip install voyageai` and a VOYAGE_API_KEY. Unlike the offline
    HashEmbedder, this understands meaning -- 'feline pets' matches 'cats'
    even with no shared words.
    """

    def __init__(self, model: str = "voyage-4-lite", dim: int = 1024):
        import voyageai  # lazy import so the package runs without it
        self._client = voyageai.Client()  # reads VOYAGE_API_KEY from the environment
        self.model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        result = self._client.embed(texts, model=self.model)
        return np.asarray(result.embeddings, dtype=np.float32)