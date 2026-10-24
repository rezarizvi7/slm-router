"""Embedding providers for the semantic cache."""

import hashlib
import re
from typing import Protocol

import httpx
import numpy as np


class Embedder(Protocol):
    dim: int

    async def embed(self, text: str) -> np.ndarray: ...


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


class OllamaEmbedder:
    """Uses Ollama's /api/embed (e.g. nomic-embed-text, 768-d, ~270MB)."""

    def __init__(self, base_url: str, model: str, client: httpx.AsyncClient | None = None, dim: int = 768):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        self._client = client or httpx.AsyncClient(timeout=10.0)

    async def embed(self, text: str) -> np.ndarray:
        r = await self._client.post(f"{self.base_url}/api/embed", json={"model": self.model, "input": text})
        r.raise_for_status()
        vec = np.asarray(r.json()["embeddings"][0], dtype=np.float32)
        self.dim = vec.shape[0]
        return _normalize(vec)


class HashingEmbedder:
    """Dependency-free fallback: hashed word uni/bi-grams. Catches near-duplicate
    phrasing ("whats a mutex" vs "what is a mutex?") but not true paraphrase.
    Good for tests and for running without an embedding model."""

    TOKEN = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 512):
        self.dim = dim

    async def embed(self, text: str) -> np.ndarray:
        toks = self.TOKEN.findall(text.lower())
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _normalize(v)
