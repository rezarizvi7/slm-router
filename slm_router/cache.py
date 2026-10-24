"""Semantic cache: return a stored answer when a new prompt is close enough (cosine
distance) to one we've already answered *in the same context*.

The context key matters: "what does it return?" means different things in different
conversations, so the vector search is filtered by an exact hash of everything except
the latest user message (system prompt + prior turns + output-shaping params).
"""

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import redis.asyncio as aioredis
from redis.exceptions import ResponseError

log = logging.getLogger(__name__)


@dataclass
class CacheHit:
    response: dict
    distance: float
    tier: str


class SemanticCache(Protocol):
    async def lookup(self, ctx: str, vec: np.ndarray) -> CacheHit | None: ...
    async def store(self, ctx: str, vec: np.ndarray, response: dict, tier: str) -> None: ...


def context_key(messages: list[dict], extra: dict | None = None) -> str:
    """Hash of everything that shapes the answer except the latest user message."""
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    prefix = [m for i, m in enumerate(messages) if i != last_user]
    blob = json.dumps({"prefix": prefix, "extra": extra or {}}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


class InMemoryCache:
    def __init__(self, max_distance: float, ttl_s: int, max_entries_per_ctx: int = 5000):
        self.max_distance = max_distance
        self.ttl_s = ttl_s
        self.max_entries = max_entries_per_ctx
        self._data: dict[str, list[tuple[float, np.ndarray, dict, str]]] = {}

    async def lookup(self, ctx, vec):
        now = time.time()
        entries = [e for e in self._data.get(ctx, []) if now - e[0] < self.ttl_s]
        self._data[ctx] = entries
        if not entries:
            return None
        mat = np.stack([e[1] for e in entries])
        dists = 1.0 - mat @ vec
        i = int(np.argmin(dists))
        if dists[i] <= self.max_distance:
            return CacheHit(entries[i][2], float(dists[i]), entries[i][3])
        return None

    async def store(self, ctx, vec, response, tier):
        bucket = self._data.setdefault(ctx, [])
        bucket.append((time.time(), vec.astype(np.float32), response, tier))
        del bucket[: max(0, len(bucket) - self.max_entries)]


class RedisSemanticCache:
    """Uses RediSearch HNSW (COSINE) when the server has it (Redis 8+, Redis Stack);
    otherwise falls back to a bounded per-context brute-force scan in numpy, so it
    still works against a plain Redis 6/7."""

    PREFIX = "slmcache:e:"
    INDEX = "slmcache:idx"

    def __init__(self, url: str, max_distance: float, ttl_s: int, scan_limit: int = 2000):
        # RESP2 pinned: FT.* reply shapes differ under RESP3 and we parse the flat form.
        self.r = aioredis.from_url(url, protocol=2)
        self.max_distance = max_distance
        self.ttl_s = ttl_s
        self.scan_limit = scan_limit
        self._mode: str | None = None  # "vector" | "scan"
        self._index_dim: int | None = None

    async def _ensure(self, dim: int) -> str:
        if self._mode == "scan" or (self._mode == "vector" and self._index_dim == dim):
            return self._mode
        try:
            await self.r.execute_command("FT._LIST")
        except ResponseError:
            log.warning("RediSearch not available; semantic cache using brute-force scan mode")
            self._mode = "scan"
            return self._mode
        try:
            info = await self.r.execute_command("FT.INFO", self.INDEX)
            # Dimension is fixed at index creation; a different embedder needs a new index.
            existing_dim = _ft_info_dim(info)
            if existing_dim is not None and existing_dim != dim:
                log.warning("cache index dim %s != embedder dim %s; recreating", existing_dim, dim)
                await self.r.execute_command("FT.DROPINDEX", self.INDEX, "DD")
                raise ResponseError("recreate")
        except ResponseError:
            await self.r.execute_command(
                "FT.CREATE", self.INDEX, "ON", "HASH", "PREFIX", "1", self.PREFIX,
                "SCHEMA",
                "ctx", "TAG",
                "vec", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", str(dim), "DISTANCE_METRIC", "COSINE",
            )
        self._mode, self._index_dim = "vector", dim
        return self._mode

    async def lookup(self, ctx, vec):
        mode = await self._ensure(vec.shape[0])
        if mode == "vector":
            res = await self.r.execute_command(
                "FT.SEARCH", self.INDEX,
                f"(@ctx:{{{ctx}}})=>[KNN 1 @vec $v AS dist]",
                "PARAMS", "2", "v", vec.astype(np.float32).tobytes(),
                "SORTBY", "dist", "RETURN", "3", "dist", "response", "tier",
                "DIALECT", "2",
            )
            if not res or res[0] == 0:
                return None
            fields = _pairs(res[2])
            dist = float(fields[b"dist"])
            if dist > self.max_distance:
                return None
            return CacheHit(json.loads(fields[b"response"]), dist, fields[b"tier"].decode())

        members = await self.r.srandmember(f"slmcache:ctx:{ctx}", self.scan_limit)
        if not members:
            return None
        pipe = self.r.pipeline()
        for k in members:
            pipe.hmget(k, "vec", "response", "tier")
        rows = await pipe.execute()
        live = [(k, row) for k, row in zip(members, rows) if row[0] is not None]
        if dead := [k for k, row in zip(members, rows) if row[0] is None]:
            await self.r.srem(f"slmcache:ctx:{ctx}", *dead)
        if not live:
            return None
        mat = np.stack([np.frombuffer(row[0], dtype=np.float32) for _, row in live])
        dists = 1.0 - mat @ vec.astype(np.float32)
        i = int(np.argmin(dists))
        if dists[i] > self.max_distance:
            return None
        row = live[i][1]
        return CacheHit(json.loads(row[1]), float(dists[i]), row[2].decode())

    async def store(self, ctx, vec, response, tier):
        mode = await self._ensure(vec.shape[0])
        key = f"{self.PREFIX}{uuid.uuid4().hex}"
        pipe = self.r.pipeline()
        pipe.hset(key, mapping={
            "ctx": ctx,
            "vec": vec.astype(np.float32).tobytes(),
            "response": json.dumps(response),
            "tier": tier,
            "created": int(time.time()),
        })
        pipe.expire(key, self.ttl_s)
        if mode == "scan":
            pipe.sadd(f"slmcache:ctx:{ctx}", key)
            pipe.expire(f"slmcache:ctx:{ctx}", self.ttl_s)
        await pipe.execute()


def _pairs(flat) -> dict:
    return {flat[i]: flat[i + 1] for i in range(0, len(flat), 2)}


def _ft_info_dim(info) -> int | None:
    """Dig the vector DIM out of FT.INFO's nested reply; None if the shape is unexpected."""
    try:
        d = _pairs(info)
        for attr in d.get(b"attributes", d.get("attributes", [])):
            a = [x.decode() if isinstance(x, bytes) else x for x in attr]
            if "VECTOR" in a:
                for i, x in enumerate(a):
                    if isinstance(x, str) and x.upper() == "DIM":
                        return int(a[i + 1])
    except Exception:  # FT.INFO's reply shape differs across Redis versions
        return None
    return None
