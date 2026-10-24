import os
import shutil
import socket
import subprocess
import time

import numpy as np
import pytest

from slm_router.cache import InMemoryCache, RedisSemanticCache, context_key
from slm_router.embeddings import HashingEmbedder


def test_context_key_ignores_last_user_only():
    a = [{"role": "system", "content": "s"}, {"role": "user", "content": "q1"}]
    b = [{"role": "system", "content": "s"}, {"role": "user", "content": "q2"}]
    c = [{"role": "system", "content": "other"}, {"role": "user", "content": "q1"}]
    assert context_key(a) == context_key(b) != context_key(c)


async def test_hashing_embedder_near_duplicates_are_close():
    e = HashingEmbedder()
    a, b, c = [await e.embed(t) for t in ("What is a mutex?", "what is a mutex", "best pizza in naples")]
    assert 1 - a @ b < 0.01 and 1 - a @ c > 0.5


async def test_in_memory_threshold():
    cache = InMemoryCache(max_distance=0.05, ttl_s=60)
    v = np.zeros(4, dtype=np.float32); v[0] = 1
    far = np.zeros(4, dtype=np.float32); far[1] = 1
    await cache.store("ctx", v, {"text": "x"}, "local")
    assert (await cache.lookup("ctx", v)).response["text"] == "x"
    assert await cache.lookup("ctx", far) is None
    assert await cache.lookup("other", v) is None


@pytest.fixture
def redis_url():
    if not shutil.which("redis-server"):
        pytest.skip("redis-server not installed")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    p = subprocess.Popen(["redis-server", "--port", str(port), "--save", "", "--appendonly", "no"],
                         stdout=subprocess.DEVNULL)
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.05)
    yield os.environ.get("TEST_REDIS_URL", f"redis://127.0.0.1:{port}/0")
    p.terminate()


async def test_redis_cache_roundtrip(redis_url):
    cache = RedisSemanticCache(redis_url, max_distance=0.05, ttl_s=60)
    e = HashingEmbedder()
    await cache.store("ctx1", await e.embed("What is a mutex?"), {"text": "a lock"}, "frontier")
    hit = await cache.lookup("ctx1", await e.embed("what is a mutex"))
    assert hit and hit.response["text"] == "a lock" and hit.tier == "frontier"
    assert await cache.lookup("ctx1", await e.embed("explain monads in haskell")) is None
    assert await cache.lookup("ctx2", await e.embed("What is a mutex?")) is None
