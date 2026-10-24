import asyncio

import pytest

from slm_router.backends import BackendError, ChatResult, StreamEvent
from slm_router.cache import InMemoryCache
from slm_router.config import Settings
from slm_router.embeddings import HashingEmbedder
from slm_router.router import Router


class FakeBackend:
    def __init__(self, model: str, text: str = "ok", fail: bool = False, delay: float = 0.0,
                 finish: str = "stop", judge: str | None = None):
        self.model = model
        self.text = text
        self.fail = fail
        self.delay = delay
        self.finish = finish
        self.judge = judge
        self.calls: list[dict] = []

    async def complete(self, req):
        self.calls.append(req)
        if self.judge and req["messages"][0].get("role") == "system" and "EASY or HARD" in req["messages"][0]["content"]:
            return ChatResult(self.judge, self.model, 10, 1)
        await asyncio.sleep(self.delay)
        if self.fail:
            raise BackendError(f"{self.model} down")
        return ChatResult(self.text, self.model, 100, 50, self.finish)

    async def stream(self, req):
        self.calls.append(req)
        await asyncio.sleep(self.delay)
        if self.fail:
            raise BackendError(f"{self.model} down")
        for w in self.text.split(" "):
            yield StreamEvent(delta=w + " ")
        yield StreamEvent(final=ChatResult(self.text, self.model, 100, 50, self.finish))


def make_settings(**kw) -> Settings:
    base = dict(judge_enabled=False, cache_backend="memory", embed_provider="hashing",
                frontier_model="claude-opus-5-5", local_ttft_budget_s=0.2, local_timeout_s=1.0)
    base.update(kw)
    return Settings(_env_file=None, **base)


@pytest.fixture
def make_router():
    def _make(local=None, frontier=None, **kw):
        s = make_settings(**kw)
        local = local or FakeBackend("qwen-local", "local answer")
        frontier = frontier or FakeBackend("claude-opus-5-5", "frontier answer")
        return Router(s, local, frontier, InMemoryCache(s.cache_max_distance, s.cache_ttl_s), HashingEmbedder())
    return _make


def user(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}
