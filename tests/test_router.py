import pytest

from slm_router.backends import BackendError
from tests.conftest import FakeBackend, user

HARD = "Design a distributed job scheduler step by step, analyze trade-offs and failure modes, then implement it."


async def test_easy_goes_local(make_router):
    r = make_router()
    res, d = await r.complete(user("What is a mutex?"))
    assert d.tier == "local" and res.text == "local answer"


async def test_hard_goes_frontier(make_router):
    r = make_router()
    res, d = await r.complete(user(HARD))
    assert d.tier == "frontier" and res.text == "frontier answer"


async def test_forced_routes(make_router):
    r = make_router()
    _, d = await r.complete(user(HARD), route="local")
    assert d.tier == "local"
    _, d = await r.complete(user("hi"), route="frontier", no_cache=True)
    assert d.tier == "frontier"


async def test_local_error_escalates(make_router):
    r = make_router(local=FakeBackend("l", fail=True))
    res, d = await r.complete(user("What is a mutex?"))
    assert d.tier == "frontier" and d.escalation == "local_error"


async def test_local_timeout_escalates(make_router):
    r = make_router(local=FakeBackend("l", delay=2.0), local_timeout_s=0.05)
    _, d = await r.complete(user("What is a mutex?"))
    assert d.tier == "frontier" and d.escalation == "local_timeout"


@pytest.mark.parametrize("text,finish,reason", [
    ("I'm not sure what that means.", "stop", "local_low_confidence"),
    ("partial answer", "length", "local_truncated"),
    ("   ", "stop", "empty_local_answer"),
])
async def test_weak_local_answer_cascades(make_router, text, finish, reason):
    r = make_router(local=FakeBackend("l", text=text, finish=finish))
    _, d = await r.complete(user("What is a mutex?"))
    assert d.tier == "frontier" and d.escalation == reason


async def test_caller_max_tokens_truncation_is_not_weak(make_router):
    r = make_router(local=FakeBackend("l", text="partial", finish="length"))
    _, d = await r.complete({**user("What is a mutex?"), "max_tokens": 10})
    assert d.tier == "local" and d.escalation is None


async def test_frontier_error_degrades_to_local(make_router):
    r = make_router(frontier=FakeBackend("f", fail=True))
    res, d = await r.complete(user(HARD))
    assert d.tier == "local" and "frontier_error_fallback" in d.reasons


async def test_forced_frontier_error_raises(make_router):
    r = make_router(frontier=FakeBackend("f", fail=True))
    with pytest.raises(BackendError):
        await r.complete(user("hi"), route="frontier")


async def test_budget_exhausted_keeps_local(make_router):
    r = make_router(frontier_daily_budget_usd=0.01)
    r.budget.add(1.0)
    _, d = await r.complete(user(HARD))
    assert d.tier == "local" and "frontier_budget_exhausted" in d.reasons


async def test_frontier_spend_is_tracked(make_router):
    r = make_router(frontier_daily_budget_usd=100)
    await r.complete(user(HARD))
    # 100 in / 50 out on claude-opus-5-5 at $4/$20 per MTok
    assert r.budget.spent == pytest.approx((100 * 4 + 50 * 20) / 1e6)


async def test_judge_used_only_in_band(make_router):
    local = FakeBackend("l", judge="HARD")
    r = make_router(local=local, judge_enabled=True, judge_band_low=0.0, judge_band_high=1.0)
    _, d = await r.complete(user("What is a mutex?"))
    assert d.tier == "frontier" and "judge=HARD" in d.reasons

    local2 = FakeBackend("l", judge="HARD")
    r2 = make_router(local=local2, judge_enabled=True, judge_band_low=0.9, judge_band_high=0.95)
    _, d2 = await r2.complete(user("What is a mutex?"))
    assert d2.tier == "local" and not any(x.startswith("judge") for x in d2.reasons)


async def test_semantic_cache_hit(make_router):
    local = FakeBackend("l", "a mutex is a lock")
    r = make_router(local=local)
    await r.complete(user("What is a mutex?"))
    res, d = await r.complete(user("what is a mutex"))
    assert d.tier == "cache" and res.text == "a mutex is a lock"
    assert len(local.calls) == 1


async def test_cache_is_scoped_to_context(make_router):
    local = FakeBackend("l")
    r = make_router(local=local)
    a = {"messages": [{"role": "system", "content": "You are a Rust expert."}, {"role": "user", "content": "What is a mutex?"}]}
    b = {"messages": [{"role": "system", "content": "You are a Go expert."}, {"role": "user", "content": "What is a mutex?"}]}
    await r.complete(a)
    _, d = await r.complete(b)
    assert d.tier == "local"


async def test_no_cache_header_bypasses(make_router):
    local = FakeBackend("l")
    r = make_router(local=local)
    await r.complete(user("What is a mutex?"))
    _, d = await r.complete(user("What is a mutex?"), no_cache=True)
    assert d.tier == "local" and len(local.calls) == 2


async def test_stream_local(make_router):
    r = make_router(local=FakeBackend("l", "hello there"))
    events, d = await r.stream(user("What is a mutex?"))
    text = "".join([e.delta async for e in events if e.delta])
    assert d.tier == "local" and text.strip() == "hello there"


async def test_stream_ttft_budget_escalates(make_router):
    r = make_router(local=FakeBackend("l", delay=1.0), local_ttft_budget_s=0.05)
    events, d = await r.stream(user("What is a mutex?"))
    text = "".join([e.delta async for e in events if e.delta])
    assert d.tier == "frontier" and d.escalation == "local_ttft_budget" and "frontier" in text


async def test_stream_then_cache(make_router):
    r = make_router()
    events, _ = await r.stream(user("What is a mutex?"))
    [e async for e in events]
    events, d = await r.stream(user("What is a mutex?"))
    assert d.tier == "cache"
