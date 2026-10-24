import json

import httpx
import pytest

from slm_router.app import create_app, parse_route


@pytest.fixture
def client(make_router):
    app = create_app(make_router())
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def test_parse_route():
    assert parse_route("gpt-4o", None) == "auto"
    assert parse_route("local", None) == "local"
    assert parse_route("auto", "frontier") == "frontier"


async def test_chat_completion_shape_and_headers(client):
    r = await client.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "What is a mutex?"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "local answer"
    assert r.headers["x-router-tier"] == "local" and "x-router-score" in r.headers


async def test_streaming_sse(client):
    async with client.stream("POST", "/v1/chat/completions", json={
            "model": "auto", "stream": True, "messages": [{"role": "user", "content": "What is a mutex?"}]}) as r:
        assert r.headers["x-router-tier"] == "local"
        lines = [l async for l in r.aiter_lines() if l.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(l[5:]) for l in lines[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert text.strip() == "local answer"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop" and chunks[-1]["usage"]["completion_tokens"] == 50


async def test_explain_and_metrics(client):
    r = await client.post("/v1/route/explain", json={"messages": [{"role": "user", "content": "Prove Fermat's little theorem step by step"}]})
    assert r.json()["heuristic_tier"] == "frontier"
    await client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}]})
    m = (await client.get("/metrics")).text
    assert "router_requests_total" in m and "router_complexity_score_bucket" in m


async def test_chat_page_served(client):
    r = await client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"] and "SLM Router" in r.text
