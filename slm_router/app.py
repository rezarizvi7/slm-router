"""OpenAI-compatible HTTP front door.

Point any tool that speaks the OpenAI Chat Completions API at http://localhost:8080/v1
and set model to "auto" (route), "local" or "frontier" (force a tier). Any other model
name is treated as "auto" so existing tool configs work unchanged.
"""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import complexity
from .backends import AnthropicBackend, BackendError, ChatResult, OpenAICompatBackend
from .cache import InMemoryCache, RedisSemanticCache
from .config import Settings, get_settings
from .embeddings import HashingEmbedder, OllamaEmbedder
from .router import Route, Router

log = logging.getLogger("slm_router")
CHAT_PAGE = Path(__file__).parent / "static" / "index.html"


def build_router(s: Settings) -> Router:
    http = httpx.AsyncClient()
    local = OpenAICompatBackend(s.local_base_url, s.local_model, s.local_api_key, s.local_timeout_s, http)
    if s.frontier_provider == "anthropic":
        frontier = AnthropicBackend(s.frontier_model, s.frontier_api_key, s.frontier_timeout_s,
                                    s.frontier_max_tokens, s.frontier_effort, s.frontier_refusal_fallback)
    else:
        frontier = OpenAICompatBackend(s.frontier_base_url, s.frontier_model, s.frontier_api_key,
                                       s.frontier_timeout_s, http)
    embedder = (OllamaEmbedder(s.embed_base_url, s.embed_model, http) if s.embed_provider == "ollama"
                else HashingEmbedder())
    cache = (RedisSemanticCache(s.redis_url, s.cache_max_distance, s.cache_ttl_s) if s.cache_backend == "redis"
             else InMemoryCache(s.cache_max_distance, s.cache_ttl_s))
    return Router(s, local, frontier, cache, embedder)


def parse_route(model: str | None, header: str | None) -> Route:
    for v in (header, model):
        if v and v.lower() in ("auto", "local", "frontier"):
            return v.lower()  # type: ignore[return-value]
    return "auto"


def completion_body(r: ChatResult, request_id: str) -> dict:
    msg: dict = {"role": "assistant", "content": r.text or None}
    if r.tool_calls:
        msg["tool_calls"] = r.tool_calls
    return {
        "id": request_id, "object": "chat.completion", "created": int(time.time()), "model": r.model,
        "choices": [{"index": 0, "message": msg, "finish_reason": r.finish_reason}],
        "usage": {"prompt_tokens": r.input_tokens, "completion_tokens": r.output_tokens,
                  "total_tokens": r.input_tokens + r.output_tokens},
    }


def create_app(router: Router | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.router = router or build_router(get_settings())
        yield

    app = FastAPI(title="slm-router", lifespan=lifespan)
    if router is not None:  # injected (tests); available even without running lifespan
        app.state.router = router

    @app.get("/", response_class=HTMLResponse)
    async def chat_page():
        """Minimal chat UI so the router can be tried without any tooling."""
        return CHAT_PAGE.read_text(encoding="utf-8")

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/metrics")
    async def prom():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "slm-router"}
                                           for m in ("auto", "local", "frontier")]}

    @app.post("/v1/route/explain")
    async def explain(req: Request):
        """Dry run: what would the heuristic do with this request? No model calls."""
        body = await req.json()
        c = complexity.score(body.get("messages", []), body.get("tools"))
        s = app.state.router.s
        return {"score": round(c.score, 4), "reasons": c.reasons, "force_frontier": c.force_frontier,
                "threshold": s.complexity_threshold,
                "in_judge_band": s.judge_enabled and s.judge_band_low <= c.score <= s.judge_band_high,
                "heuristic_tier": "frontier" if c.force_frontier or c.score >= s.complexity_threshold else "local"}

    @app.post("/v1/chat/completions")
    async def chat(req: Request,
                   x_router_route: str | None = Header(default=None),
                   cache_control: str | None = Header(default=None)):
        body = await req.json()
        if not body.get("messages"):
            raise HTTPException(400, "messages is required")
        route = parse_route(body.get("model"), x_router_route)
        no_cache = bool(cache_control and "no-cache" in cache_control.lower())
        rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        router: Router = app.state.router

        try:
            if not body.get("stream"):
                result, d = await router.complete(body, route, no_cache)
                return JSONResponse(completion_body(result, rid), headers=d.headers())
            events, d = await router.stream(body, route, no_cache)
        except BackendError as e:
            log.error("request failed: %s", e)
            raise HTTPException(502, f"upstream model error: {e}") from e

        async def sse():
            created = int(time.time())
            model = "auto"

            def chunk(delta: dict, finish=None, usage=None):
                c = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if usage:
                    c["usage"] = usage
                return f"data: {json.dumps(c)}\n\n"

            yield chunk({"role": "assistant", "content": ""})
            try:
                async for ev in events:
                    if ev.delta:
                        yield chunk({"content": ev.delta})
                    elif ev.final:
                        model = ev.final.model
                        if ev.final.tool_calls:
                            yield chunk({"tool_calls": [dict(tc, index=i) for i, tc in enumerate(ev.final.tool_calls)]})
                        yield chunk({}, ev.final.finish_reason, {
                            "prompt_tokens": ev.final.input_tokens, "completion_tokens": ev.final.output_tokens,
                            "total_tokens": ev.final.input_tokens + ev.final.output_tokens})
            except BackendError as e:
                log.error("stream failed mid-flight: %s", e)
                yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'upstream_error'}})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream", headers=d.headers())

    return app


def main():
    import uvicorn

    s = get_settings()
    logging.basicConfig(level=s.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(), host=s.host, port=s.port, log_level=s.log_level)


if __name__ == "__main__":
    main()
