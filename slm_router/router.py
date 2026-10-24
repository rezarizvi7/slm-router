"""Routing engine: semantic cache -> complexity score (+ SLM judge in the gray zone)
-> local SLM, escalating to the frontier model when the prompt is hard, the local
model fails / blows its latency budget, or its answer looks weak."""

import asyncio
import datetime as dt
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

from . import complexity, metrics
from .backends import BackendError, ChatResult, StreamEvent
from .cache import CacheHit, SemanticCache, context_key
from .config import Settings
from .embeddings import Embedder
from .pricing import cost_usd

log = logging.getLogger(__name__)

Tier = Literal["cache", "local", "frontier"]
Route = Literal["auto", "local", "frontier"]

WEAK_ANSWER = re.compile(
    r"\b(I('m| am) not (sure|certain)|I (don't|do not) know|I can(not|'t) (help|answer|determine)|"
    r"I('m| am) unable to|beyond my (capabilities|knowledge))\b", re.I)

JUDGE_SYSTEM = (
    "You triage requests for an AI system. Decide if the request can be answered well by a small "
    "assistant model (short factual answers, rewording, simple code snippets, formatting, "
    "classification) or needs a large expert model (multi-step reasoning, non-trivial code, design, "
    "math, long analysis). Reply with exactly one word: EASY or HARD."
)


@dataclass
class Decision:
    route: Route
    tier: Tier = "local"
    score: float | None = None
    reasons: list[str] = field(default_factory=list)
    escalation: str | None = None
    cache_distance: float | None = None

    def headers(self) -> dict[str, str]:
        h = {"X-Router-Tier": self.tier, "X-Router-Reasons": ",".join(self.reasons)[:500]}
        if self.score is not None:
            h["X-Router-Score"] = f"{self.score:.3f}"
        if self.escalation:
            h["X-Router-Escalation"] = self.escalation
        if self.cache_distance is not None:
            h["X-Router-Cache-Distance"] = f"{self.cache_distance:.4f}"
        return h


class DailyBudget:
    """In-process daily frontier spend tracker (UTC day). Per-process: with multiple
    workers each enforces its own cap, so divide the budget accordingly."""

    def __init__(self, limit_usd: float):
        self.limit = limit_usd
        self.day = dt.datetime.now(dt.UTC).date()
        self.spent = 0.0
        self._publish()

    def _roll(self):
        today = dt.datetime.now(dt.UTC).date()
        if today != self.day:
            self.day, self.spent = today, 0.0

    def exhausted(self) -> bool:
        self._roll()
        return self.limit > 0 and self.spent >= self.limit

    def add(self, usd: float):
        self._roll()
        self.spent += usd
        self._publish()

    def _publish(self):
        metrics.BUDGET_REMAINING.set(max(0.0, self.limit - self.spent) if self.limit > 0 else -1)


class Router:
    def __init__(self, settings: Settings, local, frontier, cache: SemanticCache | None, embedder: Embedder | None):
        self.s = settings
        self.local = local
        self.frontier = frontier
        self.cache = cache if settings.cache_enabled else None
        self.embedder = embedder
        self.budget = DailyBudget(settings.frontier_daily_budget_usd)

    # ------------------------------------------------------------------ decision

    async def decide(self, req: dict, route: Route) -> Decision:
        d = Decision(route=route)
        if route != "auto":
            d.tier = route
            d.reasons.append(f"forced_{route}")
            return d

        c = complexity.score(req["messages"], req.get("tools"))
        metrics.COMPLEXITY.observe(c.score)
        d.score, d.reasons = c.score, list(c.reasons)

        if c.force_frontier:
            d.tier = "frontier"
        elif self.s.judge_enabled and self.s.judge_band_low <= c.score <= self.s.judge_band_high:
            verdict = await self._judge(req)
            d.reasons.append(f"judge={verdict}")
            d.tier = "frontier" if verdict == "HARD" else "local" if verdict == "EASY" else (
                "frontier" if c.score >= self.s.complexity_threshold else "local")
        else:
            d.tier = "frontier" if c.score >= self.s.complexity_threshold else "local"

        if d.tier == "frontier" and self.budget.exhausted():
            d.tier = "local"
            d.reasons.append("frontier_budget_exhausted")
        return d

    async def _judge(self, req: dict) -> str:
        users = [m for m in req["messages"] if m.get("role") == "user"]
        ask = complexity.message_text(users[-1]["content"])[:2000] if users else ""
        try:
            r = await asyncio.wait_for(self.local.complete({
                "messages": [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": ask}],
                "max_tokens": 3, "temperature": 0,
            }), timeout=min(3.0, self.s.local_timeout_s))
            verdict = "HARD" if "HARD" in r.text.upper() else "EASY" if "EASY" in r.text.upper() else "UNSURE"
        except (BackendError, asyncio.TimeoutError):
            verdict = "ERROR"
        metrics.JUDGE.labels(verdict).inc()
        return verdict

    def _weak(self, r: ChatResult, req: dict) -> str | None:
        if not self.s.cascade_on_weak_answer:
            return None
        if r.tool_calls:
            return None
        if not r.text.strip():
            return "empty_local_answer"
        # A caller-imposed max_tokens truncates the frontier just the same; don't pay to repeat it.
        if r.finish_reason == "length" and not (req.get("max_tokens") or req.get("max_completion_tokens")):
            return "local_truncated"
        if WEAK_ANSWER.search(r.text[:400]):
            return "local_low_confidence"
        return None

    # ------------------------------------------------------------------ cache

    def _cacheable(self, req: dict, no_cache: bool) -> bool:
        return bool(self.cache and self.embedder and not no_cache
                    and not req.get("tools") and (req.get("n") or 1) == 1)

    async def _cache_probe(self, req: dict, route: Route):
        """Returns (ctx, vector, hit). Cache problems never fail a request."""
        users = [m for m in req["messages"] if m.get("role") == "user"]
        if not users:
            return None, None, None
        ctx = context_key(req["messages"], {"route": route, "response_format": req.get("response_format")})
        try:
            vec = await self.embedder.embed(complexity.message_text(users[-1]["content"]))
            hit = await self.cache.lookup(ctx, vec)
        except Exception as e:  # embedder down, redis down, ...
            log.warning("cache lookup failed: %r", e)
            metrics.CACHE.labels("error").inc()
            return None, None, None
        metrics.CACHE.labels("hit" if hit else "miss").inc()
        if hit:
            metrics.CACHE_DISTANCE.observe(hit.distance)
        return ctx, vec, hit

    async def _cache_store(self, ctx, vec, result: ChatResult, tier: Tier):
        if ctx is None or vec is None or result.finish_reason != "stop" or not result.text:
            return
        try:
            await self.cache.store(ctx, vec, {"text": result.text, "model": result.model,
                                              "input_tokens": result.input_tokens,
                                              "output_tokens": result.output_tokens}, tier)
        except Exception as e:
            log.warning("cache store failed: %r", e)

    # ------------------------------------------------------------------ accounting

    def _account(self, tier: Tier, r: ChatResult, latency: float, stream: bool, served_from: str | None = None):
        metrics.REQUESTS.labels(tier).inc()
        metrics.LATENCY.labels(tier, str(stream).lower()).observe(latency)
        if tier == "cache":
            # Rough: cached answers originally from the frontier save their original price.
            metrics.SAVINGS.labels("cache").inc(cost_usd(self.frontier.model, r.input_tokens, r.output_tokens)
                                                if served_from == "frontier" else 0.0)
            return
        metrics.TOKENS.labels(tier, "input").inc(r.input_tokens)
        metrics.TOKENS.labels(tier, "output").inc(r.output_tokens)
        if tier == "frontier":
            usd = cost_usd(r.model, r.input_tokens, r.output_tokens) or cost_usd(self.frontier.model, r.input_tokens, r.output_tokens)
            metrics.FRONTIER_COST.inc(usd)
            self.budget.add(usd)
        else:
            # Counterfactual: what the frontier model would have charged for the same token counts.
            # Tokenizers differ, so treat this as an order-of-magnitude estimate.
            metrics.SAVINGS.labels("local").inc(cost_usd(self.frontier.model, r.input_tokens, r.output_tokens))

    # ------------------------------------------------------------------ non-streaming

    async def complete(self, req: dict, route: Route = "auto", no_cache: bool = False) -> tuple[ChatResult, Decision]:
        t0 = time.perf_counter()
        ctx = vec = None
        if self._cacheable(req, no_cache):
            ctx, vec, hit = await self._cache_probe(req, route)
            if hit:
                return self._serve_hit(hit, route, t0, stream=False)

        d = await self.decide(req, route)
        result: ChatResult | None = None

        if d.tier == "local":
            try:
                result = await asyncio.wait_for(self.local.complete(req), timeout=self.s.local_timeout_s)
            except (BackendError, asyncio.TimeoutError) as e:
                metrics.BACKEND_ERRORS.labels("local").inc()
                d.escalation = "local_timeout" if isinstance(e, asyncio.TimeoutError) else "local_error"
            else:
                d.escalation = self._weak(result, req) if route == "auto" else None
            if d.escalation:
                if route == "local" or self.budget.exhausted():
                    if result is None:
                        raise BackendError(f"local backend failed ({d.escalation}) and escalation is not allowed")
                    d.reasons.append("escalation_blocked")
                    d.escalation = None
                else:
                    metrics.ESCALATIONS.labels(d.escalation).inc()
                    d.tier = "frontier"

        if d.tier == "frontier":
            try:
                result = await self.frontier.complete(req)
            except BackendError:
                metrics.BACKEND_ERRORS.labels("frontier").inc()
                # Degrade gracefully: a mediocre local answer beats a 502 -- unless the
                # caller explicitly asked for the frontier, or local already failed.
                if route != "auto" or d.escalation in ("local_error", "local_timeout"):
                    raise
                if result is None:
                    metrics.DEGRADATIONS.labels("frontier_error").inc()
                    result = await asyncio.wait_for(self.local.complete(req), timeout=self.s.local_timeout_s)
                d.tier = "local"
                d.reasons.append("frontier_error_fallback")

        self._account(d.tier, result, time.perf_counter() - t0, stream=False)
        await self._cache_store(ctx, vec, result, d.tier)
        return result, d

    def _serve_hit(self, hit: CacheHit, route: Route, t0: float, stream: bool):
        r = ChatResult(text=hit.response["text"], model=hit.response.get("model", "cache"),
                       input_tokens=hit.response.get("input_tokens", 0),
                       output_tokens=hit.response.get("output_tokens", 0))
        d = Decision(route=route, tier="cache", reasons=[f"cache_from_{hit.tier}"], cache_distance=hit.distance)
        self._account("cache", r, time.perf_counter() - t0, stream=stream, served_from=hit.tier)
        return r, d

    # ------------------------------------------------------------------ streaming

    async def stream(self, req: dict, route: Route = "auto", no_cache: bool = False
                     ) -> tuple[AsyncIterator[StreamEvent], Decision]:
        """Resolves the serving tier *before* returning (so headers are accurate): for the
        local tier we wait for the first token within the TTFT budget and escalate if it
        doesn't arrive. Once tokens flow we're committed -- no weak-answer cascade."""
        t0 = time.perf_counter()
        ctx = vec = None
        if self._cacheable(req, no_cache):
            ctx, vec, hit = await self._cache_probe(req, route)
            if hit:
                r, d = self._serve_hit(hit, route, t0, stream=True)
                return _replay(r), d

        d = await self.decide(req, route)

        if d.tier == "local":
            it = self.local.stream(req).__aiter__()
            try:
                first = await asyncio.wait_for(it.__anext__(), timeout=self.s.local_ttft_budget_s)
            except (BackendError, asyncio.TimeoutError, StopAsyncIteration) as e:
                await _aclose(it)
                metrics.BACKEND_ERRORS.labels("local").inc()
                reason = "local_ttft_budget" if isinstance(e, asyncio.TimeoutError) else "local_error"
                if route == "local" or self.budget.exhausted():
                    raise BackendError(f"local backend failed ({reason}) and escalation is not allowed") from e
                metrics.ESCALATIONS.labels(reason).inc()
                d.tier, d.escalation = "frontier", reason
            else:
                metrics.TTFT.labels("local").observe(time.perf_counter() - t0)
                return self._tap(_prepend(first, it), d, t0, ctx, vec), d

        it = self.frontier.stream(req).__aiter__()
        first = await it.__anext__()  # surface auth/connection errors before headers go out
        metrics.TTFT.labels("frontier").observe(time.perf_counter() - t0)
        return self._tap(_prepend(first, it), d, t0, ctx, vec), d

    async def _tap(self, events: AsyncIterator[StreamEvent], d: Decision, t0, ctx, vec):
        async for ev in events:
            if ev.final:
                self._account(d.tier, ev.final, time.perf_counter() - t0, stream=True)
                await self._cache_store(ctx, vec, ev.final, d.tier)
            yield ev


async def _prepend(first: StreamEvent, rest: AsyncIterator[StreamEvent]):
    yield first
    async for ev in rest:
        yield ev


async def _replay(r: ChatResult):
    yield StreamEvent(delta=r.text)
    yield StreamEvent(final=r)


async def _aclose(it):
    if hasattr(it, "aclose"):
        try:
            await it.aclose()
        except Exception:
            pass
