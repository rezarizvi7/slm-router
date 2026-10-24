# slm-router

A local, OpenAI-compatible reverse proxy that sends each LLM request to the cheapest tier
likely to handle it:

1. **Semantic cache** (Redis vector search): near-duplicate questions in the same context
   are answered in milliseconds at no cost.
2. **Local SLM** (Qwen 2.5 1.5B / Llama 3.2 1B via Ollama or vLLM): short, well-scoped tasks.
3. **Frontier API** (Claude via the Anthropic SDK, or any OpenAI-compatible API): multi-step
   reasoning, non-trivial code, tool calls, images, and anything the local model fumbles.

```
            ┌──────────── /v1/chat/completions (OpenAI shape) ────────────┐
client ───► │ cache probe ─hit─► return                                   │
            │   │ miss                                                    │
            │   ▼                                                         │
            │ complexity score ──(0.35–0.65: ask SLM "EASY/HARD?")        │
            │   │ easy                         │ hard / tools / images    │
            │   ▼                              ▼                          │
            │ local SLM ──error/timeout/TTFT/weak answer──► frontier API  │
            │   ▲                                   │ error → degrade      │
            │   └───────────────────────────────────┘                     │
            │ store in cache · Prometheus metrics · X-Router-* headers    │
            └─────────────────────────────────────────────────────────────┘
```

## Quick start

```bash
cp .env.example .env            # add ANTHROPIC_API_KEY
docker compose up -d --build    # ollama, model pulls, redis 8, router, prometheus, grafana
```

The first start downloads about 1.3 GB of models. Then:

```bash
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"What is a mutex?"}]}' -i | grep -i x-router
```

Point any OpenAI-compatible tool (Continue, Aider, Open WebUI, LangChain, the `openai`
SDK) at `http://localhost:8080/v1`. Set the model name to choose a route:

| model / `X-Router-Route` | behavior |
|---|---|
| `auto` (or any unknown name) | full routing |
| `local` | local only, never escalates |
| `frontier` | frontier only |

Send `Cache-Control: no-cache` to skip the semantic cache.

Grafana: http://localhost:3000 (dashboard "SLM Router") · Prometheus: http://localhost:9090

Run without Docker: `pip install -e '.[dev]' && slm-router`. This needs Ollama and Redis
running locally. For quick experiments with no Redis or embedding model, set
`ROUTER_CACHE_BACKEND=memory ROUTER_EMBED_PROVIDER=hashing`.

## Design

**Routing signal.** `complexity.py` scores each request in microseconds with a logistic
function over cheap features: log-scaled length, context size, turn count, regex families
for multi-step/design/proof/debug requests (push up) and for lookup/translate/format/classify
requests (push down), enumerated items, question count, and code blocks. Tools and images
always go to the frontier, because 1–2B models handle them poorly. The score and the features
that fired come back in `X-Router-Score` and `X-Router-Reasons`, and
`POST /v1/route/explain` gives a dry run.

**Gray-zone judge.** The heuristic is only asked to be confident at the extremes. For scores
in `[judge_band_low, judge_band_high]`, the local model gets a 3-token "EASY or HARD"
classification call. This costs one extra short local call on ambiguous requests only.

**Cascade, not just a classifier.** A local answer is escalated when:
- the local call errors or times out;
- a streamed request misses the **time-to-first-token budget** (`local_ttft_budget_s`). This
  is the latency SLO lever;
- the answer is empty, truncated (`finish_reason=length`), or hedges ("I'm not sure…").

Streaming commits to a tier once the first token arrives, so streamed requests skip the
weak-answer check.

**Graceful degradation.** If the frontier fails on an `auto` request, the router serves the
local answer instead of returning a 502. A **daily frontier budget** (`frontier_daily_budget_usd`)
keeps everything local once it is spent.

**Semantic cache.** The cache embeds the latest user message (`nomic-embed-text` via Ollama)
and runs KNN on a Redis 8 HNSW index (cosine). The search is filtered by an exact hash of the
rest of the conversation: system prompt, earlier turns, `response_format` and route. "What does
it return?" therefore never matches across unrelated conversations. On a Redis without the
query engine, the cache falls back to a bounded brute-force scan. Only clean
(`finish_reason=stop`), tool-free, `n=1` answers are stored.

**Frontier = Claude by default.** The default model is `claude-opus-5-5` at effort `medium`,
called through the official `anthropic` SDK. The router translates the OpenAI shape to the
Messages API, including system messages, multimodal parts, tools, tool calls and tool
results. It enables server-side refusal fallback (`fallbacks: "default"`), so a
safety-classifier decline is re-run on Anthropic's recommended fallback model instead of
surfacing as an error. Set `ROUTER_FRONTIER_REFUSAL_FALLBACK=false` to turn this off.
`temperature`/`top_p` are not forwarded, because current Claude models reject them. Forced
`tool_choice` is downgraded to `auto` for the same reason. Any OpenAI-compatible provider
works instead via `ROUTER_FRONTIER_PROVIDER=openai_compat`.

**Metrics** (`/metrics`): requests by serving tier, escalations by reason, frontier→local
degradations, latency and TTFT histograms by tier, tokens, frontier spend, estimated savings,
complexity-score distribution, judge verdicts, cache hit/miss/error and hit distance, and
remaining budget.

## Tuning on your own traffic

```bash
python scripts/bench.py offline scripts/prompts.jsonl                # threshold sweep, no model calls
python scripts/bench.py live scripts/prompts.jsonl --repeat 2        # real routing, latency, spend
```

`scripts/prompts.jsonl` is a 30-prompt **illustrative** set written alongside the
heuristic, so the 100% agreement it shows at threshold 0.5 is circular. To get a meaningful
number, label a few hundred real prompts from your tools: "did the local model answer this
acceptably?" Sweep the threshold, then watch `router_escalations_total{reason="local_low_confidence"}`
and the "served without frontier" panel. The share of traffic that stays local depends
entirely on your workload. Nothing here guarantees a particular figure.

## Tests

```bash
pytest                                                  # 37 tests, no network; Redis test uses a local redis-server if present
TEST_REDIS_URL=redis://localhost:6379/0 pytest tests/test_cache.py   # exercise the HNSW path on Redis 8
```

## Known limitations

- Stateless proxy: Claude's thinking blocks aren't round-tripped through the OpenAI
  format, so multi-turn conversations re-send text only.
- The budget tracker is per process. With N workers, each worker enforces the full cap.
- Savings are a counterfactual estimate: local token counts priced at frontier rates, with
  different tokenizers. Treat them as order-of-magnitude.
- The weak-answer detector is a regex. A small model can be confidently wrong, and this won't
  catch that.
- The prices in `pricing.py` were current when written. Check them against the provider's
  pricing page.
