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

## Beginner guide: no commands needed

**What this is, in one sentence:** a "front desk" on your computer that sends easy AI
questions to a free AI running on your own machine and only pays for the big AI (Claude)
when a question is actually hard. It also remembers answers it has already given.

### Step 1: Install Docker Desktop (one time)

Docker is the free app that runs all the pieces for you.

1. Go to https://www.docker.com/products/docker-desktop and download it for your computer.
2. Install it like any other app and open it.
3. Wait until Docker Desktop says it is **running**. You can leave it running in the background.

You need about **8 GB of RAM** and **10 GB of free disk space**.

### Step 2: Get an Anthropic API key (optional)

This lets hard questions go to Claude. It's pay-as-you-go and costs a few cents per hard
question. **Skip this step if you only want the free local AI.**

1. Go to https://console.anthropic.com and sign up.
2. Add billing, open **API Keys**, click **Create Key** and copy it. It starts with `sk-ant-`.
   Keep it private, like a password.

### Step 3: Download this project

1. At the top of this page on GitHub, click the green **Code** button, then **Download ZIP**.
2. Find the ZIP in your Downloads folder and unzip it (double-click on Mac; on Windows,
   right-click it and choose **Extract All**).

### Step 4: Double-click Start

Open the unzipped `slm-router` folder.

| Your computer | Double-click |
|---|---|
| **Windows** | `Start.bat` |
| **Mac** | `Start.command` |

A window opens and walks you through it:

- **The first time**, it asks for your API key. Paste it and press **Enter**, or just
  press **Enter** to skip.
- **The first start downloads about 1.5 GB** and can take **5–15 minutes**. Later starts take
  seconds.
- When it says **Ready!**, your browser opens the chat page automatically.

> **Mac says it "cannot be opened because it is from an unidentified developer"?**
> Right-click (or Control-click) `Start.command`, choose **Open**, then **Open** again. On
> newer macOS versions, if there's no **Open** button, go to **System Settings → Privacy &
> Security**, scroll down, and click **Open Anyway** next to `Start.command`. You only need to
> do this once.
>
> **Windows shows "Windows protected your PC"?** Click **More info**, then **Run anyway**.

### Step 5: Ask questions

The chat page is at **http://localhost:8080**. Type a question and press **Enter**. Under each
answer you'll see who answered it:

- 🔵 **Local model (free)**: the small AI on your computer
- 🟠 **Claude**: the big AI. This one costs money.
- 🟢 **Memory (instant, free)**: an answer it already knew

The **Who answers** menu lets you force one or the other. **New chat** starts over.

The first answer from the local AI can take 20–30 seconds while it wakes up. That's normal.

### Step 6: See the dashboard

Click **Open the dashboard** at the top of the chat page (or go to **http://localhost:3000**)
and pick **SLM Router**. It shows how many questions each AI answered, how fast it was,
and how much money was spent and saved.

### Step 7: Stop it

Double-click `Stop.bat` (Windows) or `Stop.command` (Mac). The downloaded AI models are kept,
so next time starts quickly. To start again, double-click **Start**.

### If something goes wrong

| Problem | Fix |
|---|---|
| "Docker is not installed" | Do Step 1, then double-click Start again. |
| "Docker didn't start" | Open Docker Desktop yourself, wait until it says it's running, then double-click Start again. |
| The chat page doesn't load | Wait a minute and refresh. The first start can be slow. |
| Answers always say "Local model" even for hard questions | Your API key is missing or wrong. Open the `.env` file in the folder with a text editor, fix the line `ANTHROPIC_API_KEY=...`, save, then double-click Stop and Start. |
| You want a spending limit | In `.env`, set `ROUTER_FRONTIER_DAILY_BUDGET_USD=1` for a $1/day cap, then Stop and Start. |

To see a file called `.env`, you may need to show hidden files. On Mac, press
**Cmd + Shift + .** in Finder. On Windows, open File Explorer → **View** → **Show** →
**Hidden items**.

## Quick start

```bash
cp .env.example .env            # add ANTHROPIC_API_KEY
docker compose up -d --build    # ollama, model pulls, redis 8, router, prometheus, grafana
```

The first start downloads about 1.5 GB of models. Then open http://localhost:8080 for a chat page, or:

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
pytest                                                  # 38 tests, no network; Redis test uses a local redis-server if present
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
