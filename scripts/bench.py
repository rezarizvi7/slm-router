"""Benchmark and tune the router against a labelled prompt set.

  # Offline: score the heuristic against labels and sweep the threshold (no model calls)
  python scripts/bench.py offline scripts/prompts.jsonl

  # Live: send every prompt through a running router and report tier mix, latency, spend
  python scripts/bench.py live scripts/prompts.jsonl --url http://localhost:8080 --repeat 2

Input is JSONL with {"prompt": str, "label": "easy"|"hard"}. Labels mean "a small
local model answers this acceptably" -- build them from your own traffic.
"""

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slm_router.complexity import score  # noqa: E402


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def offline(rows):
    scored = [(score([{"role": "user", "content": r["prompt"]}]).score, r["label"], r["prompt"]) for r in rows]
    print(f"{'thr':>5} {'acc':>6} {'local%':>7} {'hard→local':>11} {'easy→frontier':>14}")
    for thr in [x / 20 for x in range(4, 17)]:
        hard_local = sum(1 for s, l, _ in scored if l == "hard" and s < thr)
        easy_front = sum(1 for s, l, _ in scored if l == "easy" and s >= thr)
        acc = 1 - (hard_local + easy_front) / len(scored)
        local = sum(1 for s, _, _ in scored if s < thr) / len(scored)
        print(f"{thr:5.2f} {acc:6.1%} {local:7.1%} {hard_local:11d} {easy_front:14d}")
    print("\nMisrouted at 0.50:")
    for s, l, p in sorted(scored):
        if (l == "hard") != (s >= 0.5):
            print(f"  {s:.2f} [{l}] {p[:90]!r}")


def live(rows, url, repeat):
    import httpx

    tiers, lat, esc = Counter(), {}, Counter()
    agree = total = 0
    with httpx.Client(base_url=url, timeout=300) as c:
        for _ in range(repeat):
            for r in rows:
                t0 = time.perf_counter()
                resp = c.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": r["prompt"]}]})
                dt = time.perf_counter() - t0
                resp.raise_for_status()
                tier = resp.headers["x-router-tier"]
                tiers[tier] += 1
                lat.setdefault(tier, []).append(dt)
                if e := resp.headers.get("x-router-escalation"):
                    esc[e] += 1
                if tier != "cache":
                    total += 1
                    agree += (tier == "frontier") == (r["label"] == "hard")
    n = sum(tiers.values())
    print(f"requests: {n}")
    for t, k in tiers.most_common():
        q = sorted(lat[t])
        print(f"  {t:9s} {k / n:6.1%}  p50 {statistics.median(q) * 1000:7.0f}ms  p95 {q[int(0.95 * (len(q) - 1))] * 1000:7.0f}ms")
    print(f"served without frontier: {(tiers['local'] + tiers['cache']) / n:.1%}")
    print(f"routing agreement with labels (non-cache): {agree / max(total, 1):.1%}")
    if esc:
        print("escalations:", dict(esc))
    m = httpx.get(f"{url}/metrics").text
    for line in m.splitlines():
        if line.startswith(("router_frontier_cost_usd_total ", "router_estimated_savings_usd_total")):
            print(" ", line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["offline", "live"])
    ap.add_argument("file")
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()
    rows = load(a.file)
    offline(rows) if a.mode == "offline" else live(rows, a.url, a.repeat)
