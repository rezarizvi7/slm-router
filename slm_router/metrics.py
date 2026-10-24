from prometheus_client import Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120)

REQUESTS = Counter("router_requests_total", "Requests by the tier that finally served them", ["tier"])
ESCALATIONS = Counter("router_escalations_total", "Local -> frontier escalations", ["reason"])
DEGRADATIONS = Counter("router_degradations_total", "Frontier -> local fallbacks", ["reason"])
LATENCY = Histogram("router_request_latency_seconds", "End-to-end latency", ["tier", "stream"], buckets=LATENCY_BUCKETS)
TTFT = Histogram("router_ttft_seconds", "Time to first token (streaming)", ["tier"], buckets=LATENCY_BUCKETS)
TOKENS = Counter("router_tokens_total", "Tokens processed", ["tier", "direction"])
FRONTIER_COST = Counter("router_frontier_cost_usd_total", "Estimated frontier spend (USD)")
SAVINGS = Counter("router_estimated_savings_usd_total",
                  "Estimated frontier cost avoided by serving locally or from cache (USD)", ["tier"])
COMPLEXITY = Histogram("router_complexity_score", "Heuristic complexity score",
                       buckets=(0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0))
JUDGE = Counter("router_judge_total", "SLM judge verdicts in the ambiguous band", ["verdict"])
CACHE = Counter("router_cache_lookups_total", "Semantic cache lookups", ["result"])
CACHE_DISTANCE = Histogram("router_cache_hit_distance", "Cosine distance of cache hits",
                           buckets=(0.0, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15))
BACKEND_ERRORS = Counter("router_backend_errors_total", "Backend failures", ["tier"])
BUDGET_REMAINING = Gauge("router_frontier_budget_remaining_usd", "Remaining daily frontier budget (-1 = unlimited)")
