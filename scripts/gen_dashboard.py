"""Generates deploy/grafana/dashboards/slm-router.json. Re-run after editing panels."""

import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prom"}
# Fixed categorical order (dark-surface steps): local=slot1, frontier=slot2, cache=slot3.
TIER_COLORS = {"local": "#3987e5", "frontier": "#d95926", "cache": "#199e70"}
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]

panels, pid = [], 0


def nid():
    global pid
    pid += 1
    return pid


def tier_overrides():
    return [{"matcher": {"id": "byName", "options": t},
             "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": c}}]}
            for t, c in TIER_COLORS.items()]


def stat(title, expr, x, unit="none", decimals=None, desc=""):
    p = {"id": nid(), "type": "stat", "title": title, "description": desc, "datasource": DS,
         "gridPos": {"x": x, "y": 0, "w": 4, "h": 4},
         "targets": [{"refId": "A", "expr": expr, "datasource": DS}],
         "options": {"colorMode": "none", "graphMode": "none", "textMode": "value",
                     "reduceOptions": {"calcs": ["lastNotNull"]}},
         "fieldConfig": {"defaults": {"unit": unit, "color": {"mode": "fixed", "fixedColor": "text"}}, "overrides": []}}
    if decimals is not None:
        p["fieldConfig"]["defaults"]["decimals"] = decimals
    return p


def ts(title, targets, x, y, w=12, h=8, unit="none", overrides=None, stack=False, desc=""):
    return {"id": nid(), "type": "timeseries", "title": title, "description": desc, "datasource": DS,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [{"refId": chr(65 + i), "expr": e, "legendFormat": l, "datasource": DS}
                        for i, (e, l) in enumerate(targets)],
            "options": {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                        "tooltip": {"mode": "multi", "sort": "desc"}},
            "fieldConfig": {"defaults": {"unit": unit, "custom": {
                "lineWidth": 2, "fillOpacity": 25 if stack else 0, "showPoints": "never",
                "stacking": {"mode": "normal" if stack else "none"}},
                "color": {"mode": "palette-classic"}}, "overrides": overrides or []}}


R = "$__range"
panels += [
    stat("Served without frontier", f"sum(increase(router_requests_total{{tier=~\"local|cache\"}}[{R}])) / clamp_min(sum(increase(router_requests_total[{R}])), 1)",
         0, "percentunit", 1, "Share of requests answered by the local SLM or the semantic cache."),
    stat("Frontier spend", f"sum(increase(router_frontier_cost_usd_total[{R}]))", 4, "currencyUSD", 4),
    stat("Est. savings", f"sum(increase(router_estimated_savings_usd_total[{R}]))", 8, "currencyUSD", 4,
         "Counterfactual frontier cost of locally/cache-served requests. Order-of-magnitude: tokenizers differ."),
    stat("Cache hit rate", f"sum(increase(router_cache_lookups_total{{result=\"hit\"}}[{R}])) / clamp_min(sum(increase(router_cache_lookups_total[{R}])), 1)",
         12, "percentunit", 1),
    stat("Escalation rate", f"sum(increase(router_escalations_total[{R}])) / clamp_min(sum(increase(router_requests_total{{tier=~\"local|frontier\"}}[{R}])), 1)",
         16, "percentunit", 1, "Requests first sent to the local model that ended up on the frontier."),
    stat("Budget left today", "router_frontier_budget_remaining_usd", 20, "currencyUSD", 2, "-1 means no daily cap."),
]
panels += [
    ts("Requests / s by tier", [("sum by (tier) (rate(router_requests_total[1m]))", "{{tier}}")], 0, 4,
       unit="reqps", overrides=tier_overrides(), stack=True),
    ts("p95 latency by tier", [("histogram_quantile(0.95, sum by (le, tier) (rate(router_request_latency_seconds_bucket[5m])))", "{{tier}}")],
       12, 4, unit="s", overrides=tier_overrides()),
    ts("p50 latency by tier", [("histogram_quantile(0.5, sum by (le, tier) (rate(router_request_latency_seconds_bucket[5m])))", "{{tier}}")],
       0, 12, unit="s", overrides=tier_overrides()),
    ts("p95 time to first token (streaming)", [("histogram_quantile(0.95, sum by (le, tier) (rate(router_ttft_seconds_bucket[5m])))", "{{tier}}")],
       12, 12, unit="s", overrides=tier_overrides()),
    ts("Escalations by reason", [("sum by (reason) (increase(router_escalations_total[5m]))", "{{reason}}")], 0, 20,
       overrides=[{"matcher": {"id": "byName", "options": r},
                   "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": SERIES[i]}}]}
                  for i, r in enumerate(["local_error", "local_timeout", "local_ttft_budget",
                                         "local_low_confidence", "local_truncated", "empty_local_answer"])]),
    ts("Tokens / s by tier", [("sum by (tier, direction) (rate(router_tokens_total[1m]))", "{{tier}} {{direction}}")],
       12, 20, unit="short"),
    {"id": nid(), "type": "bargauge", "title": "Complexity score distribution (selected range)", "datasource": DS,
     "description": "Cumulative share of requests at or below each score. The router threshold defaults to 0.5.",
     "gridPos": {"x": 0, "y": 28, "w": 24, "h": 7},
     "targets": [{"refId": "A", "datasource": DS, "format": "heatmap", "legendFormat": "{{le}}",
                  "expr": f"sum by (le) (increase(router_complexity_score_bucket[{R}])) / ignoring(le) group_left sum(increase(router_complexity_score_count[{R}]))"}],
     "options": {"orientation": "vertical", "displayMode": "basic", "showUnfilled": False,
                 "reduceOptions": {"calcs": ["lastNotNull"]}},
     "fieldConfig": {"defaults": {"unit": "percentunit", "min": 0, "max": 1,
                                  "color": {"mode": "fixed", "fixedColor": TIER_COLORS["local"]}}, "overrides": []}},
]

dash = {"uid": "slm-router", "title": "SLM Router", "schemaVersion": 39, "time": {"from": "now-1h", "to": "now"},
        "refresh": "10s", "tags": ["llm", "router"], "panels": panels}
out = Path(__file__).resolve().parent.parent / "deploy/grafana/dashboards/slm-router.json"
out.write_text(json.dumps(dash, indent=2) + "\n")
print(f"wrote {out} ({len(panels)} panels)")
