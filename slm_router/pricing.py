"""Per-token prices used to estimate frontier spend and local savings.

Anthropic list prices (USD per 1M tokens, first-party API) as of 2026-10. Verify
against https://www.anthropic.com/pricing before relying on them. For any other
provider, supply prices via ROUTER_PRICING_JSON, e.g.
  ROUTER_PRICING_JSON='{"gpt-x": [1.25, 10.0]}'
Unknown models are priced at 0 and logged once, so savings are under-reported
rather than invented.
"""

import json
import logging
import os

log = logging.getLogger(__name__)

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
PRICES_PER_MTOK.update({k: tuple(v) for k, v in json.loads(os.environ.get("ROUTER_PRICING_JSON", "{}")).items()})

_warned: set[str] = set()


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price = PRICES_PER_MTOK.get(model)
    if price is None:
        if model not in _warned:
            log.warning("no price for model %r; set ROUTER_PRICING_JSON to track its cost", model)
            _warned.add(model)
        return 0.0
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
