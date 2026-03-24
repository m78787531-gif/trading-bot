"""
ai/market_bias.py  — Funding Rate & Market Bias Engine
═══════════════════════════════════════════════════════════════════════════════
Integrates exchange funding rate and open interest to determine crowd positioning.

Bias Logic (smart-money contrarian)
─────────────────────────────────────
  Funding Rate > +0.05%  → market crowded LONG → bias SHORT (retail long squeeze risk)
  Funding Rate < -0.05%  → market crowded SHORT → bias LONG (short squeeze potential)
  Funding Rate neutral   → no crowd bias

  OI rising + price rising    → trend continuation (healthy)
  OI rising + price falling   → forced liquidations possible → caution
  OI falling + price rising   → short covering, weakening rally
  OI falling + price falling  → capitulation, potential reversal

Output: MarketBiasResult
  directional_bias:   long | short | neutral
  funding_aligned:    bool — True if direction matches bias
  crowd_state:        crowded_long | crowded_short | balanced
  oi_trend:           bullish | bearish | neutral
  bias_score:         –1..+1 (positive = bullish bias)
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Funding thresholds (annualised equivalent: 0.01% per 8h ≈ 10.95% annualised)
FUNDING_CROWDED_LONG_THRESHOLD  =  0.05   # % per 8 hours
FUNDING_CROWDED_SHORT_THRESHOLD = -0.05


@dataclass
class MarketBiasResult:
    directional_bias:  str    # long | short | neutral
    funding_rate:      float  # raw 8h funding rate %
    crowd_state:       str    # crowded_long | crowded_short | balanced
    oi_trend:          str    # bullish | bearish | neutral
    bias_score:        float  # –1..+1
    funding_aligned:   bool   # True if trade direction matches bias
    rationale:         str

    def aligned_with(self, direction: str) -> bool:
        """True if market bias supports the intended trade direction."""
        if self.directional_bias == "neutral":
            return True   # no crowd signal = no contra-indicator
        return (direction == "long"  and self.directional_bias == "long") or                (direction == "short" and self.directional_bias == "short")

    def to_dict(self) -> dict:
        return {
            "directional_bias": self.directional_bias,
            "funding_rate":     round(self.funding_rate, 6),
            "crowd_state":      self.crowd_state,
            "oi_trend":         self.oi_trend,
            "bias_score":       round(self.bias_score, 4),
            "funding_aligned":  self.funding_aligned,
            "rationale":        self.rationale,
        }


def calculate_market_bias(
    funding_rate:    float,          # 8h funding rate % (positive = longs pay)
    open_interest:   Optional[float] = None,   # current OI in USD
    prev_open_interest: Optional[float] = None, # OI 1 period ago
    price_change_pct:   float = 0.0,           # recent price change %
    direction:          str   = "long",        # intended trade direction
) -> MarketBiasResult:
    """
    Calculate market bias from funding rate and open interest.

    Funding rate source: fetch from exchange REST API before each cycle.
    If unavailable, defaults to neutral (does not block trades).
    """
    reasons: list[str] = []
    bias_score = 0.0

    # ── Funding rate analysis ─────────────────────────────────────────────
    if funding_rate > FUNDING_CROWDED_LONG_THRESHOLD:
        crowd_state = "crowded_long"
        # Smart money contrarian: high funding → shorts more attractive
        # But if we're going long, this is a contra-indicator
        bias_score -= (funding_rate / 0.1) * 0.4   # penalise longs
        reasons.append(
            f"Funding {funding_rate:+.4f}% → crowded long "
            f"(longs paying premium, short-squeeze risk lower)"
        )
    elif funding_rate < FUNDING_CROWDED_SHORT_THRESHOLD:
        crowd_state = "crowded_short"
        bias_score += (abs(funding_rate) / 0.1) * 0.4
        reasons.append(
            f"Funding {funding_rate:+.4f}% → crowded short "
            f"(shorts paying premium, long-squeeze potential)"
        )
    else:
        crowd_state = "balanced"
        reasons.append(f"Funding {funding_rate:+.4f}% → balanced positioning")

    # ── Open interest analysis ────────────────────────────────────────────
    oi_trend = "neutral"
    if open_interest is not None and prev_open_interest and prev_open_interest > 0:
        oi_change_pct = (open_interest - prev_open_interest) / prev_open_interest * 100

        if oi_change_pct > 2.0:
            if price_change_pct > 0:
                oi_trend = "bullish"
                bias_score += 0.3
                reasons.append(f"OI +{oi_change_pct:.1f}% with rising price → trend continuation")
            else:
                oi_trend = "bearish"
                bias_score -= 0.2
                reasons.append(f"OI +{oi_change_pct:.1f}% with falling price → liquidation risk")
        elif oi_change_pct < -2.0:
            if price_change_pct > 0:
                oi_trend = "neutral"
                bias_score -= 0.1
                reasons.append(f"OI {oi_change_pct:.1f}% falling with rising price → short covering")
            else:
                oi_trend = "bullish"  # capitulation reversal potential
                bias_score += 0.15
                reasons.append(f"OI {oi_change_pct:.1f}% falling with falling price → capitulation")
        else:
            reasons.append(f"OI change {oi_change_pct:+.1f}% — stable positioning")

    # ── Determine directional bias ────────────────────────────────────────
    bias_score = max(-1.0, min(1.0, bias_score))

    if   bias_score >=  0.20: directional_bias = "long"
    elif bias_score <= -0.20: directional_bias = "short"
    else:                      directional_bias = "neutral"

    aligned = (
        directional_bias == "neutral" or
        (direction == "long"  and directional_bias == "long") or
        (direction == "short" and directional_bias == "short")
    )

    return MarketBiasResult(
        directional_bias = directional_bias,
        funding_rate     = funding_rate,
        crowd_state      = crowd_state,
        oi_trend         = oi_trend,
        bias_score       = round(bias_score, 4),
        funding_aligned  = aligned,
        rationale        = " | ".join(reasons),
    )


async def fetch_funding_rate(symbol: str, exchange: str = "binance") -> float:
    """
    Fetch current 8-hour funding rate from exchange REST API.
    Returns 0.0 on failure (safe default — no bias applied).
    """
    import httpx
    try:
        if exchange.lower() == "binance":
            url = "https://fapi.binance.com/fapi/v1/premiumIndex"
            async with httpx.AsyncClient(timeout=8) as c:
                r = await c.get(url, params={"symbol": symbol})
                if r.status_code == 200:
                    return float(r.json().get("lastFundingRate", 0.0)) * 100
        elif exchange.lower() == "bybit":
            url = "https://api.bybit.com/v5/market/funding/history"
            async with httpx.AsyncClient(timeout=8) as c:
                r = await c.get(url, params={"category": "linear",
                                              "symbol": symbol, "limit": 1})
                if r.status_code == 200:
                    data = r.json().get("result", {}).get("list", [])
                    if data:
                        return float(data[0].get("fundingRate", 0.0)) * 100
    except Exception as e:
        logger.debug(f"Funding rate fetch [{symbol}]: {e}")
    return 0.0   # neutral fallback
