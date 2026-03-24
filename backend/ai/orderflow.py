"""
ai/orderflow.py  — Order Flow Simulation Engine
═══════════════════════════════════════════════════════════════════════════════
Simulates institutional order flow using pure OHLCV data (no L2 feed required).

Detected patterns
─────────────────
ABSORPTION    — high volume but small candle body; price holding despite pressure
AGGRESSIVE BUY — large bull body, upper close, volume spike; institutions lifting offers
AGGRESSIVE SELL — large bear body, lower close, volume spike; hitting bids hard
EXHAUSTION    — volume diminishing on new price high/low; move running out of steam
NOISE         — inconsistent candles, no clear order flow direction

OrderFlowResult.score  0–1
  ≥ 0.65  → clear institutional flow, safe to trade with
  0.40–0.65 → mixed; use with caution
  < 0.40  → unclear / noise; signal_scorer blocks entry
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class OrderFlowResult:
    score:           float   # 0–1: clarity of order flow
    bias:            str     # bullish | bearish | neutral | noise
    pattern:         str     # absorption | aggressive_buy | aggressive_sell | exhaustion | noise
    absorption:      bool
    aggressive_buy:  bool
    aggressive_sell: bool
    exhaustion:      bool
    volume_trend:    str     # rising | falling | flat
    body_ratio:      float   # avg candle body / range ratio (0–1)
    rationale:       str

    def to_dict(self) -> dict:
        return {
            "score":           round(self.score, 4),
            "bias":            self.bias,
            "pattern":         self.pattern,
            "absorption":      self.absorption,
            "aggressive_buy":  self.aggressive_buy,
            "aggressive_sell": self.aggressive_sell,
            "exhaustion":      self.exhaustion,
            "volume_trend":    self.volume_trend,
            "body_ratio":      round(self.body_ratio, 4),
            "rationale":       self.rationale,
        }


def analyse_orderflow(
    opens:   list[float],
    highs:   list[float],
    lows:    list[float],
    closes:  list[float],
    volumes: list[float],
    lookback: int = 20,
) -> OrderFlowResult:
    """
    Analyse the last `lookback` candles for institutional order flow patterns.

    Algorithm
    ─────────
    1. Compute per-candle metrics: body %, wick ratio, bull/bear
    2. Calculate average volume and detect spikes (> 1.8× avg)
    3. Detect absorption: high volume + small body (< 30% of range)
    4. Detect aggressive buying/selling: body > 65% of range + volume spike
    5. Detect exhaustion: volume decreasing on last 3 new highs/lows
    6. Score = weighted combination of clarity signals
    """
    n = min(lookback, len(closes))
    if n < 5:
        return OrderFlowResult(
            score=0.5, bias="neutral", pattern="noise",
            absorption=False, aggressive_buy=False, aggressive_sell=False,
            exhaustion=False, volume_trend="flat", body_ratio=0.5,
            rationale="Insufficient data for order flow analysis",
        )

    o = opens[-n:];  h = highs[-n:]
    l = lows[-n:];   c = closes[-n:]
    v = volumes[-n:]

    avg_vol = sum(v) / n
    if avg_vol == 0:
        avg_vol = 1.0

    # Per-candle metrics
    bodies, bull_vols, bear_vols = [], [], []
    for i in range(n):
        rng  = h[i] - l[i]
        body = abs(c[i] - o[i])
        body_pct = body / rng if rng > 0 else 0.0
        bodies.append(body_pct)
        if c[i] >= o[i]:
            bull_vols.append(v[i])
        else:
            bear_vols.append(v[i])

    avg_body_ratio = sum(bodies) / n

    # Volume trend (last 5 vs previous 5)
    if n >= 10:
        recent_avg = sum(v[-5:]) / 5
        prior_avg  = sum(v[-10:-5]) / 5
        if   recent_avg > prior_avg * 1.15: vol_trend = "rising"
        elif recent_avg < prior_avg * 0.85: vol_trend = "falling"
        else:                                vol_trend = "flat"
    else:
        vol_trend = "flat"

    # Detect signals
    absorption = False
    agg_buy    = False
    agg_sell   = False
    exhaustion = False

    last_3 = list(zip(h[-3:], l[-3:], c[-3:], v[-3:], bodies[-3:]))

    for hi, lo, cl, vi, bp in last_3:
        spike = vi > avg_vol * 1.8

        # Absorption: big volume, tiny body
        if spike and bp < 0.30:
            absorption = True

        # Aggressive buying: big bull body + volume spike
        rng = hi - lo
        if rng > 0 and spike and bp > 0.65:
            if cl > (lo + rng * 0.65):   # closed in upper 35%
                agg_buy = True
            else:
                agg_sell = True

    # Exhaustion: price making new high/low but volume declining
    recent_vols = v[-5:]
    if len(recent_vols) >= 3:
        if c[-1] > max(c[-6:-1]) and recent_vols[-1] < recent_vols[-3] * 0.75:
            exhaustion = True
        elif c[-1] < min(c[-6:-1]) and recent_vols[-1] < recent_vols[-3] * 0.75:
            exhaustion = True

    # Determine bias from bull/bear volume dominance
    total_bull = sum(bull_vols) or 0.001
    total_bear = sum(bear_vols) or 0.001
    ratio = total_bull / (total_bull + total_bear)

    if   ratio >= 0.62: bias = "bullish"
    elif ratio <= 0.38: bias = "bearish"
    else:               bias = "neutral"

    # Determine primary pattern
    if   agg_buy:    pattern = "aggressive_buy"
    elif agg_sell:   pattern = "aggressive_sell"
    elif absorption: pattern = "absorption"
    elif exhaustion: pattern = "exhaustion"
    else:            pattern = "noise"

    # Score calculation
    score = 0.0

    # Base: clarity from body ratio (high body = clear direction)
    clarity = min(avg_body_ratio / 0.6, 1.0)
    score  += clarity * 0.30

    # Volume trend contribution
    if vol_trend == "rising":
        score += 0.25
    elif vol_trend == "flat":
        score += 0.15

    # Pattern bonuses
    if agg_buy or agg_sell:  score += 0.30
    if absorption:            score += 0.15
    if exhaustion:            score -= 0.20   # exhaustion is bearish for continuation

    # Bias strength
    bias_strength = abs(ratio - 0.5) * 2.0   # 0–1
    score += bias_strength * 0.15

    score = round(max(0.0, min(1.0, score)), 4)

    rationale = (
        f"Pattern={pattern} bias={bias} vol_trend={vol_trend} "
        f"body_ratio={avg_body_ratio:.2f} score={score:.2f}"
    )

    return OrderFlowResult(
        score=score, bias=bias, pattern=pattern,
        absorption=absorption, aggressive_buy=agg_buy, aggressive_sell=agg_sell,
        exhaustion=exhaustion, volume_trend=vol_trend,
        body_ratio=round(avg_body_ratio, 4), rationale=rationale,
    )
