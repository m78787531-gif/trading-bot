"""
ai/smc_engine.py
─────────────────────────────────────────────────────────────────────────────
Smart Money Concepts (SMC) detection engine.

Detects:
  • Order Blocks (OB)       — last bearish/bullish candle before an impulse move
  • Fair Value Gaps (FVG)   — imbalance zones price is likely to revisit
  • Liquidity Sweeps        — stop hunts above swing highs / below swing lows
  • Break of Structure (BOS) — confirms new trend direction
  • Change of Character (CHoCH) — early trend reversal signal
  • Accumulation / Distribution zones — Wyckoff-inspired range detection
  • Premium / Discount zones — current price position vs session range

All functions take OHLCV arrays (oldest → newest) and return dataclasses
that are JSON-serialisable for the signal scorer and API responses.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrderBlock:
    index: int           # candle index in the series
    type: str            # bullish | bearish
    high: float
    low: float
    open: float
    close: float
    strength: float      # 0–1  (size of subsequent move / ATR)
    mitigated: bool      # price has returned to fill the OB
    valid: bool          # still relevant (not fully mitigated)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FairValueGap:
    index: int           # middle candle index
    type: str            # bullish | bearish
    top: float
    bottom: float
    size: float          # top - bottom
    filled_pct: float    # 0–100 how much has been filled
    valid: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LiquiditySweep:
    index: int
    type: str            # buy_side (high swept) | sell_side (low swept)
    level: float         # the level that was swept
    sweep_price: float
    reversal_confirmed: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StructureBreak:
    index: int
    type: str            # bos_bullish | bos_bearish | choch_bullish | choch_bearish
    level: float
    confirmed: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MarketPhase:
    phase: str           # accumulation | distribution | markup | markdown | unknown
    confidence: float    # 0–1
    range_high: float
    range_low: float
    duration_candles: int
    bias: str            # bullish | bearish | neutral

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SMCAnalysis:
    symbol: str
    timeframe: str
    current_price: float
    order_blocks: list[OrderBlock]
    fair_value_gaps: list[FairValueGap]
    liquidity_sweeps: list[LiquiditySweep]
    structure_breaks: list[StructureBreak]
    market_phase: MarketPhase
    premium_discount: str       # premium | discount | equilibrium
    nearest_ob_support: Optional[float]
    nearest_ob_resistance: Optional[float]
    smc_score: float            # –10 to +10 (bearish → bullish)
    summary: str

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "current_price": self.current_price,
            "order_blocks": [ob.to_dict() for ob in self.order_blocks],
            "fair_value_gaps": [fvg.to_dict() for fvg in self.fair_value_gaps],
            "liquidity_sweeps": [ls.to_dict() for ls in self.liquidity_sweeps],
            "structure_breaks": [sb.to_dict() for sb in self.structure_breaks],
            "market_phase": self.market_phase.to_dict(),
            "premium_discount": self.premium_discount,
            "nearest_ob_support": self.nearest_ob_support,
            "nearest_ob_resistance": self.nearest_ob_resistance,
            "smc_score": self.smc_score,
            "summary": self.summary,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _swing_highs(highs: list[float], lows: list[float], lookback: int = 5) -> list[int]:
    """Return indices of swing highs (local maxima)."""
    result = []
    for i in range(lookback, len(highs) - lookback):
        if all(highs[i] >= highs[i - j] for j in range(1, lookback + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, lookback + 1)):
            result.append(i)
    return result


def _swing_lows(highs: list[float], lows: list[float], lookback: int = 5) -> list[int]:
    """Return indices of swing lows (local minima)."""
    result = []
    for i in range(lookback, len(lows) - lookback):
        if all(lows[i] <= lows[i - j] for j in range(1, lookback + 1)) and \
           all(lows[i] <= lows[i + j] for j in range(1, lookback + 1)):
            result.append(i)
    return result


def _candle_body_size(opens: list[float], closes: list[float], i: int) -> float:
    return abs(closes[i] - opens[i])


def _is_bullish(opens: list[float], closes: list[float], i: int) -> bool:
    return closes[i] > opens[i]


def _atr_approx(highs: list[float], lows: list[float], closes: list[float],
                period: int = 14) -> float:
    trs = [highs[i] - lows[i] for i in range(len(closes))]
    return sum(trs[-period:]) / min(period, len(trs)) if trs else closes[-1] * 0.01


# ─────────────────────────────────────────────────────────────────────────────
# 1. Order Blocks
# ─────────────────────────────────────────────────────────────────────────────

def detect_order_blocks(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    max_blocks: int = 5,
    lookback: int = 50,
) -> list[OrderBlock]:
    """
    An Order Block is the last opposing candle before a strong impulse.
    Bullish OB = last bearish candle before a strong bull run.
    Bearish OB = last bullish candle before a strong bear run.
    """
    blocks: list[OrderBlock] = []
    n = len(closes)
    start = max(0, n - lookback)
    atr = _atr_approx(highs, lows, closes)
    current = closes[-1]

    for i in range(start + 2, n - 1):
        # Look for a strong move AFTER candle i
        impulse_size = abs(closes[i + 1] - opens[i + 1])
        if impulse_size < atr * 1.5:
            continue  # not a strong enough impulse

        is_bull_impulse = closes[i + 1] > opens[i + 1]
        is_bear_impulse = not is_bull_impulse

        # Bullish OB: preceding bearish candle before bull impulse
        if is_bull_impulse and not _is_bullish(opens, closes, i):
            # Check subsequent candles confirm bullish move
            strength = impulse_size / atr
            mitigated = any(lows[j] <= closes[i] for j in range(i + 2, n))
            valid = not mitigated or (current > lows[i])
            blocks.append(OrderBlock(
                index=i, type="bullish",
                high=highs[i], low=lows[i],
                open=opens[i], close=closes[i],
                strength=min(round(strength / 5, 3), 1.0),
                mitigated=mitigated, valid=valid,
            ))

        # Bearish OB: preceding bullish candle before bear impulse
        elif is_bear_impulse and _is_bullish(opens, closes, i):
            strength = impulse_size / atr
            mitigated = any(highs[j] >= closes[i] for j in range(i + 2, n))
            valid = not mitigated or (current < highs[i])
            blocks.append(OrderBlock(
                index=i, type="bearish",
                high=highs[i], low=lows[i],
                open=opens[i], close=closes[i],
                strength=min(round(strength / 5, 3), 1.0),
                mitigated=mitigated, valid=valid,
            ))

    # Return most recent valid blocks, sorted by recency
    valid_blocks = [b for b in blocks if b.valid]
    return sorted(valid_blocks, key=lambda b: b.index, reverse=True)[:max_blocks]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fair Value Gaps
# ─────────────────────────────────────────────────────────────────────────────

def detect_fvgs(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    lookback: int = 50,
    min_size_atr_ratio: float = 0.5,
) -> list[FairValueGap]:
    """
    A Fair Value Gap (FVG) / imbalance occurs when candle[i+2].low > candle[i].high
    (bullish FVG) or candle[i+2].high < candle[i].low (bearish FVG).
    """
    fvgs: list[FairValueGap] = []
    n = len(closes)
    start = max(0, n - lookback)
    atr = _atr_approx(highs, lows, closes)
    current = closes[-1]

    for i in range(start, n - 2):
        # Bullish FVG
        if lows[i + 2] > highs[i]:
            size = lows[i + 2] - highs[i]
            if size >= atr * min_size_atr_ratio:
                # How much has been filled?
                min_low_after = min(lows[i + 2:]) if i + 2 < n else lows[i + 2]
                filled = max(0.0, highs[i] + size - max(min_low_after, highs[i]))
                filled_pct = min(100.0, filled / size * 100)
                valid = filled_pct < 100 and current > highs[i]
                fvgs.append(FairValueGap(
                    index=i + 1, type="bullish",
                    top=round(lows[i + 2], 6), bottom=round(highs[i], 6),
                    size=round(size, 6), filled_pct=round(filled_pct, 2),
                    valid=valid,
                ))

        # Bearish FVG
        elif highs[i + 2] < lows[i]:
            size = lows[i] - highs[i + 2]
            if size >= atr * min_size_atr_ratio:
                max_high_after = max(highs[i + 2:]) if i + 2 < n else highs[i + 2]
                filled = max(0.0, max_high_after - lows[i] + size)
                filled_pct = min(100.0, filled / size * 100)
                valid = filled_pct < 100 and current < lows[i]
                fvgs.append(FairValueGap(
                    index=i + 1, type="bearish",
                    top=round(lows[i], 6), bottom=round(highs[i + 2], 6),
                    size=round(size, 6), filled_pct=round(filled_pct, 2),
                    valid=valid,
                ))

    valid_fvgs = [f for f in fvgs if f.valid]
    return sorted(valid_fvgs, key=lambda f: f.index, reverse=True)[:5]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Liquidity Sweeps
# ─────────────────────────────────────────────────────────────────────────────

def detect_liquidity_sweeps(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    lookback: int = 50,
) -> list[LiquiditySweep]:
    """
    Detect stop hunts / liquidity grabs.
    A sweep occurs when price momentarily breaks a swing high/low then reverses.
    """
    sweeps: list[LiquiditySweep] = []
    n = len(closes)
    start = max(0, n - lookback)
    sh_indices = _swing_highs(highs, lows, lookback=3)
    sl_indices = _swing_lows(highs, lows, lookback=3)

    for i in range(start + 1, n):
        # Buy-side liquidity sweep (sweep above swing high then reverse down)
        for sh in sh_indices:
            if sh >= i:
                continue
            level = highs[sh]
            if highs[i] > level and closes[i] < level:
                reversal = closes[i] < closes[i - 1] if i > 0 else False
                sweeps.append(LiquiditySweep(
                    index=i, type="buy_side",
                    level=round(level, 6),
                    sweep_price=round(highs[i], 6),
                    reversal_confirmed=reversal,
                ))
                break  # one sweep per candle

        # Sell-side liquidity sweep (sweep below swing low then reverse up)
        for sl in sl_indices:
            if sl >= i:
                continue
            level = lows[sl]
            if lows[i] < level and closes[i] > level:
                reversal = closes[i] > closes[i - 1] if i > 0 else False
                sweeps.append(LiquiditySweep(
                    index=i, type="sell_side",
                    level=round(level, 6),
                    sweep_price=round(lows[i], 6),
                    reversal_confirmed=reversal,
                ))
                break

    return sorted(sweeps, key=lambda s: s.index, reverse=True)[:5]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Break of Structure / Change of Character
# ─────────────────────────────────────────────────────────────────────────────

def detect_structure_breaks(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    lookback: int = 50,
) -> list[StructureBreak]:
    """
    BOS = price closes beyond the previous structure high/low (trend continuation).
    CHoCH = price breaks the opposing structure (potential reversal).
    """
    breaks: list[StructureBreak] = []
    n = len(closes)
    start = max(0, n - lookback)
    sh_indices = _swing_highs(highs, lows, lookback=3)
    sl_indices = _swing_lows(highs, lows, lookback=3)

    for i in range(start + 5, n):
        # Bullish BOS: close above last swing high
        recent_shs = [sh for sh in sh_indices if sh < i]
        recent_sls = [sl for sl in sl_indices if sl < i]

        if recent_shs:
            last_sh_level = highs[recent_shs[-1]]
            if closes[i] > last_sh_level:
                # BOS if previous trend was up, CHoCH if previous trend was down
                prev_trend = "up" if (len(recent_sls) >= 2 and
                    lows[recent_sls[-1]] > lows[recent_sls[-2]]) else "down"
                btype = "bos_bullish" if prev_trend == "up" else "choch_bullish"
                breaks.append(StructureBreak(
                    index=i, type=btype,
                    level=round(last_sh_level, 6), confirmed=True,
                ))

        if recent_sls:
            last_sl_level = lows[recent_sls[-1]]
            if closes[i] < last_sl_level:
                prev_trend = "down" if (len(recent_shs) >= 2 and
                    highs[recent_shs[-1]] < highs[recent_shs[-2]]) else "up"
                btype = "bos_bearish" if prev_trend == "down" else "choch_bearish"
                breaks.append(StructureBreak(
                    index=i, type=btype,
                    level=round(last_sl_level, 6), confirmed=True,
                ))

    # Deduplicate consecutive same-type breaks; keep most recent
    seen: set[str] = set()
    unique_breaks: list[StructureBreak] = []
    for b in sorted(breaks, key=lambda x: x.index, reverse=True):
        if b.type not in seen:
            unique_breaks.append(b)
            seen.add(b.type)
    return unique_breaks[:4]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Market Phase — Wyckoff-inspired Accumulation/Distribution
# ─────────────────────────────────────────────────────────────────────────────

def detect_market_phase(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
    lookback: int = 60,
) -> MarketPhase:
    """
    Infer Wyckoff market phase:
    • Accumulation — price ranging near lows, rising volume on bounces
    • Distribution — price ranging near highs, rising volume on drops
    • Markup       — price trending up, increasing volume
    • Markdown     — price trending down, increasing volume
    """
    n = len(closes)
    window = min(lookback, n)
    c = closes[-window:]
    h = highs[-window:]
    lo = lows[-window:]
    v = volumes[-window:] if volumes else [1.0] * window

    hi_range = max(h)
    lo_range = min(lo)
    range_size = hi_range - lo_range
    current = closes[-1]

    if range_size == 0:
        return MarketPhase("unknown", 0.0, current, current, window, "neutral")

    # Price position in range (0 = at low, 1 = at high)
    position = (current - lo_range) / range_size

    # Slope of prices over lookback (linear regression proxy)
    mid = window // 2
    first_half_avg = sum(c[:mid]) / mid
    second_half_avg = sum(c[mid:]) / (window - mid)
    slope = (second_half_avg - first_half_avg) / first_half_avg

    # Volume trend
    vol_first = sum(v[:mid]) / mid if mid > 0 else 1
    vol_second = sum(v[mid:]) / (window - mid) if window - mid > 0 else 1
    vol_slope = (vol_second - vol_first) / vol_first if vol_first > 0 else 0

    # Range tightness (low range = potential accumulation or distribution)
    range_pct = range_size / (sum(c) / len(c)) if c else 0
    is_ranging = range_pct < 0.08 and abs(slope) < 0.02

    phase = "unknown"
    confidence = 0.5
    bias = "neutral"

    if is_ranging:
        if position < 0.35:
            phase = "accumulation"
            bias = "bullish" if vol_slope > 0 else "neutral"
            confidence = 0.6 + abs(vol_slope) * 0.3
        elif position > 0.65:
            phase = "distribution"
            bias = "bearish" if vol_slope > 0 else "neutral"
            confidence = 0.6 + abs(vol_slope) * 0.3
        else:
            phase = "accumulation" if position < 0.5 else "distribution"
            confidence = 0.45
    else:
        if slope > 0.02:
            phase = "markup"
            bias = "bullish"
            confidence = 0.5 + min(slope * 5, 0.4)
        elif slope < -0.02:
            phase = "markdown"
            bias = "bearish"
            confidence = 0.5 + min(abs(slope) * 5, 0.4)

    return MarketPhase(
        phase=phase,
        confidence=round(min(confidence, 0.95), 3),
        range_high=round(hi_range, 6),
        range_low=round(lo_range, 6),
        duration_candles=window,
        bias=bias,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Premium / Discount
# ─────────────────────────────────────────────────────────────────────────────

def premium_discount_zone(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    lookback: int = 50,
) -> str:
    """
    Premium = price above 50% of the range (potential short zone).
    Discount = price below 50% of the range (potential long zone).
    """
    h = highs[-lookback:]
    lo = lows[-lookback:]
    hi = max(h)
    lo_val = min(lo)
    mid = (hi + lo_val) / 2
    equilibrium_band = (hi - lo_val) * 0.1  # ±10% around midpoint

    current = closes[-1]
    if current > mid + equilibrium_band:
        return "premium"
    elif current < mid - equilibrium_band:
        return "discount"
    return "equilibrium"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Master SMC Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_smc(
    symbol: str,
    timeframe: str,
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
) -> SMCAnalysis:
    """
    Run all SMC detections and return a unified SMCAnalysis with a composite score.

    Score range: –10 (strongly bearish) to +10 (strongly bullish).
    """
    current = closes[-1]

    obs = detect_order_blocks(opens, highs, lows, closes)
    fvgs = detect_fvgs(highs, lows, closes)
    sweeps = detect_liquidity_sweeps(highs, lows, closes)
    structure = detect_structure_breaks(highs, lows, closes)
    phase = detect_market_phase(highs, lows, closes, volumes)
    pd_zone = premium_discount_zone(highs, lows, closes)

    # ── Score computation ──────────────────────────────────────────────────
    score = 0.0

    # Order blocks near current price
    bull_obs = [ob for ob in obs if ob.type == "bullish" and ob.low <= current <= ob.high * 1.02]
    bear_obs = [ob for ob in obs if ob.type == "bearish" and ob.low * 0.98 <= current <= ob.high]
    score += len(bull_obs) * 1.5
    score -= len(bear_obs) * 1.5

    # FVGs pulling price
    bull_fvgs = [f for f in fvgs if f.type == "bullish" and f.bottom <= current <= f.top * 1.02]
    bear_fvgs = [f for f in fvgs if f.type == "bearish" and f.bottom * 0.98 <= current <= f.top]
    score += len(bull_fvgs) * 1.0
    score -= len(bear_fvgs) * 1.0

    # Recent structure breaks (most recent = most relevant)
    for sb in structure[:2]:
        if "bullish" in sb.type:
            score += 2.0 if "bos" in sb.type else 1.0   # BOS > CHoCH
        else:
            score -= 2.0 if "bos" in sb.type else 1.0

    # Recent liquidity sweeps with reversal
    for sw in sweeps[:2]:
        if sw.reversal_confirmed:
            if sw.type == "sell_side":   # swept lows → bullish reversal
                score += 1.5
            else:                         # swept highs → bearish reversal
                score -= 1.5

    # Market phase
    phase_scores = {
        "accumulation": 2.0, "markup": 1.5,
        "distribution": -2.0, "markdown": -1.5, "unknown": 0.0,
    }
    score += phase_scores.get(phase.phase, 0) * phase.confidence

    # Premium/discount zone bias
    if pd_zone == "discount":
        score += 1.0
    elif pd_zone == "premium":
        score -= 1.0

    score = round(max(-10.0, min(10.0, score)), 2)

    # ── Nearest OB support/resistance ────────────────────────────────────
    support_obs = [ob for ob in obs if ob.type == "bullish" and ob.high < current]
    resistance_obs = [ob for ob in obs if ob.type == "bearish" and ob.low > current]
    nearest_support = max((ob.high for ob in support_obs), default=None)
    nearest_resistance = min((ob.low for ob in resistance_obs), default=None)

    # ── Summary ──────────────────────────────────────────────────────────
    parts = [
        f"Phase: {phase.phase} ({phase.confidence:.0%} conf)",
        f"Zone: {pd_zone}",
        f"{len(bull_obs)} bullish OBs + {len(bear_obs)} bearish OBs near price",
        f"{len(fvgs)} open FVGs",
    ]
    if structure:
        parts.append(f"Latest structure: {structure[0].type}")
    summary = " | ".join(parts)

    return SMCAnalysis(
        symbol=symbol,
        timeframe=timeframe,
        current_price=current,
        order_blocks=obs,
        fair_value_gaps=fvgs,
        liquidity_sweeps=sweeps,
        structure_breaks=structure,
        market_phase=phase,
        premium_discount=pd_zone,
        nearest_ob_support=nearest_support,
        nearest_ob_resistance=nearest_resistance,
        smc_score=score,
        summary=summary,
    )
