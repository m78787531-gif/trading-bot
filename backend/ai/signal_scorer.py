"""
ai/signal_scorer.py  [v3 — High-Probability, Low-Drawdown]
─────────────────────────────────────────────────────────────────────────────
v3 additions on top of v2
─────────────────────────
TRADE QUALITY SCORE  (reject < 0.70)
  Composite 0–1 score from four sub-scores:
    • RR quality         — penalise RR < 1.5, reward ≥ 2.5
    • Structure clarity  — valid OBs + FVGs + clean breaks
    • Trend strength     — ADX-based
    • ATR quality        — reward ATR expansion, penalise compression

RETEST ENTRY LOGIC  (do not enter on raw breakout)
  check_retest_proximity() validates that the current price is within
  RETEST_PROXIMITY_PCT of a valid order block or FVG boundary.
  Signals that are not near a retest zone are blocked.

MARKET STRUCTURE VALIDATION
  validate_market_structure() checks the last N swing highs/lows:
    • Long  → price must show HH + HL sequence
    • Short → price must show LH + LL sequence

SESSION TIME FILTER
  is_high_liquidity_session() blocks trades outside:
    • London open  07:00–10:59 UTC
    • NY open      13:00–16:59 UTC
  When session_filter param is True (default).

All new gates are appended to TradeSignal:
  trade_quality, quality_rejected, session_ok,
  structure_valid, retest_confirmed
"""

from __future__ import annotations
import math
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from ai.indicators import (
    atr as calc_atr,
    bollinger_bands,
    ema,
    ema_series,
    macd,
    rsi,
    stochastic_rsi,
    trend_strength,
    volume_analysis,
)
from ai.smc_engine import analyse_smc, SMCAnalysis

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class StrategyMode(str, Enum):
    TREND    = "trend"
    RANGE    = "range"
    BREAKOUT = "breakout"


class MarketRegime(str, Enum):
    BULL          = "bull"
    BEAR          = "bear"
    SIDEWAYS      = "sideways"
    VOLATILE_BULL = "volatile_bull"
    VOLATILE_BEAR = "volatile_bear"


# ─────────────────────────────────────────────────────────────────────────────
# Default Parameters  (mutated by self_optimizer.py)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PARAMS: dict = {
    # RSI
    "rsi_oversold":          35,
    "rsi_overbought":        65,
    "rsi_period":            14,
    "rsi_gate_low":          40,
    "rsi_gate_high":         60,
    # MACD
    "macd_fast":             12,
    "macd_slow":             26,
    "macd_signal":           9,
    # EMA
    "ema_fast":              20,
    "ema_mid":               50,
    "ema_slow":              200,
    # Bollinger
    "bb_period":             20,
    "bb_std":                2.0,
    # SL / TP
    "sl_atr_multiplier":     1.5,
    "tp_atr_multiplier":     3.0,
    "partial_tp_r":          1.5,
    "partial_tp_close_pct":  50,
    # Gates
    "min_confidence":        75.0,
    "volume_spike_min":      1.5,
    "atr_floor_pct":         0.003,
    "adx_floor":             20,
    # v3: Quality
    "min_trade_quality":     0.70,
    "min_rr":                1.50,
    # v3: Retest proximity tolerance (% of price)
    "retest_proximity_pct":  0.005,   # 0.5 %
    # v3: Session filter enabled by default
    "session_filter":        True,
    # v3: Market structure lookback swings
    "structure_swing_n":     5,
    # Weights
    "w_smc":                 0.35,
    "w_trend":               0.25,
    "w_momentum":            0.15,
    "w_volume":              0.10,
    "w_volatility":          0.10,
    "w_sentiment":           0.05,
    "sentiment_weight":      0.05,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IndicatorSnapshot:
    rsi_value:       float
    rsi_signal:      str
    rsi_divergence:  str
    macd_line:       float
    macd_signal:     float
    macd_histogram:  float
    macd_crossover:  str
    macd_trend:      str
    ema_20:          float
    ema_50:          float
    ema_200:         float
    ema_alignment:   str
    bb_position:     str
    bb_squeeze:      bool
    bb_width:        float
    stoch_k:         float
    stoch_d:         float
    stoch_signal:    str
    adx:             float
    trend_direction: str
    volume_ratio:    float
    volume_signal:   str
    obv_trend:       str
    atr:             float
    atr_pct:         float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GateResult:
    passed:            bool
    smc_confirmed:     bool
    ema200_aligned:    bool
    rsi_in_zone:       bool
    macd_crossover_ok: bool
    volume_spike_ok:   bool
    no_trade_reason:   str


@dataclass
class TradeQualityScore:
    """v3: Composite trade quality 0-1. Reject if total < min_trade_quality."""
    rr_quality:        float   # 0-1: penalise RR<1.5, reward ≥2.5
    structure_quality: float   # 0-1: clean OBs/FVGs/breaks
    trend_quality:     float   # 0-1: ADX-based
    atr_quality:       float   # 0-1: ATR expansion vs compression
    total:             float   # 0-1: weighted average
    passed:            bool    # total >= min_trade_quality

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScoreBreakdown:
    smc_raw:         float
    trend_raw:       float
    momentum_raw:    float
    volume_raw:      float
    volatility_raw:  float
    sentiment_raw:   float
    weighted_total:  float
    confidence:      float
    direction:       str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TradeSignal:
    symbol:               str
    timeframe:            str
    direction:            str
    confidence:           float
    entry_price:          float
    stop_loss:            float
    take_profit:          float
    take_profit_partial:  float
    risk_reward:          float
    score_breakdown:      ScoreBreakdown
    indicators:           IndicatorSnapshot
    smc:                  SMCAnalysis
    gate:                 GateResult
    # v3 fields
    trade_quality:        TradeQualityScore
    quality_rejected:     bool
    session_ok:           bool
    structure_valid:      bool
    retest_confirmed:     bool
    blocked:              bool          # True = do not trade
    block_reason:         str
    preconditions_passed: bool
    reasoning:            list
    strategy_mode:        str
    market_regime:        str
    timestamp:            str
    news_flag_active:     bool

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "timeframe": self.timeframe,
            "direction": self.direction, "confidence": self.confidence,
            "entry_price": self.entry_price, "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "take_profit_partial": self.take_profit_partial,
            "risk_reward": self.risk_reward,
            "score_breakdown": self.score_breakdown.to_dict(),
            "indicators": self.indicators.to_dict(),
            "smc": self.smc.to_dict(),
            "gate": asdict(self.gate),
            "trade_quality": self.trade_quality.to_dict(),
            "quality_rejected": self.quality_rejected,
            "session_ok": self.session_ok,
            "structure_valid": self.structure_valid,
            "retest_confirmed": self.retest_confirmed,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "preconditions_passed": self.preconditions_passed,
            "reasoning": self.reasoning,
            "strategy_mode": self.strategy_mode,
            "market_regime": self.market_regime,
            "timestamp": self.timestamp,
            "news_flag_active": self.news_flag_active,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Global news flag
# ─────────────────────────────────────────────────────────────────────────────

_news_flag_active: bool = False


def set_news_flag(active: bool) -> None:
    global _news_flag_active
    _news_flag_active = active
    logger.info(f"News flag → {'ACTIVE' if active else 'cleared'}")


def get_news_flag() -> bool:
    return _news_flag_active


# ─────────────────────────────────────────────────────────────────────────────
# v3: Session Time Filter
# ─────────────────────────────────────────────────────────────────────────────

# London open: 07:00–10:59 UTC
# New York open: 13:00–16:59 UTC
LONDON_OPEN_UTC  = (7,  11)   # [start_hour, end_hour_exclusive)
NY_OPEN_UTC      = (13, 17)


def is_high_liquidity_session(dt: Optional[datetime] = None) -> tuple[bool, str]:
    """
    Returns (is_ok, session_name).
    True during London open (07–10 UTC) or NY open (13–16 UTC).
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    h = dt.hour

    if LONDON_OPEN_UTC[0] <= h < LONDON_OPEN_UTC[1]:
        return True, f"London open ({h:02d}:XX UTC)"
    if NY_OPEN_UTC[0] <= h < NY_OPEN_UTC[1]:
        return True, f"New York open ({h:02d}:XX UTC)"

    # Asian session (00–06) and dead-zone (17–23) — low liquidity
    return False, f"Off-session ({h:02d}:XX UTC — outside London/NY open)"


# ─────────────────────────────────────────────────────────────────────────────
# v3: Market Structure Validation  (HH/HL for longs; LH/LL for shorts)
# ─────────────────────────────────────────────────────────────────────────────

def validate_market_structure(
    highs: list,
    lows:  list,
    direction: str,
    n: int = 5,
) -> tuple[bool, str]:
    """
    Validate that the last `n` candles show the required swing structure.

    Long  → Higher Highs + Higher Lows
    Short → Lower Highs  + Lower Lows

    Returns (valid, reason).
    Uses the last n candles only to avoid looking at stale structure.
    """
    if len(highs) < n + 1 or len(lows) < n + 1:
        return True, "Insufficient data for structure check — skipping"

    recent_highs = highs[-(n + 1):]
    recent_lows  = lows[-(n + 1):]

    # Pivot detection: a candle is a swing high if higher than both neighbours
    def _swing_highs(h_arr):
        pivots = []
        for i in range(1, len(h_arr) - 1):
            if h_arr[i] > h_arr[i - 1] and h_arr[i] > h_arr[i + 1]:
                pivots.append(h_arr[i])
        return pivots

    def _swing_lows(l_arr):
        pivots = []
        for i in range(1, len(l_arr) - 1):
            if l_arr[i] < l_arr[i - 1] and l_arr[i] < l_arr[i + 1]:
                pivots.append(l_arr[i])
        return pivots

    sh = _swing_highs(recent_highs)
    sl = _swing_lows(recent_lows)

    if len(sh) < 2 or len(sl) < 2:
        return True, "Insufficient swing pivots — structure check skipped"

    if direction == "long":
        hh = sh[-1] > sh[-2]   # latest high > previous high
        hl = sl[-1] > sl[-2]   # latest low  > previous low
        if hh and hl:
            return True, f"Bullish structure confirmed: HH={sh[-1]:.4f}>HH={sh[-2]:.4f}, HL={sl[-1]:.4f}>HL={sl[-2]:.4f}"
        failed = []
        if not hh: failed.append(f"No HH ({sh[-1]:.4f}<={sh[-2]:.4f})")
        if not hl: failed.append(f"No HL ({sl[-1]:.4f}<={sl[-2]:.4f})")
        return False, f"Long blocked: {'; '.join(failed)}"

    elif direction == "short":
        lh = sh[-1] < sh[-2]   # latest high < previous high
        ll = sl[-1] < sl[-2]   # latest low  < previous low
        if lh and ll:
            return True, f"Bearish structure confirmed: LH={sh[-1]:.4f}<{sh[-2]:.4f}, LL={sl[-1]:.4f}<{sl[-2]:.4f}"
        failed = []
        if not lh: failed.append(f"No LH ({sh[-1]:.4f}>={sh[-2]:.4f})")
        if not ll: failed.append(f"No LL ({sl[-1]:.4f}>={sl[-2]:.4f})")
        return False, f"Short blocked: {'; '.join(failed)}"

    return True, "Hold — no structure check required"


# ─────────────────────────────────────────────────────────────────────────────
# v3: Retest Entry Logic
# ─────────────────────────────────────────────────────────────────────────────

def check_retest_proximity(
    current_price: float,
    smc:           SMCAnalysis,
    direction:     str,
    proximity_pct: float = 0.005,   # 0.5 % of price
) -> tuple[bool, str]:
    """
    Verify price is retesting a valid OB or FVG rather than chasing a breakout.

    For a LONG: price should be at or near a bullish OB top or FVG bottom.
    For a SHORT: price should be at or near a bearish OB bottom or FVG top.

    Returns (is_near_retest, reason).
    """
    tolerance = current_price * proximity_pct

    valid_obs  = [ob for ob in smc.order_blocks if ob.valid]
    valid_fvgs = [fv for fv in smc.fvgs       if fv.valid]

    if not valid_obs and not valid_fvgs:
        return False, "No valid OBs or FVGs to retest — breakout entry avoided"

    closest_dist = float("inf")
    closest_desc = ""

    for ob in valid_obs:
        if direction == "long" and ob.type == "bullish":
            # Retest = price coming back to the top of the bullish OB
            dist = abs(current_price - ob.high)
            if dist < closest_dist:
                closest_dist = dist
                closest_desc = f"Bullish OB top ${ob.high:.4f}"
        elif direction == "short" and ob.type == "bearish":
            # Retest = price coming back to the bottom of the bearish OB
            dist = abs(current_price - ob.low)
            if dist < closest_dist:
                closest_dist = dist
                closest_desc = f"Bearish OB bottom ${ob.low:.4f}"

    for fv in valid_fvgs:
        if direction == "long" and fv.type == "bullish":
            # Retest = price entering FVG from top (filling the gap)
            dist = abs(current_price - fv.top)
            if dist < closest_dist:
                closest_dist = dist
                closest_desc = f"Bullish FVG top ${fv.top:.4f}"
        elif direction == "short" and fv.type == "bearish":
            dist = abs(current_price - fv.bottom)
            if dist < closest_dist:
                closest_dist = dist
                closest_desc = f"Bearish FVG bottom ${fv.bottom:.4f}"

    if closest_dist == float("inf"):
        # No aligned OB/FVG for this direction — try nearest of any type
        all_levels = []
        for ob in valid_obs:
            all_levels.extend([ob.high, ob.low])
        for fv in valid_fvgs:
            all_levels.extend([fv.top, fv.bottom])
        if all_levels:
            closest_dist = min(abs(current_price - lvl) for lvl in all_levels)
            closest_desc = "nearest zone (direction mismatch)"
        else:
            return False, "No valid SMC levels for retest check"

    if closest_dist <= tolerance:
        return True, f"Retest confirmed: {closest_desc} — distance {closest_dist:.4f} ≤ tolerance {tolerance:.4f}"
    else:
        return False, (
            f"Not at retest zone: {closest_desc} — distance {closest_dist:.4f} "
            f"> tolerance {tolerance:.4f} (0.5% of price). "
            f"Wait for pullback before entering."
        )


# ─────────────────────────────────────────────────────────────────────────────
# v3: Trade Quality Score
# ─────────────────────────────────────────────────────────────────────────────

def calculate_trade_quality(
    rr:        float,
    smc:       SMCAnalysis,
    snap:      IndicatorSnapshot,
    direction: str,
    p:         dict,
) -> TradeQualityScore:
    """
    Composite trade quality 0–1.  Reject trade if total < min_trade_quality (0.70).

    Sub-scores
    ──────────
    RR quality (weight 30%)
      < 1.5  → 0.0  (hard fail, redundant with RR gate but explicit)
      1.5–2.0 → 0.5
      2.0–2.5 → 0.75
      ≥ 2.5  → 1.0

    Structure quality (weight 35%)
      Counts valid OBs + FVGs + confirmed breaks weighted by strength.

    Trend quality (weight 20%)
      ADX-based: < 20 → 0, 20–25 → 0.3, 25–35 → 0.6, ≥ 35 → 1.0

    ATR quality (weight 15%)
      Penalise ATR < floor or > extreme.
      Sweet spot: 0.4–1.5% of price.
    """
    min_rr = p.get("min_rr", 1.5)

    # ── RR quality (30%) ───────────────────────────────────────────────────
    if   rr < min_rr: rr_q = 0.0
    elif rr < 2.0:    rr_q = 0.50
    elif rr < 2.5:    rr_q = 0.75
    else:             rr_q = 1.00

    # ── Structure quality (35%) ─────────────────────────────────────────────
    valid_obs  = [ob for ob in smc.order_blocks if ob.valid]
    valid_fvgs = [fv for fv in smc.fvgs       if fv.valid]
    breaks     = smc.structure_breaks

    # Score = OB strength avg + FVG fill %  + break count weight
    ob_strength  = (sum(ob.strength for ob in valid_obs) / len(valid_obs)) if valid_obs else 0.0
    fvg_bonus    = min(len(valid_fvgs) * 0.15, 0.30)
    break_bonus  = min(len(breaks)     * 0.10, 0.30)

    # Direction alignment bonus: more valid OBs in the right direction
    aligned_obs = [ob for ob in valid_obs if
                   (direction == "long"  and ob.type == "bullish") or
                   (direction == "short" and ob.type == "bearish")]
    align_bonus = min(len(aligned_obs) * 0.10, 0.20)

    struct_q = min(ob_strength * 0.50 + fvg_bonus + break_bonus + align_bonus, 1.0)

    # ── Trend quality (20%) ─────────────────────────────────────────────────
    adx = snap.adx
    if   adx >= 35: trend_q = 1.0
    elif adx >= 25: trend_q = 0.6
    elif adx >= 20: trend_q = 0.3
    else:           trend_q = 0.0

    # Penalise if ADX trend direction contradicts our trade direction
    if (direction == "long"  and snap.trend_direction == "down" and adx >= 25):
        trend_q *= 0.5
    if (direction == "short" and snap.trend_direction == "up"   and adx >= 25):
        trend_q *= 0.5

    # ── ATR quality (15%) ───────────────────────────────────────────────────
    atr_pct_val = snap.atr_pct * 100   # convert to %
    floor_pct   = p.get("atr_floor_pct", 0.003) * 100

    if   atr_pct_val < floor_pct:    atr_q = 0.0   # compressed
    elif atr_pct_val < 0.4:          atr_q = 0.3   # very tight
    elif atr_pct_val < 0.8:          atr_q = 0.7   # decent
    elif atr_pct_val <= 1.5:         atr_q = 1.0   # sweet spot
    elif atr_pct_val <= 3.0:         atr_q = 0.6   # elevated
    else:                            atr_q = 0.3   # extreme volatility

    # ── Weighted total ───────────────────────────────────────────────────────
    total = round(
        rr_q    * 0.30 +
        struct_q * 0.35 +
        trend_q  * 0.20 +
        atr_q    * 0.15,
        4,
    )

    passed = total >= p.get("min_trade_quality", 0.70) and rr_q > 0.0

    return TradeQualityScore(
        rr_quality=round(rr_q, 4),
        structure_quality=round(struct_q, 4),
        trend_quality=round(trend_q, 4),
        atr_quality=round(atr_q, 4),
        total=total,
        passed=passed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Market Regime + Strategy
# ─────────────────────────────────────────────────────────────────────────────

def detect_market_regime(snap: IndicatorSnapshot) -> MarketRegime:
    adx     = snap.adx
    align   = snap.ema_alignment
    atr_pct = snap.atr_pct
    squeeze = snap.bb_squeeze

    if adx >= 25:
        volatile = adx >= 35 and atr_pct > 0.015
        if align == "bullish":
            return MarketRegime.VOLATILE_BULL if volatile else MarketRegime.BULL
        if align == "bearish":
            return MarketRegime.VOLATILE_BEAR if volatile else MarketRegime.BEAR
        if snap.trend_direction == "up":
            return MarketRegime.VOLATILE_BULL if volatile else MarketRegime.BULL
        return MarketRegime.VOLATILE_BEAR if volatile else MarketRegime.BEAR

    if adx < 20:
        return MarketRegime.SIDEWAYS

    if squeeze:
        return MarketRegime.BULL if snap.trend_direction == "up" else MarketRegime.BEAR
    return MarketRegime.SIDEWAYS


def select_strategy(regime: MarketRegime, snap: IndicatorSnapshot) -> StrategyMode:
    if regime in (MarketRegime.BULL, MarketRegime.BEAR):
        return StrategyMode.TREND
    if regime == MarketRegime.SIDEWAYS:
        return StrategyMode.BREAKOUT if snap.bb_squeeze else StrategyMode.RANGE
    return StrategyMode.BREAKOUT


# ─────────────────────────────────────────────────────────────────────────────
# Pre-trade Gate System  (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

def _check_gates(
    snap:          IndicatorSnapshot,
    smc:           SMCAnalysis,
    p:             dict,
    current_price: float,
    strategy:      StrategyMode,
    news_flag:     bool,
) -> GateResult:
    if news_flag:
        return GateResult(False, False, False, False, False, False,
                          "High-impact news event — all trading paused")

    if strategy == StrategyMode.TREND and snap.adx < p["adx_floor"]:
        return GateResult(False, False, False, False, False, False,
                          f"ADX={snap.adx:.1f} < {p['adx_floor']} — sideways; trend disabled")

    if snap.atr_pct < p["atr_floor_pct"]:
        return GateResult(False, False, False, False, False, False,
                          f"ATR={snap.atr_pct*100:.3f}% < floor {p['atr_floor_pct']*100:.3f}%")

    if snap.volume_ratio < 0.5:
        return GateResult(False, False, False, False, False, False,
                          f"Volume={snap.volume_ratio:.2f}x — extremely thin market")

    has_break  = len(smc.structure_breaks) > 0
    valid_obs  = [ob for ob in smc.order_blocks if ob.valid]
    valid_fvgs = [fv for fv in smc.fvgs       if fv.valid]
    has_ob_fvg = bool(valid_obs or valid_fvgs)
    smc_ok     = has_break and has_ob_fvg

    e200      = snap.ema_200
    ema200_ok = True
    if not math.isnan(e200) and e200 > 0:
        ema200_ok = snap.ema_alignment in ("bullish", "bearish")

    rsi_v = snap.rsi_value
    if strategy == StrategyMode.RANGE:
        rsi_ok = rsi_v <= p["rsi_oversold"] or rsi_v >= p["rsi_overbought"]
    elif strategy == StrategyMode.BREAKOUT:
        rsi_ok = True
    else:
        rsi_ok = p["rsi_gate_low"] <= rsi_v <= p["rsi_gate_high"]

    macd_ok = True if strategy == StrategyMode.BREAKOUT else               snap.macd_crossover in ("bullish", "bearish")

    vol_ok = snap.volume_ratio >= p["volume_spike_min"]

    all_ok = smc_ok and ema200_ok and rsi_ok and macd_ok and vol_ok
    if all_ok:
        return GateResult(True, True, True, True, True, True, "")

    failed = []
    if not smc_ok:
        parts = []
        if not has_break:  parts.append("no BOS/CHoCH")
        if not has_ob_fvg: parts.append("no valid OB/FVG")
        failed.append(f"SMC ({', '.join(parts)})")
    if not ema200_ok: failed.append("EMA-200 mixed")
    if not rsi_ok:    failed.append(f"RSI={rsi_v:.1f} outside gate")
    if not macd_ok:   failed.append(f"No MACD crossover")
    if not vol_ok:    failed.append(f"Volume {snap.volume_ratio:.2f}x < {p['volume_spike_min']}x")

    return GateResult(False, smc_ok, ema200_ok, rsi_ok, macd_ok, vol_ok,
                      "Gates failed: " + "; ".join(failed))


# ─────────────────────────────────────────────────────────────────────────────
# Component Scorers  (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

def _score_smc(smc: SMCAnalysis) -> tuple:
    reasons = [f"SMC raw: {smc.smc_score:.1f} — {smc.summary}"]
    for sb in smc.structure_breaks[-2:]:
        reasons.append(f"  ↳ {sb.type.upper()} @ ${sb.level:.4f}")
    for ob in [o for o in smc.order_blocks if o.valid][-2:]:
        reasons.append(f"  ↳ Valid {ob.type} OB ${ob.low:.4f}–${ob.high:.4f} str={ob.strength:.2f}")
    for fv in [f for f in smc.fvgs if f.valid][-2:]:
        reasons.append(f"  ↳ Valid {fv.type} FVG ${fv.bottom:.4f}–${fv.top:.4f}")
    return round(max(-1.0, min(1.0, smc.smc_score / 10.0)), 4), reasons


def _score_trend(snap: IndicatorSnapshot, price: float) -> tuple:
    reasons = []
    score   = 0.0
    e20, e50, e200 = snap.ema_20, snap.ema_50, snap.ema_200

    if not math.isnan(e200) and e200 > 0:
        if e20 > e50 > e200:
            score += 0.75; reasons.append("EMA stack bullish (20>50>200)")
        elif e20 < e50 < e200:
            score -= 0.75; reasons.append("EMA stack bearish (20<50<200)")
        elif e20 > e50:
            score += 0.30; reasons.append("EMA 20>50: mild bullish")
        elif e20 < e50:
            score -= 0.30; reasons.append("EMA 20<50: mild bearish")
        if price > e200:
            score += 0.20; reasons.append(f"Price above 200 EMA (${e200:.4f})")
        else:
            score -= 0.20; reasons.append(f"Price below 200 EMA (${e200:.4f})")
    else:
        reasons.append("EMA-200 unavailable")

    adx = snap.adx
    if adx >= 40:
        bonus = 0.15 if snap.trend_direction == "up" else -0.15
        score += bonus; reasons.append(f"Strong trend ADX={adx:.1f}")
    elif adx >= 25:
        bonus = 0.08 if snap.trend_direction == "up" else -0.08
        score += bonus; reasons.append(f"Solid trend ADX={adx:.1f}")

    return round(max(-1.0, min(1.0, score)), 4), reasons


def _score_momentum(snap: IndicatorSnapshot, p: dict) -> tuple:
    reasons = []
    score   = 0.0
    rsi_v   = snap.rsi_value

    if rsi_v <= p["rsi_oversold"]:
        score += 0.35; reasons.append(f"RSI oversold {rsi_v:.1f}")
    elif rsi_v >= p["rsi_overbought"]:
        score -= 0.35; reasons.append(f"RSI overbought {rsi_v:.1f}")
    elif rsi_v < p["rsi_gate_low"]:
        score += 0.10
    elif rsi_v > p["rsi_gate_high"]:
        score -= 0.10

    if snap.rsi_divergence == "bullish":
        score += 0.20; reasons.append("Bullish RSI divergence")
    elif snap.rsi_divergence == "bearish":
        score -= 0.20; reasons.append("Bearish RSI divergence")

    if snap.macd_crossover == "bullish":
        score += 0.30; reasons.append("MACD bullish crossover")
    elif snap.macd_crossover == "bearish":
        score -= 0.30; reasons.append("MACD bearish crossover")
    elif snap.macd_trend == "bullish":
        score += 0.10
    else:
        score -= 0.10

    if snap.stoch_signal == "oversold" and snap.stoch_k > snap.stoch_d:
        score += 0.15; reasons.append("StochRSI K>D from oversold")
    elif snap.stoch_signal == "overbought" and snap.stoch_k < snap.stoch_d:
        score -= 0.15; reasons.append("StochRSI K<D from overbought")

    return round(max(-1.0, min(1.0, score)), 4), reasons


def _score_volume(snap: IndicatorSnapshot, p: dict) -> tuple:
    reasons = []
    score   = 0.0
    ratio   = snap.volume_ratio

    if ratio >= p["volume_spike_min"] * 2:
        score += 0.60; reasons.append(f"Extreme volume {ratio:.1f}x")
    elif ratio >= p["volume_spike_min"]:
        score += 0.40; reasons.append(f"Volume spike {ratio:.1f}x")
    elif ratio < 0.8:
        score -= 0.30; reasons.append(f"Thin volume {ratio:.1f}x")

    if snap.obv_trend == "up":
        score += 0.40; reasons.append("OBV rising: accumulation")
    elif snap.obv_trend == "down":
        score -= 0.40; reasons.append("OBV falling: distribution")

    return round(max(-1.0, min(1.0, score)), 4), reasons


def _score_volatility(snap: IndicatorSnapshot) -> tuple:
    reasons = []
    score   = 0.0
    pos     = snap.bb_position

    if pos == "below_lower":
        score += 0.70; reasons.append("Below lower BB: oversell extreme")
    elif pos == "near_lower":
        score += 0.30; reasons.append("Near lower BB: pullback zone")
    elif pos == "above_upper":
        score -= 0.70; reasons.append("Above upper BB: overbuy extreme")
    elif pos == "near_upper":
        score -= 0.30; reasons.append("Near upper BB: distribution zone")

    if snap.bb_squeeze:
        reasons.append(f"BB squeeze: breakout imminent")

    return round(max(-1.0, min(1.0, score)), 4), reasons


def _score_sentiment(sentiment_score: float, p: dict) -> tuple:
    reasons = []
    s = max(-1.0, min(1.0, sentiment_score))
    if s > 0.5:   reasons.append(f"Strong bullish sentiment (+{s:.2f})")
    elif s > 0.2: reasons.append(f"Mild bullish sentiment (+{s:.2f})")
    elif s < -0.5: reasons.append(f"Strong bearish sentiment ({s:.2f})")
    elif s < -0.2: reasons.append(f"Mild bearish sentiment ({s:.2f})")
    return round(s, 4), reasons


def _adjust_for_strategy(smc_s, trend_s, mom_s, vol_s, vola_s, sent_s, strategy):
    clip = lambda v: round(max(-1.0, min(1.0, v)), 4)
    if strategy == StrategyMode.RANGE:
        trend_s = 0.0
        mom_s   = clip(mom_s  * 1.20)
        vola_s  = clip(vola_s * 1.20)
    elif strategy == StrategyMode.BREAKOUT:
        vol_s   = clip(vol_s  * 1.30)
        vola_s  = clip(vola_s * 1.20)
        trend_s = clip(trend_s * 0.80)
    return smc_s, trend_s, mom_s, vol_s, vola_s, sent_s


# ─────────────────────────────────────────────────────────────────────────────
# Main Signal Generator  (v3)
# ─────────────────────────────────────────────────────────────────────────────

def generate_signal(
    symbol:          str,
    timeframe:       str,
    opens:           list,
    highs:           list,
    lows:            list,
    closes:          list,
    volumes:         list,
    sentiment_score: float = 0.0,
    params:          Optional[dict] = None,
    news_flag:       Optional[bool] = None,
) -> TradeSignal:
    """
    v3 signal pipeline:
      1–9. Same as v2 (indicators, SMC, regime, gates, score, direction, EMA guard, levels)
     10.  Market Structure Validation (HH/HL or LH/LL)
     11.  Retest Entry Check (near OB/FVG)
     12.  Session Time Filter (London/NY open)
     13.  Trade Quality Score (≥ 0.70 required)
     14.  Return TradeSignal with all v3 fields populated
    """
    p       = {**DEFAULT_PARAMS, **(params or {})}
    current = closes[-1]
    nf      = news_flag if news_flag is not None else _news_flag_active

    # ── 1. Indicators ─────────────────────────────────────────────────────
    rsi_res   = rsi(closes, p["rsi_period"])
    macd_res  = macd(closes, p["macd_fast"], p["macd_slow"], p["macd_signal"])
    e20       = ema(closes, p["ema_fast"])
    e50       = ema(closes, p["ema_mid"])
    e200      = ema(closes, p["ema_slow"])
    bb_res    = bollinger_bands(closes, p["bb_period"], p["bb_std"])
    stoch_res = stochastic_rsi(closes)
    vol_res   = volume_analysis(closes, volumes)
    adx_val, trend_dir = trend_strength(highs, lows, closes)
    atr_val   = calc_atr(highs, lows, closes)
    atr_pct   = atr_val / current if current > 0 else 0.0

    ema_align = (
        "bullish" if (not math.isnan(e200) and e20 > e50 > e200) else
        "bearish" if (not math.isnan(e200) and e20 < e50 < e200) else
        "mixed"
    )

    snap = IndicatorSnapshot(
        rsi_value=rsi_res.value,            rsi_signal=rsi_res.signal,
        rsi_divergence=rsi_res.divergence,
        macd_line=macd_res.macd_line,       macd_signal=macd_res.signal_line,
        macd_histogram=macd_res.histogram,  macd_crossover=macd_res.crossover,
        macd_trend=macd_res.trend,
        ema_20=e20, ema_50=e50, ema_200=e200, ema_alignment=ema_align,
        bb_position=bb_res.position,        bb_squeeze=bb_res.squeeze,
        bb_width=bb_res.width,
        stoch_k=stoch_res.k,                stoch_d=stoch_res.d,
        stoch_signal=stoch_res.signal,
        adx=adx_val,                        trend_direction=trend_dir,
        volume_ratio=vol_res.ratio,         volume_signal=vol_res.signal,
        obv_trend=vol_res.obv_trend,
        atr=atr_val,                        atr_pct=atr_pct,
    )

    # ── 2. SMC ────────────────────────────────────────────────────────────
    smc = analyse_smc(symbol, timeframe, opens, highs, lows, closes, volumes)

    # ── 3. Regime + Strategy ──────────────────────────────────────────────
    regime   = detect_market_regime(snap)
    strategy = select_strategy(regime, snap)

    # ── 4. Gate Checks ────────────────────────────────────────────────────
    gate = _check_gates(snap, smc, p, current, strategy, nf)

    all_reasons = [
        f"Market regime: {regime.value} | Strategy: {strategy.value}",
        f"ADX={snap.adx:.1f} | ATR={atr_pct*100:.2f}% | Vol={snap.volume_ratio:.2f}x",
    ]

    # Build a zero quality placeholder for early returns
    def _zero_quality():
        return TradeQualityScore(0.0, 0.0, 0.0, 0.0, 0.0, False)

    if not gate.passed:
        all_reasons.append(f"⛔ {gate.no_trade_reason}")
        return _build_hold(symbol, timeframe, current, snap, smc, gate,
                           strategy, regime, all_reasons, nf, p, atr_val,
                           _zero_quality(), session_ok=True,
                           struct_valid=True, retest_ok=True)

    # ── 5–7. Score + Weighted sum ─────────────────────────────────────────
    smc_s,  r1 = _score_smc(smc)
    tr_s,   r2 = _score_trend(snap, current)
    mom_s,  r3 = _score_momentum(snap, p)
    vol_s,  r4 = _score_volume(snap, p)
    vola_s, r5 = _score_volatility(snap)
    sent_s, r6 = _score_sentiment(sentiment_score, p)
    all_reasons.extend(r1 + r2 + r3 + r4 + r5 + r6)

    smc_s, tr_s, mom_s, vol_s, vola_s, sent_s = _adjust_for_strategy(
        smc_s, tr_s, mom_s, vol_s, vola_s, sent_s, strategy)

    weighted = (
        smc_s  * p["w_smc"]        +
        tr_s   * p["w_trend"]      +
        mom_s  * p["w_momentum"]   +
        vol_s  * p["w_volume"]     +
        vola_s * p["w_volatility"] +
        sent_s * p["w_sentiment"]
    )
    weighted   = round(max(-1.0, min(1.0, weighted)), 6)
    confidence = round((weighted + 1.0) / 2.0 * 100.0, 2)

    # ── 8. Direction ──────────────────────────────────────────────────────
    min_conf = p["min_confidence"]
    if   weighted > 0 and confidence >= min_conf:
        direction = "long"
    elif weighted < 0 and (100 - confidence) >= min_conf:
        direction = "short"
    else:
        direction = "hold"
        all_reasons.append(f"Below threshold: conf={confidence:.1f} (need ≥{min_conf})")

    # ── 9. EMA-200 direction guard ────────────────────────────────────────
    if not math.isnan(e200) and e200 > 0 and direction != "hold":
        if direction == "long" and current < e200 and ema_align == "bearish":
            direction = "hold"
            all_reasons.append("⚠️ Long blocked — price below bearish EMA-200")
        elif direction == "short" and current > e200 and ema_align == "bullish":
            direction = "hold"
            all_reasons.append("⚠️ Short blocked — price above bullish EMA-200")

    # ── 10. Levels ────────────────────────────────────────────────────────
    sl_mult, tp_mult = p["sl_atr_multiplier"], p["tp_atr_multiplier"]
    if direction == "long":
        stop_loss   = round(current - atr_val * sl_mult, 6)
        take_profit = round(current + atr_val * tp_mult, 6)
        if smc.nearest_ob_support and smc.nearest_ob_support > stop_loss:
            stop_loss = round(smc.nearest_ob_support * 0.998, 6)
    elif direction == "short":
        stop_loss   = round(current + atr_val * sl_mult, 6)
        take_profit = round(current - atr_val * tp_mult, 6)
        if smc.nearest_ob_resistance and smc.nearest_ob_resistance < stop_loss:
            stop_loss = round(smc.nearest_ob_resistance * 1.002, 6)
    else:
        stop_loss   = round(current - atr_val * sl_mult, 6)
        take_profit = round(current + atr_val * tp_mult, 6)

    risk_dist   = abs(current - stop_loss)
    tp_partial  = (round(current + risk_dist * p["partial_tp_r"], 6) if direction == "long"
                   else round(current - risk_dist * p["partial_tp_r"], 6) if direction == "short"
                   else take_profit)
    reward_dist = abs(take_profit - current)
    rr          = round(reward_dist / risk_dist, 2) if risk_dist > 0 else 0.0

    if direction != "hold":
        all_reasons.append(
            f"Entry ${current:.4f} | SL ${stop_loss:.4f} | "
            f"PTP ${tp_partial:.4f} | TP ${take_profit:.4f} | R:R {rr}"
        )

    breakdown = ScoreBreakdown(
        smc_raw=smc_s, trend_raw=tr_s, momentum_raw=mom_s,
        volume_raw=vol_s, volatility_raw=vola_s, sentiment_raw=sent_s,
        weighted_total=weighted, confidence=confidence, direction=direction,
    )

    # ── 11. v3: Market Structure Validation ──────────────────────────────
    struct_valid = True
    struct_reason = ""
    if direction != "hold":
        struct_valid, struct_reason = validate_market_structure(
            highs, lows, direction, p.get("structure_swing_n", 5))
        if not struct_valid:
            all_reasons.append(f"⛔ Structure: {struct_reason}")

    # ── 12. v3: Retest Entry Check ────────────────────────────────────────
    retest_ok = True
    retest_reason = ""
    if direction != "hold" and struct_valid:
        retest_ok, retest_reason = check_retest_proximity(
            current, smc, direction, p.get("retest_proximity_pct", 0.005))
        all_reasons.append(f"Retest: {retest_reason}")

    # ── 13. v3: Session Time Filter ───────────────────────────────────────
    session_ok   = True
    session_name = "session filter disabled"
    if p.get("session_filter", True) and direction != "hold":
        session_ok, session_name = is_high_liquidity_session()
        if not session_ok:
            all_reasons.append(f"⛔ Session: {session_name}")

    # ── 14. v3: Trade Quality Score ───────────────────────────────────────
    quality = calculate_trade_quality(rr, smc, snap, direction, p)
    quality_rejected = False
    if direction != "hold":
        if not quality.passed:
            all_reasons.append(
                f"⛔ Quality {quality.total:.2f} < {p.get('min_trade_quality', 0.70)} "
                f"(RR={quality.rr_quality:.2f}, struct={quality.structure_quality:.2f}, "
                f"trend={quality.trend_quality:.2f}, ATR={quality.atr_quality:.2f})"
            )
            quality_rejected = True

    # ── 15. Final blocked decision ────────────────────────────────────────
    blocked = False
    block_reason = ""
    final_direction = direction

    if direction != "hold":
        if not struct_valid:
            blocked = True; block_reason = struct_reason
        elif not retest_ok:
            blocked = True; block_reason = retest_reason
        elif not session_ok and p.get("session_filter", True):
            blocked = True; block_reason = session_name
        elif quality_rejected:
            blocked = True
            block_reason = (f"Trade quality {quality.total:.2f} < "
                            f"{p.get('min_trade_quality', 0.70)}")

    if blocked:
        final_direction = "hold"

    return TradeSignal(
        symbol=symbol, timeframe=timeframe,
        direction=final_direction, confidence=confidence,
        entry_price=round(current, 6),
        stop_loss=stop_loss, take_profit=take_profit,
        take_profit_partial=tp_partial, risk_reward=rr,
        score_breakdown=breakdown, indicators=snap, smc=smc, gate=gate,
        trade_quality=quality,
        quality_rejected=quality_rejected,
        session_ok=session_ok,
        structure_valid=struct_valid,
        retest_confirmed=retest_ok,
        blocked=blocked,
        block_reason=block_reason,
        preconditions_passed=gate.passed and not blocked,
        reasoning=all_reasons,
        strategy_mode=strategy.value, market_regime=regime.value,
        timestamp=datetime.now(timezone.utc).isoformat(),
        news_flag_active=nf,
    )


def _build_hold(
    symbol, timeframe, current, snap, smc, gate,
    strategy, regime, reasons, nf, p, atr_val,
    quality, session_ok=True, struct_valid=True, retest_ok=True,
) -> TradeSignal:
    sl  = round(current - atr_val * p["sl_atr_multiplier"], 6)
    tp  = round(current + atr_val * p["tp_atr_multiplier"], 6)
    rd  = abs(current - sl) or 1.0
    tpp = round(current + rd * p["partial_tp_r"], 6)
    rr  = round(abs(tp - current) / rd, 2)
    bd  = ScoreBreakdown(0, 0, 0, 0, 0, 0, 0.0, 50.0, "hold")
    return TradeSignal(
        symbol=symbol, timeframe=timeframe, direction="hold", confidence=50.0,
        entry_price=round(current, 6), stop_loss=sl, take_profit=tp,
        take_profit_partial=tpp, risk_reward=rr,
        score_breakdown=bd, indicators=snap, smc=smc, gate=gate,
        trade_quality=quality,
        quality_rejected=False, session_ok=session_ok,
        structure_valid=struct_valid, retest_confirmed=retest_ok,
        blocked=not gate.passed, block_reason=gate.no_trade_reason,
        preconditions_passed=False, reasoning=reasons,
        strategy_mode=strategy.value, market_regime=regime.value,
        timestamp=datetime.now(timezone.utc).isoformat(),
        news_flag_active=nf,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Timeframe System  (Req 6)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations as _ann


@dataclass
class MTFAnalysis:
    """Result of multi-timeframe confluence check."""
    htf_regime:   str   # 1H/4H — bull | bear | sideways
    mtf_structure: str  # 15m   — bullish | bearish | neutral
    ltf_bias:      str  # 5m    — long | short | neutral
    aligned:       bool # True if all three point same direction
    direction:     str  # long | short | neutral
    score:         float  # 0–1
    reason:        str

    def to_dict(self) -> dict:
        return {
            "htf_regime":    self.htf_regime,
            "mtf_structure": self.mtf_structure,
            "ltf_bias":      self.ltf_bias,
            "aligned":       self.aligned,
            "direction":     self.direction,
            "score":         round(self.score, 4),
            "reason":        self.reason,
        }


def analyse_mtf(
    htf_closes:  list,   # 1H or 4H closes (≥ 50 bars)
    mtf_closes:  list,   # 15m closes  (≥ 50 bars)
    ltf_closes:  list,   # 5m closes   (≥ 20 bars)
    htf_highs:   list  = None,
    htf_lows:    list  = None,
    mtf_highs:   list  = None,
    mtf_lows:    list  = None,
) -> MTFAnalysis:
    """
    Determine trend alignment across three timeframes.

    HTF (1H/4H): EMA 20 vs EMA 50 alignment → bull/bear/sideways
    MTF (15m):   Price vs EMA 20 + EMA 50 cross → structure
    LTF (5m):    Last 5 candles momentum → entry bias
    """
    import math as _m

    def _safe_ema(prices, period):
        if len(prices) < period:
            return float("nan")
        k = 2.0 / (period + 1)
        v = sum(prices[:period]) / period
        for p in prices[period:]:
            v = p * k + v * (1 - k)
        return v

    # HTF regime
    htf_e20 = _safe_ema(htf_closes, 20)
    htf_e50 = _safe_ema(htf_closes, 50)
    if _m.isnan(htf_e20) or _m.isnan(htf_e50):
        htf_regime = "unknown"
    elif htf_e20 > htf_e50 * 1.002:
        htf_regime = "bull"
    elif htf_e20 < htf_e50 * 0.998:
        htf_regime = "bear"
    else:
        htf_regime = "sideways"

    # MTF structure
    mtf_e20 = _safe_ema(mtf_closes, 20)
    mtf_price = mtf_closes[-1] if mtf_closes else 0.0
    if _m.isnan(mtf_e20):
        mtf_structure = "neutral"
    elif mtf_price > mtf_e20 * 1.001:
        mtf_structure = "bullish"
    elif mtf_price < mtf_e20 * 0.999:
        mtf_structure = "bearish"
    else:
        mtf_structure = "neutral"

    # LTF bias (last 5 candle momentum)
    if len(ltf_closes) >= 6:
        ltf_delta = ltf_closes[-1] - ltf_closes[-6]
        avg_move  = sum(abs(ltf_closes[i] - ltf_closes[i-1]) for i in range(-5,0)) / 5
        if ltf_delta > avg_move * 0.5:
            ltf_bias = "long"
        elif ltf_delta < -avg_move * 0.5:
            ltf_bias = "short"
        else:
            ltf_bias = "neutral"
    else:
        ltf_bias = "neutral"

    # Alignment check
    htf_bull = htf_regime == "bull"
    htf_bear = htf_regime == "bear"
    mtf_bull = mtf_structure == "bullish"
    mtf_bear = mtf_structure == "bearish"
    ltf_bull = ltf_bias == "long"
    ltf_bear = ltf_bias == "short"

    if htf_bull and mtf_bull and ltf_bull:
        aligned   = True
        direction = "long"
        score     = 1.0
        reason    = "Perfect bullish MTF alignment (HTF bull + MTF bullish + LTF long)"
    elif htf_bear and mtf_bear and ltf_bear:
        aligned   = True
        direction = "short"
        score     = 1.0
        reason    = "Perfect bearish MTF alignment (HTF bear + MTF bearish + LTF short)"
    elif htf_bull and mtf_bull:
        aligned   = True
        direction = "long"
        score     = 0.70
        reason    = f"HTF+MTF bullish, LTF={ltf_bias} — partial alignment"
    elif htf_bear and mtf_bear:
        aligned   = True
        direction = "short"
        score     = 0.70
        reason    = f"HTF+MTF bearish, LTF={ltf_bias} — partial alignment"
    elif htf_bull:
        aligned   = False
        direction = "long"
        score     = 0.35
        reason    = f"Only HTF bullish, MTF={mtf_structure} — weak alignment"
    elif htf_bear:
        aligned   = False
        direction = "short"
        score     = 0.35
        reason    = f"Only HTF bearish, MTF={mtf_structure} — weak alignment"
    else:
        aligned   = False
        direction = "neutral"
        score     = 0.20
        reason    = f"HTF sideways — no clear direction"

    return MTFAnalysis(
        htf_regime=htf_regime, mtf_structure=mtf_structure, ltf_bias=ltf_bias,
        aligned=aligned, direction=direction, score=round(score, 4), reason=reason,
    )


def is_noise_environment(
    adx:          float,
    atr_pct:      float,
    volume_ratio: float,
    bb_width:     float,
    adx_min:      float = 20.0,
    atr_min_pct:  float = 0.002,
    vol_min:      float = 0.7,
) -> tuple:
    """
    Returns (is_noisy: bool, reason: str).
    Blocks signal if market is in a high-noise environment:
      • ADX < threshold (trending strength absent)
      • ATR too low (compressed, bad R:R)
      • Volume too low (thin, illiquid)
      • Conflicting signals indicated by very narrow BB
    """
    if adx < adx_min:
        return True, f"ADX {adx:.1f} < {adx_min} — no trend, ranging noise"
    if atr_pct < atr_min_pct:
        return True, f"ATR {atr_pct*100:.3f}% < {atr_min_pct*100:.3f}% — compressed, no room"
    if volume_ratio < vol_min:
        return True, f"Volume {volume_ratio:.2f}× < {vol_min}× — illiquid session"
    if bb_width < 0.005:
        return True, f"BB width {bb_width*100:.2f}% — extreme compression, pre-breakout noise"
    return False, ""
