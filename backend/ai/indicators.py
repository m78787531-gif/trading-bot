"""
ai/indicators.py
─────────────────────────────────────────────────────────────────────────────
Pure-Python technical indicator library.  No TA-Lib dependency required —
all formulas implemented from scratch for maximum portability.

All functions accept a list/deque of floats (closing prices, highs, lows,
volumes) and return a single float (the most-recent value) unless noted.

Usage:
    from ai.indicators import rsi, macd, ema, bollinger_bands, atr

    closes = [42100, 42300, 41900, ...]   # oldest → newest
    r = rsi(closes)
    fast, slow, hist = macd(closes)
    upper, mid, lower = bollinger_bands(closes)
"""

from __future__ import annotations
import math
import statistics
from typing import NamedTuple


# ─────────────────────────────────────────────────────────────────────────────
# Basic smoothing helpers
# ─────────────────────────────────────────────────────────────────────────────

def ema_series(prices: list[float], period: int) -> list[float]:
    """
    Return a list of EMA values (same length as prices, NaN-padded at start).
    Uses Wilder smoothing multiplier: k = 2 / (period + 1).
    """
    if len(prices) < period:
        return [float("nan")] * len(prices)

    k = 2.0 / (period + 1)
    result: list[float] = [float("nan")] * len(prices)

    # seed with SMA
    seed = sum(prices[:period]) / period
    result[period - 1] = seed

    for i in range(period, len(prices)):
        result[i] = prices[i] * k + result[i - 1] * (1 - k)

    return result


def sma_series(prices: list[float], period: int) -> list[float]:
    result: list[float] = []
    for i in range(len(prices)):
        if i < period - 1:
            result.append(float("nan"))
        else:
            result.append(sum(prices[i - period + 1 : i + 1]) / period)
    return result


def rma_series(prices: list[float], period: int) -> list[float]:
    """Wilder's smoothed moving average (used in RSI)."""
    if len(prices) < period:
        return [float("nan")] * len(prices)

    k = 1.0 / period
    result = [float("nan")] * len(prices)
    result[period - 1] = sum(prices[:period]) / period

    for i in range(period, len(prices)):
        result[i] = prices[i] * k + result[i - 1] * (1 - k)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# EMA (single value)
# ─────────────────────────────────────────────────────────────────────────────

def ema(prices: list[float], period: int = 20) -> float:
    """Most-recent EMA value."""
    series = ema_series(prices, period)
    vals = [v for v in series if not math.isnan(v)]
    return vals[-1] if vals else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# RSI — Relative Strength Index
# ─────────────────────────────────────────────────────────────────────────────

class RSIResult(NamedTuple):
    value: float        # 0–100
    signal: str         # oversold | overbought | neutral
    divergence: str     # bullish | bearish | none   (price vs RSI direction)


def rsi(prices: list[float], period: int = 14) -> RSIResult:
    """
    Wilder RSI.  Needs at least period+1 data points.
    Returns RSIResult(value, signal, divergence).
    """
    if len(prices) < period + 1:
        return RSIResult(50.0, "neutral", "none")

    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [abs(min(d, 0.0)) for d in deltas]

    avg_gain_series = rma_series(gains, period)
    avg_loss_series = rma_series(losses, period)

    # Most-recent
    ag = avg_gain_series[-1]
    al = avg_loss_series[-1]

    if math.isnan(ag) or math.isnan(al):
        return RSIResult(50.0, "neutral", "none")

    rs = ag / al if al != 0 else float("inf")
    value = 100.0 - (100.0 / (1.0 + rs)) if al != 0 else 100.0

    signal = "neutral"
    if value <= 30:
        signal = "oversold"
    elif value >= 70:
        signal = "overbought"

    # Basic divergence: last 5 candles price direction vs RSI direction
    divergence = "none"
    if len(prices) >= 6:
        price_dir = prices[-1] - prices[-6]
        rsi_prev = rsi(prices[:-1], period).value
        rsi_dir = value - rsi_prev
        if price_dir < 0 and rsi_dir > 0:
            divergence = "bullish"
        elif price_dir > 0 and rsi_dir < 0:
            divergence = "bearish"

    return RSIResult(round(value, 2), signal, divergence)


# ─────────────────────────────────────────────────────────────────────────────
# MACD — Moving Average Convergence/Divergence
# ─────────────────────────────────────────────────────────────────────────────

class MACDResult(NamedTuple):
    macd_line: float     # fast EMA – slow EMA
    signal_line: float   # EMA of macd_line
    histogram: float     # macd_line – signal_line
    crossover: str       # bullish | bearish | none  (just crossed)
    trend: str           # bullish | bearish


def macd(
    prices: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> MACDResult:
    """
    Standard MACD.  Returns MACDResult with crossover detection.
    """
    if len(prices) < slow + signal:
        return MACDResult(0.0, 0.0, 0.0, "none", "neutral")

    fast_ema = ema_series(prices, fast)
    slow_ema = ema_series(prices, slow)

    # MACD line where both EMAs are valid
    macd_line_series: list[float] = []
    for f, s in zip(fast_ema, slow_ema):
        if math.isnan(f) or math.isnan(s):
            macd_line_series.append(float("nan"))
        else:
            macd_line_series.append(f - s)

    valid_macd = [v for v in macd_line_series if not math.isnan(v)]
    if len(valid_macd) < signal:
        return MACDResult(0.0, 0.0, 0.0, "none", "neutral")

    signal_series = ema_series(valid_macd, signal)
    sig_val = signal_series[-1]
    macd_val = valid_macd[-1]
    hist_val = macd_val - sig_val if not math.isnan(sig_val) else 0.0

    # Crossover detection (compare last two valid pairs)
    crossover = "none"
    if len(valid_macd) >= 2 and len(signal_series) >= 2:
        prev_macd = valid_macd[-2]
        prev_sig = signal_series[-2]
        if not math.isnan(prev_sig):
            if prev_macd <= prev_sig and macd_val > sig_val:
                crossover = "bullish"
            elif prev_macd >= prev_sig and macd_val < sig_val:
                crossover = "bearish"

    trend = "bullish" if macd_val > 0 else "bearish"

    return MACDResult(
        round(macd_val, 6),
        round(sig_val, 6) if not math.isnan(sig_val) else 0.0,
        round(hist_val, 6),
        crossover,
        trend,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bollinger Bands
# ─────────────────────────────────────────────────────────────────────────────

class BBResult(NamedTuple):
    upper: float
    middle: float
    lower: float
    width: float        # (upper - lower) / middle  — volatility proxy
    position: str       # above_upper | below_lower | near_upper | near_lower | middle
    squeeze: bool       # width < 20-period avg width * 0.75


def bollinger_bands(
    prices: list[float],
    period: int = 20,
    std_dev: float = 2.0,
) -> BBResult:
    if len(prices) < period:
        p = prices[-1] if prices else 0
        return BBResult(p, p, p, 0.0, "middle", False)

    window = prices[-period:]
    mid = sum(window) / period
    variance = sum((x - mid) ** 2 for x in window) / period
    std = math.sqrt(variance)

    upper = mid + std_dev * std
    lower = mid - std_dev * std
    width = (upper - lower) / mid if mid else 0.0

    current = prices[-1]
    if current > upper:
        position = "above_upper"
    elif current < lower:
        position = "below_lower"
    elif current > mid + (upper - mid) * 0.7:
        position = "near_upper"
    elif current < mid - (mid - lower) * 0.7:
        position = "near_lower"
    else:
        position = "middle"

    # Squeeze: current width < 75% of recent average width
    squeeze = False
    if len(prices) >= period * 2:
        widths = []
        for i in range(period, len(prices)):
            w = prices[i - period : i]
            m = sum(w) / period
            s = math.sqrt(sum((x - m) ** 2 for x in w) / period)
            widths.append((m + std_dev * s - (m - std_dev * s)) / m if m else 0)
        avg_width = sum(widths) / len(widths) if widths else width
        squeeze = width < avg_width * 0.75

    return BBResult(
        round(upper, 4),
        round(mid, 4),
        round(lower, 4),
        round(width, 6),
        position,
        squeeze,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ATR — Average True Range
# ─────────────────────────────────────────────────────────────────────────────

def atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> float:
    """Wilder ATR.  Used for dynamic SL/TP placement."""
    if len(closes) < period + 1:
        return closes[-1] * 0.01  # fallback 1%

    true_ranges: list[float] = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        true_ranges.append(tr)

    rma = rma_series(true_ranges, period)
    vals = [v for v in rma if not math.isnan(v)]
    return round(vals[-1], 6) if vals else closes[-1] * 0.01


# ─────────────────────────────────────────────────────────────────────────────
# VWAP — Volume-Weighted Average Price
# ─────────────────────────────────────────────────────────────────────────────

def vwap(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
) -> float:
    """Intraday VWAP reset per session."""
    if not volumes or sum(volumes) == 0:
        return closes[-1] if closes else 0.0

    typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    numerator = sum(tp * v for tp, v in zip(typical_prices, volumes))
    denominator = sum(volumes)
    return round(numerator / denominator, 6) if denominator else closes[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Stochastic RSI
# ─────────────────────────────────────────────────────────────────────────────

class StochRSIResult(NamedTuple):
    k: float     # %K (0–100)
    d: float     # %D = SMA(K, 3)
    signal: str  # oversold | overbought | neutral


def stochastic_rsi(
    prices: list[float],
    rsi_period: int = 14,
    stoch_period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> StochRSIResult:
    if len(prices) < rsi_period + stoch_period + smooth_k + smooth_d:
        return StochRSIResult(50.0, 50.0, "neutral")

    # Build RSI series
    rsi_vals: list[float] = []
    for i in range(rsi_period + 1, len(prices) + 1):
        r = rsi(prices[:i], rsi_period)
        rsi_vals.append(r.value)

    if len(rsi_vals) < stoch_period:
        return StochRSIResult(50.0, 50.0, "neutral")

    # Stochastic of RSI
    stoch_vals: list[float] = []
    for i in range(stoch_period, len(rsi_vals) + 1):
        window = rsi_vals[i - stoch_period : i]
        lo, hi = min(window), max(window)
        stoch = (rsi_vals[i - 1] - lo) / (hi - lo) * 100 if hi != lo else 50.0
        stoch_vals.append(stoch)

    # Smooth K
    k_series = sma_series(stoch_vals, smooth_k)
    d_series = sma_series([v for v in k_series if not math.isnan(v)], smooth_d)

    k = k_series[-1] if k_series and not math.isnan(k_series[-1]) else 50.0
    d = d_series[-1] if d_series and not math.isnan(d_series[-1]) else 50.0

    signal = "neutral"
    if k < 20 and d < 20:
        signal = "oversold"
    elif k > 80 and d > 80:
        signal = "overbought"

    return StochRSIResult(round(k, 2), round(d, 2), signal)


# ─────────────────────────────────────────────────────────────────────────────
# Volume Analysis
# ─────────────────────────────────────────────────────────────────────────────

class VolumeResult(NamedTuple):
    current: float
    avg_20: float
    ratio: float         # current / avg_20
    signal: str          # high | low | normal
    obv_trend: str       # up | down | flat  (On-Balance Volume)


def volume_analysis(
    closes: list[float],
    volumes: list[float],
    period: int = 20,
) -> VolumeResult:
    if len(volumes) < period:
        return VolumeResult(0, 0, 1.0, "normal", "flat")

    current = volumes[-1]
    avg = sum(volumes[-period:]) / period
    ratio = current / avg if avg > 0 else 1.0

    signal = "normal"
    if ratio > 1.5:
        signal = "high"
    elif ratio < 0.5:
        signal = "low"

    # OBV trend (last 5)
    obv = 0.0
    obv_series: list[float] = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv += volumes[i]
        elif closes[i] < closes[i - 1]:
            obv -= volumes[i]
        obv_series.append(obv)

    obv_trend = "flat"
    if len(obv_series) >= 5:
        delta = obv_series[-1] - obv_series[-5]
        avg_obv = sum(abs(obv_series[i] - obv_series[i-1]) for i in range(1, len(obv_series))) / max(len(obv_series)-1, 1)
        if delta > avg_obv * 0.5:
            obv_trend = "up"
        elif delta < -avg_obv * 0.5:
            obv_trend = "down"

    return VolumeResult(
        round(current, 2),
        round(avg, 2),
        round(ratio, 3),
        signal,
        obv_trend,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Trend Strength (ADX-like approximation without TA-Lib)
# ─────────────────────────────────────────────────────────────────────────────

def trend_strength(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> tuple[float, str]:
    """
    Returns (adx_value 0–100, direction 'up'|'down'|'sideways').
    Simplified ADX: uses DM/TR ratios.
    """
    if len(closes) < period * 2:
        return 25.0, "sideways"

    dm_plus, dm_minus, trs = [], [], []
    for i in range(1, len(closes)):
        h_diff = highs[i] - highs[i - 1]
        l_diff = lows[i - 1] - lows[i]
        dm_plus.append(h_diff if h_diff > l_diff and h_diff > 0 else 0.0)
        dm_minus.append(l_diff if l_diff > h_diff and l_diff > 0 else 0.0)
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)

    atr_s = rma_series(trs, period)
    dmp_s = rma_series(dm_plus, period)
    dmm_s = rma_series(dm_minus, period)

    valid = [(a, p, m) for a, p, m in zip(atr_s, dmp_s, dmm_s)
             if not any(math.isnan(x) for x in (a, p, m)) and a > 0]
    if not valid:
        return 25.0, "sideways"

    a, p, m = valid[-1]
    di_plus = (p / a) * 100
    di_minus = (m / a) * 100
    dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100 if (di_plus + di_minus) > 0 else 0

    adx_vals = [abs((p/a - m/a) / (p/a + m/a)) * 100 for a, p, m in valid if p+m > 0]
    adx_series = rma_series(adx_vals, period) if len(adx_vals) >= period else adx_vals
    adx = adx_series[-1] if adx_series and not math.isnan(adx_series[-1]) else 25.0

    direction = "sideways"
    if adx > 25:
        direction = "up" if di_plus > di_minus else "down"

    return round(adx, 2), direction
