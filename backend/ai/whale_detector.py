"""
ai/whale_detector.py  — Whale Activity Detector
═══════════════════════════════════════════════════════════════════════════════
Detects abnormal institutional / whale activity using pure OHLCV data.

Heuristics (no order book required)
─────────────────────────────────────
1. VOLUME EXPLOSION  — candle volume > 3× 20-period average
2. SUDDEN BIG CANDLE — body > 3× average body size
3. RAPID REVERSAL   — price retraces > 60% of prior move in 1–2 candles
4. STOP HUNT SPIKE  — wick > 2× body, followed by close in opposite direction
5. MOMENTUM SURGE   — consecutive candles with accelerating volume

When whale activity is detected:
  • Entry is blocked until COOLDOWN_CANDLES pass with normalising volume
  • Alert sent via Telegram
  • MetaAI receives whale_activity=True → auto-REJECT
"""

from __future__ import annotations
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

COOLDOWN_CANDLES = 3   # candles to wait after whale detected before re-enabling


@dataclass
class WhaleResult:
    detected:          bool
    signal_type:       str    # volume_explosion | big_candle | rapid_reversal | stop_hunt | momentum_surge | none
    severity:          float  # 0–1 (1 = most extreme)
    cooldown_candles:  int    # remaining candles to wait
    rationale:         str

    def to_dict(self) -> dict:
        return {
            "detected":         self.detected,
            "signal_type":      self.signal_type,
            "severity":         round(self.severity, 4),
            "cooldown_candles": self.cooldown_candles,
            "rationale":        self.rationale,
        }


def detect_whale_activity(
    highs:   list[float],
    lows:    list[float],
    closes:  list[float],
    volumes: list[float],
    vol_multiplier:   float = 3.0,    # volume spike threshold
    body_multiplier:  float = 3.0,    # body size spike threshold
    reversal_pct:     float = 0.60,   # rapid reversal threshold
    lookback:         int   = 20,
) -> WhaleResult:
    """
    Scan the last `lookback` candles for whale / institutional activity.
    Returns WhaleResult.  If detected=True, do NOT enter new trades.
    """
    n = min(lookback, len(closes))
    if n < 5:
        return WhaleResult(False, "none", 0.0, 0, "Insufficient data")

    h = highs[-n:];  l = lows[-n:]
    c = closes[-n:]; v = volumes[-n:]

    avg_vol = sum(v[:-3]) / max(len(v[:-3]), 1) or 1.0
    bodies  = [abs(c[i] - (h[i] + l[i] - c[i])) for i in range(n)]   # approx
    # Better: |close - open| but we don't have opens; use |close - prev_close|
    body_sizes = [abs(c[i] - c[i-1]) for i in range(1, n)]
    avg_body   = sum(body_sizes[:-3]) / max(len(body_sizes[:-3]), 1) or 1.0

    last_vol  = v[-1]
    last_body = abs(c[-1] - c[-2]) if n >= 2 else 0.0
    last_rng  = h[-1] - l[-1]

    # ── 1. Volume explosion ───────────────────────────────────────────────
    if last_vol > avg_vol * vol_multiplier:
        severity = min((last_vol / avg_vol) / (vol_multiplier * 2), 1.0)
        return WhaleResult(
            True, "volume_explosion", severity, COOLDOWN_CANDLES,
            f"Volume {last_vol:.0f} = {last_vol/avg_vol:.1f}× avg {avg_vol:.0f}"
        )

    # ── 2. Sudden big candle ──────────────────────────────────────────────
    if last_body > avg_body * body_multiplier and last_vol > avg_vol * 1.5:
        severity = min((last_body / avg_body) / (body_multiplier * 2), 1.0)
        return WhaleResult(
            True, "big_candle", severity, COOLDOWN_CANDLES,
            f"Candle body {last_body:.2f} = {last_body/avg_body:.1f}× avg; vol spike"
        )

    # ── 3. Rapid reversal ─────────────────────────────────────────────────
    if n >= 4:
        prior_move = abs(c[-3] - c[-4])
        reversal   = abs(c[-1] - c[-2])
        if prior_move > avg_body * 2 and reversal > prior_move * reversal_pct:
            severity = min(reversal / (prior_move or 1.0), 1.0)
            return WhaleResult(
                True, "rapid_reversal", severity, COOLDOWN_CANDLES,
                f"Reversal {reversal:.2f} = {severity:.0%} of prior move {prior_move:.2f}"
            )

    # ── 4. Stop hunt spike ────────────────────────────────────────────────
    if last_rng > 0:
        wick_total = last_rng - last_body
        wick_ratio = wick_total / last_rng
        if wick_ratio > 0.70 and last_vol > avg_vol * 1.8:
            severity = wick_ratio * min(last_vol / (avg_vol * 2), 1.0)
            return WhaleResult(
                True, "stop_hunt", round(severity, 4), COOLDOWN_CANDLES,
                f"Wick {wick_ratio:.0%} of range, vol {last_vol/avg_vol:.1f}× avg — stop hunt"
            )

    # ── 5. Momentum surge (3 accelerating candles) ────────────────────────
    if n >= 5:
        last3_vols = v[-3:]
        if (last3_vols[1] > last3_vols[0] * 1.3 and
                last3_vols[2] > last3_vols[1] * 1.3 and
                last3_vols[2] > avg_vol * 2.0):
            severity = min(last3_vols[2] / (avg_vol * 3), 1.0)
            return WhaleResult(
                True, "momentum_surge", severity, COOLDOWN_CANDLES,
                f"3-candle vol surge: {[round(x,0) for x in last3_vols]}"
            )

    return WhaleResult(False, "none", 0.0, 0, "No whale activity detected")
