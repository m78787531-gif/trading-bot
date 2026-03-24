"""
ai/meta_ai.py  — Meta-AI Decision Engine
═══════════════════════════════════════════════════════════════════════════════
Top-level autonomous decision layer that overrides all signals.

should_trade() is the single entry point. It synthesises:
  • Signal confidence + trade quality
  • Market regime + volatility state
  • Sentiment score + whale activity flag
  • Current drawdown + loss streak
  • Recent win rate + adaptive mode

Returns TradeDecision(verdict, score, reasons) where verdict ∈ {APPROVE, REJECT}.

Adaptive Modes
──────────────
  NORMAL      — standard thresholds, full position size
  SAFE        — tightened thresholds, 50% size after 2+ losses or DD > 3%
  AGGRESSIVE  — relaxed slightly when win-rate > 65% and DD < 1%
  PROFIT_LOCK — locks in gains; only top-tier signals pass (conf ≥ 90)

10-Trade / 3-Hour Batch Cycle
──────────────────────────────
  After 10 live trades in any 3-hour window the engine enters REST state.
  REST lasts exactly 180 minutes from the 10th trade.
  During REST all new entries are blocked; position management continues.
  Auto-resumes without manual intervention.
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Mode
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveMode(str, Enum):
    NORMAL      = "normal"
    SAFE        = "safe"
    AGGRESSIVE  = "aggressive"
    PROFIT_LOCK = "profit_lock"


# ─────────────────────────────────────────────────────────────────────────────
# Batch Cycle State  (10 trades / 3 hours)
# ─────────────────────────────────────────────────────────────────────────────

BATCH_LIMIT        = 10          # trades per batch
BATCH_REST_SECONDS = 3 * 3600   # 3-hour rest after batch completes


@dataclass
class BatchCycleState:
    """Tracks the 10-trade / 3-hour rest batch cycle."""
    trades_this_batch:  int            = 0
    batch_start_time:   Optional[float] = None   # unix ts of first trade in batch
    rest_until:         Optional[float] = None   # unix ts when rest ends
    total_batches:      int            = 0

    def record_trade(self) -> None:
        now = time.time()
        if self.batch_start_time is None:
            self.batch_start_time = now
        self.trades_this_batch += 1
        if self.trades_this_batch >= BATCH_LIMIT:
            self.rest_until = now + BATCH_REST_SECONDS
            self.total_batches += 1
            logger.warning(
                f"Batch #{self.total_batches} complete ({BATCH_LIMIT} trades). "
                f"REST for 3 hours — auto-resumes at "
                f"{datetime.fromtimestamp(self.rest_until, tz=timezone.utc).strftime('%H:%M UTC')}"
            )

    def is_resting(self) -> bool:
        if self.rest_until is None:
            return False
        if time.time() >= self.rest_until:
            # Auto-resume: reset for next batch
            self.trades_this_batch = 0
            self.batch_start_time  = None
            self.rest_until        = None
            logger.info("Batch REST complete — resuming live trading")
            return False
        return True

    def rest_minutes_remaining(self) -> float:
        if not self.is_resting():
            return 0.0
        return (self.rest_until - time.time()) / 60.0

    def to_dict(self) -> dict:
        resting = self.is_resting()
        return {
            "trades_this_batch":    self.trades_this_batch,
            "batch_limit":          BATCH_LIMIT,
            "rest_period_hours":    BATCH_REST_SECONDS / 3600,
            "is_resting":           resting,
            "rest_minutes_remaining": round(self.rest_minutes_remaining(), 1),
            "total_batches":        self.total_batches,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Trade Decision
# ─────────────────────────────────────────────────────────────────────────────

class TradeVerdict(str, Enum):
    APPROVE = "APPROVE"
    REJECT  = "REJECT"


@dataclass
class TradeDecision:
    verdict:      TradeVerdict
    score:        float          # 0–100 composite meta-score
    adaptive_mode: AdaptiveMode
    reasons:      list[str]
    size_multiplier: float       # 0.0–1.5 — applied to position size
    timestamp:    str

    @property
    def approved(self) -> bool:
        return self.verdict == TradeVerdict.APPROVE

    def to_dict(self) -> dict:
        return {
            "verdict":         self.verdict.value,
            "score":           round(self.score, 2),
            "adaptive_mode":   self.adaptive_mode.value,
            "reasons":         self.reasons,
            "size_multiplier": round(self.size_multiplier, 3),
            "timestamp":       self.timestamp,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds per adaptive mode
# ─────────────────────────────────────────────────────────────────────────────

_THRESHOLDS = {
    AdaptiveMode.NORMAL:      {"min_meta": 60.0, "min_conf": 75.0, "min_quality": 0.70, "size": 1.00},
    AdaptiveMode.SAFE:        {"min_meta": 70.0, "min_conf": 80.0, "min_quality": 0.75, "size": 0.50},
    AdaptiveMode.AGGRESSIVE:  {"min_meta": 55.0, "min_conf": 72.0, "min_quality": 0.65, "size": 1.20},
    AdaptiveMode.PROFIT_LOCK: {"min_meta": 78.0, "min_conf": 90.0, "min_quality": 0.82, "size": 0.75},
}


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Mode Selector
# ─────────────────────────────────────────────────────────────────────────────

def select_adaptive_mode(
    drawdown_pct:       float,   # current session drawdown %
    consecutive_losses: int,
    recent_win_rate:    float,   # last 20 trades, 0–1
    daily_pnl_pct:      float,   # today's PnL %
    session_pnl_pct:    float,   # total session PnL %
) -> AdaptiveMode:
    """
    PROFIT_LOCK — session up > 3% AND win rate > 60%: protect gains
    AGGRESSIVE  — win rate > 65% AND drawdown < 1%: press the edge
    SAFE        — consecutive losses ≥ 2 OR drawdown > 3%: play defence
    NORMAL      — everything else
    """
    if session_pnl_pct >= 3.0 and recent_win_rate >= 0.60:
        return AdaptiveMode.PROFIT_LOCK

    if recent_win_rate >= 0.65 and drawdown_pct < 1.0 and consecutive_losses == 0:
        return AdaptiveMode.AGGRESSIVE

    if consecutive_losses >= 2 or drawdown_pct >= 3.0 or daily_pnl_pct <= -2.0:
        return AdaptiveMode.SAFE

    return AdaptiveMode.NORMAL


# ─────────────────────────────────────────────────────────────────────────────
# Meta-Score Calculation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_meta_score(
    confidence:          float,   # 0–100 signal confidence
    quality:             float,   # 0–1 trade quality score
    regime:              str,     # bull/bear/sideways/volatile_*
    volatility_pct:      float,   # ATR as % of price
    sentiment:           float,   # –1..+1
    direction:           str,     # long | short
    drawdown_pct:        float,
    consecutive_losses:  int,
    orderflow_score:     float,   # 0–1 from orderflow.py
    whale_clear:         bool,    # True = no whale activity detected
    funding_aligned:     bool,    # True = funding bias matches direction
    mtf_aligned:         bool,    # True = all timeframes aligned
) -> tuple[float, list[str]]:
    """
    Builds a 0–100 meta-score from 8 weighted sub-scores.

    Weights
    ───────
    Signal confidence   25 %
    Trade quality       20 %
    Regime alignment    15 %
    Order flow          15 %
    Multi-TF alignment  10 %
    Sentiment           5  %
    Funding bias        5  %
    Risk state          5  %
    """
    reasons: list[str] = []
    score    = 0.0

    # 1. Signal confidence (25%)
    conf_norm = min(confidence / 100.0, 1.0)
    score += conf_norm * 25.0
    reasons.append(f"Confidence {confidence:.1f}% → +{conf_norm*25:.1f}")

    # 2. Trade quality (20%)
    score += quality * 20.0
    reasons.append(f"Quality {quality:.2f} → +{quality*20:.1f}")

    # 3. Regime alignment (15%)
    regime_score = 0.0
    if regime in ("bull", "bear"):
        if (regime == "bull" and direction == "long") or            (regime == "bear" and direction == "short"):
            regime_score = 1.0
            reasons.append(f"Regime {regime} aligned with {direction} → +15.0")
        else:
            regime_score = 0.2
            reasons.append(f"Regime {regime} contra {direction} → +3.0 (penalised)")
    elif regime in ("volatile_bull", "volatile_bear"):
        if (regime == "volatile_bull" and direction == "long") or            (regime == "volatile_bear" and direction == "short"):
            regime_score = 0.70
        else:
            regime_score = 0.15
        reasons.append(f"Volatile regime, partial credit → +{regime_score*15:.1f}")
    else:  # sideways / unknown
        regime_score = 0.35
        reasons.append(f"Sideways/unknown regime → +{regime_score*15:.1f}")
    score += regime_score * 15.0

    # 4. Order flow (15%)
    score += orderflow_score * 15.0
    reasons.append(f"Order flow {orderflow_score:.2f} → +{orderflow_score*15:.1f}")

    # 5. Multi-timeframe alignment (10%)
    mtf_bonus = 10.0 if mtf_aligned else 3.0
    score += mtf_bonus
    reasons.append(f"MTF aligned={mtf_aligned} → +{mtf_bonus:.1f}")

    # 6. Sentiment (5%)
    if direction == "long":
        sent_aligned = (sentiment + 1.0) / 2.0
    else:
        sent_aligned = (1.0 - sentiment) / 2.0
    score += sent_aligned * 5.0
    reasons.append(f"Sentiment {sentiment:+.2f} aligned={sent_aligned:.2f} → +{sent_aligned*5:.1f}")

    # 7. Funding bias (5%)
    if funding_aligned:
        score += 5.0
        reasons.append("Funding rate aligned → +5.0")
    else:
        reasons.append("Funding rate contra → +0.0")

    # 8. Risk state (5%)
    risk_score = 5.0
    if consecutive_losses >= 3:
        risk_score = 0.0
        reasons.append(f"Consecutive losses {consecutive_losses} → risk score zero")
    elif consecutive_losses >= 2:
        risk_score = 2.0
        reasons.append(f"Consecutive losses {consecutive_losses} → risk score halved")
    elif drawdown_pct >= 5.0:
        risk_score = 1.0
        reasons.append(f"Drawdown {drawdown_pct:.1f}% → risk score minimal")
    else:
        reasons.append(f"Risk state clean → +{risk_score:.1f}")
    score += risk_score

    return round(min(score, 100.0), 2), reasons


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def should_trade(
    confidence:          float,
    quality:             float,
    regime:              str,
    direction:           str,
    volatility_pct:      float,
    sentiment:           float,
    drawdown_pct:        float,
    consecutive_losses:  int,
    recent_win_rate:     float,
    daily_pnl_pct:       float,
    session_pnl_pct:     float,
    orderflow_score:     float   = 0.7,
    whale_activity:      bool    = False,
    funding_aligned:     bool    = True,
    mtf_aligned:         bool    = True,
    batch_state:         Optional[BatchCycleState] = None,
) -> TradeDecision:
    """
    Master autonomous trade-approval function.

    Returns TradeDecision. decision.approved must be True for execution.

    Gate order
    ──────────
    1. Batch rest check    — 3-hour cooldown after 10 trades
    2. Whale activity      — avoid entries during abnormal moves
    3. Adaptive mode       — select NORMAL/SAFE/AGGRESSIVE/PROFIT_LOCK
    4. Meta-score          — 0–100 composite from 8 sub-scores
    5. Mode thresholds     — score, confidence, quality gates per mode
    6. Finalise            — size multiplier + verdict
    """
    now = datetime.now(timezone.utc).isoformat()
    reasons: list[str] = []

    # ── Gate 1: Batch rest ────────────────────────────────────────────────
    if batch_state and batch_state.is_resting():
        mins = batch_state.rest_minutes_remaining()
        return TradeDecision(
            verdict=TradeVerdict.REJECT,
            score=0.0,
            adaptive_mode=AdaptiveMode.SAFE,
            reasons=[f"BATCH REST: {mins:.0f} min remaining after {BATCH_LIMIT}-trade cycle"],
            size_multiplier=0.0,
            timestamp=now,
        )

    # ── Gate 2: Whale activity ────────────────────────────────────────────
    if whale_activity:
        return TradeDecision(
            verdict=TradeVerdict.REJECT,
            score=0.0,
            adaptive_mode=AdaptiveMode.SAFE,
            reasons=["WHALE ALERT: abnormal activity detected — waiting for stabilisation"],
            size_multiplier=0.0,
            timestamp=now,
        )

    # ── Gate 3: Adaptive mode ─────────────────────────────────────────────
    mode = select_adaptive_mode(
        drawdown_pct, consecutive_losses,
        recent_win_rate, daily_pnl_pct, session_pnl_pct,
    )
    thresh = _THRESHOLDS[mode]
    reasons.append(f"Adaptive mode: {mode.value.upper()}")

    # ── Gate 4: Meta-score ────────────────────────────────────────────────
    meta_score, score_reasons = _compute_meta_score(
        confidence, quality, regime, volatility_pct, sentiment, direction,
        drawdown_pct, consecutive_losses, orderflow_score,
        not whale_activity, funding_aligned, mtf_aligned,
    )
    reasons.extend(score_reasons)

    # ── Gate 5: Threshold checks ──────────────────────────────────────────
    rejects: list[str] = []

    if meta_score < thresh["min_meta"]:
        rejects.append(
            f"Meta-score {meta_score:.1f} < {thresh['min_meta']} ({mode.value})")

    if confidence < thresh["min_conf"]:
        rejects.append(
            f"Confidence {confidence:.1f} < {thresh['min_conf']} ({mode.value})")

    if quality < thresh["min_quality"]:
        rejects.append(
            f"Quality {quality:.2f} < {thresh['min_quality']} ({mode.value})")

    if volatility_pct < 0.002:
        rejects.append(f"ATR {volatility_pct*100:.3f}% too compressed — no room for SL/TP")

    if volatility_pct > 0.04:
        rejects.append(f"ATR {volatility_pct*100:.2f}% extreme — position size risk unacceptable")

    if orderflow_score < 0.40:
        rejects.append(f"Order flow {orderflow_score:.2f} unclear — smart-money direction ambiguous")

    if not mtf_aligned and mode in (AdaptiveMode.SAFE, AdaptiveMode.PROFIT_LOCK):
        rejects.append(f"MTF misaligned in {mode.value} mode — require full confluence")

    # ── Gate 6: Final verdict ─────────────────────────────────────────────
    if rejects:
        reasons.extend(rejects)
        return TradeDecision(
            verdict=TradeVerdict.REJECT,
            score=meta_score,
            adaptive_mode=mode,
            reasons=reasons,
            size_multiplier=0.0,
            timestamp=now,
        )

    # Size multiplier: mode base × score boost
    score_boost  = min((meta_score - thresh["min_meta"]) / 40.0, 0.5)
    size_mult    = round(min(thresh["size"] + score_boost, 1.5), 3)
    reasons.append(
        f"APPROVED — meta={meta_score:.1f} conf={confidence:.1f} "
        f"qual={quality:.2f} size×{size_mult}"
    )

    return TradeDecision(
        verdict=TradeVerdict.APPROVE,
        score=meta_score,
        adaptive_mode=mode,
        reasons=reasons,
        size_multiplier=size_mult,
        timestamp=now,
    )
