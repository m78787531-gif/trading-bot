"""
ai/self_optimizer.py  [v2 — Anti-Overfitting Optimized]
─────────────────────────────────────────────────────────────────────────────
Loss-learning & self-optimisation engine — v2.

v2 changes
──────────
MINIMUM TRADES: 10 → 50
  No optimisation runs until at least 50 closed trades are recorded.
  Prevents overfitting to small, statistically insignificant samples.

LEARNING RATE: 0.15 → 0.01
  Parameters adjust by at most 1% per optimisation run (was 15%).
  Prevents overcorrection to recent noise.

OVERFITTING SAFEGUARDS
  • MAX_PARAM_CHANGE_PCT  5%  — hard cap on drift per param per run
  • OVERFITTING_CORR_THRESHOLD 0.70 — require ≥70% bucket win rate to act
  • ROLLING_WINDOW 200  — only analyse last 200 trades
  • MIN_BUCKET_SAMPLES 10 — require 10 samples before treating as significant
  • Convergence lock: if WR hasn't improved in 3 consecutive opt runs →
    freeze params for next 25 trades

STRATEGY PERFORMANCE
  get_strategy_performance() returns per-mode win rate + PnL breakdown.

How it works
────────────
  1. record_trade_result()   → called after every closed trade
  2. optimise_parameters()   → triggered every 10 trades (min 50)
  3. Pattern analysis        → RSI / ADX / confidence / volume buckets
  4. Micro-adjustments       → ≤1% drift per param per run
  5. Convergence guard       → lock if no improvement detected
"""

from __future__ import annotations
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ── Tuning constants ─────────────────────────────────────────────────────────
MIN_TRADES_FOR_OPTIMISATION  = 50      # v2: was 10
ADJUSTMENT_RATE              = 0.01    # v2: was 0.15
MAX_HISTORY                  = 500
ROLLING_WINDOW               = 200
MIN_BUCKET_SAMPLES           = 10
OVERFITTING_CORR_THRESHOLD   = 0.70
MAX_CONSECUTIVE_OPT_LOCK     = 25
MAX_PARAM_CHANGE_PCT         = 0.05
OPT_TRIGGER_EVERY_N          = 10


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    trade_id: int
    symbol: str
    direction: str
    pnl: float
    pnl_pct: float
    won: bool
    confidence_at_entry: float
    rsi_at_entry: float
    macd_trend_at_entry: str
    ema_alignment_at_entry: str
    smc_score_at_entry: float
    sentiment_at_entry: float
    adx_at_entry: float
    volume_ratio_at_entry: float
    bb_position_at_entry: str
    strategy_mode: str              # trend | range | breakout | unknown
    opened_at: str
    closed_at: str


@dataclass
class PerformancePattern:
    factor: str
    bucket: str
    wins: int
    losses: int
    total_pnl: float
    win_rate: float
    avg_pnl: float

    @property
    def edge(self) -> float:
        return (self.win_rate - 0.5) * abs(self.avg_pnl)

    @property
    def is_significant(self) -> bool:
        """True only when sample is large enough and win rate is decisive."""
        return (
            (self.wins + self.losses) >= MIN_BUCKET_SAMPLES
            and abs(self.win_rate - 0.5) >= (1.0 - OVERFITTING_CORR_THRESHOLD)
        )


@dataclass
class OptimisationResult:
    previous_params: dict
    new_params: dict
    changes: list
    performance_summary: dict
    trades_analysed: int
    analysis_timestamp: str
    locked: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# In-memory state
# ─────────────────────────────────────────────────────────────────────────────

_trade_history: list = []           # list[TradeRecord]
_current_params: dict = {}

_opt_run_count: int          = 0
_last_opt_win_rate: float    = 0.0
_no_improvement_runs: int    = 0
_locked_until_trade_n: int   = 0


def set_params(params: dict) -> None:
    global _current_params
    _current_params = params.copy()


def get_params() -> dict:
    from ai.signal_scorer import DEFAULT_PARAMS
    return _current_params if _current_params else DEFAULT_PARAMS.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Record Trade Result
# ─────────────────────────────────────────────────────────────────────────────

def record_trade_result(
    trade_id: int,
    symbol: str,
    direction: str,
    pnl: float,
    pnl_pct: float,
    signal_snapshot: Optional[dict] = None,
    strategy_mode: str = "unknown",
    opened_at: str = "",
    closed_at: str = "",
) -> bool:
    """
    Record a closed trade and trigger auto-optimisation when conditions are met.
    Returns True if optimisation ran this call.
    """
    snap      = signal_snapshot or {}
    indic     = snap.get("indicators", {})
    breakdown = snap.get("score_breakdown", {})
    smc       = snap.get("smc", {})

    record = TradeRecord(
        trade_id=trade_id,
        symbol=symbol,
        direction=direction,
        pnl=pnl,
        pnl_pct=pnl_pct,
        won=(pnl > 0),
        confidence_at_entry=snap.get("confidence", 75.0),
        rsi_at_entry=indic.get("rsi_value", 50.0),
        macd_trend_at_entry=indic.get("macd_trend", "neutral"),
        ema_alignment_at_entry=indic.get("ema_alignment", "mixed"),
        smc_score_at_entry=smc.get("smc_score", 0.0),
        sentiment_at_entry=breakdown.get("sentiment_score", 0.0),
        adx_at_entry=indic.get("adx", 25.0),
        volume_ratio_at_entry=indic.get("volume_ratio", 1.0),
        bb_position_at_entry=indic.get("bb_position", "middle"),
        strategy_mode=snap.get("strategy_mode", strategy_mode),
        opened_at=opened_at,
        closed_at=closed_at,
    )

    _trade_history.append(record)
    if len(_trade_history) > MAX_HISTORY:
        _trade_history.pop(0)

    logger.info(
        f"Recorded #{trade_id} [{symbol}/{strategy_mode}] "
        f"{'WIN' if pnl > 0 else 'LOSS'} {pnl_pct:+.2f}%  "
        f"(total history: {len(_trade_history)}, opt starts at {MIN_TRADES_FOR_OPTIMISATION})"
    )

    n = len(_trade_history)
    if n >= MIN_TRADES_FOR_OPTIMISATION and n % OPT_TRIGGER_EVERY_N == 0:
        result = optimise_parameters()
        if not result.locked:
            logger.info(f"Auto-optimisation complete: {len(result.changes)} change(s)")
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Bucket Classifiers
# ─────────────────────────────────────────────────────────────────────────────

def _bucket_rsi(v: float) -> str:
    if v < 25:  return "extreme_oversold"
    if v < 35:  return "oversold"
    if v < 45:  return "below_neutral"
    if v < 55:  return "neutral"
    if v < 65:  return "above_neutral"
    if v < 75:  return "overbought"
    return "extreme_overbought"


def _bucket_adx(v: float) -> str:
    if v < 15:  return "very_weak"
    if v < 25:  return "weak"
    if v < 35:  return "moderate"
    if v < 50:  return "strong"
    return "very_strong"


def _bucket_confidence(v: float) -> str:
    if v < 75:  return "below_threshold"    # shouldn't appear post-v2
    if v < 80:  return "threshold"
    if v < 88:  return "good"
    return "excellent"


def _bucket_volume(v: float) -> str:
    if v < 0.7:  return "low"
    if v < 1.0:  return "below_avg"
    if v < 1.5:  return "normal"
    if v < 2.5:  return "spike"
    return "very_high"


# ─────────────────────────────────────────────────────────────────────────────
# Pattern Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_patterns(trades: list) -> list:
    """Build PerformancePattern objects per factor × bucket."""
    patterns = []

    def _process(factor: str, bucket_fn, key_fn):
        buckets: dict = defaultdict(list)
        for t in trades:
            buckets[bucket_fn(key_fn(t))].append(t)
        for bucket, group in buckets.items():
            wins  = sum(1 for t in group if t.won)
            total = len(group)
            losses = total - wins
            wr = wins / total if total > 0 else 0.0
            ap = sum(t.pnl_pct for t in group) / total if total > 0 else 0.0
            patterns.append(PerformancePattern(
                factor=factor, bucket=bucket, wins=wins, losses=losses,
                total_pnl=sum(t.pnl_pct for t in group),
                win_rate=round(wr, 4), avg_pnl=round(ap, 4),
            ))

    _process("rsi",        _bucket_rsi,        lambda t: t.rsi_at_entry)
    _process("adx",        _bucket_adx,        lambda t: t.adx_at_entry)
    _process("confidence", _bucket_confidence, lambda t: t.confidence_at_entry)
    _process("volume",     _bucket_volume,     lambda t: t.volume_ratio_at_entry)

    # EMA alignment
    ema_buckets: dict = defaultdict(list)
    for t in trades:
        ema_buckets[t.ema_alignment_at_entry].append(t)
    for bucket, group in ema_buckets.items():
        wins  = sum(1 for t in group if t.won)
        total = len(group)
        patterns.append(PerformancePattern(
            factor="ema", bucket=bucket, wins=wins, losses=total - wins,
            total_pnl=sum(t.pnl_pct for t in group),
            win_rate=round(wins / total, 4) if total > 0 else 0.0,
            avg_pnl=round(sum(t.pnl_pct for t in group) / max(total, 1), 4),
        ))

    return patterns


# ─────────────────────────────────────────────────────────────────────────────
# Micro-Adjustment Engine  (anti-overfitting core)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_adjust(current: float, target: float, bounds: tuple) -> float:
    """
    Move current → target by ADJUSTMENT_RATE (0.01), capped by:
      1. ±MAX_PARAM_CHANGE_PCT (5%) of current value per run
      2. Hard bounds
    """
    if current == 0.0:
        return current
    raw_step = (target - current) * ADJUSTMENT_RATE
    max_step = abs(current) * MAX_PARAM_CHANGE_PCT
    step     = max(-max_step, min(max_step, raw_step))
    new_val  = current + step
    return round(max(bounds[0], min(bounds[1], new_val)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Main Optimisation Engine
# ─────────────────────────────────────────────────────────────────────────────

def optimise_parameters() -> OptimisationResult:
    """
    Analyse recent trades and apply micro-adjustments to signal parameters.

    Overfitting safeguards (applied in order):
      1. Minimum 50 trades required
      2. Convergence lock — freeze if 3 runs with no win-rate improvement
      3. Significant patterns only — MIN_BUCKET_SAMPLES + decisive win rate
      4. MAX_PARAM_CHANGE_PCT cap on every adjustment
      5. Hard parameter bounds to prevent extreme drift
    """
    global _opt_run_count, _last_opt_win_rate, _no_improvement_runs, _locked_until_trade_n

    from ai.signal_scorer import DEFAULT_PARAMS
    ts_now = datetime.now(timezone.utc).isoformat()

    trades = _trade_history[-ROLLING_WINDOW:]
    n      = len(trades)

    if n < MIN_TRADES_FOR_OPTIMISATION:
        return OptimisationResult(
            previous_params={}, new_params={}, changes=[],
            performance_summary={
                "message": f"Need {MIN_TRADES_FOR_OPTIMISATION} trades, have {n}.",
                "trades_until_opt": MIN_TRADES_FOR_OPTIMISATION - n,
            },
            trades_analysed=n, analysis_timestamp=ts_now,
        )

    # Convergence lock
    if len(_trade_history) <= _locked_until_trade_n:
        logger.info(
            f"Optimisation locked until trade #{_locked_until_trade_n} "
            f"(current: {len(_trade_history)})"
        )
        return OptimisationResult(
            previous_params={}, new_params={}, changes=[],
            performance_summary={"message": "Convergence lock active — params frozen."},
            trades_analysed=n, analysis_timestamp=ts_now, locked=True,
        )

    old_params = get_params().copy()
    new_params  = old_params.copy()
    changes: list = []
    patterns   = analyse_patterns(trades)

    overall_wr  = sum(1 for t in trades if t.won) / n
    overall_pnl = sum(t.pnl_pct for t in trades) / n

    logger.info(
        f"Optimising {n} trades | WR={overall_wr:.1%} | "
        f"AvgPnL={overall_pnl:+.2f}% | Run #{_opt_run_count + 1}"
    )

    # ── 1. RSI oversold / overbought thresholds ──────────────────────────────
    rsi_map = {
        "extreme_oversold": 22.5, "oversold": 30,
        "below_neutral": 40,      "above_neutral": 60,
        "overbought": 70,         "extreme_overbought": 77.5,
    }
    rsi_sig = [p for p in patterns if p.factor == "rsi" and p.is_significant]
    t_os    = old_params.get("rsi_oversold",  30.0)
    t_ob    = old_params.get("rsi_overbought", 70.0)

    for pat in rsi_sig:
        if pat.bucket in ("extreme_oversold", "oversold") and pat.win_rate > 0.70:
            t_os = rsi_map.get(pat.bucket, t_os) + 2
        if pat.bucket in ("overbought", "extreme_overbought") and pat.win_rate > 0.70:
            t_ob = rsi_map.get(pat.bucket, t_ob) - 2

    new_os = _safe_adjust(old_params.get("rsi_oversold",  30.0), t_os, (20.0, 42.0))
    new_ob = _safe_adjust(old_params.get("rsi_overbought", 70.0), t_ob, (58.0, 80.0))

    if abs(new_os - old_params.get("rsi_oversold",  30.0)) > 0.05:
        new_params["rsi_oversold"] = new_os
        changes.append(f"RSI oversold {old_params.get('rsi_oversold', 30):.2f} → {new_os:.2f}")
    if abs(new_ob - old_params.get("rsi_overbought", 70.0)) > 0.05:
        new_params["rsi_overbought"] = new_ob
        changes.append(f"RSI overbought {old_params.get('rsi_overbought', 70):.2f} → {new_ob:.2f}")

    # ── 2. RSI entry zone  ────────────────────────────────────────────────────
    # Only ever widen the zone if trades in the 40-60 window are profitable
    zone_sig = [p for p in patterns if p.factor == "rsi"
                and p.bucket in ("below_neutral", "above_neutral") and p.is_significant]
    best_zone_wr = max((p.win_rate for p in zone_sig), default=0.0)
    if best_zone_wr > 0.70:
        nr_min = _safe_adjust(old_params.get("rsi_entry_min", 40.0), 38.0, (35.0, 45.0))
        nr_max = _safe_adjust(old_params.get("rsi_entry_max", 60.0), 62.0, (55.0, 65.0))
        if abs(nr_min - old_params.get("rsi_entry_min", 40.0)) > 0.05:
            new_params["rsi_entry_min"] = nr_min
            changes.append(f"RSI entry min {old_params.get('rsi_entry_min', 40):.2f} → {nr_min:.2f}")
        if abs(nr_max - old_params.get("rsi_entry_max", 60.0)) > 0.05:
            new_params["rsi_entry_max"] = nr_max
            changes.append(f"RSI entry max {old_params.get('rsi_entry_max', 60):.2f} → {nr_max:.2f}")

    # ── 3. Min confidence — only raise, never lower below 75 ─────────────────
    conf_sig = [p for p in patterns if p.factor == "confidence"]
    thresh_p = next((p for p in conf_sig if p.bucket == "threshold" and p.is_significant), None)
    if thresh_p and thresh_p.win_rate < 0.45:
        new_mc = _safe_adjust(old_params.get("min_confidence", 75.0), 78.0, (75.0, 88.0))
        if abs(new_mc - old_params.get("min_confidence", 75.0)) > 0.05:
            new_params["min_confidence"] = new_mc
            changes.append(f"Min confidence {old_params.get('min_confidence', 75):.2f} → {new_mc:.2f}")

    # ── 4. ADX minimum ────────────────────────────────────────────────────────
    adx_sig  = [p for p in patterns if p.factor == "adx" and p.is_significant]
    weak_adx = next((p for p in adx_sig if p.bucket in ("very_weak", "weak")), None)
    if weak_adx and weak_adx.win_rate < 0.40:
        new_adx = _safe_adjust(old_params.get("adx_min", 20.0), 22.0, (18.0, 28.0))
        if abs(new_adx - old_params.get("adx_min", 20.0)) > 0.05:
            new_params["adx_min"] = new_adx
            changes.append(f"ADX min {old_params.get('adx_min', 20):.2f} → {new_adx:.2f}")

    # ── 5. Volume spike minimum ───────────────────────────────────────────────
    vol_sig = [p for p in patterns if p.factor == "volume" and p.is_significant]
    low_vol = next((p for p in vol_sig if p.bucket in ("low", "below_avg")), None)
    if low_vol and low_vol.win_rate < 0.40:
        new_vs = _safe_adjust(old_params.get("volume_spike_min", 1.5), 1.7, (1.3, 2.5))
        if abs(new_vs - old_params.get("volume_spike_min", 1.5)) > 0.01:
            new_params["volume_spike_min"] = new_vs
            changes.append(f"Volume spike min {old_params.get('volume_spike_min', 1.5):.3f} → {new_vs:.3f}")

    # ── 6. SL / TP multipliers ────────────────────────────────────────────────
    sl_hits  = sum(1 for t in trades if not t.won and t.pnl_pct > -3.0)
    tp_small = sum(1 for t in trades if t.won and t.pnl_pct < 1.0)

    if sl_hits / n > 0.35:
        new_sl = _safe_adjust(old_params.get("sl_atr_multiplier", 1.5), 1.7, (1.2, 2.5))
        if abs(new_sl - old_params.get("sl_atr_multiplier", 1.5)) > 0.01:
            new_params["sl_atr_multiplier"] = new_sl
            changes.append(f"SL ATR mult {old_params.get('sl_atr_multiplier', 1.5):.3f} → {new_sl:.3f} (tight SL)")

    if tp_small / n > 0.30:
        new_tp = _safe_adjust(old_params.get("tp_atr_multiplier", 3.0), 2.7, (2.0, 4.5))
        if abs(new_tp - old_params.get("tp_atr_multiplier", 3.0)) > 0.01:
            new_params["tp_atr_multiplier"] = new_tp
            changes.append(f"TP ATR mult {old_params.get('tp_atr_multiplier', 3.0):.3f} → {new_tp:.3f} (small wins)")

    # ── 7. Convergence guard ──────────────────────────────────────────────────
    _opt_run_count += 1

    if overall_wr >= _last_opt_win_rate * 0.95:
        _no_improvement_runs = 0
    else:
        _no_improvement_runs += 1
        logger.warning(
            f"No WR improvement: run {_opt_run_count} "
            f"({_last_opt_win_rate:.1%} → {overall_wr:.1%}), "
            f"consecutive={_no_improvement_runs}"
        )

    if _no_improvement_runs >= 3:
        _locked_until_trade_n = len(_trade_history) + MAX_CONSECUTIVE_OPT_LOCK
        _no_improvement_runs  = 0
        msg = f"Convergence lock: params frozen for next {MAX_CONSECUTIVE_OPT_LOCK} trades"
        changes.append(f"⚠ {msg}")
        logger.warning(msg)

    _last_opt_win_rate = overall_wr

    if changes:
        set_params(new_params)
        logger.info(f"Parameters updated: {changes}")
    else:
        logger.info("Optimisation complete — no adjustments needed (params stable)")

    return OptimisationResult(
        previous_params=old_params,
        new_params=new_params,
        changes=changes,
        performance_summary={
            "total_trades":    n,
            "win_rate":        round(overall_wr,  4),
            "avg_pnl_pct":     round(overall_pnl, 4),
            "total_pnl":       round(sum(t.pnl for t in trades), 4),
            "best_trade_pct":  round(max(t.pnl_pct for t in trades), 4) if trades else 0,
            "worst_trade_pct": round(min(t.pnl_pct for t in trades), 4) if trades else 0,
            "opt_run":         _opt_run_count,
            "locked_until_trade": _locked_until_trade_n,
        },
        trades_analysed=n,
        analysis_timestamp=datetime.now(timezone.utc).isoformat(),
        locked=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Performance  (called by bot_engine & performance API)
# ─────────────────────────────────────────────────────────────────────────────

def get_strategy_performance() -> dict:
    """Per-strategy win rate + PnL + avg confidence breakdown."""
    if not _trade_history:
        return {}

    result: dict = {}
    for t in _trade_history:
        mode = t.strategy_mode or "unknown"
        if mode not in result:
            result[mode] = {"wins": 0, "losses": 0, "count": 0,
                            "total_pnl_pct": 0.0, "avg_confidence": 0.0}
        s = result[mode]
        s["count"]          += 1
        s["total_pnl_pct"]  += t.pnl_pct
        s["avg_confidence"] += t.confidence_at_entry
        if t.won: s["wins"]   += 1
        else:     s["losses"] += 1

    for mode, s in result.items():
        c = s["count"]
        s["win_rate"]       = round(s["wins"] / c, 4) if c > 0 else 0.0
        s["avg_pnl_pct"]    = round(s["total_pnl_pct"] / c, 4) if c > 0 else 0.0
        s["avg_confidence"] = round(s["avg_confidence"] / c, 2) if c > 0 else 0.0
        del s["total_pnl_pct"]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Summary API (for /bot/performance endpoint)
# ─────────────────────────────────────────────────────────────────────────────

def get_performance_summary() -> dict:
    trades = _trade_history
    if not trades:
        return {
            "message": (
                f"No trades recorded yet. "
                f"Self-optimisation starts after {MIN_TRADES_FOR_OPTIMISATION} closed trades."
            ),
            "min_trades_required":  MIN_TRADES_FOR_OPTIMISATION,
            "trades_recorded":      0,
        }

    won  = [t for t in trades if t.won]
    lost = [t for t in trades if not t.won]
    n    = len(trades)

    return {
        "total_trades":               n,
        "wins":                       len(won),
        "losses":                     len(lost),
        "win_rate":                   round(len(won) / n, 4),
        "total_pnl":                  round(sum(t.pnl for t in trades), 4),
        "avg_win_pct":                round(sum(t.pnl_pct for t in won)  / max(len(won),  1), 4),
        "avg_loss_pct":               round(sum(t.pnl_pct for t in lost) / max(len(lost), 1), 4),
        "trades_until_first_opt":     max(0, MIN_TRADES_FOR_OPTIMISATION - n),
        "opt_runs_completed":         _opt_run_count,
        "convergence_lock_until_n":   _locked_until_trade_n,
        "learning_rate":              ADJUSTMENT_RATE,
        "current_params":             get_params(),
        "last_trade_closed_at":       trades[-1].closed_at if trades else None,
        "strategy_breakdown":         get_strategy_performance(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Live Learning Engine  (Req 13)
# ─────────────────────────────────────────────────────────────────────────────

_live_learning_buffer: list = []   # rolling last-N for real-time adaptation
_live_win_count:  int   = 0
_live_loss_count: int   = 0
_live_session_wr: float = 0.5      # session win rate (exponential moving average)

LIVE_EMA_ALPHA = 0.15   # smoothing for EMA win rate update


def live_update(pnl: float, signal_snapshot: dict) -> dict:
    """
    Called immediately after each trade closes.
    Returns updated weight recommendations for signal_scorer.

    Updates:
      • Session EMA win rate (_live_session_wr)
      • Rolling 10-trade performance buffer
      • Adjusts weight hints if pattern is clear (≥5 trades)
    """
    global _live_win_count, _live_loss_count, _live_session_wr

    won = pnl > 0
    if won: _live_win_count  += 1
    else:   _live_loss_count += 1

    # EMA win rate
    _live_session_wr = LIVE_EMA_ALPHA * (1.0 if won else 0.0) + (1 - LIVE_EMA_ALPHA) * _live_session_wr

    indic     = signal_snapshot.get("indicators", {})
    breakdown = signal_snapshot.get("score_breakdown", {})

    _live_learning_buffer.append({
        "won":        won,
        "pnl":        pnl,
        "smc_raw":    breakdown.get("smc_raw",      0.0),
        "trend_raw":  breakdown.get("trend_raw",    0.0),
        "momentum":   breakdown.get("momentum_raw", 0.0),
        "volume":     breakdown.get("volume_raw",   0.0),
        "adx":        indic.get("adx", 25.0),
        "rsi":        indic.get("rsi_value", 50.0),
    })
    if len(_live_learning_buffer) > 20:
        _live_learning_buffer.pop(0)

    weight_hints: dict = {}

    if len(_live_learning_buffer) >= 5:
        wins  = [t for t in _live_learning_buffer if t["won"]]
        losses= [t for t in _live_learning_buffer if not t["won"]]

        def _avg(lst, key):
            return sum(t[key] for t in lst) / len(lst) if lst else 0.0

        # If winning trades have higher SMC score → increase SMC weight hint
        win_smc  = _avg(wins,   "smc_raw")
        loss_smc = _avg(losses, "smc_raw")
        if win_smc > loss_smc + 0.15:
            weight_hints["w_smc"] = min(0.40, 0.35 + (win_smc - loss_smc) * 0.1)

        # If winning trades have stronger trend → adjust trend weight
        win_trend  = _avg(wins,   "trend_raw")
        loss_trend = _avg(losses, "trend_raw")
        if win_trend > loss_trend + 0.15:
            weight_hints["w_trend"] = min(0.30, 0.25 + (win_trend - loss_trend) * 0.1)

        logger.info(
            f"Live learning update: WR={_live_session_wr:.1%} "
            f"({_live_win_count}W/{_live_loss_count}L) hints={weight_hints}"
        )

    return {
        "session_win_rate":  round(_live_session_wr, 4),
        "total_wins":        _live_win_count,
        "total_losses":      _live_loss_count,
        "weight_hints":      weight_hints,
    }


def get_live_session_stats() -> dict:
    """Return current live-session learning stats for API/monitoring."""
    return {
        "session_win_rate":  round(_live_session_wr, 4),
        "total_wins":        _live_win_count,
        "total_losses":      _live_loss_count,
        "recent_buffer_n":   len(_live_learning_buffer),
        "live_ema_alpha":    LIVE_EMA_ALPHA,
    }
