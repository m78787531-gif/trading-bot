"""
ai/risk_manager.py  [v3 — Intelligent Trade Control]
─────────────────────────────────────────────────────────────────────────────
v3 additions
────────────
TRADING MODE MANAGEMENT  (live / paper with auto-switch)
  TradingMode enum: LIVE | PAPER
  Daily loss protection:
    • daily_loss_count ≥ 9   → switch to paper for the day
    • daily_drawdown ≥ 3%    → switch to paper immediately
    • Consecutive losses ≥ 4 → pause 2h (unchanged threshold raised to 4)
  Recovery mode:
    After a loss day, risk_pct drops to 0.5% and recovers +0.05% per win
    until restored to base.
  Next-day reset:
    reset_daily_if_needed() switches back to LIVE at midnight.

LIVE TRADE BATCH CYCLE  (max N live trades per cycle)
  live_trade_count_cycle: int   — resets when user manually switches to live
  max_trades_per_live_cycle: 5  — after 5, auto-switch to paper
  Adaptive: if avg trade_quality < 0.65 → reduce to 3
  Safety gate before re-enabling live:
    daily_drawdown < 3%, consecutive_losses < 3, market_quality ≥ 0.70

DRAWDOWN-ADAPTIVE SIZING
  current_drawdown ≥ 5%  → 50% position size reduction
  current_drawdown ≥ 10% → kill switch (unchanged from v2)

CONFIDENCE-BASED RISK TIERS  (replaces flat 1%)
  conf ≥ 85 → risk 1.5%
  conf ≥ 75 → risk 1.0%
  else       → reject (returns quantity=0)

SYMBOL EXPOSURE LIMIT
  max_open_per_symbol: 2  — max 2 open trades per symbol

CORRELATION CONTROL
  Correlated groups detected; same-direction second trade in a group
  gets 50% size reduction or is skipped if already at corr limit.

VOLATILITY FILTER (position-level)
  Extreme ATR (> 3% of price) → 50% size reduction.

MARKET QUALITY FILTER
  market_quality_score (from TradeQualityScore.total) must be ≥ 0.70.
  Passed in from signal and checked in assess_trade_risk().

LOSS PATTERN BLOCKER
  check_loss_pattern() inspects self_optimizer pattern data and returns
  True if current signal matches a statistically losing pattern.
  Integrated into assess_trade_risk().

Telegram alerts added for:
  • Switch to paper mode
  • Daily reset / resume live
  • Recovery mode activation
"""

from __future__ import annotations
import logging
import math
import statistics
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Trading Mode
# ─────────────────────────────────────────────────────────────────────────────

class TradingMode(str, Enum):
    LIVE  = "live"
    PAPER = "paper"


# ─────────────────────────────────────────────────────────────────────────────
# Correlation Groups
# ─────────────────────────────────────────────────────────────────────────────

CORRELATION_GROUPS: dict = {
    "btc_cluster":  {"BTCUSDT", "BTCBUSD", "WBTCUSDT"},
    "eth_cluster":  {"ETHUSDT", "ETHBUSD", "STETHUSDT"},
    "layer1":       {"SOLUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT", "NEARUSDT"},
    "defi":         {"UNIUSDT", "AAVEUSDT", "CRVUSDT", "COMPUSDT"},
    "meme":         {"DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "FLOKIUSDT"},
    "layer2":       {"MATICUSDT", "ARBUSDT", "OPUSDT"},
    "exchange":     {"BNBUSDT", "OKBUSDT"},
}


def get_correlation_group(symbol: str) -> Optional[str]:
    """Return the correlation group name for a symbol, or None."""
    for group, members in CORRELATION_GROUPS.items():
        if symbol in members:
            return group
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Risk Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskConfig:
    # ── Position sizing ────────────────────────────────────────────────────
    max_risk_per_trade_pct:    float = 1.0       # base; overridden by confidence tier
    risk_tier_high_pct:        float = 1.5       # conf ≥ 85
    risk_tier_standard_pct:    float = 1.0       # conf ≥ 75
    kelly_fraction:            float = 0.25
    use_kelly:                 bool  = True

    # ── Portfolio limits ───────────────────────────────────────────────────
    max_open_positions:        int   = 5
    max_portfolio_risk_pct:    float = 5.0
    max_open_per_symbol:       int   = 2         # v3: max 2 per symbol
    max_correlated_exposure:   int   = 2         # max same-direction trades in a corr group

    # ── Drawdown protection ────────────────────────────────────────────────
    max_daily_loss_pct:        float = 3.0
    max_total_drawdown_pct:    float = 10.0      # kill switch
    drawdown_half_size_pct:    float = 5.0       # v3: halve size at 5% DD

    # ── Leverage ───────────────────────────────────────────────────────────
    max_leverage:              float = 3.0
    default_leverage:          float = 1.0

    # ── Consecutive / daily loss control ──────────────────────────────────
    max_consecutive_losses:    int   = 4         # v3: was 3, raise to 4 for pause
    consecutive_loss_pause_hours: float = 2.0
    max_daily_loss_count:      int   = 9         # v3: switch to paper after 9 loss trades/day

    # ── Live trade batch cycle ─────────────────────────────────────────────
    max_trades_per_live_cycle: int   = 5         # v3: after 5 live trades → paper
    low_quality_cycle_limit:   int   = 3         # reduce to 3 if avg quality < 0.65

    # ── Market quality floor ───────────────────────────────────────────────
    market_quality_min:        float = 0.70      # TradeQualityScore.total floor

    # ── Recovery mode ──────────────────────────────────────────────────────
    recovery_risk_pct:         float = 0.50      # risk after loss day
    recovery_win_increment:    float = 0.05      # risk restored per winning trade
    recovery_trades_to_restore: int  = 10        # max recovery trades before full restore

    # ── SL management ──────────────────────────────────────────────────────
    breakeven_at_r:            float = 1.0
    partial_profit_at_r:       float = 1.5
    partial_profit_pct:        float = 50.0
    min_risk_reward:           float = 1.5

    # ── Safety ─────────────────────────────────────────────────────────────
    withdrawals_enabled:       bool  = False     # ALWAYS False
    auto_close_on_dd:          bool  = True


# ─────────────────────────────────────────────────────────────────────────────
# Runtime Risk State
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskState:
    """
    Per-session mutable risk state.  Updated on every trade open/close.
    v3 adds: trading_mode, daily_loss_count, live_trade_count_cycle,
             in_recovery_mode, recovery_risk_pct, symbol_open_counts,
             last_live_switch_time, cycle_quality_scores.
    """
    user_id: int

    # Balance tracking
    session_high_balance:   float = 0.0
    daily_start_balance:    float = 0.0
    daily_date:             str   = ""

    # P&L
    daily_pnl:              float = 0.0
    daily_pnl_pct:          float = 0.0
    session_pnl:            float = 0.0
    current_drawdown_pct:   float = 0.0

    # Loss counters
    consecutive_losses:     int   = 0
    consecutive_wins:       int   = 0
    daily_loss_count:       int   = 0       # v3: # losing trades today
    daily_win_count:        int   = 0

    # Kill switch
    kill_switch_triggered:  bool  = False
    kill_switch_reason:     str   = ""
    kill_switch_at:         Optional[float] = None

    # Consecutive loss pause
    pause_until:            Optional[float] = None

    # v3: Trading mode
    trading_mode:           str   = TradingMode.LIVE.value
    mode_switched_at:       Optional[float] = None
    mode_switch_reason:     str   = ""

    # v3: Live trade batch cycle
    live_trade_count_cycle: int   = 0
    last_live_switch_time:  Optional[float] = None
    cycle_quality_scores:   list  = field(default_factory=list)

    # v3: Recovery mode
    in_recovery_mode:       bool  = False
    current_recovery_risk:  float = 0.0     # active risk % (0 = use base)
    recovery_trades_done:   int   = 0

    # v3: Per-symbol open trade counts {symbol: count}
    symbol_open_counts:     dict  = field(default_factory=dict)

    # v3: Per-correlation-group directional counts {group: {side: count}}
    corr_group_counts:      dict  = field(default_factory=dict)

    def reset_daily_if_needed(self, current_balance: float) -> Optional[str]:
        """
        Called at start of each cycle.  Returns an alert message if reset occurred.
        """
        today = datetime.now(timezone.utc).date().isoformat()
        if self.daily_date == today:
            return None

        msg = None
        was_loss_day = self.daily_loss_count >= 9 or self.daily_pnl_pct < -3.0
        was_paper    = self.trading_mode == TradingMode.PAPER.value

        self.daily_date          = today
        self.daily_start_balance = current_balance
        self.daily_pnl           = 0.0
        self.daily_pnl_pct       = 0.0
        self.daily_loss_count    = 0
        self.daily_win_count     = 0

        # Auto-resume live at day start
        if was_paper and not self.kill_switch_triggered:
            self.trading_mode       = TradingMode.LIVE.value
            self.mode_switch_reason = "Daily reset — resuming live trading"
            self.last_live_switch_time = time.time()
            self.live_trade_count_cycle = 0

            if was_loss_day:
                self.in_recovery_mode      = True
                self.current_recovery_risk = 0.5
                self.recovery_trades_done  = 0
                msg = (
                    f"🌅 <b>Daily Reset — Recovery Mode Active</b>\n"
                    f"User: {self.user_id}\n"
                    f"Risk reduced to 0.5% per trade.\n"
                    f"Will gradually restore to base risk over winning trades."
                )
            else:
                msg = (
                    f"🌅 <b>Daily Reset</b> — Live trading resumed\n"
                    f"User: {self.user_id} | Balance: ${current_balance:.2f}"
                )

        logger.info(f"[user={self.user_id}] Daily reset → {today} | balance=${current_balance:.2f}")
        return msg

    def record_trade_open(self, symbol: str, side: str, corr_group: Optional[str]) -> None:
        """Track per-symbol and per-correlation-group open counts."""
        self.symbol_open_counts[symbol] = self.symbol_open_counts.get(symbol, 0) + 1
        if self.trading_mode == TradingMode.LIVE.value:
            self.live_trade_count_cycle += 1
        if corr_group:
            if corr_group not in self.corr_group_counts:
                self.corr_group_counts[corr_group] = {"long": 0, "short": 0}
            direction_key = "long" if side in ("long", "buy") else "short"
            self.corr_group_counts[corr_group][direction_key] =                 self.corr_group_counts[corr_group].get(direction_key, 0) + 1

    def record_trade_close(self, symbol: str, pnl: float, side: str,
                           account_balance: float, corr_group: Optional[str]) -> None:
        """Update all state on trade close."""
        # Balance / drawdown
        self.daily_pnl   += pnl
        self.session_pnl += pnl
        if self.daily_start_balance > 0:
            self.daily_pnl_pct = self.daily_pnl / self.daily_start_balance * 100
        if account_balance > self.session_high_balance:
            self.session_high_balance = account_balance
        if self.session_high_balance > 0:
            self.current_drawdown_pct = max(
                0.0,
                (self.session_high_balance - account_balance) / self.session_high_balance * 100,
            )
        # Win/loss counters
        if pnl <= 0:
            self.consecutive_losses += 1
            self.consecutive_wins    = 0
            self.daily_loss_count   += 1
        else:
            self.consecutive_wins   += 1
            self.consecutive_losses  = 0
            self.daily_win_count    += 1
            # Recovery progression
            if self.in_recovery_mode:
                self.recovery_trades_done  += 1
                self.current_recovery_risk += 0.05
                if (self.current_recovery_risk >= 1.0 or
                        self.recovery_trades_done >= 10):
                    self.in_recovery_mode     = False
                    self.current_recovery_risk = 0.0
                    logger.info(f"[user={self.user_id}] Recovery mode complete")
        # Symbol count
        self.symbol_open_counts[symbol] = max(
            0, self.symbol_open_counts.get(symbol, 1) - 1)
        # Corr group count
        if corr_group and corr_group in self.corr_group_counts:
            dk = "long" if side in ("long", "buy") else "short"
            self.corr_group_counts[corr_group][dk] = max(
                0, self.corr_group_counts[corr_group].get(dk, 1) - 1)

    # Compatibility alias (called by old bot_engine code)
    def record_trade_result(self, pnl: float, account_balance: float) -> None:
        self.record_trade_close("UNKNOWN", pnl, "long", account_balance, None)

    def add_cycle_quality(self, quality_score: float) -> None:
        self.cycle_quality_scores.append(quality_score)
        if len(self.cycle_quality_scores) > 20:
            self.cycle_quality_scores.pop(0)

    def avg_cycle_quality(self) -> float:
        if not self.cycle_quality_scores:
            return 1.0
        return sum(self.cycle_quality_scores) / len(self.cycle_quality_scores)

    def trigger_consecutive_loss_pause(self, pause_hours: float) -> None:
        self.pause_until = time.time() + pause_hours * 3600
        logger.warning(
            f"[user={self.user_id}] Consecutive loss pause: "
            f"{self.consecutive_losses} → pausing {pause_hours}h"
        )

    def is_paused(self) -> bool:
        if self.pause_until is None:
            return False
        return time.time() < self.pause_until

    def pause_minutes_remaining(self) -> float:
        if not self.pause_until or time.time() >= self.pause_until:
            return 0.0
        return (self.pause_until - time.time()) / 60.0

    def switch_to_paper(self, reason: str) -> None:
        self.trading_mode       = TradingMode.PAPER.value
        self.mode_switched_at   = time.time()
        self.mode_switch_reason = reason
        logger.warning(f"[user={self.user_id}] → PAPER MODE: {reason}")

    def reset_kill_switch(self) -> None:
        self.kill_switch_triggered = False
        self.kill_switch_reason    = ""
        self.kill_switch_at        = None
        logger.info(f"[user={self.user_id}] Kill switch reset")

    def to_dict(self) -> dict:
        return {
            "user_id":                  self.user_id,
            "trading_mode":             self.trading_mode,
            "daily_pnl":                round(self.daily_pnl, 4),
            "daily_pnl_pct":            round(self.daily_pnl_pct, 4),
            "daily_loss_count":         self.daily_loss_count,
            "daily_win_count":          self.daily_win_count,
            "session_pnl":              round(self.session_pnl, 4),
            "current_drawdown_pct":     round(self.current_drawdown_pct, 4),
            "consecutive_losses":       self.consecutive_losses,
            "consecutive_wins":         self.consecutive_wins,
            "live_trade_count_cycle":   self.live_trade_count_cycle,
            "avg_cycle_quality":        round(self.avg_cycle_quality(), 3),
            "in_recovery_mode":         self.in_recovery_mode,
            "current_recovery_risk":    round(self.current_recovery_risk, 3),
            "kill_switch_triggered":    self.kill_switch_triggered,
            "kill_switch_reason":       self.kill_switch_reason,
            "paused":                   self.is_paused(),
            "pause_minutes_remaining":  round(self.pause_minutes_remaining(), 1),
            "mode_switch_reason":       self.mode_switch_reason,
            "symbol_open_counts":       self.symbol_open_counts,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Kill Switch  (same as v2 + paper-mode switch at 3% daily DD)
# ─────────────────────────────────────────────────────────────────────────────

async def check_kill_switch(
    risk_state:      RiskState,
    config:          RiskConfig,
    account_balance: float,
    alert_fn=None,
) -> bool:
    if risk_state.kill_switch_triggered:
        return True

    fired  = False
    reason = ""

    if risk_state.current_drawdown_pct >= config.max_total_drawdown_pct:
        fired  = True
        reason = (f"Total drawdown {risk_state.current_drawdown_pct:.2f}% ≥ "
                  f"kill-switch threshold {config.max_total_drawdown_pct}%")
    elif risk_state.daily_pnl_pct <= -config.max_daily_loss_pct:
        fired  = True
        reason = (f"Daily loss {risk_state.daily_pnl_pct:.2f}% breached "
                  f"-{config.max_daily_loss_pct}%")

    if fired:
        risk_state.kill_switch_triggered = True
        risk_state.kill_switch_reason    = reason
        risk_state.kill_switch_at        = time.time()
        msg = (
            f"🚨 <b>KILL SWITCH</b>\nUser: {risk_state.user_id}\n"
            f"Reason: {reason}\nBalance: ${account_balance:.2f}\n"
            f"DD: {risk_state.current_drawdown_pct:.2f}% | "
            f"Daily: {risk_state.daily_pnl_pct:.2f}%\n"
            f"<b>Admin reset required before trading resumes.</b>"
        )
        logger.critical(f"KILL SWITCH [user={risk_state.user_id}]: {reason}")
        if alert_fn:
            try: await alert_fn(msg)
            except Exception as e: logger.error(f"Kill switch alert failed: {e}")
    return fired


# ─────────────────────────────────────────────────────────────────────────────
# v3: Daily Loss Protection  — switch to paper when thresholds breached
# ─────────────────────────────────────────────────────────────────────────────

async def check_daily_loss_protection(
    risk_state: RiskState,
    config:     RiskConfig,
    alert_fn=None,
) -> Optional[str]:
    """
    Returns a block reason if trading should be switched to paper, else None.
    Also mutates risk_state.trading_mode when switching.

    Triggers (any one sufficient):
      • daily_loss_count ≥ max_daily_loss_count (9)
      • daily_pnl_pct ≤ -max_daily_loss_pct (3%)
    """
    if risk_state.trading_mode == TradingMode.PAPER.value:
        return None   # already in paper — no further action

    reason = None

    if risk_state.daily_loss_count >= config.max_daily_loss_count:
        reason = (f"Daily loss count {risk_state.daily_loss_count} "
                  f"≥ {config.max_daily_loss_count} — switching to paper for rest of day")

    elif risk_state.daily_pnl_pct <= -config.max_daily_loss_pct:
        reason = (f"Daily drawdown {risk_state.daily_pnl_pct:.2f}% "
                  f"≤ -{config.max_daily_loss_pct}% — switching to paper immediately")

    if reason:
        risk_state.switch_to_paper(reason)
        msg = (
            f"📄 <b>PAPER MODE ACTIVATED</b>\n"
            f"User: {risk_state.user_id}\n"
            f"Reason: {reason}\n"
            f"Signals will continue scanning. Live trading resumes tomorrow."
        )
        logger.warning(f"Paper mode [user={risk_state.user_id}]: {reason}")
        if alert_fn:
            try: await alert_fn(msg)
            except Exception as e: logger.error(f"Paper switch alert failed: {e}")

    return reason


# ─────────────────────────────────────────────────────────────────────────────
# v3: Live Trade Batch Cycle Control
# ─────────────────────────────────────────────────────────────────────────────

def check_live_cycle_limit(
    risk_state: RiskState,
    config:     RiskConfig,
) -> tuple:
    """
    Returns (limit_reached: bool, cycle_limit: int, reason: str).

    Adaptive limit:
      avg trade quality < 0.65 → limit = low_quality_cycle_limit (3)
      else                     → limit = max_trades_per_live_cycle (5)
    """
    avg_q = risk_state.avg_cycle_quality()
    limit = (config.low_quality_cycle_limit
             if avg_q < 0.65
             else config.max_trades_per_live_cycle)

    if risk_state.live_trade_count_cycle >= limit:
        reason = (
            f"Live trade cycle limit reached: {risk_state.live_trade_count_cycle}/{limit} "
            f"(avg quality={avg_q:.2f}) — switching to paper"
        )
        return True, limit, reason

    return False, limit, ""


def can_switch_to_live(
    risk_state:           RiskState,
    config:               RiskConfig,
    market_quality_score: float,
) -> tuple:
    """
    Safety gate before re-enabling live trading.
    Returns (allowed: bool, reason: str).

    All three must pass:
      1. daily_pnl_pct > -max_daily_loss_pct (< 3% daily loss)
      2. consecutive_losses < 3
      3. market_quality_score >= market_quality_min (0.70)
    """
    if risk_state.kill_switch_triggered:
        return False, "Kill switch active — admin reset required"

    if risk_state.daily_pnl_pct <= -config.max_daily_loss_pct:
        return False, (f"Daily drawdown {risk_state.daily_pnl_pct:.2f}% "
                       f"≥ limit {config.max_daily_loss_pct}%")

    if risk_state.consecutive_losses >= 3:
        return False, f"{risk_state.consecutive_losses} consecutive losses — wait for win streak"

    if market_quality_score < config.market_quality_min:
        return False, (f"Market quality {market_quality_score:.2f} "
                       f"< {config.market_quality_min} — market not suitable")

    return True, "All safety checks passed"


def switch_to_live(risk_state: RiskState) -> None:
    """Manual user-triggered switch back to live. Resets cycle counter."""
    risk_state.trading_mode          = TradingMode.LIVE.value
    risk_state.live_trade_count_cycle = 0
    risk_state.last_live_switch_time  = time.time()
    risk_state.mode_switch_reason     = "Manual switch to live by user"
    logger.info(f"[user={risk_state.user_id}] Switched to LIVE — cycle counter reset")


# ─────────────────────────────────────────────────────────────────────────────
# v3: Loss Pattern Blocker
# ─────────────────────────────────────────────────────────────────────────────

def check_loss_pattern(
    signal_snapshot:  dict,
    optimizer_patterns: list,   # list[PerformancePattern] from self_optimizer
    min_significance: int = 10,
) -> tuple:
    """
    Compare the current signal's indicator state against known losing patterns.
    Returns (is_losing_pattern: bool, reason: str).

    Checks:
      • RSI bucket matches a losing bucket with WR < 40% + ≥10 samples
      • ADX bucket matches a losing bucket with WR < 40% + ≥10 samples
    """
    if not optimizer_patterns:
        return False, ""

    indic = signal_snapshot.get("indicators", {})
    rsi_v = indic.get("rsi_value", 50.0)
    adx_v = indic.get("adx", 25.0)

    def _rsi_bucket(v):
        if v < 25: return "extreme_oversold"
        if v < 35: return "oversold"
        if v < 45: return "below_neutral"
        if v < 55: return "neutral"
        if v < 65: return "above_neutral"
        if v < 75: return "overbought"
        return "extreme_overbought"

    def _adx_bucket(v):
        if v < 15: return "very_weak"
        if v < 25: return "weak"
        if v < 35: return "moderate"
        if v < 50: return "strong"
        return "very_strong"

    cur_rsi_bucket = _rsi_bucket(rsi_v)
    cur_adx_bucket = _adx_bucket(adx_v)

    for pat in optimizer_patterns:
        total = pat.wins + pat.losses
        if total < min_significance:
            continue
        if pat.win_rate >= 0.40:
            continue

        if pat.factor == "rsi" and pat.bucket == cur_rsi_bucket:
            return True, (
                f"Loss pattern: RSI={rsi_v:.1f} in bucket '{cur_rsi_bucket}' "
                f"has WR={pat.win_rate:.1%} over {total} trades — blocked"
            )
        if pat.factor == "adx" and pat.bucket == cur_adx_bucket:
            return True, (
                f"Loss pattern: ADX={adx_v:.1f} in bucket '{cur_adx_bucket}' "
                f"has WR={pat.win_rate:.1%} over {total} trades — blocked"
            )

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Position Sizing  (v3: confidence tiers + recovery + drawdown reduction)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PositionSize:
    quantity:       float
    notional_value: float
    risk_amount:    float
    risk_pct:       float
    leverage:       float
    rationale:      str
    rejected:       bool = False   # v3: True if confidence too low to trade


@dataclass
class RiskAssessment:
    approved:            bool
    position_size:       Optional[PositionSize]
    rejection_reason:    Optional[str]
    warnings:            list
    portfolio_risk_after: float
    mode_override:       str = ""   # "paper" if forced to paper


def get_confidence_risk_pct(confidence: float, config: RiskConfig) -> Optional[float]:
    """
    v3 confidence-based risk tiers.
    Returns risk_pct or None if confidence too low to trade.
    """
    if confidence >= 85:
        return config.risk_tier_high_pct     # 1.5%
    if confidence >= 75:
        return config.risk_tier_standard_pct  # 1.0%
    return None   # reject


def _apply_drawdown_size_multiplier(
    risk_pct:     float,
    risk_state:   RiskState,
    config:       RiskConfig,
    warnings:     list,
) -> float:
    """Halve size at 5% drawdown. Kill switch handles 10%."""
    dd = risk_state.current_drawdown_pct
    if dd >= config.drawdown_half_size_pct:
        warnings.append(
            f"Drawdown {dd:.2f}% ≥ {config.drawdown_half_size_pct}% "
            f"→ position size halved"
        )
        return risk_pct * 0.50
    return risk_pct


def _apply_atr_volatility_adjustment(
    risk_pct:  float,
    atr_pct:   float,   # ATR as % of price
    warnings:  list,
) -> float:
    """Halve position in extreme volatility (ATR > 3% of price)."""
    if atr_pct > 0.03:
        warnings.append(f"Extreme ATR {atr_pct*100:.2f}% → size halved for volatility")
        return risk_pct * 0.50
    return risk_pct


def kelly_position_size(
    win_rate:      float,
    avg_win:       float,
    avg_loss:      float,
    kelly_fraction: float = 0.25,
) -> float:
    if avg_loss <= 0 or avg_win <= 0:
        return 0.01
    b = avg_win / avg_loss
    p = max(0.0, min(1.0, win_rate))
    kelly = (p * b - (1 - p)) / b
    return round(max(0.005, min(0.05, kelly * kelly_fraction)), 4)


def fixed_fraction_position_size(
    account_balance: float,
    entry_price:     float,
    stop_loss:       float,
    risk_pct:        float = 1.0,
    leverage:        float = 1.0,
) -> PositionSize:
    if entry_price <= 0 or stop_loss <= 0:
        return PositionSize(0, 0, 0, 0, leverage, "Invalid prices", True)
    risk_amount = account_balance * (risk_pct / 100.0)
    sl_distance = abs(entry_price - stop_loss)
    if sl_distance == 0:
        return PositionSize(0, 0, 0, 0, leverage, "SL equals entry", True)
    quantity = (risk_amount / sl_distance) * leverage
    notional = quantity * entry_price / leverage
    return PositionSize(
        quantity=round(quantity, 6),
        notional_value=round(notional, 2),
        risk_amount=round(risk_amount, 2),
        risk_pct=round(risk_pct, 3),
        leverage=leverage,
        rationale=(f"Fixed {risk_pct:.2f}% risk=${risk_amount:.2f} | "
                   f"SL dist=${sl_distance:.4f} | {quantity:.6f} units"),
    )


def calculate_position_size(
    account_balance: float,
    entry_price:     float,
    stop_loss:       float,
    confidence:      float,
    win_rate:        float,
    avg_win_pct:     float,
    avg_loss_pct:    float,
    config:          RiskConfig,
    risk_state:      Optional[RiskState] = None,
    atr_pct:         float = 0.0,
) -> PositionSize:
    """
    v3 sizing pipeline:
      1. Confidence tier: conf<75 → reject, 75-84 → 1%, ≥85 → 1.5%
      2. Recovery mode:  override to recovery_risk_pct
      3. Kelly adjustment (optional cap)
      4. Drawdown reduction: ≥5% DD → halve
      5. ATR volatility reduction: ATR>3% → halve
      6. Leverage: conservative, conf-scaled, capped at max_leverage
    """
    warnings: list = []

    # Step 1: Confidence tier
    base_risk = get_confidence_risk_pct(confidence, config)
    if base_risk is None:
        return PositionSize(0, 0, 0, 0, 1.0,
                            f"Confidence {confidence:.1f} < 75 — trade rejected",
                            rejected=True)

    # Step 2: Recovery mode override
    if risk_state and risk_state.in_recovery_mode and risk_state.current_recovery_risk > 0:
        base_risk = min(base_risk, risk_state.current_recovery_risk)
        warnings.append(f"Recovery mode: risk capped at {base_risk:.2f}%")

    # Step 3: Kelly optional cap
    if config.use_kelly and win_rate > 0:
        kf        = kelly_position_size(win_rate, avg_win_pct, avg_loss_pct, config.kelly_fraction)
        kelly_r   = kf * 100 * (abs(entry_price - stop_loss) / max(entry_price, 1))
        base_risk = min(base_risk, kelly_r) if kelly_r > 0 else base_risk

    # Step 4: Drawdown reduction
    if risk_state:
        base_risk = _apply_drawdown_size_multiplier(base_risk, risk_state, config, warnings)

    # Step 5: ATR volatility
    if atr_pct > 0:
        base_risk = _apply_atr_volatility_adjustment(base_risk, atr_pct, warnings)

    # Step 6: Leverage
    leverage = min(config.max_leverage, max(1.0, 1.0 + (confidence - 75) / 25.0))
    leverage = round(leverage, 1)

    ps = fixed_fraction_position_size(account_balance, entry_price, stop_loss, base_risk, leverage)
    if warnings:
        ps.rationale += " | " + " | ".join(warnings)
    return ps


# ─────────────────────────────────────────────────────────────────────────────
# Trade Risk Gate  (v3: 12 checks)
# ─────────────────────────────────────────────────────────────────────────────

def assess_trade_risk(
    signal_direction:     str,
    symbol:               str,
    entry_price:          float,
    stop_loss:            float,
    position_size:        PositionSize,
    account_balance:      float,
    open_trades:          list,
    daily_pnl_pct:        float,
    total_drawdown_pct:   float,
    config:               RiskConfig,
    risk_state:           Optional[RiskState] = None,
    trade_quality_score:  float = 1.0,
    optimizer_patterns:   Optional[list] = None,
    atr_pct:              float = 0.0,
) -> RiskAssessment:
    """
    v3 risk gate — 12 checks in order.
    """
    warnings: list = []

    def _reject(reason: str, mode: str = "") -> RiskAssessment:
        return RiskAssessment(False, None, reason, [], 0.0, mode)

    # 1. Safety
    if config.withdrawals_enabled:
        logger.critical("SAFETY VIOLATION: withdrawals_enabled=True")
        return _reject("SAFETY: withdrawals must remain disabled")

    # 2. Kill switch
    if risk_state and risk_state.kill_switch_triggered:
        return _reject(f"Kill switch active: {risk_state.kill_switch_reason}")

    # 3. Consecutive loss pause
    if risk_state and risk_state.is_paused():
        mins = risk_state.pause_minutes_remaining()
        return _reject(f"Consecutive loss pause active ({mins:.0f} min remaining)")

    # 4. Trading mode — if paper, signal is allowed through but flagged
    mode_override = ""
    if risk_state and risk_state.trading_mode == TradingMode.PAPER.value:
        mode_override = "paper"
        # Paper mode: still proceed to size/approve, but execution layer forces paper

    # 5. Daily loss limit (hard)
    if daily_pnl_pct <= -config.max_daily_loss_pct:
        return _reject(
            f"Daily loss limit {config.max_daily_loss_pct}% breached "
            f"({daily_pnl_pct:.2f}%)", "paper"
        )

    # 6. Total drawdown (hard)
    if total_drawdown_pct >= config.max_total_drawdown_pct:
        return _reject(
            f"Max drawdown {config.max_total_drawdown_pct}% breached "
            f"({total_drawdown_pct:.2f}%)", "paper"
        )

    # 7. Position count
    live_trades = [t for t in open_trades if t.get("mode") != "paper"]
    if len(live_trades) >= config.max_open_positions:
        return _reject(f"Max open positions ({config.max_open_positions}) reached")

    # 8. v3: Symbol exposure limit (max 2 per symbol)
    sym_open = sum(1 for t in open_trades if t.get("symbol") == symbol)
    if sym_open >= config.max_open_per_symbol:
        return _reject(f"Symbol {symbol}: already {sym_open} open trades "
                       f"(max {config.max_open_per_symbol})")

    # 9. v3: Correlation control — same direction in correlated group
    corr_group = get_correlation_group(symbol)
    if corr_group and risk_state:
        grp_counts = risk_state.corr_group_counts.get(corr_group, {})
        dir_key    = "long" if signal_direction in ("long", "buy") else "short"
        same_dir_n = grp_counts.get(dir_key, 0)
        if same_dir_n >= config.max_correlated_exposure:
            return _reject(
                f"Correlation limit: {same_dir_n} {dir_key} trades in "
                f"'{corr_group}' group (max {config.max_correlated_exposure})"
            )
        elif same_dir_n >= 1:
            # Soft warning — reduce size by 50%
            warnings.append(
                f"Correlated {dir_key} trade in '{corr_group}' "
                f"({same_dir_n} existing) — size halved"
            )
            position_size.quantity      = round(position_size.quantity * 0.50, 6)
            position_size.risk_amount   = round(position_size.risk_amount * 0.50, 2)
            position_size.risk_pct      = round(position_size.risk_pct * 0.50, 4)

    # 10. v3: Market quality floor
    if trade_quality_score < config.market_quality_min:
        return _reject(
            f"Trade quality {trade_quality_score:.2f} < "
            f"minimum {config.market_quality_min}"
        )

    # 11. v3: Loss pattern blocker
    if optimizer_patterns:
        is_losing, pattern_reason = check_loss_pattern({}, optimizer_patterns)
        if is_losing:
            return _reject(f"Loss pattern match: {pattern_reason}")

    # 12. Leverage cap
    if position_size.leverage > config.max_leverage:
        warnings.append(f"Leverage capped: {position_size.leverage}x → {config.max_leverage}x")
        position_size.leverage = config.max_leverage

    if position_size.quantity <= 0 or position_size.rejected:
        return _reject("Position size is zero — confidence too low or risk calc failed")

    sl_dist = abs(entry_price - stop_loss)
    if sl_dist == 0:
        return _reject("SL equals entry price")

    total_risk = sum(t.get("risk_pct", 0) for t in live_trades)
    new_total  = total_risk + position_size.risk_pct
    if new_total > config.max_portfolio_risk_pct:
        return _reject(f"Portfolio risk {new_total:.1f}% > max {config.max_portfolio_risk_pct}%")

    if warnings:
        logger.warning(f"[{symbol}] Risk warnings: {warnings}")

    return RiskAssessment(
        approved=True,
        position_size=position_size,
        rejection_reason=None,
        warnings=warnings,
        portfolio_risk_after=round(new_total, 3),
        mode_override=mode_override,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic SL / TP  (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

def trailing_stop(
    entry_price:   float,
    current_price: float,
    current_sl:    float,
    atr:           float,
    side:          str,
    trail_atr_mult: float = 1.5,
) -> float:
    trail_dist = atr * trail_atr_mult
    if side == "buy":
        return round(max(current_sl, current_price - trail_dist), 6)
    return round(min(current_sl, current_price + trail_dist), 6)


def breakeven_stop(
    entry_price:   float,
    current_price: float,
    current_sl:    float,
    side:          str,
    trigger_r:     float = 1.0,
    initial_sl:    float = None,
) -> tuple:
    if initial_sl is None:
        initial_sl = current_sl
    initial_risk = abs(entry_price - initial_sl)
    if initial_risk == 0:
        return current_sl, False
    profit = (current_price - entry_price) if side == "buy" else (entry_price - current_price)
    if profit < initial_risk * trigger_r:
        return current_sl, False
    buffer = initial_risk * 0.10
    if side == "buy":
        new_sl = round(max(current_sl, entry_price + buffer), 6)
    else:
        new_sl = round(min(current_sl, entry_price - buffer), 6)
    return new_sl, new_sl != current_sl


def get_default_partial_profit_levels(
    entry_price: float,
    take_profit: float,
    side:        str,
) -> list:
    total_move = abs(take_profit - entry_price)
    levels = [(0.50, 50, "TP1 1.5R"), (0.75, 30, "TP2 2.5R"), (1.00, 20, "TP3 full")]
    result = []
    for frac, pct, label in levels:
        price = round(
            entry_price + total_move * frac if side == "buy"
            else entry_price - total_move * frac,
            6,
        )
        result.append({"price": price, "close_pct": pct, "label": label})
    return result


def partial_profit_levels(entry_price, take_profit, side, levels=None):
    if levels is None:
        return get_default_partial_profit_levels(entry_price, take_profit, side)
    total = abs(take_profit - entry_price)
    pcts  = [50, 30, 20]
    return [
        {"price": round(entry_price + total * f if side == "buy"
                        else entry_price - total * f, 6),
         "close_pct": pcts[i] if i < len(pcts) else 0,
         "label": f"TP{i+1}"}
        for i, f in enumerate(levels)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Performance Analytics  (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PerformanceReport:
    total_trades:        int
    closed_trades:       int
    wins:                int
    losses:              int
    win_rate:            float
    avg_win_pct:         float
    avg_loss_pct:        float
    profit_factor:       float
    total_pnl:           float
    total_pnl_pct:       float
    max_drawdown_pct:    float
    current_drawdown_pct: float
    daily_pnl_pct:       float
    avg_risk_reward:     float
    best_trade_pct:      float
    worst_trade_pct:     float
    strategy_breakdown:  dict
    consecutive_losses:  int
    sharpe_approx:       float

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_performance_report(
    trades:          list,
    account_balance: float,
    daily_pnl_pct:   float = 0.0,
    risk_state:      Optional[RiskState] = None,
) -> PerformanceReport:
    closed = [t for t in trades if t.get("status") == "closed"]
    if not closed:
        return PerformanceReport(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, daily_pnl_pct, 0.0, 0.0, 0.0,
                                 {}, 0, 0.0)
    wins   = [t for t in closed if t.get("pnl", 0) > 0]
    losses = [t for t in closed if t.get("pnl", 0) <= 0]
    n      = len(closed)
    wr     = len(wins) / n
    aw     = sum(t.get("pnl_percent", 0) for t in wins)   / max(len(wins),   1)
    al     = abs(sum(t.get("pnl_percent", 0) for t in losses)) / max(len(losses), 1)
    gw     = sum(t.get("pnl", 0) for t in wins)
    gl     = abs(sum(t.get("pnl", 0) for t in losses))
    pf     = round(gw / gl, 4) if gl > 0 else float("inf")
    tp     = sum(t.get("pnl", 0) for t in closed)
    avgrr  = sum(t.get("risk_reward", 0) for t in closed) / n
    pcts   = [t.get("pnl_percent", 0) for t in closed]

    cum, peak, mdd = 0.0, 0.0, 0.0
    for t in sorted(closed, key=lambda x: x.get("closed_at") or ""):
        cum += t.get("pnl", 0)
        if cum > peak: peak = cum
        mdd = max(mdd, peak - cum)
    mdd_pct = -(mdd / account_balance * 100) if account_balance > 0 else 0.0

    strat: dict = {}
    for t in closed:
        m = t.get("strategy_mode", "unknown")
        if m not in strat:
            strat[m] = {"wins": 0, "losses": 0, "count": 0, "pnl": 0.0}
        strat[m]["count"] += 1
        strat[m]["pnl"]   += t.get("pnl", 0)
        if t.get("pnl", 0) > 0: strat[m]["wins"] += 1
        else:                    strat[m]["losses"] += 1
    for m, s in strat.items():
        s["win_rate"] = round(s["wins"] / s["count"], 4) if s["count"] > 0 else 0.0

    sharpe = 0.0
    if len(pcts) >= 2:
        import statistics as _st
        mn, sd = _st.mean(pcts), _st.stdev(pcts)
        sharpe = round(mn / sd * (252 ** 0.5), 4) if sd > 0 else 0.0

    curr_dd = risk_state.current_drawdown_pct if risk_state else abs(mdd_pct)
    consec  = risk_state.consecutive_losses   if risk_state else 0

    return PerformanceReport(
        total_trades=len(trades), closed_trades=n, wins=len(wins), losses=len(losses),
        win_rate=round(wr, 4), avg_win_pct=round(aw, 4), avg_loss_pct=round(al, 4),
        profit_factor=pf, total_pnl=round(tp, 4),
        total_pnl_pct=round(tp / account_balance * 100, 4) if account_balance > 0 else 0.0,
        max_drawdown_pct=round(abs(mdd_pct), 4), current_drawdown_pct=round(curr_dd, 4),
        daily_pnl_pct=round(daily_pnl_pct, 4), avg_risk_reward=round(avgrr, 4),
        best_trade_pct=round(max(pcts), 4), worst_trade_pct=round(min(pcts), 4),
        strategy_breakdown=strat, consecutive_losses=consec, sharpe_approx=sharpe,
    )


def calculate_account_stats(trades: list, account_balance: float) -> dict:
    r = calculate_performance_report(trades, account_balance)
    return {
        "win_rate": r.win_rate, "avg_win_pct": r.avg_win_pct,
        "avg_loss_pct": r.avg_loss_pct, "profit_factor": r.profit_factor,
        "daily_pnl_pct": r.daily_pnl_pct, "total_drawdown_pct": r.max_drawdown_pct,
    }
