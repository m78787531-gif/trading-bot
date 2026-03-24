"""
bot_routes.py  [v3 — Unified Production Build]
─────────────────────────────────────────────────────────────────────────────
FastAPI router for AI bot management.  Mount in main.py:

    from bot_routes import router as bot_router
    app.include_router(bot_router, prefix="/bot", tags=["AI Bot"])

Complete endpoint map
─────────────────────
Bot Control
  POST  /bot/start            Start bot session
  POST  /bot/stop             Stop bot session
  GET   /bot/status           Full status (mode, quality, session info)

Signal Analysis
  POST  /bot/signal           One-shot signal analysis (no trade placed)
  GET   /bot/signal/{symbol}  Quick signal GET alias
  GET   /bot/signals          Signal history

Trading Mode  (v3)
  GET   /bot/mode             Current mode + daily stats + cycle info
  POST  /bot/mode/live        Request switch back to live (safety gate)
  POST  /bot/mode/paper       Manually force paper mode

Risk & Drawdown  (v3)
  GET   /bot/risk-state       Full RiskState snapshot (all 18 variables)
  GET   /bot/drawdown         Drawdown dashboard vs all thresholds
  GET   /bot/recovery         Recovery mode status + progress

Quality & Filters  (v3)
  GET   /bot/quality          Trade quality stats + cycle health
  GET   /bot/session-filter   Active London/NY session window status
  POST  /bot/news-flag        Set/clear high-impact news flag
  GET   /bot/news-flag        Current news flag status

Exposure  (v3)
  GET   /bot/correlations     Open correlation group exposure map

Performance & Config
  GET   /bot/performance      P&L, win rate, R:R, drawdown, strategy
  POST  /bot/optimise         Manually trigger self-optimiser
  PATCH /bot/params           Override AI signal parameters
  GET   /bot/config           Get bot configuration
  PATCH /bot/config           Update bot configuration
  GET   /bot/strategy         Per-strategy (trend/range/breakout) breakdown
  GET   /bot/logs             Bot run logs
  POST  /bot/cycle            Manually trigger one cycle (debug)

Admin
  GET   /bot/admin/sessions   List all active sessions
  POST  /bot/admin/stop/{id}  Force-stop a user's bot
  POST  /bot/admin/reset-kill/{id}  Reset kill switch (admin only)
"""

from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, validator
from sqlalchemy.orm import Session

from database import get_db
from models import User, APIKey, Trade, TradeStatus
from auth import get_approved_user, get_admin_user
from ai.bot_engine import (
    start_bot, stop_bot, get_session, list_active_sessions,
    fetch_ohlcv, BotSession, DEFAULT_SYMBOLS, TIMEFRAMES,
    get_session_performance, reset_kill_switch, request_live_switch,
)
from ai.bot_models import BotConfig, BotStatus, SignalHistory, BotRunLog
from ai.signal_scorer import (
    generate_signal, DEFAULT_PARAMS,
    is_high_liquidity_session, set_news_flag, get_news_flag,
)
from ai.sentiment import get_sentiment
from ai.risk_manager import (
    RiskConfig, TradingMode,
    can_switch_to_live, switch_to_live,
    check_daily_loss_protection, check_live_cycle_limit,
    get_correlation_group, CORRELATION_GROUPS,
)
from ai.self_optimizer import (
    optimise_parameters, get_performance_summary,
    get_params, set_params, get_strategy_performance,
    analyse_patterns,
)
from ai.alerts import alert_risk_warning

logger = logging.getLogger(__name__)
router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class BotStartRequest(BaseModel):
    exchange:                 str        = "binance"
    symbols:                  List[str]  = DEFAULT_SYMBOLS[:4]
    timeframe:                str        = "1h"
    interval_seconds:         int        = 300
    paper_mode:               bool       = True

    # Risk config — v3 defaults
    max_risk_per_trade_pct:   float = 1.0
    max_open_positions:       int   = 5
    max_daily_loss_pct:       float = 3.0
    max_leverage:             float = 3.0
    max_consecutive_losses:   int   = 3
    max_daily_loss_count:     int   = 9
    max_trades_per_live_cycle: int  = 5
    market_quality_min:       float = 0.70

    @validator("timeframe")
    def valid_tf(cls, v):
        if v not in TIMEFRAMES:
            raise ValueError(f"timeframe must be one of {TIMEFRAMES}")
        return v

    @validator("interval_seconds")
    def valid_interval(cls, v):
        if v < 60:
            raise ValueError("interval_seconds must be ≥ 60")
        return v

    @validator("symbols")
    def valid_symbols(cls, v):
        if not v:
            raise ValueError("At least one symbol required")
        if len(v) > 20:
            raise ValueError("Max 20 symbols")
        return [s.upper() for s in v]


class BotConfigUpdate(BaseModel):
    symbols:              Optional[List[str]] = None
    timeframe:            Optional[str]       = None
    interval_seconds:     Optional[int]       = None
    paper_mode:           Optional[bool]      = None
    telegram_enabled:     Optional[bool]      = None
    notify_all_signals:   Optional[bool]      = None
    min_confidence_alert: Optional[float]     = None


class SignalRequest(BaseModel):
    symbol:   str
    timeframe: str  = "1h"
    exchange: str   = "binance"


class ManualParamUpdate(BaseModel):
    params: dict


class NewsFlagRequest(BaseModel):
    active: bool
    reason: Optional[str] = ""


class LiveSwitchRequest(BaseModel):
    market_quality_override: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cfg(user_id: int, db: Session) -> BotConfig:
    """Get or create bot config for a user."""
    cfg = db.query(BotConfig).filter(BotConfig.user_id == user_id).first()
    if not cfg:
        cfg = BotConfig(
            user_id=user_id,
            symbols=DEFAULT_SYMBOLS[:4],
            signal_params=DEFAULT_PARAMS.copy(),
        )
        db.add(cfg); db.commit(); db.refresh(cfg)
    return cfg


def _api_key(user_id: int, exchange: str, db: Session) -> Optional[APIKey]:
    return db.query(APIKey).filter(
        APIKey.user_id == user_id,
        APIKey.exchange == exchange,
        APIKey.is_active == True,
    ).first()


def _session_or_404(user_id: int) -> BotSession:
    s = get_session(user_id)
    if not s:
        raise HTTPException(404, "No active bot session. Call POST /bot/start first.")
    return s


def _next_session_hint(h: int) -> str:
    """Human-readable hint for when the next liquidity window opens."""
    if h < 7:   return f"London open in {7 - h}h (07:00 UTC)"
    if h < 13:  return f"NY open in {13 - h}h (13:00 UTC)"
    if h < 17:  return "NY open now"
    return f"London open in {7 + 24 - h}h (tomorrow 07:00 UTC)"


# ─────────────────────────────────────────────────────────────────────────────
# Bot Control
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/start", summary="Start the AI trading bot")
async def bot_start(
    req: BotStartRequest,
    current_user: User = Depends(get_approved_user),
    db: Session = Depends(get_db),
):
    """
    Start the AI trading bot for the current user.

    - `paper_mode=true`  → signals execute as paper trades regardless of API keys
    - `paper_mode=false` → requires an active exchange API key; uses live orders

    The bot continues scanning and learning even when forced into paper mode by
    risk controls (daily loss, drawdown, cycle limit).
    """
    ak_rec  = _api_key(current_user.id, req.exchange, db)
    if not req.paper_mode and not ak_rec:
        raise HTTPException(
            400,
            f"No active {req.exchange} API key found. "
            "Add one in Settings or use paper_mode=true."
        )

    ak      = ak_rec.api_key    if ak_rec else ""
    sk      = ak_rec.api_secret if ak_rec else ""
    testnet = ak_rec.is_testnet if ak_rec else False

    risk_cfg = RiskConfig(
        max_risk_per_trade_pct   = req.max_risk_per_trade_pct,
        max_open_positions       = req.max_open_positions,
        max_daily_loss_pct       = req.max_daily_loss_pct,
        max_leverage             = req.max_leverage,
        max_consecutive_losses   = req.max_consecutive_losses,
        max_daily_loss_count     = req.max_daily_loss_count,
        max_trades_per_live_cycle= req.max_trades_per_live_cycle,
        market_quality_min       = req.market_quality_min,
        withdrawals_enabled      = False,   # ALWAYS False
    )

    session = start_bot(
        user_id=current_user.id, exchange=req.exchange,
        api_key=ak, api_secret=sk, is_testnet=testnet,
        paper_mode=req.paper_mode, symbols=req.symbols,
        timeframe=req.timeframe, interval_seconds=req.interval_seconds,
        risk_config=risk_cfg,
    )

    # Persist config
    c = _cfg(current_user.id, db)
    c.status          = BotStatus.running
    c.exchange        = req.exchange
    c.paper_mode      = req.paper_mode
    c.symbols         = req.symbols
    c.timeframe       = req.timeframe
    c.interval_seconds= req.interval_seconds
    c.is_testnet      = testnet
    db.commit()

    mode = "paper" if req.paper_mode else "live"
    logger.info(f"Bot started: user={current_user.username} mode={mode}")

    return {
        "message": f"Bot started ({mode} mode)",
        "session": session.to_status_dict(),
        "risk_config": {
            "max_risk_pct":     req.max_risk_per_trade_pct,
            "max_daily_loss":   req.max_daily_loss_pct,
            "kill_switch_at":   10.0,
            "live_cycle_limit": req.max_trades_per_live_cycle,
            "quality_floor":    req.market_quality_min,
        },
    }


@router.post("/stop", summary="Stop the bot")
async def bot_stop(
    current_user: User = Depends(get_approved_user),
    db: Session = Depends(get_db),
):
    stopped = stop_bot(current_user.id)
    c = _cfg(current_user.id, db)
    c.status = BotStatus.stopped
    db.commit()
    return {"message": "Bot stopped", "stopped": stopped}


@router.get("/status", summary="Full bot session status")
async def bot_status(
    current_user: User = Depends(get_approved_user),
    db: Session = Depends(get_db),
):
    s   = get_session(current_user.id)
    c   = _cfg(current_user.id, db)
    ok, sess_name = is_high_liquidity_session()
    return {
        "running":            bool(s),
        "session":            s.to_status_dict() if s else None,
        "current_session":    sess_name,
        "high_liquidity_now": ok,
        "news_flag_active":   get_news_flag(),
        "config": {
            "paper_mode":       c.paper_mode,
            "symbols":          c.symbols,
            "timeframe":        c.timeframe,
            "telegram_enabled": c.telegram_enabled,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Signal Analysis
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/signal", summary="Analyse a symbol (no trade placed)")
async def get_signal(
    req: SignalRequest,
    current_user: User = Depends(get_approved_user),
    db: Session = Depends(get_db),
):
    """
    Generate a full AI signal for any symbol.
    All 9 entry filters are evaluated. No trade is executed.
    """
    sym   = req.symbol.upper()
    ohlcv = await fetch_ohlcv(sym, req.exchange, req.timeframe, 200)
    if not ohlcv or len(ohlcv.get("closes", [])) < 60:
        raise HTTPException(400, f"Insufficient candle data for {sym}")

    sentiment = await get_sentiment(sym)
    params    = get_params()

    signal = generate_signal(
        symbol=sym, timeframe=req.timeframe,
        opens=ohlcv["opens"], highs=ohlcv["highs"],
        lows=ohlcv["lows"],   closes=ohlcv["closes"],
        volumes=ohlcv["volumes"],
        sentiment_score=sentiment.composite_score if sentiment else 0.0,
        params=params,
    )

    # Persist to signal history
    try:
        sh = SignalHistory(
            user_id=current_user.id, symbol=signal.symbol,
            timeframe=signal.timeframe, direction=signal.direction,
            confidence=signal.confidence, entry_price=signal.entry_price,
            stop_loss=signal.stop_loss, take_profit=signal.take_profit,
            risk_reward=signal.risk_reward,
            score_breakdown=signal.score_breakdown.to_dict(),
            indicator_snapshot=signal.indicators.to_dict(),
            smc_data=signal.smc.to_dict(),
            sentiment_score=sentiment.composite_score if sentiment else 0.0,
            sentiment_label=sentiment.label if sentiment else "neutral",
            executed=False, execution_reason="manual_preview",
            params_snapshot=params,
        )
        db.add(sh); db.commit()
    except Exception as e:
        logger.warning(f"Signal history save failed: {e}")

    return {
        "signal": signal.to_dict(),
        "filters": {
            "gate_passed":       signal.gate.passed,
            "gate_reason":       signal.gate.no_trade_reason,
            "quality_score":     signal.trade_quality.total if hasattr(signal, "trade_quality") else None,
            "quality_passed":    signal.trade_quality.passed if hasattr(signal, "trade_quality") else None,
            "session_ok":        signal.session_ok if hasattr(signal, "session_ok") else True,
            "structure_valid":   signal.structure_valid if hasattr(signal, "structure_valid") else True,
            "retest_confirmed":  signal.retest_confirmed if hasattr(signal, "retest_confirmed") else True,
            "blocked":           signal.blocked if hasattr(signal, "blocked") else False,
            "block_reason":      signal.block_reason if hasattr(signal, "block_reason") else None,
        },
        "sentiment": {
            "score":       sentiment.composite_score if sentiment else 0.0,
            "label":       sentiment.label if sentiment else "neutral",
            "confidence":  sentiment.confidence if sentiment else 0.0,
            "headlines":   (sentiment.top_headlines[:3] if sentiment else []),
        },
    }


@router.get("/signal/{symbol}", summary="Quick signal check (GET)")
async def get_quick_signal(
    symbol:   str,
    timeframe: str = "1h",
    exchange: str  = "binance",
    current_user: User = Depends(get_approved_user),
    db: Session = Depends(get_db),
):
    return await get_signal(
        SignalRequest(symbol=symbol, timeframe=timeframe, exchange=exchange),
        current_user=current_user, db=db,
    )


@router.get("/signals", summary="Signal history")
async def signal_history(
    limit:     int            = 50,
    symbol:    Optional[str]  = None,
    direction: Optional[str]  = None,
    current_user: User = Depends(get_approved_user),
    db: Session = Depends(get_db),
):
    q = db.query(SignalHistory).filter(SignalHistory.user_id == current_user.id)
    if symbol:    q = q.filter(SignalHistory.symbol == symbol.upper())
    if direction: q = q.filter(SignalHistory.direction == direction)
    rows = q.order_by(SignalHistory.generated_at.desc()).limit(limit).all()
    return [{
        "id": r.id, "symbol": r.symbol, "timeframe": r.timeframe,
        "direction": r.direction, "confidence": r.confidence,
        "entry_price": r.entry_price, "stop_loss": r.stop_loss,
        "take_profit": r.take_profit, "risk_reward": r.risk_reward,
        "sentiment_score": r.sentiment_score, "sentiment_label": r.sentiment_label,
        "executed": r.executed,
        "generated_at": r.generated_at.isoformat() if r.generated_at else None,
    } for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# v3: Trading Mode Management
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/mode", summary="Current trading mode + daily stats")
async def get_trading_mode(
    current_user: User = Depends(get_approved_user),
):
    """
    Returns the current trading mode (live/paper) with all daily counters,
    cycle info, recovery status, and pause details.

    Key fields:
    - `trading_mode`          — "live" or "paper"
    - `daily_loss_count`      — losses today; switches to paper at max_daily_loss_count (9)
    - `daily_pnl_pct`         — daily drawdown %; switches to paper at max_daily_loss_pct (3%)
    - `live_trade_count_cycle`— live trades in current cycle; pauses at limit (5)
    - `consecutive_losses`    — streak; pause at 3
    - `in_recovery_mode`      — True = reduced risk per trade (0.5%)
    """
    s  = _session_or_404(current_user.id)
    rs = s.risk_state
    avg_q = rs.avg_cycle_quality()
    adaptive_limit = (
        s.risk_config.low_quality_cycle_limit
        if avg_q < 0.65
        else s.risk_config.max_trades_per_live_cycle
    )
    return {
        "trading_mode":             rs.trading_mode,
        "effective_paper":          s.effective_paper(),
        "mode_switch_reason":       rs.mode_switch_reason,
        "daily_loss_count":         rs.daily_loss_count,
        "daily_win_count":          rs.daily_win_count,
        "daily_pnl_pct":            round(rs.daily_pnl_pct, 4),
        "consecutive_losses":       rs.consecutive_losses,
        "consecutive_wins":         rs.consecutive_wins,
        "current_drawdown_pct":     round(rs.current_drawdown_pct, 4),
        "kill_switch_triggered":    rs.kill_switch_triggered,
        "kill_switch_reason":       rs.kill_switch_reason,
        "paused":                   rs.is_paused(),
        "pause_minutes_remaining":  round(rs.pause_minutes_remaining(), 1),
        "in_recovery_mode":         rs.in_recovery_mode,
        "current_recovery_risk_pct": round(rs.current_recovery_risk, 3),
        "live_trade_count_cycle":   rs.live_trade_count_cycle,
        "adaptive_cycle_limit":     adaptive_limit,
        "avg_cycle_quality":        round(avg_q, 3),
        "thresholds": {
            "paper_at_loss_count":  s.risk_config.max_daily_loss_count,
            "paper_at_dd_pct":      s.risk_config.max_daily_loss_pct,
            "pause_after_losses":   s.risk_config.max_consecutive_losses,
            "kill_switch_at_dd":    s.risk_config.max_total_drawdown_pct,
            "half_size_at_dd":      s.risk_config.drawdown_half_size_pct,
        },
    }


@router.post("/mode/live", summary="Request switch to live trading (safety-gated)")
async def request_live_mode(
    req: LiveSwitchRequest,
    current_user: User = Depends(get_approved_user),
):
    """
    Request switch from paper → live trading.

    Three conditions must ALL pass before the switch is allowed:
      1. daily_pnl_pct > -3%       (no significant daily loss)
      2. consecutive_losses < 3    (no active losing streak)
      3. market_quality_score ≥ 0.70  (market suitable for trading)

    If `market_quality_override` is provided, it is used instead of the
    bot's averaged cycle quality score.
    """
    s = _session_or_404(current_user.id)
    quality = (req.market_quality_override
               if req.market_quality_override is not None
               else s.risk_state.avg_cycle_quality())

    allowed, reason = request_live_switch(current_user.id, quality)

    if allowed:
        asyncio.ensure_future(alert_risk_warning(
            f"✅ <b>Live Mode Resumed</b>\n"
            f"User: {current_user.id}\n"
            f"Cycle counter reset. Trade carefully.\n"
            f"Quality score: {quality:.2f}"
        ))
        return {
            "switched":              True,
            "reason":                reason,
            "current_mode":          s.risk_state.trading_mode,
            "live_trade_count_cycle": s.risk_state.live_trade_count_cycle,
        }

    return {
        "switched":     False,
        "reason":       reason,
        "current_mode": s.risk_state.trading_mode,
        "advice": (
            "Resolve the blocking condition before retrying. "
            "The bot continues scanning in paper mode — signals and learning are unaffected."
        ),
        "current_quality": round(quality, 3),
        "thresholds": {
            "daily_dd_must_be_above": f"-{s.risk_config.max_daily_loss_pct}%",
            "consecutive_losses_must_be_below": 3,
            "quality_must_be_above": s.risk_config.market_quality_min,
        },
    }


@router.post("/mode/paper", summary="Manually force paper mode")
async def force_paper_mode(
    reason: str = Body("Manual override by user", embed=True),
    current_user: User = Depends(get_approved_user),
):
    """
    Force the bot into paper trading mode immediately.

    Scanning, signal generation, and self-learning continue unaffected.
    No live orders will be placed until the user calls POST /bot/mode/live
    and the safety gate passes.
    """
    s = _session_or_404(current_user.id)
    s.risk_state.switch_to_paper(f"Manual: {reason}")
    asyncio.ensure_future(alert_risk_warning(
        f"📄 <b>Paper Mode (Manual)</b>\n"
        f"User: {current_user.id}\nReason: {reason}"
    ))
    return {
        "switched_to_paper": True,
        "reason":            reason,
        "message":           (
            "Bot continues scanning. "
            "Call POST /bot/mode/live to resume live trading."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# v3: Risk State & Drawdown Dashboard
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/risk-state", summary="Full RiskState snapshot")
async def get_risk_state(
    current_user: User = Depends(get_approved_user),
):
    """
    Complete RiskState dump — all 18 tracked variables.
    Useful for debugging, monitoring dashboards, and audit logs.
    """
    s = _session_or_404(current_user.id)
    return s.risk_state.to_dict()


@router.get("/drawdown", summary="Drawdown dashboard vs all protection thresholds")
async def get_drawdown_dashboard(
    current_user: User = Depends(get_approved_user),
):
    """
    Shows every drawdown and loss metric alongside its threshold.

    Protection layers in order:
      1. Consecutive losses ≥ 3    → 2-hour pause
      2. Daily drawdown ≥ 3%       → paper mode for the day
      3. Daily loss count ≥ 9      → paper mode for the day
      4. Session drawdown ≥ 5%     → position size halved
      5. Session drawdown ≥ 10%    → kill switch (admin reset required)
    """
    s   = _session_or_404(current_user.id)
    rs  = s.risk_state
    cfg = s.risk_config
    return {
        "session_drawdown": {
            "current_pct":       round(rs.current_drawdown_pct, 4),
            "half_size_at_pct":  cfg.drawdown_half_size_pct,
            "kill_switch_at_pct": cfg.max_total_drawdown_pct,
            "size_halved_now":   rs.current_drawdown_pct >= cfg.drawdown_half_size_pct,
            "kill_switch_active": rs.kill_switch_triggered,
            "kill_switch_reason": rs.kill_switch_reason,
        },
        "daily": {
            "pnl_pct":            round(rs.daily_pnl_pct, 4),
            "loss_count":         rs.daily_loss_count,
            "win_count":          rs.daily_win_count,
            "paper_at_loss_count": cfg.max_daily_loss_count,
            "paper_at_dd_pct":    -cfg.max_daily_loss_pct,
            "paper_triggered":    rs.trading_mode == TradingMode.PAPER.value,
        },
        "consecutive": {
            "losses":             rs.consecutive_losses,
            "wins":               rs.consecutive_wins,
            "pause_after_n":      cfg.max_consecutive_losses,
            "paused":             rs.is_paused(),
            "pause_min_remaining": round(rs.pause_minutes_remaining(), 1),
        },
        "recovery": {
            "active":             rs.in_recovery_mode,
            "current_risk_pct":   round(rs.current_recovery_risk, 3),
            "base_risk_pct":      cfg.risk_tier_standard_pct,
            "trades_completed":   rs.recovery_trades_done,
            "trades_to_restore":  cfg.recovery_trades_to_restore,
        },
        "protection_layers": [
            {"layer": 1, "trigger": f"consecutive_losses ≥ {cfg.max_consecutive_losses}",
             "effect": f"pause {cfg.consecutive_loss_pause_hours}h",
             "active": rs.is_paused()},
            {"layer": 2, "trigger": f"daily_pnl_pct ≤ -{cfg.max_daily_loss_pct}%",
             "effect": "paper mode for the day",
             "active": rs.trading_mode == TradingMode.PAPER.value},
            {"layer": 3, "trigger": f"daily_loss_count ≥ {cfg.max_daily_loss_count}",
             "effect": "paper mode for the day",
             "active": rs.daily_loss_count >= cfg.max_daily_loss_count},
            {"layer": 4, "trigger": f"drawdown ≥ {cfg.drawdown_half_size_pct}%",
             "effect": "50% position size reduction",
             "active": rs.current_drawdown_pct >= cfg.drawdown_half_size_pct},
            {"layer": 5, "trigger": f"drawdown ≥ {cfg.max_total_drawdown_pct}%",
             "effect": "kill switch — admin reset required",
             "active": rs.kill_switch_triggered},
        ],
    }


@router.get("/recovery", summary="Recovery mode status and progress")
async def get_recovery_status(
    current_user: User = Depends(get_approved_user),
):
    """
    After a loss day, recovery mode reduces risk to 0.5% per trade
    and gradually restores to base risk (+0.05% per winning trade).

    Deactivates automatically after 10 winning trades or risk reaches 1.0%.
    """
    s   = _session_or_404(current_user.id)
    rs  = s.risk_state
    cfg = s.risk_config

    if not rs.in_recovery_mode:
        return {
            "active":  False,
            "message": "Not in recovery mode — full risk active.",
            "standard_risk_pct": cfg.risk_tier_standard_pct,
        }

    span    = cfg.risk_tier_standard_pct - cfg.recovery_risk_pct
    current = rs.current_recovery_risk - cfg.recovery_risk_pct
    progress = round(min(100.0, current / span * 100), 1) if span > 0 else 0.0

    return {
        "active":              True,
        "current_risk_pct":    round(rs.current_recovery_risk, 3),
        "start_risk_pct":      cfg.recovery_risk_pct,
        "target_risk_pct":     cfg.risk_tier_standard_pct,
        "progress_pct":        progress,
        "trades_completed":    rs.recovery_trades_done,
        "trades_remaining":    max(0, cfg.recovery_trades_to_restore - rs.recovery_trades_done),
        "message": (
            f"Risk at {rs.current_recovery_risk:.2f}% ({progress:.0f}% restored). "
            f"Win {max(0, cfg.recovery_trades_to_restore - rs.recovery_trades_done)} more "
            f"trades to reach full {cfg.risk_tier_standard_pct}% risk."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# v3: Quality & Filter Status
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/quality", summary="Trade quality stats + cycle health")
async def get_quality_dashboard(
    current_user: User = Depends(get_approved_user),
):
    """
    Shows trade quality scoring health and adaptive cycle limit status.

    trade_quality_score is calculated per signal:
      - RR quality     30%  (RR < 1.5 → 0.0)
      - Structure      35%  (OB/FVG clarity + aligned breaks)
      - Trend strength 20%  (ADX-based)
      - ATR quality    15%  (expansion vs compression)
    Minimum: 0.70

    The live cycle limit is adaptive:
      - avg_quality ≥ 0.65 → 5 trades per cycle
      - avg_quality < 0.65 → 3 trades per cycle (low-quality throttle)
    """
    s   = _session_or_404(current_user.id)
    rs  = s.risk_state
    cfg = s.risk_config
    avg_q = rs.avg_cycle_quality()
    adaptive = (cfg.low_quality_cycle_limit if avg_q < 0.65
                else cfg.max_trades_per_live_cycle)
    ok, sess_name = is_high_liquidity_session()

    return {
        "quality": {
            "avg_cycle_score":  round(avg_q, 4),
            "recent_scores":    [round(q, 3) for q in rs.cycle_quality_scores[-10:]],
            "minimum":          cfg.market_quality_min,
            "above_threshold":  avg_q >= cfg.market_quality_min,
            "components": {
                "rr_quality":        "30% — penalise RR<1.5, reward ≥2.5",
                "structure_quality": "35% — OB/FVG clarity + directional alignment",
                "trend_quality":     "20% — ADX: <20=0, 20-25=0.3, 25-35=0.6, ≥35=1.0",
                "atr_quality":       "15% — 0.4-1.5% of price = 1.0 (sweet spot)",
            },
        },
        "cycle": {
            "live_trades_this_cycle": rs.live_trade_count_cycle,
            "adaptive_limit":         adaptive,
            "at_limit":               rs.live_trade_count_cycle >= adaptive,
            "low_quality_throttle":   avg_q < 0.65,
        },
        "session_filter": {
            "high_liquidity_now": ok,
            "active_session":     sess_name,
            "windows":            {"london": "07:00–10:59 UTC", "ny": "13:00–16:59 UTC"},
            "next_window":        _next_session_hint(datetime.now(timezone.utc).hour),
        },
        "entry_gates": {
            "smc_required":       "BOS/CHoCH + active OB or FVG",
            "ema200_required":    "Price on correct side of 200 EMA",
            "rsi_zone":           "40–60 for trend entries",
            "macd_required":      "Fresh crossover or aligned histogram",
            "volume_required":    "≥ 1.5× 20-period average",
            "retest_required":    "Within 0.5% of OB/FVG boundary",
            "structure_required": "HH+HL for long; LH+LL for short",
        },
    }


@router.get("/session-filter", summary="London/NY session window status")
async def get_session_filter(
    current_user: User = Depends(get_approved_user),
):
    """
    Shows whether the current UTC time is within a high-liquidity trading window.
    Entries are blocked outside these windows (when session_filter param is True).
    """
    now = datetime.now(timezone.utc)
    ok, name = is_high_liquidity_session(now)
    return {
        "current_utc":        now.strftime("%H:%M UTC (%A)"),
        "high_liquidity_now": ok,
        "active_session":     name,
        "windows": {
            "london_open": "07:00–10:59 UTC",
            "ny_open":     "13:00–16:59 UTC",
        },
        "next_window": _next_session_hint(now.hour),
        "note": "Session filter can be disabled per-user via signal params: session_filter=false",
    }


@router.post("/news-flag", summary="Activate/clear high-impact news flag")
async def set_news_flag_endpoint(
    req: NewsFlagRequest,
    current_user: User = Depends(get_approved_user),
):
    """
    Manually set or clear the high-impact news flag.

    When active=True:
      - Gate #0 fires on every signal — all new entries are blocked
      - Open positions continue to be managed (SL, partial TP, trailing)
      - Self-optimizer continues learning from open position closes

    Use for: CPI / FOMC / Fed speeches / flash crash / exchange maintenance.
    Clear immediately after the event resolves.
    """
    set_news_flag(req.active)
    action = "ACTIVATED" if req.active else "CLEARED"
    logger.info(f"News flag {action} by {current_user.username}: {req.reason or '(no reason)'}")

    if req.active:
        asyncio.ensure_future(alert_risk_warning(
            f"📰 <b>News Flag Active</b>\n"
            f"User: {current_user.id}\n"
            f"Reason: {req.reason or 'Manual'}\n"
            f"All new entries paused until cleared."
        ))

    return {
        "news_flag_active": req.active,
        "reason":           req.reason,
        "effect":           ("All new entries blocked — positions still managed"
                             if req.active else "Normal trading resumed"),
    }


@router.get("/news-flag", summary="Current news flag status")
async def get_news_flag_status(
    current_user: User = Depends(get_approved_user),
):
    active = get_news_flag()
    return {
        "news_flag_active": active,
        "effect": "All new entries blocked" if active else "No restriction from news flag",
    }


# ─────────────────────────────────────────────────────────────────────────────
# v3: Correlation Exposure
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/correlations", summary="Open correlation group exposure")
async def get_correlation_exposure(
    current_user: User = Depends(get_approved_user),
):
    """
    Shows directional exposure per correlated asset group.

    If a group already has 1 same-direction trade, the next one gets 50% size.
    If it has 2, the trade is blocked entirely.

    Groups: btc_cluster, eth_cluster, layer1 (SOL/AVAX/ADA),
            defi, meme, layer2 (MATIC/ARB/OP), exchange (BNB/OKB).
    """
    s   = _session_or_404(current_user.id)
    rs  = s.risk_state
    cfg = s.risk_config

    exposure = {}
    for group, members in CORRELATION_GROUPS.items():
        counts   = rs.corr_group_counts.get(group, {"long": 0, "short": 0})
        longs    = counts.get("long",  0)
        shorts   = counts.get("short", 0)
        open_sym = [p.symbol for p in s.open_positions.values()
                    if p.symbol in members]
        exposure[group] = {
            "members":        sorted(members),
            "open_symbols":   open_sym,
            "long_count":     longs,
            "short_count":    shorts,
            "at_limit":       max(longs, shorts) >= cfg.max_correlated_exposure,
            "max_allowed":    cfg.max_correlated_exposure,
            "next_trade_size": (
                "blocked"  if max(longs, shorts) >= cfg.max_correlated_exposure else
                "50% size" if max(longs, shorts) >= 1 else
                "full size"
            ),
        }

    return {
        "correlation_exposure": exposure,
        "symbol_open_counts":   rs.symbol_open_counts,
        "max_per_symbol":       cfg.max_open_per_symbol,
        "note": (
            "1 existing same-direction trade in a group → next is 50% size. "
            "2 existing → blocked."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Performance & Optimisation
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/performance", summary="Full performance report")
async def bot_performance(
    current_user: User = Depends(get_approved_user),
    db: Session = Depends(get_db),
):
    """
    Combined performance report from:
    - Database trade records (all-time)
    - Self-optimizer in-memory history (last 500 trades)
    - Per-strategy breakdown (trend / range / breakout)
    """
    perf = get_performance_summary()
    c    = _cfg(current_user.id, db)

    trades  = db.query(Trade).filter(Trade.user_id == current_user.id).all()
    closed  = [t for t in trades if (t.status == "closed" or
               (hasattr(t.status, "value") and t.status.value == "closed"))]
    wins    = [t for t in closed if (t.pnl or 0) > 0]
    losses  = [t for t in closed if (t.pnl or 0) <= 0]
    open_t  = [t for t in trades if (t.status == "open" or
               (hasattr(t.status, "value") and t.status.value == "open"))]

    # Session-level perf if bot is running
    session = get_session(current_user.id)
    session_perf = None
    if session:
        from ai.risk_manager import calculate_performance_report
        tdicts = [{"pnl": t.pnl or 0.0, "pnl_percent": t.pnl_percent or 0.0,
                   "status": "closed" if t in closed else "open",
                   "symbol": t.symbol,
                   "risk_reward": getattr(t, "risk_reward", 0) or 0.0,
                   "strategy_mode": getattr(t, "strategy_mode", "trend") or "trend",
                   "closed_at": str(getattr(t, "updated_at", "") or "")}
                  for t in trades]
        report = calculate_performance_report(
            tdicts, session.account_balance,
            session.risk_state.daily_pnl_pct,
            session.risk_state,
        )
        session_perf = report.to_dict()

    return {
        "optimizer":   perf,
        "session":     session_perf,
        "db_summary": {
            "total_trades": len(trades),
            "open":         len(open_t),
            "closed":       len(closed),
            "wins":         len(wins),
            "losses":       len(losses),
            "win_rate":     round(len(wins) / max(len(closed), 1), 4),
            "total_pnl":    round(sum(t.pnl or 0 for t in closed), 4),
        },
        "current_params": get_params(),
        "config": {
            "symbols":    c.symbols,
            "timeframe":  c.timeframe,
            "paper_mode": c.paper_mode,
        },
    }


@router.post("/optimise", summary="Manually trigger self-optimiser")
async def trigger_optimisation(
    current_user: User = Depends(get_approved_user),
):
    """
    Manually trigger the self-optimiser.
    Requires ≥ 50 closed trades in memory. Runs with all overfitting safeguards:
    - Learning rate 0.01 (max 1% drift per param per run)
    - Max 5% parameter change per run
    - Convergence lock if no improvement over 3 consecutive runs
    """
    result = optimise_parameters()
    return {
        "locked":          result.locked,
        "changes":         result.changes,
        "trades_analysed": result.trades_analysed,
        "performance":     result.performance_summary,
        "new_params":      result.new_params,
        "timestamp":       result.analysis_timestamp,
    }


@router.patch("/params", summary="Override AI signal parameters")
async def update_params(
    req: ManualParamUpdate,
    current_user: User = Depends(get_approved_user),
    db: Session = Depends(get_db),
):
    """
    Manually override any signal parameter from DEFAULT_PARAMS.
    Unrecognised keys are silently ignored.
    Changes persist in-memory and are saved to the user's BotConfig.
    """
    allowed = set(DEFAULT_PARAMS.keys())
    filtered = {k: v for k, v in req.params.items() if k in allowed}
    if not filtered:
        raise HTTPException(400, f"No valid params. Allowed: {sorted(allowed)}")

    current = get_params()
    current.update(filtered)
    set_params(current)

    c = _cfg(current_user.id, db)
    c.signal_params = current
    db.commit()

    return {
        "message": f"Updated {len(filtered)} parameter(s)",
        "updated": filtered,
        "full_params": current,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Performance
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/strategy", summary="Per-strategy performance breakdown")
async def get_strategy_breakdown(
    current_user: User = Depends(get_approved_user),
    db: Session = Depends(get_db),
):
    """
    Win rate, avg PnL, and confidence by strategy mode
    (trend / range / breakout / unknown).

    Two data sources:
    - optimizer_breakdown: from in-memory self-optimizer (more detailed)
    - db_breakdown: from DB trade records (persistent, survives restarts)
    """
    s        = get_session(current_user.id)
    strategy = get_strategy_performance()

    db_t = db.query(Trade).filter(Trade.user_id == current_user.id).all()
    closed = [t for t in db_t if (t.status == "closed" or
              (hasattr(t.status, "value") and t.status.value == "closed"))]
    db_by_strat = {}
    for t in closed:
        mode = getattr(t, "strategy_mode", None) or "unknown"
        if mode not in db_by_strat:
            db_by_strat[mode] = {"count": 0, "wins": 0, "pnl": 0.0}
        db_by_strat[mode]["count"] += 1
        db_by_strat[mode]["pnl"]   += t.pnl or 0.0
        if (t.pnl or 0) > 0:
            db_by_strat[mode]["wins"] += 1
    for mode, stats in db_by_strat.items():
        c = stats["count"]
        stats["win_rate"] = round(stats["wins"] / c, 4) if c > 0 else 0.0

    return {
        "optimizer_breakdown":  strategy,
        "db_breakdown":         db_by_strat,
        "current_strategy":     s.current_strategy_mode if s else "unknown",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Config Management
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/config", summary="Get bot configuration")
async def get_config(
    current_user: User = Depends(get_approved_user),
    db: Session = Depends(get_db),
):
    c = _cfg(current_user.id, db)
    return {
        "status":             c.status,
        "exchange":           c.exchange,
        "paper_mode":         c.paper_mode,
        "is_testnet":         c.is_testnet,
        "symbols":            c.symbols,
        "timeframe":          c.timeframe,
        "interval_seconds":   c.interval_seconds,
        "telegram_enabled":   c.telegram_enabled,
        "notify_all_signals": c.notify_all_signals,
        "min_confidence_alert": c.min_confidence_alert,
        "signal_params":      c.signal_params or DEFAULT_PARAMS,
        "total_cycles":       c.total_cycles,
        "total_signals":      c.total_signals,
        "total_trades":       c.total_trades,
    }


@router.patch("/config", summary="Update bot configuration")
async def update_config(
    req: BotConfigUpdate,
    current_user: User = Depends(get_approved_user),
    db: Session = Depends(get_db),
):
    c = _cfg(current_user.id, db)
    if req.symbols is not None:
        c.symbols = [s.upper() for s in req.symbols]
    if req.timeframe is not None:
        if req.timeframe not in TIMEFRAMES:
            raise HTTPException(400, f"timeframe must be one of {TIMEFRAMES}")
        c.timeframe = req.timeframe
    if req.interval_seconds is not None:
        c.interval_seconds = max(60, req.interval_seconds)
    if req.paper_mode is not None:
        c.paper_mode = req.paper_mode
    if req.telegram_enabled is not None:
        c.telegram_enabled = req.telegram_enabled
    if req.notify_all_signals is not None:
        c.notify_all_signals = req.notify_all_signals
    if req.min_confidence_alert is not None:
        c.min_confidence_alert = req.min_confidence_alert
    db.commit()
    return {"message": "Config updated"}


# ─────────────────────────────────────────────────────────────────────────────
# Logs & Debug
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/logs", summary="Bot run logs")
async def get_logs(
    limit: int = 20,
    current_user: User = Depends(get_approved_user),
    db: Session = Depends(get_db),
):
    logs = (db.query(BotRunLog)
              .filter(BotRunLog.user_id == current_user.id)
              .order_by(BotRunLog.ran_at.desc())
              .limit(limit).all())
    return [{
        "id":          l.id,
        "cycle":       l.cycle_number,
        "signals":     l.signals_found,
        "trades":      l.trades_opened,
        "actions":     l.position_actions,
        "errors":      l.errors,
        "duration_ms": l.duration_ms,
        "ran_at":      l.ran_at.isoformat() if l.ran_at else None,
    } for l in logs]


@router.post("/cycle", summary="Manually trigger one bot cycle (debug)")
async def manual_cycle(
    current_user: User = Depends(get_approved_user),
    db: Session = Depends(get_db),
):
    """
    Trigger one complete scan-and-execute cycle immediately.
    Useful for testing signal quality and risk gate behaviour.
    Creates a temporary session from saved config if no active session exists.
    """
    s = get_session(current_user.id)
    if not s:
        c   = _cfg(current_user.id, db)
        ak  = _api_key(current_user.id, c.exchange, db)
        s   = BotSession(
            user_id=current_user.id, exchange=c.exchange,
            api_key=ak.api_key if ak else "",
            api_secret=ak.api_secret if ak else "",
            is_testnet=ak.is_testnet if ak else False,
            paper_mode=c.paper_mode,
            symbols=c.symbols or DEFAULT_SYMBOLS[:2],
            timeframe=c.timeframe,
        )

    from ai.bot_engine import run_cycle
    result = await run_cycle(s, db)

    try:
        log = BotRunLog(
            user_id=current_user.id,
            cycle_number=result.get("cycle", 0),
            signals_found=len(result.get("signals_found", [])),
            trades_opened=len(result.get("trades_opened", [])),
            position_actions=len(result.get("trades_closed", [])),
            errors=result.get("errors", []),
            duration_ms=result.get("duration_ms", 0),
            summary=result,
        )
        db.add(log); db.commit()
    except Exception as e:
        logger.warning(f"BotRunLog save failed: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Admin Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/admin/sessions", summary="Admin: list all active bot sessions")
async def admin_sessions(admin: User = Depends(get_admin_user)):
    return list_active_sessions()


@router.post("/admin/stop/{user_id}", summary="Admin: force-stop a user's bot")
async def admin_stop_bot(
    user_id: int,
    admin:   User    = Depends(get_admin_user),
    db:      Session = Depends(get_db),
):
    stopped = stop_bot(user_id)
    cfg = db.query(BotConfig).filter(BotConfig.user_id == user_id).first()
    if cfg:
        cfg.status = BotStatus.stopped
        db.commit()
    return {"stopped": stopped, "user_id": user_id}


@router.post("/admin/reset-kill/{user_id}", summary="Admin: reset kill switch")
async def admin_reset_kill_switch(
    user_id: int,
    admin:   User    = Depends(get_admin_user),
):
    """
    Reset the kill switch for a user after manual risk review.
    The bot does NOT automatically restart — the user must call POST /bot/start.
    """
    ok = reset_kill_switch(user_id)
    if not ok:
        raise HTTPException(404, f"No active session for user {user_id}")

    asyncio.ensure_future(alert_risk_warning(
        f"🔑 <b>Kill Switch Reset</b>\n"
        f"Admin: {admin.username} reset kill switch for user {user_id}.\n"
        f"User must restart the bot manually."
    ))
    return {
        "reset": True,
        "user_id": user_id,
        "message": "Kill switch cleared. User must restart bot with POST /bot/start.",
    }
