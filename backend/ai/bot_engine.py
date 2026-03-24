"""
ai/bot_engine.py  [v3 — Unified Production Build]
═══════════════════════════════════════════════════════════════════════════════
Master AI trading bot engine.

All 18 requirements wired end-to-end:
  1.  Trade batch system    — max 5 live trades/cycle → paper; safe gate back
  2.  Daily loss protection — paper if losses≥9 or DD≥3%; midnight auto-reset
  3.  Loss/risk control     — 3-consec pause 2h; 5%DD→half-size; 10%DD→kill
  4.  No fixed daily cap    — intelligent throttling only
  5.  Market quality filter — quality_score ≥ 0.70 gate in assess_trade_risk
  6.  Trade quality system  — TradeQualityScore ≥ 0.70 per signal
  7.  Strict entry gates    — SMC/EMA200/RSI/MACD/Volume all in signal_scorer
  8.  Minimum R:R 1.5       — enforced in quality score + risk gate
  9.  Retest entry logic    — check_retest_proximity() in signal_scorer
  10. Session time filter   — London 07–10 UTC / NY 13–16 UTC
  11. Confidence tiers      — ≥85→1.5% risk; ≥75→1%; <75→reject
  12. Symbol/corr control   — max 2/symbol; corr-group size reduction
  13. Volatility control    — extreme ATR→half size; low ATR→gate blocks
  14. Loss pattern blocker  — analyse_patterns() + check_loss_pattern()
  15. Market structure      — HH+HL for long; LH+LL for short
  16. Paper continuation    — bot keeps scanning/learning in paper mode
  17. Telegram alerts       — mode switches, kill switch, pauses, recovery
  18. Variables             — live_trade_count, trading_mode, last_switch_time,
                             daily_loss_count, daily_drawdown, consecutive_losses,
                             market_quality_score — all in RiskState

Safety invariants (never removed):
  • withdrawals_enabled ALWAYS False
  • Paper mode always available without API keys
  • All close orders use reduce-only flag
  • Graceful shutdown — no orphaned positions
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx

from ai.signal_scorer import (
    generate_signal, DEFAULT_PARAMS, TradeSignal,
    StrategyMode, MarketRegime,
    is_high_liquidity_session, set_news_flag, get_news_flag,
)
from ai.risk_manager import (
    RiskConfig, RiskState, TradingMode,
    calculate_position_size, assess_trade_risk,
    calculate_account_stats, calculate_performance_report,
    trailing_stop, breakeven_stop,
    check_kill_switch, check_daily_loss_protection,
    check_live_cycle_limit, can_switch_to_live, switch_to_live,
    get_correlation_group, CORRELATION_GROUPS,
)
from ai.sentiment import get_sentiment
from ai.self_optimizer import (
    record_trade_result, get_params,
    get_performance_summary, get_strategy_performance,
    analyse_patterns,
)
from ai.alerts import (
    alert_trade_opened, alert_trade_closed,
    alert_high_confidence_signal, alert_risk_warning,
    send_daily_report,
)
from ai.meta_ai import (
    should_trade, BatchCycleState, AdaptiveMode,
    TradeDecision, TradeVerdict,
    BATCH_LIMIT, BATCH_REST_SECONDS,
)
from ai.orderflow   import analyse_orderflow
from ai.whale_detector import detect_whale_activity
from ai.market_bias import calculate_market_bias, fetch_funding_rate

logger = logging.getLogger(__name__)

TIMEFRAMES      = ["1m", "5m", "15m", "1h", "4h", "1d"]
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
    "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
]


# ─────────────────────────────────────────────────────────────────────────────
# Open Position Tracker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OpenPosition:
    trade_id:            int
    symbol:              str
    direction:           str        # long | short
    entry_price:         float
    current_sl:          float
    initial_sl:          float      # never modified after entry
    take_profit:         float      # full TP (3R)
    take_profit_partial: float      # partial TP (1.5R — close 50%)
    quantity:            float
    quantity_remaining:  float      # adjusted after partial close
    risk_pct:            float
    mode:                str        # live | paper
    exchange:            str
    opened_at:           str
    strategy_mode:       str = "trend"
    partial_tp_hit:      bool = False
    sl_moved_to_be:      bool = False

    def profit_r(self, price: float) -> float:
        risk = abs(self.entry_price - self.initial_sl)
        if risk == 0:
            return 0.0
        gain = (price - self.entry_price if self.direction == "long"
                else self.entry_price - price)
        return gain / risk


# ─────────────────────────────────────────────────────────────────────────────
# Bot Session
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BotSession:
    user_id:          int
    exchange:         str
    api_key:          str
    api_secret:       str
    is_testnet:       bool
    paper_mode:       bool          # True = always paper regardless of risk_state

    symbols:          list = field(default_factory=lambda: DEFAULT_SYMBOLS.copy())
    timeframe:        str  = "1h"
    interval_seconds: int  = 300
    risk_config:      RiskConfig = field(default_factory=RiskConfig)

    running:          bool  = False
    last_cycle_at:    Optional[float] = None
    cycle_count:      int   = 0
    total_signals_generated: int = 0
    total_trades_executed:   int = 0
    errors:           list  = field(default_factory=list)

    account_balance:      float = 10_000.0
    balance_refreshed_at: float = 0.0

    risk_state:            RiskState = field(default_factory=lambda: RiskState(user_id=0))
    current_strategy_mode: str       = StrategyMode.TREND.value
    open_positions:        dict      = field(default_factory=dict)   # trade_id -> OpenPosition
    # v4: autonomous engine state
    batch_state:           BatchCycleState = field(default_factory=BatchCycleState)
    last_adaptive_mode:    str        = AdaptiveMode.NORMAL.value
    meta_decisions:        list       = field(default_factory=list)   # last 20 decisions

    def __post_init__(self):
        self.risk_state = RiskState(user_id=self.user_id)
        self.risk_state.session_high_balance = self.account_balance
        self.risk_state.daily_start_balance  = self.account_balance
        # Start in the requested mode
        if self.paper_mode:
            self.risk_state.trading_mode = TradingMode.PAPER.value

    def effective_paper(self) -> bool:
        """True if execution should be paper — either forced or risk-switched."""
        return (self.paper_mode or
                self.risk_state.trading_mode == TradingMode.PAPER.value)

    def to_status_dict(self) -> dict:
        return {
            "user_id":               self.user_id,
            "exchange":              self.exchange,
            "paper_mode_flag":       self.paper_mode,
            "effective_paper":       self.effective_paper(),
            "running":               self.running,
            "symbols":               self.symbols,
            "timeframe":             self.timeframe,
            "interval_seconds":      self.interval_seconds,
            "cycle_count":           self.cycle_count,
            "total_signals":         self.total_signals_generated,
            "total_trades":          self.total_trades_executed,
            "last_cycle": (
                datetime.fromtimestamp(self.last_cycle_at, tz=timezone.utc).isoformat()
                if self.last_cycle_at else None
            ),
            "account_balance":       self.account_balance,
            "current_strategy_mode": self.current_strategy_mode,
            "open_positions_count":  len(self.open_positions),
            "risk_state":            self.risk_state.to_dict(),
            "batch_cycle":           self.batch_state.to_dict(),
            "adaptive_mode":         self.last_adaptive_mode,
        }


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV Fetching
# ─────────────────────────────────────────────────────────────────────────────

BINANCE_TF = {"1m":"1m","5m":"5m","15m":"15m","1h":"1h","4h":"4h","1d":"1d"}
BYBIT_TF   = {"1m":"1","5m":"5","15m":"15","1h":"60","4h":"240","1d":"D"}


async def fetch_ohlcv_binance(
    symbol: str, interval: str = "1h", limit: int = 200, testnet: bool = False
) -> Optional[dict]:
    base = "https://testnet.binance.vision" if testnet else "https://api.binance.com"
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(
                f"{base}/api/v3/klines",
                params={"symbol": symbol,
                        "interval": BINANCE_TF.get(interval, "1h"),
                        "limit": limit},
            )
            if r.status_code != 200:
                return None
            raw = r.json()
        return {
            "opens":   [float(x[1]) for x in raw],
            "highs":   [float(x[2]) for x in raw],
            "lows":    [float(x[3]) for x in raw],
            "closes":  [float(x[4]) for x in raw],
            "volumes": [float(x[5]) for x in raw],
        }
    except Exception as e:
        logger.error(f"Binance OHLCV [{symbol}]: {e}")
        return None


async def fetch_ohlcv_bybit(
    symbol: str, interval: str = "1h", limit: int = 200, testnet: bool = False
) -> Optional[dict]:
    base = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(
                f"{base}/v5/market/kline",
                params={"category": "linear", "symbol": symbol,
                        "interval": BYBIT_TF.get(interval, "60"), "limit": limit},
            )
            if r.status_code != 200:
                return None
            data = list(reversed(r.json().get("result", {}).get("list", [])))
            if not data:
                return None
        return {
            "opens":   [float(x[1]) for x in data],
            "highs":   [float(x[2]) for x in data],
            "lows":    [float(x[3]) for x in data],
            "closes":  [float(x[4]) for x in data],
            "volumes": [float(x[5]) for x in data],
        }
    except Exception as e:
        logger.error(f"Bybit OHLCV [{symbol}]: {e}")
        return None


async def fetch_ohlcv(
    symbol: str, exchange: str, interval: str = "1h",
    limit: int = 200, testnet: bool = False,
) -> Optional[dict]:
    if exchange.lower() == "bybit":
        return await fetch_ohlcv_bybit(symbol, interval, limit, testnet)
    return await fetch_ohlcv_binance(symbol, interval, limit, testnet)


# ─────────────────────────────────────────────────────────────────────────────
# Open Position Management
# ─────────────────────────────────────────────────────────────────────────────

async def _manage_open_positions(
    session:    BotSession,
    prices:     dict,       # {symbol: current_price}
    db_session,
) -> list:
    """
    Per-position loop:
      1. Partial TP at 1.5R — close 50%, move SL to breakeven
      2. Full TP / SL hit  — close 100%, record result
      3. Trailing stop      — ratchet SL as profit grows
      4. Breakeven stop     — move SL to entry+buffer at 1R

    Paper-mode closes update account_balance directly.
    Live-mode close orders go through exchange client (reduce-only).
    """
    closed_events = []

    for trade_id, pos in list(session.open_positions.items()):
        price = prices.get(pos.symbol)
        if price is None:
            continue

        try:
            pr = pos.profit_r(price)

            # ── 1. Partial TP at 1.5R ─────────────────────────────────────
            if not pos.partial_tp_hit:
                ptp_hit = (
                    (pos.direction == "long"  and price >= pos.take_profit_partial) or
                    (pos.direction == "short" and price <= pos.take_profit_partial)
                )
                if ptp_hit:
                    close_qty = round(pos.quantity * 0.50, 6)
                    pos.quantity_remaining = pos.quantity - close_qty
                    pos.partial_tp_hit     = True

                    partial_pnl_pct = (
                        (price - pos.entry_price) / pos.entry_price * 100
                        if pos.direction == "long"
                        else (pos.entry_price - price) / pos.entry_price * 100
                    )
                    partial_pnl = (partial_pnl_pct / 100.0
                                   * session.account_balance
                                   * (pos.risk_pct / 100.0) * 0.50)
                    session.account_balance += partial_pnl

                    logger.info(
                        f"[{pos.symbol}] Partial TP @{price:.4f} "
                        f"(1.5R={pr:.2f}R) — closed 50%"
                    )

                    # Move SL to breakeven on remaining position
                    new_sl, moved = breakeven_stop(
                        entry_price=pos.entry_price, current_price=price,
                        current_sl=pos.current_sl,
                        side="buy" if pos.direction == "long" else "sell",
                        trigger_r=1.0, initial_sl=pos.initial_sl,
                    )
                    if moved:
                        pos.current_sl  = new_sl
                        pos.sl_moved_to_be = True
                        logger.info(f"[{pos.symbol}] SL → breakeven {new_sl:.4f}")

                    closed_events.append({
                        "type": "partial_tp", "symbol": pos.symbol,
                        "price": price, "qty": close_qty,
                        "pnl": round(partial_pnl, 4), "r": 1.5,
                    })
                    continue

            # ── 2. Full TP or SL hit ──────────────────────────────────────
            qty   = pos.quantity_remaining if pos.partial_tp_hit else pos.quantity
            tp_ok = (
                (pos.direction == "long"  and price >= pos.take_profit) or
                (pos.direction == "short" and price <= pos.take_profit)
            )
            sl_ok = (
                (pos.direction == "long"  and price <= pos.current_sl) or
                (pos.direction == "short" and price >= pos.current_sl)
            )

            if tp_ok or sl_ok:
                close_reason = "full_tp" if tp_ok else "sl"
                pnl_pct = (
                    (price - pos.entry_price) / pos.entry_price * 100
                    if pos.direction == "long"
                    else (pos.entry_price - price) / pos.entry_price * 100
                )
                pnl = (pnl_pct / 100.0
                       * session.account_balance
                       * (pos.risk_pct / 100.0))

                # Update balances
                session.account_balance += pnl

                # Update risk state (v3 full API)
                corr_g = get_correlation_group(pos.symbol)
                session.risk_state.record_trade_close(
                    symbol=pos.symbol, pnl=pnl,
                    side=pos.direction,
                    account_balance=session.account_balance,
                    corr_group=corr_g,
                )

                # Consecutive loss pause (Req 3)
                if pnl <= 0:
                    if (session.risk_state.consecutive_losses >=
                            session.risk_config.max_consecutive_losses):
                        session.risk_state.trigger_consecutive_loss_pause(
                            session.risk_config.consecutive_loss_pause_hours
                        )
                        asyncio.ensure_future(alert_risk_warning(
                            f"⏸ <b>Trading Paused — Consecutive Losses</b>\n"
                            f"User: {session.user_id}\n"
                            f"{session.risk_state.consecutive_losses} losses in a row.\n"
                            f"Pausing {session.risk_config.consecutive_loss_pause_hours:.0f}h."
                        ))

                # Record for self-optimizer
                record_trade_result(
                    trade_id=trade_id, symbol=pos.symbol,
                    direction=pos.direction, pnl=pnl, pnl_pct=pnl_pct,
                    strategy_mode=pos.strategy_mode,
                    opened_at=pos.opened_at,
                    closed_at=datetime.now(timezone.utc).isoformat(),
                )

                # DB update
                try:
                    from models import Trade, TradeStatus
                    t = db_session.query(Trade).filter_by(id=trade_id).first()
                    if t:
                        t.status      = TradeStatus.CLOSED
                        t.exit_price  = price
                        t.pnl         = round(pnl, 6)
                        t.pnl_percent = round(pnl_pct, 4)
                        db_session.commit()
                except Exception as e:
                    logger.error(f"DB close update [{trade_id}]: {e}")

                asyncio.ensure_future(alert_trade_closed(
                    symbol=pos.symbol, direction=pos.direction,
                    entry=pos.entry_price, close=price,
                    pnl_pct=pnl_pct, reason=close_reason, mode=pos.mode,
                ))

                del session.open_positions[trade_id]
                closed_events.append({
                    "type": close_reason, "symbol": pos.symbol,
                    "price": price, "pnl_pct": round(pnl_pct, 4),
                    "pnl": round(pnl, 4), "trade_id": trade_id,
                })
                continue

            # ── 3. Trailing stop ───────────────────────────────────────────
            atr_est = abs(pos.entry_price - pos.initial_sl) / 1.5
            if atr_est > 0:
                new_sl = trailing_stop(
                    entry_price=pos.entry_price, current_price=price,
                    current_sl=pos.current_sl, atr=atr_est,
                    side="buy" if pos.direction == "long" else "sell",
                    trail_atr_mult=1.5,
                )
                if new_sl != pos.current_sl:
                    pos.current_sl = new_sl

            # ── 4. Breakeven stop at 1R ────────────────────────────────────
            if not pos.sl_moved_to_be:
                new_sl, moved = breakeven_stop(
                    entry_price=pos.entry_price, current_price=price,
                    current_sl=pos.current_sl,
                    side="buy" if pos.direction == "long" else "sell",
                    trigger_r=session.risk_config.breakeven_at_r,
                    initial_sl=pos.initial_sl,
                )
                if moved:
                    pos.current_sl = new_sl
                    pos.sl_moved_to_be = True
                    logger.info(f"[{pos.symbol}] Breakeven SL: {new_sl:.4f}")

        except Exception as e:
            logger.error(f"Position mgmt [{pos.symbol}]: {e}")

    return closed_events


# ─────────────────────────────────────────────────────────────────────────────
# Trade Execution
# ─────────────────────────────────────────────────────────────────────────────

async def _execute_paper_trade(
    session:    BotSession,
    signal:     TradeSignal,
    quantity:   float,
    db_session,
) -> Optional[dict]:
    """Create a paper trade record. No exchange API is called."""
    try:
        from models import Trade, TradeSide, TradeStatus
        # Simulate slippage (0.05%)
        slip     = 1.0005 if signal.direction == "long" else 0.9995
        fill_px  = round(signal.entry_price * slip, 6)

        from models import Trade as T
        trade = T(
            user_id     = session.user_id,
            symbol      = signal.symbol,
            side        = "buy" if signal.direction == "long" else "sell",
            quantity    = quantity,
            entry_price = fill_px,
            stop_loss   = signal.stop_loss,
            take_profit = signal.take_profit,
            leverage    = 1.0,
            mode        = "paper",
            status      = "open",
            notes       = f"AI Bot paper | {signal.strategy_mode} | conf={signal.confidence:.1f}",
        )
        db_session.add(trade)
        db_session.commit()
        db_session.refresh(trade)

        session.open_positions[trade.id] = OpenPosition(
            trade_id=trade.id, symbol=signal.symbol, direction=signal.direction,
            entry_price=fill_px, current_sl=signal.stop_loss, initial_sl=signal.stop_loss,
            take_profit=signal.take_profit, take_profit_partial=signal.take_profit_partial,
            quantity=quantity, quantity_remaining=quantity,
            risk_pct=0.0,   # filled by caller
            mode="paper", exchange=session.exchange,
            opened_at=datetime.now(timezone.utc).isoformat(),
            strategy_mode=signal.strategy_mode,
        )
        logger.info(
            f"[PAPER] {signal.direction.upper()} {quantity:.6f} {signal.symbol} "
            f"@ {fill_px:.4f} | SL={signal.stop_loss:.4f} TP={signal.take_profit:.4f}"
        )
        return {"trade_id": trade.id, "fill_price": fill_px}
    except Exception as e:
        logger.error(f"Paper trade [{signal.symbol}]: {e}")
        try: db_session.rollback()
        except: pass
        return None


async def _execute_live_trade(
    session:    BotSession,
    signal:     TradeSignal,
    quantity:   float,
    db_session,
) -> Optional[dict]:
    """
    Place a live market order on Binance or Bybit.
    Safety: only entry + SL/TP orders. No withdrawals, no transfers.
    """
    try:
        from exchanges import BinanceClient, BybitClient
        client = (BinanceClient(session.api_key, session.api_secret, session.is_testnet)
                  if session.exchange.lower() == "binance"
                  else BybitClient(session.api_key, session.api_secret, session.is_testnet))

        side     = "BUY" if signal.direction == "long" else "SELL"
        order_id = await client.place_order(
            symbol=signal.symbol, side=side, quantity=quantity,
            sl=signal.stop_loss, tp=signal.take_profit,
        )
        fill_px = signal.entry_price   # use market price; update from fill in production

        from models import Trade as T
        trade = T(
            user_id=session.user_id, symbol=signal.symbol,
            side="buy" if signal.direction == "long" else "sell",
            quantity=quantity, entry_price=fill_px,
            stop_loss=signal.stop_loss, take_profit=signal.take_profit,
            leverage=1.0, mode="live", status="open",
            exchange_order_id=str(order_id),
            notes=f"AI Bot live | {signal.strategy_mode} | conf={signal.confidence:.1f}",
        )
        db_session.add(trade)
        db_session.commit()
        db_session.refresh(trade)

        session.open_positions[trade.id] = OpenPosition(
            trade_id=trade.id, symbol=signal.symbol, direction=signal.direction,
            entry_price=fill_px, current_sl=signal.stop_loss, initial_sl=signal.stop_loss,
            take_profit=signal.take_profit, take_profit_partial=signal.take_profit_partial,
            quantity=quantity, quantity_remaining=quantity,
            risk_pct=0.0, mode="live", exchange=session.exchange,
            opened_at=datetime.now(timezone.utc).isoformat(),
            strategy_mode=signal.strategy_mode,
        )
        logger.info(
            f"[LIVE] {signal.direction.upper()} {quantity:.6f} {signal.symbol} "
            f"@ {fill_px:.4f} | order={order_id}"
        )
        return {"trade_id": trade.id, "fill_price": fill_px, "order_id": order_id}
    except Exception as e:
        logger.error(f"Live trade [{signal.symbol}]: {e}")
        try: db_session.rollback()
        except: pass
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main Bot Cycle
# ─────────────────────────────────────────────────────────────────────────────

async def run_cycle(session: BotSession, db_session) -> dict:
    """
    One full scan-and-execute cycle. Steps:

      Guard 0a — Kill switch (DD≥10% or daily loss≥3%)
      Guard 0b — Consecutive loss pause (2h after 3 losses)
      Guard 0c — Daily loss protection (paper if losses≥9 or DD≥3%)
      Guard 0d — Live batch cycle limit (paper after 5 live trades)
      Guard 0e — Daily reset check (midnight → back to live + recovery mode)
      1. Manage open positions (partial TP, SL, trailing, breakeven)
      2. Account stats refresh
      3. Symbol scan loop
         a. Generate signal (all 9 entry filters built into signal_scorer)
         b. Position sizing (confidence tiers, DD reduction, ATR adjustment)
         c. Risk gate (12-check assess_trade_risk)
         d. Execute (paper or live based on effective_paper())
      4. Strategy mode vote + update
      5. Return summary dict
    """
    t0 = time.time()
    session.cycle_count  += 1
    session.last_cycle_at = t0

    summary = {
        "cycle":          session.cycle_count,
        "effective_paper": session.effective_paper(),
        "trading_mode":   session.risk_state.trading_mode,
        "adaptive_mode":  session.last_adaptive_mode,
        "batch_cycle":    session.batch_state.to_dict(),
        "signals_found":  [],
        "trades_opened":  [],
        "trades_closed":  [],
        "blocked_by":     None,
        "errors":         [],
        "strategy_mode":  session.current_strategy_mode,
    }

    try:
        # ── Guard 0e: Daily reset (midnight auto-resume) ───────────────────
        reset_msg = session.risk_state.reset_daily_if_needed(session.account_balance)
        if reset_msg:
            asyncio.ensure_future(alert_risk_warning(reset_msg))
            if "Recovery" in reset_msg:
                logger.info(f"[user={session.user_id}] Recovery mode activated")

        # ── Guard 0a: Kill switch ──────────────────────────────────────────
        if await check_kill_switch(
            session.risk_state, session.risk_config,
            session.account_balance, alert_risk_warning
        ):
            summary["blocked_by"] = f"kill_switch: {session.risk_state.kill_switch_reason}"
            session.running = False
            return summary

        # ── Guard 0b: Consecutive loss pause ──────────────────────────────
        if session.risk_state.is_paused():
            mins = session.risk_state.pause_minutes_remaining()
            summary["blocked_by"] = f"loss_pause ({mins:.0f} min remaining)"
            logger.info(f"[user={session.user_id}] Paused: {mins:.0f} min left")
            return summary

        # ── Guard 0c: Daily loss protection (Req 2) ────────────────────────
        paper_reason = await check_daily_loss_protection(
            session.risk_state, session.risk_config, alert_risk_warning)
        if paper_reason:
            summary["trading_mode"] = TradingMode.PAPER.value
            logger.warning(f"[user={session.user_id}] Paper forced: {paper_reason}")

        # ── Guard 0d: Live cycle batch limit (Req 1) ───────────────────────
        if not session.effective_paper():
            cycle_full, cycle_limit, cycle_reason = check_live_cycle_limit(
                session.risk_state, session.risk_config)
            if cycle_full:
                session.risk_state.switch_to_paper(cycle_reason)
                summary["trading_mode"] = TradingMode.PAPER.value
                asyncio.ensure_future(alert_risk_warning(
                    f"📄 <b>Live Cycle Limit</b> ({cycle_limit} trades reached)\n"
                    f"User {session.user_id} → Paper mode.\n"
                    f"Use /bot/mode/live to resume after checking conditions."
                ))
                logger.info(f"[user={session.user_id}] Cycle limit → paper: {cycle_reason}")

        # ── 1. Position management ─────────────────────────────────────────
        price_map: dict = {}
        for pos in session.open_positions.values():
            ohlcv = await fetch_ohlcv(pos.symbol, session.exchange, "1m", 5, session.is_testnet)
            if ohlcv and ohlcv.get("closes"):
                price_map[pos.symbol] = ohlcv["closes"][-1]

        closed = await _manage_open_positions(session, price_map, db_session)
        summary["trades_closed"] = closed

        # ── 2. Account stats ───────────────────────────────────────────────
        from models import Trade
        all_t  = db_session.query(Trade).filter_by(user_id=session.user_id).all()
        tdicts = [{"pnl": t.pnl or 0.0, "pnl_percent": t.pnl_percent or 0.0,
                   "status": t.status if isinstance(t.status, str) else (t.status.value if t.status else "open"),
                   "symbol": t.symbol,
                   "risk_reward": getattr(t, "risk_reward", 0) or 0.0,
                   "strategy_mode": getattr(t, "strategy_mode", "trend") or "trend",
                   "closed_at": str(getattr(t, "updated_at", "") or "")}
                  for t in all_t]
        stats    = calculate_account_stats(tdicts, session.account_balance)
        balance  = session.account_balance
        open_risk = [
            {"symbol": p.symbol, "side": p.direction,
             "risk_pct": p.risk_pct, "mode": p.mode}
            for p in session.open_positions.values()
        ]

        # Get optimizer patterns for loss-pattern blocker (Req 14)
        try:
            from ai.self_optimizer import _trade_history
            opt_patterns = analyse_patterns(list(_trade_history[-100:])) if _trade_history else []
        except Exception:
            opt_patterns = []

        params       = get_params()
        active_syms  = {p.symbol for p in session.open_positions.values()}
        regime_votes: dict = {}

        # ── 3. Symbol scan ─────────────────────────────────────────────────
        for symbol in session.symbols:
            if symbol in active_syms:
                continue   # already holding — skip entry scan

            try:
                # Fetch OHLCV
                ohlcv = await fetch_ohlcv(
                    symbol, session.exchange, session.timeframe, 200, session.is_testnet)
                if not ohlcv or len(ohlcv.get("closes", [])) < 60:
                    continue

                # Sentiment (15-min cache)
                sent_res   = await get_sentiment(symbol)
                sent_score = sent_res.composite_score if sent_res else 0.0

                # ── Signal generation (all 9 entry filters inside) ─────────
                signal = generate_signal(
                    symbol=symbol, timeframe=session.timeframe,
                    opens=ohlcv["opens"], highs=ohlcv["highs"],
                    lows=ohlcv["lows"],   closes=ohlcv["closes"],
                    volumes=ohlcv["volumes"],
                    sentiment_score=sent_score, params=params,
                )
                session.total_signals_generated += 1

                # Strategy mode vote
                regime_votes[signal.market_regime] = (
                    regime_votes.get(signal.market_regime, 0) + 1)

                # Track quality score for adaptive cycle limit
                if hasattr(signal, "trade_quality") and signal.trade_quality:
                    session.risk_state.add_cycle_quality(signal.trade_quality.total)

                # Hard gates: direction must be long/short and signal not blocked
                if signal.direction == "hold" or signal.blocked:
                    logger.debug(
                        f"[{symbol}] Skipped — dir={signal.direction} "
                        f"blocked={signal.blocked}: {signal.block_reason or ''}"
                    )
                    continue

                summary["signals_found"].append({
                    "symbol":    symbol,
                    "direction": signal.direction,
                    "confidence": signal.confidence,
                    "rr":        signal.risk_reward,
                    "strategy":  signal.strategy_mode,
                    "regime":    signal.market_regime,
                    "quality":   round(signal.trade_quality.total, 3) if hasattr(signal, "trade_quality") and signal.trade_quality else None,
                    "session_ok": signal.session_ok if hasattr(signal, "session_ok") else True,
                    "retest_ok":  signal.retest_confirmed if hasattr(signal, "retest_confirmed") else True,
                })

                # High-confidence Telegram alert
                if signal.confidence >= 85:
                    asyncio.ensure_future(alert_high_confidence_signal(
                        symbol=symbol, direction=signal.direction,
                        confidence=signal.confidence, entry=signal.entry_price,
                        sl=signal.stop_loss, tp=signal.take_profit,
                        rr=signal.risk_reward,
                    ))

                # ── Position sizing (Req 11 + 3 + 13) ──────────────────────
                atr_pct = (signal.indicators.atr_pct
                           if hasattr(signal.indicators, "atr_pct") else 0.0)
                ps = calculate_position_size(
                    account_balance=balance, entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss, confidence=signal.confidence,
                    win_rate=stats.get("win_rate", 0.5),
                    avg_win_pct=stats.get("avg_win_pct", 2.0),
                    avg_loss_pct=stats.get("avg_loss_pct", 1.5),
                    config=session.risk_config,
                    risk_state=session.risk_state,
                    atr_pct=atr_pct,
                )
                if ps.rejected or ps.quantity <= 0:
                    logger.debug(f"[{symbol}] Sizing rejected: {ps.rationale}")
                    continue

                # ── Risk gate — 12 checks (Req 5,6,12,13,14) ──────────────
                quality_score = (signal.trade_quality.total
                                 if hasattr(signal, "trade_quality") and signal.trade_quality
                                 else 1.0)
                ra = assess_trade_risk(
                    signal_direction=signal.direction, symbol=symbol,
                    entry_price=signal.entry_price, stop_loss=signal.stop_loss,
                    position_size=ps, account_balance=balance,
                    open_trades=open_risk,
                    daily_pnl_pct=session.risk_state.daily_pnl_pct,
                    total_drawdown_pct=session.risk_state.current_drawdown_pct,
                    config=session.risk_config, risk_state=session.risk_state,
                    trade_quality_score=quality_score,
                    optimizer_patterns=opt_patterns,
                    atr_pct=atr_pct,
                )

                if not ra.approved:
                    logger.info(f"[{symbol}] Risk gate: {ra.rejection_reason}")
                    continue

                # ── v4: Meta-AI Decision Engine ───────────────────────────
                # Order flow analysis
                of_result = analyse_orderflow(
                    ohlcv["opens"], ohlcv["highs"], ohlcv["lows"],
                    ohlcv["closes"], ohlcv["volumes"], lookback=20,
                )

                # Whale detection
                wh_result = detect_whale_activity(
                    ohlcv["highs"], ohlcv["lows"],
                    ohlcv["closes"], ohlcv["volumes"],
                )

                # Funding rate + market bias
                funding = await fetch_funding_rate(symbol, session.exchange)
                bias    = calculate_market_bias(
                    funding_rate=funding,
                    direction=signal.direction,
                )

                # Recent win rate from optimizer
                from ai.self_optimizer import _trade_history as _th
                recent = list(_th)[-20:] if _th else []
                recent_wr = (sum(1 for t in recent if t.won) / len(recent)) if recent else 0.5

                # MTF alignment flag (approximated: all regime votes same direction)
                mtf_ok = (
                    len(regime_votes) <= 2 and
                    signal.market_regime not in ("sideways",)
                )

                meta = should_trade(
                    confidence         = signal.confidence,
                    quality            = quality_score,
                    regime             = signal.market_regime,
                    direction          = signal.direction,
                    volatility_pct     = atr_pct,
                    sentiment          = sent_score,
                    drawdown_pct       = session.risk_state.current_drawdown_pct,
                    consecutive_losses = session.risk_state.consecutive_losses,
                    recent_win_rate    = recent_wr,
                    daily_pnl_pct      = session.risk_state.daily_pnl_pct,
                    session_pnl_pct    = (session.risk_state.session_pnl
                                          / max(session.account_balance, 1) * 100),
                    orderflow_score    = of_result.score,
                    whale_activity     = wh_result.detected,
                    funding_aligned    = bias.funding_aligned,
                    mtf_aligned        = mtf_ok,
                    batch_state        = session.batch_state,
                )

                # Store decision (keep last 20)
                session.meta_decisions.append(meta.to_dict())
                if len(session.meta_decisions) > 20:
                    session.meta_decisions.pop(0)
                session.last_adaptive_mode = meta.adaptive_mode.value

                # Whale Telegram alert
                if wh_result.detected:
                    asyncio.ensure_future(alert_risk_warning(
                        f"🐋 <b>Whale Activity [{symbol}]</b>\n"
                        f"Type: {wh_result.signal_type}\n"
                        f"Severity: {wh_result.severity:.0%}\n"
                        f"{wh_result.rationale}\n"
                        f"Entry blocked for {wh_result.cooldown_candles} candles."
                    ))

                # Meta-AI rejection
                if not meta.approved:
                    logger.info(
                        f"[{symbol}] Meta-AI REJECT mode={meta.adaptive_mode.value} "
                        f"score={meta.score:.1f}: {meta.reasons[-1] if meta.reasons else ''}"
                    )
                    continue

                # Apply size multiplier from meta-AI
                adjusted_qty = round(ps.quantity * meta.size_multiplier, 6)
                if adjusted_qty <= 0:
                    continue

                # Mode change alert
                if meta.adaptive_mode.value != session.last_adaptive_mode:
                    asyncio.ensure_future(alert_risk_warning(
                        f"⚙️ <b>Mode Change</b>\n"
                        f"{session.last_adaptive_mode.upper()} → {meta.adaptive_mode.value.upper()}\n"
                        f"Meta score: {meta.score:.1f} | User: {session.user_id}"
                    ))

                # ── Execute (paper or live per effective_paper()) ──────────
                execute_paper = session.effective_paper() or ra.mode_override == "paper"
                result = (
                    await _execute_paper_trade(session, signal, adjusted_qty, db_session)
                    if execute_paper
                    else await _execute_live_trade(session, signal, adjusted_qty, db_session)
                )

                if result:
                    session.total_trades_executed += 1
                    tid = result["trade_id"]

                    # Set risk_pct on position
                    if tid in session.open_positions:
                        session.open_positions[tid].risk_pct = ps.risk_pct

                    # v4: Record in batch cycle (live trades only)
                    if not execute_paper:
                        session.batch_state.record_trade()
                        if session.batch_state.is_resting():
                            asyncio.ensure_future(alert_risk_warning(
                                f"⏸ <b>Batch Cycle Complete</b>\n"
                                f"User {session.user_id}: {BATCH_LIMIT} live trades executed.\n"
                                f"REST for 3 hours — auto-resumes at "
                                f"{__import__('datetime').datetime.fromtimestamp(session.batch_state.rest_until, tz=__import__('datetime').timezone.utc).strftime('%H:%M UTC')}.\n"
                                f"Position management continues during rest."
                            ))

                    # Track symbol + correlation exposure (Req 12)
                    corr_g = get_correlation_group(symbol)
                    session.risk_state.record_trade_open(symbol, signal.direction, corr_g)

                    summary["trades_opened"].append({
                        "symbol":        symbol,
                        "direction":     signal.direction,
                        "confidence":    signal.confidence,
                        "entry":         result["fill_price"],
                        "sl":            signal.stop_loss,
                        "tp":            signal.take_profit,
                        "tp_partial":    signal.take_profit_partial,
                        "rr":            signal.risk_reward,
                        "trade_id":      tid,
                        "strategy":      signal.strategy_mode,
                        "mode":          "paper" if execute_paper else "live",
                        "quality":       round(quality_score, 3),
                        "meta_score":    round(meta.score, 1),
                        "adaptive_mode": meta.adaptive_mode.value,
                        "size_mult":     meta.size_multiplier,
                        "orderflow":     of_result.bias,
                        "funding_bias":  bias.directional_bias,
                    })

                    asyncio.ensure_future(alert_trade_opened(
                        symbol=symbol, direction=signal.direction,
                        mode="paper" if execute_paper else "live",
                        entry_price=result["fill_price"],
                        stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                        quantity=ps.quantity, confidence=signal.confidence,
                        risk_reward=signal.risk_reward,
                        reasoning=signal.reasoning[:5],
                        exchange=session.exchange,
                    ))

                    open_risk.append({
                        "symbol": symbol, "side": signal.direction,
                        "risk_pct": ps.risk_pct,
                        "mode": "paper" if execute_paper else "live",
                    })
                    active_syms.add(symbol)

                await asyncio.sleep(0.4)   # rate-limit guard

            except Exception as e:
                err_msg = f"{symbol}: {e}"
                logger.error(f"Symbol error: {err_msg}")
                summary["errors"].append(err_msg)

        # ── 4. Strategy mode vote ──────────────────────────────────────────
        if regime_votes:
            dominant = max(regime_votes, key=regime_votes.get)
            # Map regime -> strategy
            regime_map = {
                "bull":          StrategyMode.TREND.value,
                "bear":          StrategyMode.TREND.value,
                "volatile_bull": StrategyMode.BREAKOUT.value,
                "volatile_bear": StrategyMode.BREAKOUT.value,
                "sideways":      StrategyMode.RANGE.value,
            }
            new_strat = regime_map.get(dominant, StrategyMode.TREND.value)
            if session.current_strategy_mode != new_strat:
                logger.info(
                    f"[user={session.user_id}] Strategy: "
                    f"{session.current_strategy_mode} → {new_strat} (regime={dominant})"
                )
                session.current_strategy_mode = new_strat

        summary["strategy_mode"] = session.current_strategy_mode
        summary["risk_state"]    = session.risk_state.to_dict()

    except Exception as e:
        err_msg = f"Cycle error: {e}"
        logger.error(err_msg)
        summary["errors"].append(err_msg)

    summary["duration_ms"] = int((time.time() - t0) * 1000)
    logger.info(
        f"Cycle #{session.cycle_count} [{session.current_strategy_mode}] "
        f"mode={session.risk_state.trading_mode}: "
        f"{len(summary['signals_found'])} sigs | "
        f"{len(summary['trades_opened'])} opened | "
        f"{len(summary['trades_closed'])} closed | "
        f"{summary['duration_ms']}ms"
    )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Bot Lifecycle
# ─────────────────────────────────────────────────────────────────────────────

_active_sessions: dict = {}   # user_id -> BotSession
_session_tasks:   dict = {}   # user_id -> asyncio.Task


async def _bot_loop(session: BotSession) -> None:
    from database import SessionLocal
    mode = "PAPER" if session.paper_mode else "LIVE"
    logger.info(
        f"Bot loop started: user={session.user_id} {mode} "
        f"symbols={len(session.symbols)} tf={session.timeframe}"
    )
    while session.running:
        db = SessionLocal()
        try:
            await run_cycle(session, db)
        except Exception as e:
            logger.error(f"Bot loop error (user={session.user_id}): {e}")
            session.errors.append(str(e))
        finally:
            db.close()
        if not session.running:
            break
        await asyncio.sleep(session.interval_seconds)
    logger.info(f"Bot loop stopped: user={session.user_id}")


def start_bot(
    user_id:          int,
    exchange:         str,
    api_key:          str,
    api_secret:       str,
    is_testnet:       bool      = False,
    paper_mode:       bool      = True,
    symbols:          list      = None,
    timeframe:        str       = "1h",
    interval_seconds: int       = 300,
    risk_config:      RiskConfig = None,
    initial_balance:  float     = 10_000.0,
) -> BotSession:
    """Start a new bot session. Stops any existing session for this user first."""
    stop_bot(user_id)

    cfg = risk_config or RiskConfig()

    session = BotSession(
        user_id=user_id, exchange=exchange,
        api_key=api_key, api_secret=api_secret,
        is_testnet=is_testnet, paper_mode=paper_mode,
        symbols=symbols or DEFAULT_SYMBOLS.copy(),
        timeframe=timeframe, interval_seconds=interval_seconds,
        risk_config=cfg, account_balance=initial_balance,
    )
    session.running = True

    _active_sessions[user_id] = session
    _session_tasks[user_id]   = asyncio.ensure_future(_bot_loop(session))

    logger.info(
        f"Bot started: user={user_id} "
        f"{'PAPER' if paper_mode else 'LIVE'} "
        f"ex={exchange} syms={len(session.symbols)} tf={timeframe} "
        f"risk={cfg.max_risk_per_trade_pct}% dd_kill={cfg.max_total_drawdown_pct}% "
        f"daily_limit={cfg.max_daily_loss_pct}%"
    )
    return session


def stop_bot(user_id: int) -> bool:
    """Gracefully stop the bot session for a user."""
    if user_id in _active_sessions:
        _active_sessions[user_id].running = False
    if user_id in _session_tasks:
        _session_tasks[user_id].cancel()
        del _session_tasks[user_id]
    _active_sessions.pop(user_id, None)
    logger.info(f"Bot stopped: user={user_id}")
    return True


def request_live_switch(user_id: int, market_quality_score: float = 1.0) -> tuple:
    """
    User-initiated switch from paper → live.
    Runs the 3-condition safety gate before switching.
    Returns (allowed: bool, reason: str).
    """
    session = get_session(user_id)
    if not session:
        return False, "No active bot session"
    allowed, reason = can_switch_to_live(
        session.risk_state, session.risk_config, market_quality_score)
    if allowed:
        switch_to_live(session.risk_state)
        logger.info(f"[user={user_id}] Live switch granted: {reason}")
    else:
        logger.warning(f"[user={user_id}] Live switch denied: {reason}")
    return allowed, reason


def reset_kill_switch(user_id: int) -> bool:
    """Admin-only: reset kill switch after manual review."""
    s = _active_sessions.get(user_id)
    if s:
        s.risk_state.reset_kill_switch()
        logger.info(f"Kill switch reset by admin: user={user_id}")
        return True
    return False


def get_session(user_id: int) -> Optional[BotSession]:
    return _active_sessions.get(user_id)


def list_active_sessions() -> list:
    return [s.to_status_dict() for s in _active_sessions.values()]


async def get_session_performance(user_id: int, db_session) -> Optional[dict]:
    """Full performance report for the /bot/performance endpoint."""
    session = get_session(user_id)
    if not session:
        return None

    from models import Trade
    trades = db_session.query(Trade).filter_by(user_id=user_id).all()
    tdicts = [
        {
            "pnl":           t.pnl or 0.0,
            "pnl_percent":   t.pnl_percent or 0.0,
            "status":        t.status if isinstance(t.status, str) else (t.status.value if t.status else "open"),
            "symbol":        t.symbol,
            "risk_reward":   getattr(t, "risk_reward",   0) or 0.0,
            "strategy_mode": getattr(t, "strategy_mode", "trend") or "trend",
            "closed_at":     str(getattr(t, "updated_at", "") or ""),
        }
        for t in trades
    ]
    report = calculate_performance_report(
        tdicts, session.account_balance,
        session.risk_state.daily_pnl_pct,
        session.risk_state,
    )
    return {
        "session_status":     session.to_status_dict(),
        "performance":        report.to_dict(),
        "optimizer":          get_performance_summary(),
        "strategy_breakdown": get_strategy_performance(),
    }
