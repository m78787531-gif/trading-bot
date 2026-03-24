"""
ai/scheduler.py
─────────────────────────────────────────────────────────────────────────────
Background task scheduler for:
  • Daily PnL reports (via Telegram)
  • Weekly performance digests
  • Automatic parameter optimisation sweeps
  • Sentiment cache warming

Start it in main.py lifespan:
    from ai.scheduler import start_scheduler, stop_scheduler
    asyncio.ensure_future(start_scheduler(app))
"""

from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone, time as dt_time
from typing import Optional

logger = logging.getLogger(__name__)

_scheduler_task: Optional[asyncio.Task] = None
_running = False

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DAILY_REPORT_HOUR_UTC  = 23     # Send daily report at 23:00 UTC
WEEKLY_DIGEST_WEEKDAY  = 6      # Sunday (0=Mon, 6=Sun)
OPTIMISE_INTERVAL_H    = 6      # Run optimisation sweep every 6 hours
CACHE_WARM_INTERVAL_H  = 1      # Warm sentiment cache every hour


# ─────────────────────────────────────────────────────────────────────────────
# Helper: seconds until next target hour (UTC)
# ─────────────────────────────────────────────────────────────────────────────

def _seconds_until_hour(target_hour: int) -> float:
    now = datetime.now(timezone.utc)
    target = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
    if target <= now:
        target = target.replace(day=target.day + 1)
    return (target - now).total_seconds()


# ─────────────────────────────────────────────────────────────────────────────
# Daily Report
# ─────────────────────────────────────────────────────────────────────────────

async def _send_daily_reports():
    """Compile and send daily report for all users who have Telegram configured."""
    try:
        from database import SessionLocal
        from models import User, Trade, UserStatus
        from ai.alerts import send_daily_report
        from ai.self_optimizer import optimise_parameters, get_performance_summary
        import os

        # Only send if Telegram is configured
        if not (os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID")):
            logger.debug("Telegram not configured — skipping daily report")
            return

        db = SessionLocal()
        try:
            users = db.query(User).filter(User.status == UserStatus.approved).all()
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            for user in users:
                try:
                    # Run optimisation for this user
                    opt_result = optimise_parameters()
                    param_changes = opt_result.changes

                    # Gather today's trades
                    from datetime import timedelta
                    today_start = datetime.now(timezone.utc).replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                    today_trades = db.query(Trade).filter(
                        Trade.user_id == user.id,
                        Trade.closed_at >= today_start,
                    ).all()

                    if not today_trades:
                        continue   # No activity today — skip

                    wins = [t for t in today_trades if t.pnl > 0]
                    losses = [t for t in today_trades if t.pnl <= 0]
                    total_pnl = sum(t.pnl for t in today_trades)

                    best = max(today_trades, key=lambda t: t.pnl, default=None)
                    worst = min(today_trades, key=lambda t: t.pnl, default=None)

                    open_trades = db.query(Trade).filter(
                        Trade.user_id == user.id,
                        Trade.status == "open",
                    ).count()

                    # Account balance approximation
                    balance = 10000.0   # fallback

                    await send_daily_report(
                        date=today_str,
                        total_trades=len(today_trades),
                        wins=len(wins),
                        losses=len(losses),
                        total_pnl=total_pnl,
                        total_pnl_pct=total_pnl / balance * 100,
                        best_trade={"symbol": best.symbol, "pnl_pct": best.pnl_percent} if best else {},
                        worst_trade={"symbol": worst.symbol, "pnl_pct": worst.pnl_percent} if worst else {},
                        open_trades=open_trades,
                        account_balance=balance,
                        param_changes=param_changes,
                        top_signals=[],
                    )
                    logger.info(f"Daily report sent for user {user.username}")

                except Exception as e:
                    logger.error(f"Daily report error for user {user.id}: {e}")

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Daily report scheduler error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Weekly Digest
# ─────────────────────────────────────────────────────────────────────────────

async def _send_weekly_digests():
    """Send weekly performance digest if today is the configured weekday."""
    today = datetime.now(timezone.utc).weekday()
    if today != WEEKLY_DIGEST_WEEKDAY:
        return

    try:
        from database import SessionLocal
        from models import User, Trade, UserStatus
        from ai.alerts import send_weekly_digest
        from ai.self_optimizer import optimise_parameters

        db = SessionLocal()
        try:
            from datetime import timedelta
            week_start = datetime.now(timezone.utc) - timedelta(days=7)
            week_str = week_start.strftime("%b %d") + " – " + datetime.now(timezone.utc).strftime("%b %d %Y")

            users = db.query(User).filter(User.status == UserStatus.approved).all()
            for user in users:
                week_trades = db.query(Trade).filter(
                    Trade.user_id == user.id,
                    Trade.closed_at >= week_start,
                ).all()

                if not week_trades:
                    continue

                wins = [t for t in week_trades if t.pnl > 0]
                total_pnl = sum(t.pnl for t in week_trades)
                gross_profit = sum(t.pnl for t in wins)
                gross_loss = abs(sum(t.pnl for t in week_trades if t.pnl <= 0))

                # Per-symbol performance
                sym_pnl: dict[str, float] = {}
                sym_cnt: dict[str, int] = {}
                for t in week_trades:
                    sym_pnl[t.symbol] = sym_pnl.get(t.symbol, 0) + t.pnl_percent
                    sym_cnt[t.symbol] = sym_cnt.get(t.symbol, 0) + 1

                sym_avg = {s: sym_pnl[s] / sym_cnt[s] for s in sym_pnl}
                sorted_syms = sorted(sym_avg.items(), key=lambda x: x[1], reverse=True)

                opt = optimise_parameters()

                await send_weekly_digest(
                    week_str=week_str,
                    metrics={
                        "total_pnl": total_pnl,
                        "total_trades": len(week_trades),
                        "win_rate": len(wins) / max(len(week_trades), 1),
                        "profit_factor": gross_profit / max(gross_loss, 0.01),
                    },
                    top_symbols=sorted_syms[:3],
                    worst_symbols=list(reversed(sorted_syms))[:3],
                    optimisation_summary={"changes": opt.changes},
                )
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Weekly digest error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment Cache Warmer
# ─────────────────────────────────────────────────────────────────────────────

async def _warm_sentiment_cache():
    """Pre-fetch and cache sentiment for all actively-watched symbols."""
    try:
        from ai.bot_engine import _active_sessions
        from ai.sentiment import get_sentiment

        symbols: set[str] = set()
        for session in _active_sessions.values():
            symbols.update(session.symbols)

        if not symbols:
            return

        logger.debug(f"Warming sentiment cache for {len(symbols)} symbols")
        for sym in list(symbols)[:10]:   # max 10 to avoid rate limits
            try:
                await get_sentiment(sym)
                await asyncio.sleep(1)   # be gentle with APIs
            except Exception as e:
                logger.debug(f"Cache warm failed for {sym}: {e}")

    except Exception as e:
        logger.debug(f"Sentiment cache warm error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Scheduler Loop
# ─────────────────────────────────────────────────────────────────────────────

async def _scheduler_loop():
    global _running
    _running = True
    logger.info("Scheduler started")

    last_optimise = datetime.now(timezone.utc)
    last_cache_warm = datetime.now(timezone.utc)
    daily_report_sent_date: Optional[str] = None
    weekly_sent_date: Optional[str] = None

    while _running:
        try:
            now = datetime.now(timezone.utc)

            # ── Daily report ──────────────────────────────────────────────
            today_key = now.strftime("%Y-%m-%d")
            if now.hour == DAILY_REPORT_HOUR_UTC and daily_report_sent_date != today_key:
                logger.info("Running daily report...")
                await _send_daily_reports()
                daily_report_sent_date = today_key

            # ── Weekly digest ─────────────────────────────────────────────
            week_key = now.strftime("%Y-%W")
            if now.weekday() == WEEKLY_DIGEST_WEEKDAY and weekly_sent_date != week_key:
                await _send_weekly_digests()
                weekly_sent_date = week_key

            # ── Periodic optimisation ─────────────────────────────────────
            hours_since_opt = (now - last_optimise).total_seconds() / 3600
            if hours_since_opt >= OPTIMISE_INTERVAL_H:
                from ai.self_optimizer import optimise_parameters
                result = optimise_parameters()
                if result.changes:
                    logger.info(f"Scheduled optimisation: {result.changes}")
                last_optimise = now

            # ── Sentiment cache warming ────────────────────────────────────
            hours_since_warm = (now - last_cache_warm).total_seconds() / 3600
            if hours_since_warm >= CACHE_WARM_INTERVAL_H:
                await _warm_sentiment_cache()
                last_cache_warm = now

        except Exception as e:
            logger.error(f"Scheduler loop error: {e}")

        # Check every 60 seconds
        await asyncio.sleep(60)

    logger.info("Scheduler stopped")


async def start_scheduler():
    global _scheduler_task
    if _scheduler_task and not _scheduler_task.done():
        return
    _scheduler_task = asyncio.ensure_future(_scheduler_loop())
    logger.info("AI scheduler task created")


def stop_scheduler():
    global _running, _scheduler_task
    _running = False
    if _scheduler_task:
        _scheduler_task.cancel()
    logger.info("AI scheduler stopped")
