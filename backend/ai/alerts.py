"""
ai/alerts.py
─────────────────────────────────────────────────────────────────────────────
Telegram integration for real-time trade alerts and daily reports.

Setup:
  1. Create a Telegram bot via @BotFather → get TELEGRAM_BOT_TOKEN
  2. Get your chat ID: message the bot, then call
     https://api.telegram.org/bot<TOKEN>/getUpdates
  3. Set env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

Features:
  • Trade opened / closed notifications
  • Signal analysis alerts (high-confidence setups)
  • Risk warnings (drawdown, daily loss limit hit)
  • Daily PnL report (scheduled at configurable time)
  • Weekly performance digest
  • Admin-only sensitive alerts (configurable second chat ID)
"""

from __future__ import annotations
import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID     = os.getenv("TELEGRAM_CHAT_ID", "")
ADMIN_CHAT  = os.getenv("TELEGRAM_ADMIN_CHAT_ID", CHAT_ID)

BASE_URL    = f"https://api.telegram.org/bot{BOT_TOKEN}"
MAX_RETRIES = 3


def _is_configured() -> bool:
    return bool(BOT_TOKEN and CHAT_ID)


# ─────────────────────────────────────────────────────────────────────────────
# Core send function
# ─────────────────────────────────────────────────────────────────────────────

async def _send_message(
    text: str,
    chat_id: str = None,
    parse_mode: str = "HTML",
    disable_preview: bool = True,
) -> bool:
    """
    Send a Telegram message with retry logic.
    Returns True on success.
    """
    if not _is_configured():
        logger.debug("Telegram not configured — skipping alert")
        return False

    cid = chat_id or CHAT_ID
    url = f"{BASE_URL}/sendMessage"
    payload = {
        "chat_id": cid,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": disable_preview,
    }

    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(url, json=payload)
                if r.status_code == 200:
                    return True
                if r.status_code == 429:  # rate limited
                    retry_after = r.json().get("parameters", {}).get("retry_after", 5)
                    await asyncio.sleep(retry_after)
                    continue
                logger.warning(f"Telegram error {r.status_code}: {r.text[:200]}")
                return False
        except Exception as e:
            logger.warning(f"Telegram attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)

    return False


def _icon(value: float, zero_neutral: bool = True) -> str:
    if value > 0:   return "🟢"
    if value < 0:   return "🔴"
    return "⚪" if zero_neutral else "🟡"


def _dir_icon(direction: str) -> str:
    return "📈" if direction == "long" else "📉"


# ─────────────────────────────────────────────────────────────────────────────
# Alert Templates
# ─────────────────────────────────────────────────────────────────────────────

async def alert_trade_opened(
    symbol: str,
    direction: str,
    mode: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    quantity: float,
    confidence: float,
    risk_reward: float,
    reasoning: list[str],
    exchange: str = "binance",
) -> bool:
    icon = _dir_icon(direction)
    mode_tag = "📄 PAPER" if mode == "paper" else "⚡ LIVE"
    dir_tag  = "LONG" if direction == "long" else "SHORT"

    sl_pct = abs(entry_price - stop_loss) / entry_price * 100
    tp_pct = abs(take_profit - entry_price) / entry_price * 100
    reason_text = "\n".join(f"  • {r}" for r in reasoning[:5])

    msg = (
        f"{icon} <b>Trade Opened — {dir_tag}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💎 <b>{symbol}</b> | {exchange.upper()} | {mode_tag}\n\n"
        f"📊 <b>Entry:</b>     <code>${entry_price:,.6f}</code>\n"
        f"🛑 <b>Stop Loss:</b> <code>${stop_loss:,.6f}</code> ({sl_pct:.2f}%)\n"
        f"🎯 <b>Take Profit:</b><code>${take_profit:,.6f}</code> ({tp_pct:.2f}%)\n"
        f"📦 <b>Quantity:</b>  <code>{quantity:.4f}</code>\n\n"
        f"🧠 <b>Confidence:</b> {confidence:.1f}%\n"
        f"⚖️ <b>R:R Ratio:</b>  1:{risk_reward:.2f}\n\n"
        f"<b>Analysis:</b>\n{reason_text}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</i>"
    )
    return await _send_message(msg)


async def alert_trade_closed(
    symbol: str,
    direction: str,
    mode: str,
    entry_price: float,
    exit_price: float,
    pnl: float,
    pnl_pct: float,
    close_reason: str = "manual",   # manual | stop_loss | take_profit | trailing_stop
    hold_duration: str = "",
) -> bool:
    icon = _icon(pnl)
    won = pnl > 0
    dir_tag = "LONG" if direction == "long" else "SHORT"
    mode_tag = "📄 PAPER" if mode == "paper" else "⚡ LIVE"

    reason_icons = {
        "stop_loss": "🛑",
        "take_profit": "🎯",
        "trailing_stop": "🔁",
        "manual": "👤",
    }
    close_icon = reason_icons.get(close_reason, "✅")

    msg = (
        f"{icon} <b>Trade Closed — {'WIN 🏆' if won else 'LOSS 💔'}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💎 <b>{symbol}</b> {dir_tag} | {mode_tag}\n\n"
        f"📥 <b>Entry:</b>   <code>${entry_price:,.6f}</code>\n"
        f"📤 <b>Exit:</b>    <code>${exit_price:,.6f}</code>\n"
        f"{'🟢' if won else '🔴'} <b>PnL:</b>    <code>{pnl:+.4f} USDT ({pnl_pct:+.2f}%)</code>\n\n"
        f"{close_icon} <b>Reason:</b> {close_reason.replace('_', ' ').title()}\n"
    )
    if hold_duration:
        msg += f"⏱️ <b>Duration:</b> {hold_duration}\n"
    msg += (
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</i>"
    )
    return await _send_message(msg)


async def alert_high_confidence_signal(
    symbol: str,
    direction: str,
    confidence: float,
    entry: float,
    sl: float,
    tp: float,
    smc_summary: str,
    sentiment_label: str,
    risk_reward: float,
) -> bool:
    if confidence < 75:
        return False   # only send high-confidence alerts

    icon = _dir_icon(direction)
    msg = (
        f"{icon} <b>⚡ High-Confidence Signal ({confidence:.0f}%)</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💎 <b>{symbol}</b> — {'LONG' if direction == 'long' else 'SHORT'}\n\n"
        f"💰 <b>Entry:</b>    <code>${entry:,.6f}</code>\n"
        f"🛑 <b>SL:</b>       <code>${sl:,.6f}</code>\n"
        f"🎯 <b>TP:</b>       <code>${tp:,.6f}</code>\n"
        f"⚖️ <b>R:R:</b>      1:{risk_reward:.2f}\n\n"
        f"🏦 <b>SMC:</b> {smc_summary[:100]}\n"
        f"📰 <b>Sentiment:</b> {sentiment_label.replace('_', ' ').title()}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Auto-signal — not financial advice</i>"
    )
    return await _send_message(msg)


async def alert_risk_warning(
    warning_type: str,
    message: str,
    current_pnl_pct: float,
    threshold_pct: float,
) -> bool:
    """Send to admin chat for critical risk events."""
    msg = (
        f"⚠️ <b>RISK WARNING — {warning_type}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 {message}\n\n"
        f"📉 Current PnL: <code>{current_pnl_pct:+.2f}%</code>\n"
        f"🚧 Threshold:   <code>{threshold_pct:.1f}%</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Action: Trading paused automatically</b>\n"
        f"<i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</i>"
    )
    return await _send_message(msg, chat_id=ADMIN_CHAT)


async def send_daily_report(
    date: str,
    total_trades: int,
    wins: int,
    losses: int,
    total_pnl: float,
    total_pnl_pct: float,
    best_trade: dict,
    worst_trade: dict,
    open_trades: int,
    account_balance: float,
    param_changes: list[str],
    top_signals: list[dict],
) -> bool:
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    pnl_icon = _icon(total_pnl)

    best = f"{best_trade.get('symbol', 'N/A')} +{best_trade.get('pnl_pct', 0):.2f}%" if best_trade else "N/A"
    worst = f"{worst_trade.get('symbol', 'N/A')} {worst_trade.get('pnl_pct', 0):.2f}%" if worst_trade else "N/A"

    param_section = ""
    if param_changes:
        param_section = "\n🔧 <b>AI Parameter Updates:</b>\n"
        param_section += "\n".join(f"  • {c}" for c in param_changes[:5])
        param_section += "\n"

    signals_section = ""
    if top_signals:
        signals_section = "\n📡 <b>Top Signals (next 24h):</b>\n"
        for s in top_signals[:3]:
            signals_section += (
                f"  {_dir_icon(s.get('direction','long'))} {s.get('symbol','')} "
                f"{s.get('direction','').upper()} — {s.get('confidence',0):.0f}% conf\n"
            )

    msg = (
        f"📊 <b>Daily Performance Report — {date}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"💰 <b>Total PnL:</b>    {pnl_icon} <code>{total_pnl:+.4f} ({total_pnl_pct:+.2f}%)</code>\n"
        f"🔢 <b>Trades:</b>       {total_trades} ({wins}W / {losses}L)\n"
        f"🎯 <b>Win Rate:</b>     {win_rate:.1f}%\n"
        f"💼 <b>Balance:</b>      <code>${account_balance:,.2f}</code>\n"
        f"📂 <b>Open Positions:</b> {open_trades}\n\n"
        f"🏆 <b>Best Trade:</b>  {best}\n"
        f"💔 <b>Worst Trade:</b> {worst}\n"
        f"{param_section}"
        f"{signals_section}"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>TradeOS AI Engine | {datetime.now(timezone.utc).strftime('%H:%M UTC')}</i>"
    )
    return await _send_message(msg)


async def send_weekly_digest(
    week_str: str,
    metrics: dict,
    top_symbols: list[tuple[str, float]],
    worst_symbols: list[tuple[str, float]],
    optimisation_summary: dict,
) -> bool:
    pnl_icon = _icon(metrics.get("total_pnl", 0))
    best_syms = "\n".join(f"  🥇 {sym}: +{pnl:.2f}%" for sym, pnl in top_symbols[:3])
    worst_syms = "\n".join(f"  💔 {sym}: {pnl:.2f}%" for sym, pnl in worst_symbols[:3])

    msg = (
        f"📅 <b>Weekly Digest — {week_str}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"💰 <b>Week PnL:</b>    {pnl_icon} <code>{metrics.get('total_pnl', 0):+.4f}</code>\n"
        f"🔢 <b>Total Trades:</b> {metrics.get('total_trades', 0)}\n"
        f"🎯 <b>Win Rate:</b>     {metrics.get('win_rate', 0)*100:.1f}%\n"
        f"📈 <b>Profit Factor:</b> {metrics.get('profit_factor', 1):.2f}\n\n"
        f"<b>Best Symbols:</b>\n{best_syms}\n\n"
        f"<b>Worst Symbols:</b>\n{worst_syms}\n\n"
        f"🧠 <b>AI Optimisation:</b> {len(optimisation_summary.get('changes', []))} params updated\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>TradeOS AI Engine Weekly Report</i>"
    )
    return await _send_message(msg)


async def alert_system_status(message: str, is_error: bool = False) -> bool:
    icon = "🚨" if is_error else "ℹ️"
    msg = f"{icon} <b>System {'Error' if is_error else 'Status'}</b>\n{message}"
    return await _send_message(msg, chat_id=ADMIN_CHAT)


# ─────────────────────────────────────────────────────────────────────────────
# Sync wrapper for use in non-async contexts
# ─────────────────────────────────────────────────────────────────────────────

def send_alert_sync(coro) -> bool:
    """Run an alert coroutine safely from sync code."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(coro)
            return True
        return loop.run_until_complete(coro)
    except Exception as e:
        logger.error(f"Alert send error: {e}")
        return False
