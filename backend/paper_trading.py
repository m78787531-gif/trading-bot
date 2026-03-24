"""
Paper trading simulation engine.
Simulates trade execution, PnL, and market movements without real money.
"""

import asyncio
import random
import logging
from datetime import datetime
from typing import Dict, List
from sqlalchemy.orm import Session

from models import Trade, TradeStatus
from exchanges import get_public_prices

try:
    from exchanges import TRACKED_SYMBOLS
except ImportError:
    TRACKED_SYMBOLS = None

logger = logging.getLogger(__name__)

# Fallback prices if API is unavailable
FALLBACK_PRICES = {
    "BTCUSDT": 43250.0,
    "ETHUSDT": 2280.0,
    "BNBUSDT": 305.0,
    "SOLUSDT": 98.5,
    "XRPUSDT": 0.52,
    "ADAUSDT": 0.38,
    "DOGEUSDT": 0.078,
    "AVAXUSDT": 28.4,
    "DOTUSDT": 6.8,
    "MATICUSDT": 0.71,
}


async def get_current_price(symbol: str) -> float:
    """Fetch live price; fall back to simulated if unavailable."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbol": symbol}
            )
            if r.status_code == 200:
                return float(r.json()["price"])
    except Exception:
        pass

    # Simulate small random movement around fallback
    base = FALLBACK_PRICES.get(symbol, 100.0)
    return base * (1 + random.uniform(-0.005, 0.005))


def calculate_pnl(trade: Trade, current_price: float) -> tuple[float, float]:
    """Calculate unrealized PnL for an open trade."""
    if trade.side.value == "buy":
        pnl = (current_price - trade.entry_price) * trade.quantity * trade.leverage
    else:
        pnl = (trade.entry_price - current_price) * trade.quantity * trade.leverage

    cost_basis = trade.entry_price * trade.quantity
    pnl_percent = (pnl / cost_basis * 100) if cost_basis > 0 else 0.0
    return round(pnl, 4), round(pnl_percent, 4)


def check_sl_tp(trade: Trade, current_price: float) -> str | None:
    """Returns 'sl' or 'tp' if triggered, else None."""
    if trade.stop_loss:
        if trade.side.value == "buy" and current_price <= trade.stop_loss:
            return "sl"
        if trade.side.value == "sell" and current_price >= trade.stop_loss:
            return "sl"
    if trade.take_profit:
        if trade.side.value == "buy" and current_price >= trade.take_profit:
            return "tp"
        if trade.side.value == "sell" and current_price <= trade.take_profit:
            return "tp"
    return None


async def update_paper_trades(db: Session, user_id: int) -> List[Dict]:
    """
    Update all open paper trades for a user with current prices.
    Returns list of updates (for WebSocket broadcasting).
    """
    open_trades = db.query(Trade).filter(
        Trade.user_id == user_id,
        Trade.status == TradeStatus.open,
        Trade.mode == "paper"
    ).all()

    updates = []
    for trade in open_trades:
        try:
            price = await get_current_price(trade.symbol)
            pnl, pnl_percent = calculate_pnl(trade, price)

            trade.current_price = price
            trade.pnl = pnl
            trade.pnl_percent = pnl_percent

            trigger = check_sl_tp(trade, price)
            if trigger:
                trade.status = TradeStatus.closed
                trade.exit_price = price
                trade.closed_at = datetime.utcnow()
                logger.info(f"Paper trade {trade.id} closed by {'stop-loss' if trigger=='sl' else 'take-profit'}")

            db.add(trade)
            updates.append({
                "trade_id": trade.id,
                "symbol": trade.symbol,
                "current_price": price,
                "pnl": pnl,
                "pnl_percent": pnl_percent,
                "status": trade.status.value,
                "trigger": trigger,
            })
        except Exception as e:
            logger.warning(f"Error updating paper trade {trade.id}: {e}")

    db.commit()
    return updates


def simulate_paper_order_fill(symbol: str, side: str, quantity: float, price: float = None) -> Dict:
    """Simulate instant order fill for paper trading."""
    fill_price = price or FALLBACK_PRICES.get(symbol, 100.0)
    # Simulate small slippage
    slippage = random.uniform(0.0001, 0.0005)
    if side == "buy":
        fill_price *= (1 + slippage)
    else:
        fill_price *= (1 - slippage)

    fee = fill_price * quantity * 0.001  # 0.1% fee simulation

    return {
        "order_id": f"PAPER-{int(datetime.utcnow().timestamp())}-{random.randint(1000, 9999)}",
        "fill_price": round(fill_price, 8),
        "fee": round(fee, 8),
        "status": "filled",
    }
