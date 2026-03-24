"""
WebSocket connection manager.
Broadcasts price updates, trade status changes, and portfolio updates to connected clients.
"""

import asyncio
import json
import logging
from typing import Dict, Set
from fastapi import WebSocket

from exchanges import get_public_24h_stats

logger = logging.getLogger(__name__)

# Symbols to track in the public price feed
TRACKED_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
    "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    "DOTUSDT", "MATICUSDT"
]


class ConnectionManager:
    def __init__(self):
        # user_id -> set of websocket connections
        self.active: Dict[int, Set[WebSocket]] = {}
        # admin connections (user_id=0 for unauthenticated price feed)
        self.price_subscribers: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        if user_id not in self.active:
            self.active[user_id] = set()
        self.active[user_id].add(websocket)
        logger.info(f"WS connected: user={user_id}, total={self.total_connections}")

    async def connect_price_feed(self, websocket: WebSocket):
        await websocket.accept()
        self.price_subscribers.add(websocket)
        logger.info(f"Price feed subscriber connected, total={len(self.price_subscribers)}")

    def disconnect(self, websocket: WebSocket, user_id: int):
        if user_id in self.active:
            self.active[user_id].discard(websocket)
            if not self.active[user_id]:
                del self.active[user_id]

    def disconnect_price_feed(self, websocket: WebSocket):
        self.price_subscribers.discard(websocket)

    @property
    def total_connections(self) -> int:
        return sum(len(v) for v in self.active.values())

    async def send_to_user(self, user_id: int, message: dict):
        """Send a message to all connections of a specific user."""
        if user_id not in self.active:
            return
        dead = set()
        for ws in self.active[user_id]:
            try:
                await ws.send_json(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.active[user_id].discard(ws)

    async def broadcast_prices(self, price_data: dict):
        """Broadcast price updates to all price feed subscribers."""
        dead = set()
        for ws in self.price_subscribers:
            try:
                await ws.send_json(price_data)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.price_subscribers.discard(ws)

    async def broadcast_all(self, message: dict):
        """Broadcast to ALL connected users (e.g., system announcements)."""
        for user_id in list(self.active.keys()):
            await self.send_to_user(user_id, message)


manager = ConnectionManager()


async def price_broadcast_loop():
    """
    Background task: fetch public prices every 3 seconds and broadcast.
    Runs for the lifetime of the application.
    """
    logger.info("Price broadcast loop started")
    while True:
        try:
            if manager.price_subscribers or manager.total_connections > 0:
                stats = await get_public_24h_stats(TRACKED_SYMBOLS)
                if stats:
                    payload = {
                        "type": "prices",
                        "data": stats
                    }
                    await manager.broadcast_prices(payload)
                    await manager.broadcast_all(payload)
        except Exception as e:
            logger.warning(f"Price broadcast error: {e}")
        await asyncio.sleep(3)


async def send_trade_update(user_id: int, trade_id: int, pnl: float,
                             pnl_percent: float, current_price: float, status: str):
    await manager.send_to_user(user_id, {
        "type": "trade_update",
        "trade_id": trade_id,
        "status": status,
        "pnl": pnl,
        "pnl_percent": pnl_percent,
        "current_price": current_price,
    })


async def send_portfolio_update(user_id: int, total_value: float,
                                 total_pnl: float, total_pnl_percent: float):
    await manager.send_to_user(user_id, {
        "type": "portfolio_update",
        "total_value": total_value,
        "total_pnl": total_pnl,
        "total_pnl_percent": total_pnl_percent,
    })


async def send_notification(user_id: int, level: str, message: str):
    """level: info | success | warning | error"""
    await manager.send_to_user(user_id, {
        "type": "notification",
        "level": level,
        "message": message,
    })
