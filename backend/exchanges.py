"""
Exchange integrations for Binance and Bybit.
Fetches real trades, positions, and prices from user accounts.
"""

import hmac
import hashlib
import time
import httpx
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class ExchangeTrade:
    order_id: str
    symbol: str
    side: str          # buy | sell
    quantity: float
    entry_price: float
    exit_price: Optional[float]
    pnl: float
    fee: float
    status: str        # open | closed
    timestamp: int


@dataclass
class ExchangePosition:
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    leverage: float


# ─── Binance ─────────────────────────────────────────────────────────────────

class BinanceClient:
    BASE_URL = "https://api.binance.com"
    TESTNET_URL = "https://testnet.binance.vision"
    FUTURES_URL = "https://fapi.binance.com"
    FUTURES_TESTNET_URL = "https://testnet.binancefuture.com"

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base = self.TESTNET_URL if testnet else self.BASE_URL
        self.futures_base = self.FUTURES_TESTNET_URL if testnet else self.FUTURES_URL
        self.headers = {"X-MBX-APIKEY": api_key}

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        query = "&".join(f"{k}={v}" for k, v in params.items())
        signature = hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params

    async def get_account(self) -> Dict:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{self.base}/api/v3/account",
                headers=self.headers,
                params=self._sign({})
            )
            r.raise_for_status()
            return r.json()

    async def get_spot_trades(self, symbol: str = None, limit: int = 50) -> List[ExchangeTrade]:
        params = {"limit": limit}
        if symbol:
            params["symbol"] = symbol
        async with httpx.AsyncClient() as client:
            endpoint = f"{self.base}/api/v3/myTrades"
            if not symbol:
                # Need symbol for this endpoint; get recent from all orders instead
                endpoint = f"{self.base}/api/v3/allOrders"
                params["limit"] = limit
            r = await client.get(
                endpoint,
                headers=self.headers,
                params=self._sign(params)
            )
            r.raise_for_status()
            raw = r.json()

        trades = []
        for t in raw:
            trades.append(ExchangeTrade(
                order_id=str(t.get("orderId", t.get("id", ""))),
                symbol=t.get("symbol", ""),
                side=t.get("side", "buy").lower(),
                quantity=float(t.get("qty", t.get("executedQty", 0))),
                entry_price=float(t.get("price", 0)),
                exit_price=None,
                pnl=0.0,
                fee=float(t.get("commission", 0)),
                status="closed" if t.get("status") == "FILLED" else "open",
                timestamp=t.get("time", 0)
            ))
        return trades

    async def get_futures_positions(self) -> List[ExchangePosition]:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{self.futures_base}/fapi/v2/positionRisk",
                headers=self.headers,
                params=self._sign({})
            )
            r.raise_for_status()
            raw = r.json()

        positions = []
        for p in raw:
            qty = float(p.get("positionAmt", 0))
            if qty == 0:
                continue
            positions.append(ExchangePosition(
                symbol=p["symbol"],
                side="buy" if qty > 0 else "sell",
                quantity=abs(qty),
                entry_price=float(p["entryPrice"]),
                current_price=float(p["markPrice"]),
                unrealized_pnl=float(p["unRealizedProfit"]),
                leverage=float(p["leverage"])
            ))
        return positions

    async def get_ticker_price(self, symbol: str) -> float:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{self.base}/api/v3/ticker/price",
                params={"symbol": symbol}
            )
            r.raise_for_status()
            return float(r.json()["price"])

    async def get_24h_stats(self, symbol: str = None) -> List[Dict]:
        params = {}
        if symbol:
            params["symbol"] = symbol
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{self.base}/api/v3/ticker/24hr",
                params=params
            )
            r.raise_for_status()
            data = r.json()
            return [data] if symbol else data


# ─── Bybit ───────────────────────────────────────────────────────────────────

class BybitClient:
    BASE_URL = "https://api.bybit.com"
    TESTNET_URL = "https://api-testnet.bybit.com"

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base = self.TESTNET_URL if testnet else self.BASE_URL

    def _sign(self, params: dict) -> dict:
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        param_str = timestamp + self.api_key + recv_window
        # Sort params for query string
        sorted_params = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        param_str += sorted_params

        signature = hmac.new(
            self.api_secret.encode(), param_str.encode(), hashlib.sha256
        ).hexdigest()

        params["api_key"] = self.api_key
        params["timestamp"] = timestamp
        params["recv_window"] = recv_window
        params["sign"] = signature
        return params

    def _headers(self, timestamp: str, signature: str, recv_window: str = "5000") -> dict:
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": signature,
            "X-BAPI-RECV-WINDOW": recv_window,
        }

    def _make_headers(self, params: dict = None, body: str = "") -> dict:
        ts = str(int(time.time() * 1000))
        recv_window = "5000"
        query_str = "&".join(f"{k}={v}" for k, v in (params or {}).items())
        sign_str = ts + self.api_key + recv_window + (query_str or body)
        sig = hmac.new(self.api_secret.encode(), sign_str.encode(), hashlib.sha256).hexdigest()
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-SIGN": sig,
            "X-BAPI-RECV-WINDOW": recv_window,
        }

    async def get_wallet_balance(self, account_type: str = "UNIFIED") -> Dict:
        params = {"accountType": account_type}
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{self.base}/v5/account/wallet-balance",
                headers=self._make_headers(params),
                params=params
            )
            r.raise_for_status()
            return r.json()

    async def get_positions(self, category: str = "linear", symbol: str = None) -> List[ExchangePosition]:
        params = {"category": category, "limit": 50}
        if symbol:
            params["symbol"] = symbol
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{self.base}/v5/position/list",
                headers=self._make_headers(params),
                params=params
            )
            r.raise_for_status()
            data = r.json()

        positions = []
        for p in data.get("result", {}).get("list", []):
            qty = float(p.get("size", 0))
            if qty == 0:
                continue
            positions.append(ExchangePosition(
                symbol=p["symbol"],
                side=p.get("side", "Buy").lower(),
                quantity=qty,
                entry_price=float(p.get("avgPrice", 0)),
                current_price=float(p.get("markPrice", 0)),
                unrealized_pnl=float(p.get("unrealisedPnl", 0)),
                leverage=float(p.get("leverage", 1))
            ))
        return positions

    async def get_trade_history(self, category: str = "linear", limit: int = 50) -> List[ExchangeTrade]:
        params = {"category": category, "limit": limit}
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{self.base}/v5/execution/list",
                headers=self._make_headers(params),
                params=params
            )
            r.raise_for_status()
            data = r.json()

        trades = []
        for t in data.get("result", {}).get("list", []):
            trades.append(ExchangeTrade(
                order_id=t.get("orderId", ""),
                symbol=t.get("symbol", ""),
                side=t.get("side", "Buy").lower(),
                quantity=float(t.get("execQty", 0)),
                entry_price=float(t.get("execPrice", 0)),
                exit_price=None,
                pnl=float(t.get("closedPnl", 0)),
                fee=float(t.get("execFee", 0)),
                status="closed",
                timestamp=int(t.get("execTime", 0))
            ))
        return trades

    async def get_ticker(self, category: str = "spot", symbol: str = None) -> List[Dict]:
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{self.base}/v5/market/tickers", params=params)
            r.raise_for_status()
            return r.json().get("result", {}).get("list", [])


# ─── Factory ─────────────────────────────────────────────────────────────────

def get_exchange_client(exchange: str, api_key: str, api_secret: str, testnet: bool = False):
    if exchange == "binance":
        return BinanceClient(api_key, api_secret, testnet)
    elif exchange == "bybit":
        return BybitClient(api_key, api_secret, testnet)
    raise ValueError(f"Unknown exchange: {exchange}")


# ─── Public Price Feed (no auth needed) ──────────────────────────────────────

async def get_public_prices(symbols: List[str]) -> Dict[str, float]:
    """Fetch current prices from Binance public API (no auth needed)."""
    prices = {}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get("https://api.binance.com/api/v3/ticker/price")
            if r.status_code == 200:
                for item in r.json():
                    if item["symbol"] in symbols:
                        prices[item["symbol"]] = float(item["price"])
    except Exception:
        pass
    return prices


async def get_public_24h_stats(symbols: List[str]) -> Dict[str, Dict]:
    """Fetch 24h stats from Binance public API."""
    stats = {}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get("https://api.binance.com/api/v3/ticker/24hr")
            if r.status_code == 200:
                for item in r.json():
                    if item["symbol"] in symbols:
                        stats[item["symbol"]] = {
                            "price": float(item["lastPrice"]),
                            "change": float(item["priceChangePercent"]),
                            "volume": float(item["volume"]),
                            "high": float(item["highPrice"]),
                            "low": float(item["lowPrice"]),
                        }
    except Exception:
        pass
    return stats
