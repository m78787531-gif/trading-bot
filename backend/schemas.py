from pydantic import BaseModel, EmailStr, validator
from typing import Optional, List
from datetime import datetime
from models import UserRole, UserStatus, TradeStatus, TradeSide, TradeMode


# ─── Auth ───────────────────────────────────────────────────────────────────

class UserRegister(BaseModel):
    email: EmailStr
    username: str
    password: str

    @validator("username")
    def username_alphanumeric(cls, v):
        if not v.isalnum():
            raise ValueError("Username must be alphanumeric")
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters")
        return v

    @validator("password")
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    user_id: Optional[int] = None


# ─── Users ───────────────────────────────────────────────────────────────────

class UserOut(BaseModel):
    id: int
    email: str
    username: str
    role: UserRole
    status: UserStatus
    is_active: bool
    paper_trading: bool
    created_at: datetime

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    paper_trading: Optional[bool] = None


class AdminUserUpdate(BaseModel):
    status: Optional[UserStatus] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


# ─── API Keys ────────────────────────────────────────────────────────────────

class APIKeyCreate(BaseModel):
    exchange: str
    api_key: str
    api_secret: str
    is_testnet: bool = False
    label: str = ""

    @validator("exchange")
    def exchange_valid(cls, v):
        if v not in ("binance", "bybit"):
            raise ValueError("Exchange must be 'binance' or 'bybit'")
        return v


class APIKeyOut(BaseModel):
    id: int
    exchange: str
    label: str
    is_testnet: bool
    is_active: bool
    created_at: datetime
    api_key_preview: str = ""

    class Config:
        from_attributes = True


# ─── Trades ──────────────────────────────────────────────────────────────────

class TradeCreate(BaseModel):
    exchange: str
    symbol: str
    side: TradeSide
    mode: TradeMode = TradeMode.paper
    quantity: float
    entry_price: float
    leverage: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    notes: str = ""


class TradeOut(BaseModel):
    id: int
    exchange: str
    symbol: str
    side: TradeSide
    mode: TradeMode
    status: TradeStatus
    order_id: Optional[str]
    quantity: float
    entry_price: float
    exit_price: Optional[float]
    current_price: Optional[float]
    pnl: float
    pnl_percent: float
    leverage: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    fee: float
    notes: str
    opened_at: datetime
    closed_at: Optional[datetime]

    class Config:
        from_attributes = True


class TradeClose(BaseModel):
    exit_price: float


# ─── WebSocket ───────────────────────────────────────────────────────────────

class WSPriceUpdate(BaseModel):
    type: str = "price_update"
    symbol: str
    price: float
    change_24h: float


class WSTradeUpdate(BaseModel):
    type: str = "trade_update"
    trade_id: int
    status: str
    pnl: float
    pnl_percent: float
    current_price: float


class WSPortfolioUpdate(BaseModel):
    type: str = "portfolio_update"
    total_value: float
    total_pnl: float
    total_pnl_percent: float


# ─── Portfolio ───────────────────────────────────────────────────────────────

class PortfolioStats(BaseModel):
    total_trades: int
    open_trades: int
    closed_trades: int
    total_pnl: float
    total_pnl_percent: float
    win_rate: float
    best_trade: Optional[float]
    worst_trade: Optional[float]
    paper_mode: bool
