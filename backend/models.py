from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Enum, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from database import Base


class UserRole(str, enum.Enum):
    admin = "admin"
    user = "user"


class UserStatus(str, enum.Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class TradeStatus(str, enum.Enum):
    open = "open"
    closed = "closed"
    cancelled = "cancelled"


class TradeSide(str, enum.Enum):
    buy = "buy"
    sell = "sell"


class TradeMode(str, enum.Enum):
    live = "live"
    paper = "paper"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(UserRole), default=UserRole.user)
    status = Column(Enum(UserStatus), default=UserStatus.pending)
    is_active = Column(Boolean, default=True)
    paper_trading = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="user", cascade="all, delete-orphan")


class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    exchange = Column(String, nullable=False)       # binance | bybit
    api_key = Column(String, nullable=False)
    api_secret = Column(String, nullable=False)     # stored encrypted in production
    is_testnet = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    label = Column(String, default="")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="api_keys")


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    exchange = Column(String, nullable=False)
    symbol = Column(String, nullable=False, index=True)
    side = Column(Enum(TradeSide), nullable=False)
    mode = Column(Enum(TradeMode), default=TradeMode.live)
    status = Column(Enum(TradeStatus), default=TradeStatus.open)
    order_id = Column(String, index=True)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    current_price = Column(Float, nullable=True)
    pnl = Column(Float, default=0.0)
    pnl_percent = Column(Float, default=0.0)
    leverage = Column(Float, default=1.0)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    fee = Column(Float, default=0.0)
    notes = Column(Text, default="")
    opened_at = Column(DateTime(timezone=True), server_default=func.now())
    closed_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="trades")
