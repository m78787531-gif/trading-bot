"""
Trading Platform API
FastAPI backend with JWT auth, PostgreSQL, WebSockets, Binance/Bybit integration.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

from fastapi import (
    FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect,
    status, BackgroundTasks
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from models import User, APIKey, Trade, UserRole, UserStatus, TradeStatus, TradeMode
from schemas import (
    UserRegister, UserOut, UserUpdate, AdminUserUpdate,
    APIKeyCreate, APIKeyOut,
    TradeCreate, TradeOut, TradeClose,
    Token, PortfolioStats
)
from auth import (
    hash_password, verify_password, create_access_token,
    get_current_user, get_approved_user, get_admin_user
)
from exchanges import get_exchange_client, get_public_24h_stats
from paper_trading import simulate_paper_order_fill, update_paper_trades, calculate_pnl, get_current_price
from websocket_manager import manager, price_broadcast_loop, send_trade_update, send_notification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Startup / Shutdown ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create all tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified")

    # Ensure a default admin exists
    db = next(get_db())
    try:
        admin = db.query(User).filter(User.role == UserRole.admin).first()
        if not admin:
            admin = User(
                email="admin@trading.local",
                username="admin",
                hashed_password=hash_password("Admin@1234"),
                role=UserRole.admin,
                status=UserStatus.approved,
                paper_trading=False,
            )
            db.add(admin)
            db.commit()
            logger.info("Default admin created: admin / Admin@1234")
    finally:
        db.close()

    # Start background price broadcast
    price_task = asyncio.create_task(price_broadcast_loop())

    yield

    price_task.cancel()
    try:
        await price_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Trading Platform API",
    version="1.0.0",
    description="Real-time crypto trading platform with Binance/Bybit integration",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Auth Routes ─────────────────────────────────────────────────────────────

@app.post("/auth/register", response_model=UserOut, status_code=201)
async def register(data: UserRegister, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(400, "Email already registered")
    if db.query(User).filter(User.username == data.username).first():
        raise HTTPException(400, "Username already taken")

    user = User(
        email=data.email,
        username=data.username,
        hashed_password=hash_password(data.password),
        status=UserStatus.pending,
        paper_trading=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info(f"New user registered: {user.username} (pending approval)")
    return user


@app.post("/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    if not user.is_active:
        raise HTTPException(400, "Account disabled")

    token = create_access_token({"sub": str(user.id), "role": user.role.value})
    return {"access_token": token, "token_type": "bearer"}


# ─── User Routes ─────────────────────────────────────────────────────────────

@app.get("/user", response_model=UserOut)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user


@app.patch("/user", response_model=UserOut)
async def update_me(
    data: UserUpdate,
    current_user: User = Depends(get_approved_user),
    db: Session = Depends(get_db)
):
    if data.paper_trading is not None:
        current_user.paper_trading = data.paper_trading
    db.commit()
    db.refresh(current_user)
    return current_user


# ─── API Key Routes ───────────────────────────────────────────────────────────

@app.post("/add-api", response_model=APIKeyOut, status_code=201)
async def add_api_key(
    data: APIKeyCreate,
    current_user: User = Depends(get_approved_user),
    db: Session = Depends(get_db)
):
    # Deactivate any existing key for this exchange
    existing = db.query(APIKey).filter(
        APIKey.user_id == current_user.id,
        APIKey.exchange == data.exchange,
        APIKey.is_active == True
    ).first()
    if existing:
        existing.is_active = False
        db.add(existing)

    # Verify key works (optional but recommended)
    try:
        client = get_exchange_client(data.exchange, data.api_key, data.api_secret, data.is_testnet)
        if data.exchange == "binance":
            await client.get_account()
        else:
            await client.get_wallet_balance()
    except Exception as e:
        raise HTTPException(400, f"API key verification failed: {str(e)}")

    key = APIKey(
        user_id=current_user.id,
        exchange=data.exchange,
        api_key=data.api_key,
        api_secret=data.api_secret,   # TODO: encrypt with Fernet in production
        is_testnet=data.is_testnet,
        label=data.label,
    )
    db.add(key)
    db.commit()
    db.refresh(key)

    # Auto-disable paper trading if live key added
    if not data.is_testnet:
        current_user.paper_trading = False
        db.commit()
        await send_notification(current_user.id, "success", f"{data.exchange.title()} API connected. Live trading enabled.")

    result = APIKeyOut(
        id=key.id,
        exchange=key.exchange,
        label=key.label,
        is_testnet=key.is_testnet,
        is_active=key.is_active,
        created_at=key.created_at,
        api_key_preview=f"{key.api_key[:4]}...{key.api_key[-4:]}"
    )
    return result


@app.get("/api-keys", response_model=List[APIKeyOut])
async def list_api_keys(
    current_user: User = Depends(get_approved_user),
    db: Session = Depends(get_db)
):
    keys = db.query(APIKey).filter(
        APIKey.user_id == current_user.id,
        APIKey.is_active == True
    ).all()
    return [APIKeyOut(
        id=k.id, exchange=k.exchange, label=k.label,
        is_testnet=k.is_testnet, is_active=k.is_active,
        created_at=k.created_at,
        api_key_preview=f"{k.api_key[:4]}...{k.api_key[-4:]}"
    ) for k in keys]


@app.delete("/api-keys/{key_id}")
async def delete_api_key(
    key_id: int,
    current_user: User = Depends(get_approved_user),
    db: Session = Depends(get_db)
):
    key = db.query(APIKey).filter(APIKey.id == key_id, APIKey.user_id == current_user.id).first()
    if not key:
        raise HTTPException(404, "API key not found")
    key.is_active = False
    db.commit()
    return {"message": "API key removed"}


# ─── Trade Routes ─────────────────────────────────────────────────────────────

@app.get("/trades", response_model=List[TradeOut])
async def get_trades(
    status_filter: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 100,
    current_user: User = Depends(get_approved_user),
    db: Session = Depends(get_db)
):
    q = db.query(Trade).filter(Trade.user_id == current_user.id)
    if status_filter:
        q = q.filter(Trade.status == status_filter)
    if symbol:
        q = q.filter(Trade.symbol == symbol.upper())
    trades = q.order_by(Trade.opened_at.desc()).limit(limit).all()
    return trades


@app.get("/trade/{symbol}", response_model=List[TradeOut])
async def get_trades_by_symbol(
    symbol: str,
    current_user: User = Depends(get_approved_user),
    db: Session = Depends(get_db)
):
    trades = db.query(Trade).filter(
        Trade.user_id == current_user.id,
        Trade.symbol == symbol.upper()
    ).order_by(Trade.opened_at.desc()).all()
    return trades


@app.post("/trades", response_model=TradeOut, status_code=201)
async def open_trade(
    data: TradeCreate,
    current_user: User = Depends(get_approved_user),
    db: Session = Depends(get_db)
):
    mode = data.mode

    # Force paper trading if user doesn't have live API keys
    api_key = db.query(APIKey).filter(
        APIKey.user_id == current_user.id,
        APIKey.exchange == data.exchange,
        APIKey.is_active == True
    ).first()

    if not api_key or current_user.paper_trading:
        mode = TradeMode.paper

    fill_price = data.entry_price
    order_id = None
    fee = 0.0

    if mode == TradeMode.paper:
        fill = simulate_paper_order_fill(data.symbol, data.side.value, data.quantity, data.entry_price)
        fill_price = fill["fill_price"]
        order_id = fill["order_id"]
        fee = fill["fee"]
    else:
        # Live trade via exchange API
        try:
            client = get_exchange_client(data.exchange, api_key.api_key, api_key.api_secret, api_key.is_testnet)
            # Implement actual order placement here
            # For now we record the intended price; real implementation would call place_order()
            order_id = f"LIVE-{int(datetime.utcnow().timestamp())}"
        except Exception as e:
            raise HTTPException(500, f"Exchange error: {str(e)}")

    trade = Trade(
        user_id=current_user.id,
        exchange=data.exchange,
        symbol=data.symbol.upper(),
        side=data.side,
        mode=mode,
        order_id=order_id,
        quantity=data.quantity,
        entry_price=fill_price,
        current_price=fill_price,
        leverage=data.leverage,
        stop_loss=data.stop_loss,
        take_profit=data.take_profit,
        fee=fee,
        notes=data.notes,
        pnl=0.0,
        pnl_percent=0.0,
    )
    db.add(trade)
    db.commit()
    db.refresh(trade)
    await send_notification(current_user.id, "success",
        f"{'Paper' if mode == TradeMode.paper else 'Live'} trade opened: {data.side.value.upper()} {data.symbol}")
    return trade


@app.patch("/trades/{trade_id}/close", response_model=TradeOut)
async def close_trade(
    trade_id: int,
    data: TradeClose,
    current_user: User = Depends(get_approved_user),
    db: Session = Depends(get_db)
):
    trade = db.query(Trade).filter(
        Trade.id == trade_id,
        Trade.user_id == current_user.id,
        Trade.status == TradeStatus.open
    ).first()
    if not trade:
        raise HTTPException(404, "Open trade not found")

    pnl, pnl_percent = calculate_pnl(trade, data.exit_price)
    trade.exit_price = data.exit_price
    trade.current_price = data.exit_price
    trade.pnl = pnl
    trade.pnl_percent = pnl_percent
    trade.status = TradeStatus.closed
    trade.closed_at = datetime.utcnow()
    db.commit()
    db.refresh(trade)

    await send_trade_update(current_user.id, trade.id, pnl, pnl_percent, data.exit_price, "closed")
    return trade


@app.get("/trades/sync/exchange")
async def sync_exchange_trades(
    exchange: str,
    current_user: User = Depends(get_approved_user),
    db: Session = Depends(get_db)
):
    """Pull real trades from exchange and save to DB."""
    api_key = db.query(APIKey).filter(
        APIKey.user_id == current_user.id,
        APIKey.exchange == exchange,
        APIKey.is_active == True
    ).first()
    if not api_key:
        raise HTTPException(400, f"No active {exchange} API key found")

    client = get_exchange_client(exchange, api_key.api_key, api_key.api_secret, api_key.is_testnet)
    synced = 0

    try:
        if exchange == "binance":
            positions = await client.get_futures_positions()
        else:
            positions = await client.get_positions()

        for pos in positions:
            existing = db.query(Trade).filter(
                Trade.user_id == current_user.id,
                Trade.symbol == pos.symbol,
                Trade.status == TradeStatus.open,
                Trade.mode == TradeMode.live
            ).first()
            if not existing:
                trade = Trade(
                    user_id=current_user.id,
                    exchange=exchange,
                    symbol=pos.symbol,
                    side=pos.side,
                    mode=TradeMode.live,
                    quantity=pos.quantity,
                    entry_price=pos.entry_price,
                    current_price=pos.current_price,
                    leverage=pos.leverage,
                    pnl=pos.unrealized_pnl,
                    pnl_percent=0.0,
                )
                db.add(trade)
                synced += 1

        db.commit()
    except Exception as e:
        raise HTTPException(500, f"Sync failed: {str(e)}")

    return {"synced": synced, "exchange": exchange}


# ─── Portfolio Route ──────────────────────────────────────────────────────────

@app.get("/portfolio", response_model=PortfolioStats)
async def get_portfolio(
    current_user: User = Depends(get_approved_user),
    db: Session = Depends(get_db)
):
    trades = db.query(Trade).filter(Trade.user_id == current_user.id).all()
    open_trades = [t for t in trades if t.status == TradeStatus.open]
    closed_trades = [t for t in trades if t.status == TradeStatus.closed]

    total_pnl = sum(t.pnl for t in trades)
    total_cost = sum(t.entry_price * t.quantity for t in trades)
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0

    winners = [t for t in closed_trades if t.pnl > 0]
    win_rate = (len(winners) / len(closed_trades) * 100) if closed_trades else 0.0
    pnls = [t.pnl for t in closed_trades]

    return PortfolioStats(
        total_trades=len(trades),
        open_trades=len(open_trades),
        closed_trades=len(closed_trades),
        total_pnl=round(total_pnl, 4),
        total_pnl_percent=round(total_pnl_pct, 4),
        win_rate=round(win_rate, 2),
        best_trade=max(pnls) if pnls else None,
        worst_trade=min(pnls) if pnls else None,
        paper_mode=current_user.paper_trading,
    )


# ─── Market Data ─────────────────────────────────────────────────────────────

@app.get("/market/prices")
async def market_prices():
    """Public endpoint: current prices + 24h stats."""
    symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
        "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    ]
    stats = await get_public_24h_stats(symbols)
    return stats


# ─── Admin Routes ─────────────────────────────────────────────────────────────

@app.get("/admin/users", response_model=List[UserOut])
async def admin_list_users(
    status_filter: Optional[str] = None,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    q = db.query(User)
    if status_filter:
        q = q.filter(User.status == status_filter)
    return q.order_by(User.created_at.desc()).all()


@app.patch("/admin/users/{user_id}", response_model=UserOut)
async def admin_update_user(
    user_id: int,
    data: AdminUserUpdate,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    if user.id == admin.id:
        raise HTTPException(400, "Cannot modify your own admin account")

    if data.status is not None:
        user.status = data.status
        await send_notification(
            user.id, "success" if data.status == UserStatus.approved else "warning",
            f"Your account has been {'approved' if data.status == UserStatus.approved else 'rejected'} by an admin."
        )
    if data.role is not None:
        user.role = data.role
    if data.is_active is not None:
        user.is_active = data.is_active

    db.commit()
    db.refresh(user)
    return user


@app.get("/admin/stats")
async def admin_stats(
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    total_users = db.query(User).count()
    pending = db.query(User).filter(User.status == UserStatus.pending).count()
    approved = db.query(User).filter(User.status == UserStatus.approved).count()
    total_trades = db.query(Trade).count()
    open_trades = db.query(Trade).filter(Trade.status == TradeStatus.open).count()
    paper_trades = db.query(Trade).filter(Trade.mode == TradeMode.paper).count()
    live_trades = db.query(Trade).filter(Trade.mode == TradeMode.live).count()

    return {
        "users": {"total": total_users, "pending": pending, "approved": approved},
        "trades": {
            "total": total_trades, "open": open_trades,
            "paper": paper_trades, "live": live_trades,
        },
        "connections": manager.total_connections,
    }


# ─── WebSocket ───────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = None):
    """
    Authenticated WebSocket for per-user updates.
    Connect with: ws://localhost:8000/ws?token=<JWT>
    """
    user_id = 0
    db = next(get_db())
    try:
        if token:
            from auth import decode_token
            token_data = decode_token(token)
            if token_data:
                user = db.query(User).filter(User.id == token_data.user_id).first()
                if user and user.is_active and user.status == UserStatus.approved:
                    user_id = user.id

        await manager.connect(websocket, user_id)

        # Send initial prices on connect
        from exchanges import get_public_24h_stats
        stats = await get_public_24h_stats([
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"
        ])
        await websocket.send_json({"type": "prices", "data": stats})

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                msg = __import__('json').loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif msg.get("type") == "subscribe" and user_id:
                    # Update paper trades on demand
                    updates = await update_paper_trades(db, user_id)
                    await websocket.send_json({"type": "trade_updates", "data": updates})
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"WS error: {e}")
    finally:
        manager.disconnect(websocket, user_id)
        db.close()


@app.websocket("/ws/prices")
async def price_feed(websocket: WebSocket):
    """Public WebSocket for real-time price feed (no auth required)."""
    await manager.connect_price_feed(websocket)
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=60)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect_price_feed(websocket)


# ─── Health Check ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "connections": manager.total_connections
    }
