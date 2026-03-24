"""
INTEGRATION GUIDE — Add AI Bot to existing main.py
═══════════════════════════════════════════════════════════════════════════════

Make the following 4 changes to your existing backend/main.py.
Each change is marked with its exact location in the file.

───────────────────────────────────────────────────────────────────────────────
CHANGE 1 — Add imports (after existing imports, around line 35)
───────────────────────────────────────────────────────────────────────────────

Add these lines after your existing import block:

    # ── AI Bot Engine ──────────────────────────────────────────────────────
    from bot_routes import router as bot_router
    from ai.bot_models import BotConfig, SignalHistory, BotRunLog
    from ai.scheduler import start_scheduler, stop_scheduler


───────────────────────────────────────────────────────────────────────────────
CHANGE 2 — Register AI tables and start scheduler in lifespan() (around line 44)
───────────────────────────────────────────────────────────────────────────────

Inside the lifespan() function, AFTER the existing Base.metadata.create_all() call:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified")

        # ← ADD THESE LINES ↓
        from ai.bot_models import BotConfig, SignalHistory, BotRunLog
        from database import Base as AIBase
        AIBase.metadata.create_all(bind=engine)
        logger.info("AI bot tables created/verified")

        # Start background scheduler (daily reports, optimisation sweeps)
        await start_scheduler()
        logger.info("AI scheduler started")
        # ← ADD THESE LINES ↑

        # ... rest of existing lifespan code ...

        yield   # ← existing yield

        # ← ADD THIS LINE inside the shutdown section ↓
        stop_scheduler()
        logger.info("AI scheduler stopped")


───────────────────────────────────────────────────────────────────────────────
CHANGE 3 — Mount the bot router (after existing app = FastAPI(...) block)
───────────────────────────────────────────────────────────────────────────────

After the line:  app.add_middleware(CORSMiddleware, ...)

    # ← ADD THIS LINE ↓
    app.include_router(bot_router, prefix="/bot", tags=["AI Bot"])


───────────────────────────────────────────────────────────────────────────────
CHANGE 4 — Optional: add /bot/signal endpoint shortcut to existing routes
───────────────────────────────────────────────────────────────────────────────

Optionally, add this convenience route anywhere in main.py:

    @app.get("/market/signal/{symbol}")
    async def market_signal_quick(
        symbol: str,
        timeframe: str = "1h",
        current_user: User = Depends(get_approved_user),
    ):
        \"\"\"Quick AI signal for market data page.\"\"\"
        from ai.bot_engine import fetch_ohlcv
        from ai.signal_scorer import generate_signal
        from ai.sentiment import get_sentiment
        from ai.self_optimizer import get_params

        ohlcv = await fetch_ohlcv(symbol.upper(), "binance", timeframe, 200)
        if not ohlcv:
            raise HTTPException(400, "Could not fetch candles")
        sentiment = await get_sentiment(symbol.upper())
        signal = generate_signal(
            symbol=symbol.upper(), timeframe=timeframe,
            opens=ohlcv["opens"], highs=ohlcv["highs"],
            lows=ohlcv["lows"], closes=ohlcv["closes"],
            volumes=ohlcv["volumes"],
            sentiment_score=sentiment.composite_score,
            params=get_params(),
        )
        return {
            "direction": signal.direction,
            "confidence": signal.confidence,
            "entry": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "rr": signal.risk_reward,
            "reasoning": signal.reasoning[:3],
        }


═══════════════════════════════════════════════════════════════════════════════
ENVIRONMENT VARIABLES — add to your .env or docker-compose.yml
═══════════════════════════════════════════════════════════════════════════════

# Telegram Alerts
TELEGRAM_BOT_TOKEN=your_bot_token_from_botfather
TELEGRAM_CHAT_ID=your_chat_id
TELEGRAM_ADMIN_CHAT_ID=your_admin_chat_id   # optional, defaults to TELEGRAM_CHAT_ID

# Sentiment APIs (all optional — gracefully degrade if missing)
TWITTER_BEARER_TOKEN=your_twitter_v2_bearer_token
CRYPTOPANIC_API_KEY=your_cryptopanic_api_key   # free at cryptopanic.com

# Reddit (optional — public API works without credentials for basic use)
REDDIT_CLIENT_ID=your_reddit_app_client_id
REDDIT_SECRET=your_reddit_app_secret


═══════════════════════════════════════════════════════════════════════════════
NEW API ENDPOINTS SUMMARY
═══════════════════════════════════════════════════════════════════════════════

All routes under /bot — require JWT (approved user):

  POST  /bot/start           Start the AI bot
  POST  /bot/stop            Stop the AI bot
  GET   /bot/status          Session status + config
  POST  /bot/signal          Analyse a symbol (no trade placed)
  GET   /bot/signal/{sym}    Quick signal check
  GET   /bot/signals         Signal history
  GET   /bot/performance     Win rate, PnL, AI params
  POST  /bot/optimise        Trigger self-optimisation manually
  PATCH /bot/params          Override AI parameters
  GET   /bot/config          Get full bot config
  PATCH /bot/config          Update bot config
  GET   /bot/logs            Bot run logs
  POST  /bot/cycle           Manually trigger one cycle (debug)

Admin only:
  GET   /bot/admin/sessions  All active bot sessions
  POST  /bot/admin/stop/{id} Force-stop a user's bot


═══════════════════════════════════════════════════════════════════════════════
EXAMPLE: Start Bot via API
═══════════════════════════════════════════════════════════════════════════════

# 1. Login and get JWT
curl -X POST http://localhost:8000/auth/login \\
  -d "username=alice&password=Alice@1234"

# 2. Start bot (paper mode, 1h timeframe, 5-minute scan interval)
curl -X POST http://localhost:8000/bot/start \\
  -H "Authorization: Bearer <JWT>" \\
  -H "Content-Type: application/json" \\
  -d '{
    "exchange": "binance",
    "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "timeframe": "1h",
    "interval_seconds": 300,
    "paper_mode": true,
    "max_risk_per_trade_pct": 1.5,
    "max_open_positions": 5,
    "max_daily_loss_pct": 5.0
  }'

# 3. Get a signal for BTCUSDT right now
curl -X POST http://localhost:8000/bot/signal \\
  -H "Authorization: Bearer <JWT>" \\
  -H "Content-Type: application/json" \\
  -d '{"symbol": "BTCUSDT", "timeframe": "4h"}'

# 4. View performance
curl -X GET http://localhost:8000/bot/performance \\
  -H "Authorization: Bearer <JWT>"

# 5. Stop bot
curl -X POST http://localhost:8000/bot/stop \\
  -H "Authorization: Bearer <JWT>"
"""
