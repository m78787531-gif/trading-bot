# TradeOS — Quick Start Guide

## Prerequisites
- Docker + Docker Compose
- Git (to clone, if applicable)

## 1. Configure Environment

Copy the example env file and fill in your values:

```bash
cp .env.example .env
# Edit .env with your Telegram token, DB credentials, etc.
```

**Minimum required** (for paper trading):
- `DATABASE_URL` — already set in docker-compose.yml
- `SECRET_KEY`   — change to any random string

**Optional** (for Telegram alerts):
- `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`

**Only needed for live trading**:
- Exchange API keys (Binance or Bybit)

## 2. Start the Platform

```bash
docker compose up --build
```

Services start at:
- Frontend:  http://localhost:3000
- Backend:   http://localhost:8000
- API Docs:  http://localhost:8000/docs

Default admin login: `admin` / `Admin@1234`
**Change this immediately in production.**

## 3. Start the AI Bot (Paper Mode)

```bash
# Login first
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=Admin@1234"

# Start bot in paper mode
curl -X POST http://localhost:8000/bot/start \
  -H "Authorization: Bearer <YOUR_JWT>" \
  -H "Content-Type: application/json" \
  -d '{
    "exchange": "binance",
    "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT"],
    "timeframe": "1h",
    "interval_seconds": 300,
    "paper_mode": true
  }'
```

## 4. Monitor the Bot

```bash
# Full status
GET /bot/status

# Current trading mode + daily counters
GET /bot/mode

# Drawdown protection dashboard
GET /bot/drawdown

# Trade quality + session filter
GET /bot/quality

# Signal history
GET /bot/signals

# Performance report
GET /bot/performance
```

## 5. Switch to Live Trading

After verifying paper results:

```bash
# Add exchange API key
POST /add-api  {"exchange": "binance", "api_key": "...", "api_secret": "..."}

# Request live mode (safety gate checks automatically)
POST /bot/mode/live  {"market_quality_override": null}
```

The safety gate requires ALL three:
1. Daily drawdown < 3%
2. Consecutive losses < 3  
3. Market quality score ≥ 0.70

## 6. Run Tests

```bash
cd backend
pip install pytest pytest-asyncio --break-system-packages
pytest tests/test_ai_engine.py -v --tb=short
```

## Protection Layers (automatic, no config needed)

| Trigger | Effect |
|---------|--------|
| 3 consecutive losses | 2-hour trading pause |
| 5 live trades in cycle | Auto-switch to paper; reset manually |
| Daily losses ≥ 9 trades | Paper mode for rest of day |
| Daily drawdown ≥ 3% | Paper mode for rest of day |
| Session drawdown ≥ 5% | Position size halved |
| Session drawdown ≥ 10% | Kill switch — admin reset required |
| Loss day | Recovery mode: 0.5% risk, restored +0.05%/win |

## Signal Entry Requirements (all must pass)

1. SMC: BOS/CHoCH + active Order Block or Fair Value Gap
2. EMA 200 trend alignment (above for long, below for short)
3. RSI in 40–60 zone
4. MACD crossover confirmation
5. Volume spike ≥ 1.5× average
6. Price retesting OB/FVG (not chasing breakouts)
7. Market structure: HH+HL for long; LH+LL for short
8. Trade during London (07–10 UTC) or NY (13–16 UTC) session
9. Trade quality score ≥ 0.70 (RR/structure/trend/ATR composite)

## Architecture

```
backend/
  main.py               FastAPI app entry point
  bot_routes.py         AI bot API endpoints (/bot/*)
  ai/
    signal_scorer.py    Signal generation + all 9 entry filters
    risk_manager.py     Risk state machine + position sizing
    bot_engine.py       Cycle orchestration + execution
    self_optimizer.py   Loss-learning (50 trades min, rate=0.01)
    smc_engine.py       Smart Money Concepts detection
    indicators.py       RSI, MACD, EMA, BB, ATR, ADX, StochRSI
    sentiment.py        Twitter + Reddit + CryptoPanic
    alerts.py           Telegram bot integration
    scheduler.py        Daily reports + auto-optimisation
  tests/
    test_ai_engine.py   ~80 tests across 20 test classes

frontend/
  TradingDashboard.jsx  React dashboard
  
docker-compose.yml      PostgreSQL + Redis + Backend + Frontend
```
