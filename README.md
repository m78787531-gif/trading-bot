# TradeOS — Full-Stack Crypto Trading Platform

A production-ready trading platform with real-time WebSocket data, JWT authentication,
admin approval workflow, Binance/Bybit integration, paper trading mode, and PostgreSQL persistence.

---

## Architecture

```
trading-platform/
├── backend/                   # FastAPI Python backend
│   ├── main.py                # All routes (auth, trades, admin, WebSocket)
│   ├── models.py              # SQLAlchemy ORM models
│   ├── schemas.py             # Pydantic request/response schemas
│   ├── database.py            # PostgreSQL connection
│   ├── auth.py                # JWT authentication + role guards
│   ├── exchanges.py           # Binance + Bybit API clients
│   ├── paper_trading.py       # Simulated trade engine
│   ├── websocket_manager.py   # WebSocket broadcast system
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── TradingDashboard.jsx   # Full React dashboard
│   ├── src/
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── package.json
└── docker-compose.yml
```

---

## Quick Start (Docker — Recommended)

```bash
# 1. Clone and enter the project
cd trading-platform

# 2. Start everything (Postgres + Redis + Backend + Frontend)
docker compose up --build

# 3. Open the app
open http://localhost:3000

# Default admin credentials:
#   Username: admin
#   Password: Admin@1234
```

---

## Manual Setup

### Prerequisites
- Python 3.12+
- Node.js 20+
- PostgreSQL 15+

### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://trader:trader_pass@localhost:5432/tradingdb"
export JWT_SECRET_KEY="your-very-secret-key-change-this"

# Create the database
createdb tradingdb
# Or via psql:
# psql -U postgres -c "CREATE USER trader WITH PASSWORD 'trader_pass';"
# psql -U postgres -c "CREATE DATABASE tradingdb OWNER trader;"

# Start the backend (tables auto-created on first run)
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend

npm install
npm run dev
# Opens on http://localhost:3000
```

---

## API Reference

### Authentication
| Method | Path | Description |
|--------|------|-------------|
| POST | `/auth/register` | Register new user (status: pending) |
| POST | `/auth/login` | Login → returns JWT token |

### User
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/user` | ✓ | Get current user profile |
| PATCH | `/user` | ✓ | Update profile (paper_trading toggle) |

### API Keys
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/add-api` | approved | Add exchange API key (verified on save) |
| GET | `/api-keys` | approved | List your active API keys |
| DELETE | `/api-keys/{id}` | approved | Remove an API key |

### Trades
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/trades` | approved | List all your trades (filter by status/symbol) |
| GET | `/trade/{symbol}` | approved | Get trades for a specific symbol |
| POST | `/trades` | approved | Open a new trade (paper or live) |
| PATCH | `/trades/{id}/close` | approved | Close an open trade |
| GET | `/trades/sync/exchange` | approved | Pull live trades from exchange |

### Portfolio
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/portfolio` | approved | Win rate, PnL, trade stats |
| GET | `/market/prices` | public | Live prices + 24h stats |

### Admin
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/admin/users` | admin | List all users |
| PATCH | `/admin/users/{id}` | admin | Approve/reject/suspend users |
| GET | `/admin/stats` | admin | Platform-wide stats |

### WebSocket
| Endpoint | Auth | Description |
|----------|------|-------------|
| `ws://localhost:8000/ws?token=<JWT>` | Bearer token | Per-user real-time updates |
| `ws://localhost:8000/ws/prices` | None | Public price feed |

---

## WebSocket Message Types

```json
// Price update (broadcast every 3s)
{ "type": "prices", "data": { "BTCUSDT": { "price": 43250, "change": 1.23 } } }

// Trade update
{ "type": "trade_update", "trade_id": 42, "pnl": 123.4, "pnl_percent": 2.1, "status": "open" }

// Portfolio update
{ "type": "portfolio_update", "total_pnl": 1234.5, "total_pnl_percent": 3.2 }

// Notification
{ "type": "notification", "level": "success", "message": "Trade opened: BUY BTCUSDT" }
```

---

## User Flow

```
Register → pending
    ↓
Admin approves → approved
    ↓
User logs in → can trade

Paper mode: instant simulated fills, live Binance price feed
Live mode: real orders via exchange API (Binance/Bybit)
```

---

## Database Schema

```sql
-- users: id, email, username, hashed_password, role, status, paper_trading
-- api_keys: id, user_id, exchange, api_key, api_secret, is_testnet
-- trades: id, user_id, exchange, symbol, side, mode, status, entry_price,
--         current_price, exit_price, pnl, pnl_percent, leverage, sl, tp, fee
```

---

## Security Notes

> ⚠️ Before deploying to production:

1. **Change `JWT_SECRET_KEY`** to a strong random string (e.g. `openssl rand -hex 32`)
2. **Encrypt API secrets at rest** — add Fernet encryption in `exchanges.py`
3. **Change default admin password** immediately after first login
4. **Set `CORS` origins** in `main.py` to your actual domain only
5. **Use HTTPS** for all production traffic (reverse proxy with nginx + certbot)
6. **Rate limiting** — add `slowapi` or nginx rate limits on `/auth/*`

---

## Connecting Real Exchange APIs

### Binance
1. Go to [binance.com/en/my/settings/api-management](https://www.binance.com/en/my/settings/api-management)
2. Create API Key → enable "Read Info" + "Enable Spot & Margin Trading"
3. Whitelist your server IP
4. Enter key in Settings → Exchange API Keys

### Bybit
1. Go to [bybit.com/app/user/api-management](https://www.bybit.com/app/user/api-management)
2. Create key with "Read" + "Trade" permissions
3. Enter key in Settings → Exchange API Keys

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite, Tailwind CSS |
| Backend | FastAPI, Python 3.12 |
| Database | PostgreSQL 16 + SQLAlchemy ORM |
| Auth | JWT (python-jose) + bcrypt |
| Real-time | WebSocket (FastAPI native) |
| Exchange | Binance REST API, Bybit V5 API |
| Containers | Docker + Docker Compose |
