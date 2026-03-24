import { useState, useEffect, useRef, useCallback } from "react";

const API = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws";

// ─── API Client ──────────────────────────────────────────────────────────────

function useApi() {
  const token = localStorage.getItem("token");
  const headers = {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };

  const request = async (method, path, body = null) => {
    const res = await fetch(`${API}${path}`, {
      method,
      headers,
      body: body ? JSON.stringify(body) : null,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");
    return data;
  };

  return {
    get: (path) => request("GET", path),
    post: (path, body) => request("POST", path, body),
    patch: (path, body) => request("PATCH", path, body),
    del: (path) => request("DELETE", path),
  };
}

// ─── WebSocket Hook ──────────────────────────────────────────────────────────

function useWebSocket(token, onMessage) {
  const ws = useRef(null);
  const reconnectTimer = useRef(null);

  const connect = useCallback(() => {
    const url = token ? `${WS_URL}?token=${token}` : `${WS_URL}`;
    ws.current = new WebSocket(url);

    ws.current.onopen = () => {
      console.log("WS connected");
      ws.current.send(JSON.stringify({ type: "subscribe" }));
    };

    ws.current.onmessage = (e) => {
      try { onMessage(JSON.parse(e.data)); } catch {}
    };

    ws.current.onclose = () => {
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    ws.current.onerror = () => ws.current?.close();
  }, [token, onMessage]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      ws.current?.close();
    };
  }, [connect]);

  const send = (msg) => ws.current?.readyState === 1 && ws.current.send(JSON.stringify(msg));
  return { send };
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

const fmt = {
  usd: (n) => (n == null ? "—" : `$${parseFloat(n).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`),
  pct: (n) => (n == null ? "—" : `${n >= 0 ? "+" : ""}${parseFloat(n).toFixed(2)}%`),
  date: (d) => d ? new Date(d).toLocaleString() : "—",
  sym: (s) => s?.replace("USDT", "") ?? s,
};

const cl = (...args) => args.filter(Boolean).join(" ");

// ─── Components ──────────────────────────────────────────────────────────────

function Badge({ children, variant = "default" }) {
  const styles = {
    default: "bg-slate-700 text-slate-200",
    success: "bg-emerald-900/60 text-emerald-300 border border-emerald-700/50",
    danger: "bg-red-900/60 text-red-300 border border-red-700/50",
    warning: "bg-amber-900/60 text-amber-300 border border-amber-700/50",
    info: "bg-blue-900/60 text-blue-300 border border-blue-700/50",
    paper: "bg-violet-900/60 text-violet-300 border border-violet-700/50",
  };
  return (
    <span className={cl("px-2 py-0.5 rounded text-xs font-mono font-semibold", styles[variant] || styles.default)}>
      {children}
    </span>
  );
}

function StatCard({ label, value, sub, trend, color = "default" }) {
  const colors = {
    default: "text-slate-100",
    green: "text-emerald-400",
    red: "text-red-400",
    blue: "text-blue-400",
    violet: "text-violet-400",
  };
  return (
    <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4 backdrop-blur">
      <p className="text-xs text-slate-400 font-mono uppercase tracking-widest mb-1">{label}</p>
      <p className={cl("text-2xl font-bold font-mono", colors[color])}>{value}</p>
      {sub && <p className="text-xs text-slate-500 mt-1">{sub}</p>}
    </div>
  );
}

function Spinner() {
  return (
    <div className="flex items-center justify-center h-32">
      <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
    </div>
  );
}

function Toast({ toasts, remove }) {
  const colors = {
    success: "border-emerald-500/50 bg-emerald-950/90 text-emerald-300",
    error: "border-red-500/50 bg-red-950/90 text-red-300",
    warning: "border-amber-500/50 bg-amber-950/90 text-amber-300",
    info: "border-blue-500/50 bg-blue-950/90 text-blue-300",
  };
  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 pointer-events-none">
      {toasts.map((t) => (
        <div key={t.id} className={cl("border rounded-lg px-4 py-3 text-sm font-mono shadow-xl backdrop-blur pointer-events-auto flex items-center gap-3 animate-in slide-in-from-right", colors[t.level] || colors.info)}>
          <span>{t.message}</span>
          <button onClick={() => remove(t.id)} className="ml-auto opacity-60 hover:opacity-100">✕</button>
        </div>
      ))}
    </div>
  );
}

// ─── Auth Screen ─────────────────────────────────────────────────────────────

function AuthScreen({ onAuth }) {
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({ username: "", password: "", email: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setError(""); setLoading(true);
    try {
      if (mode === "login") {
        const body = new URLSearchParams({ username: form.username, password: form.password });
        const res = await fetch(`${API}/auth/login`, {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body,
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail);
        localStorage.setItem("token", data.access_token);
        onAuth(data.access_token);
      } else {
        const res = await fetch(`${API}/auth/register`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(form),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail);
        setMode("login");
        setError("Registered! Awaiting admin approval. Then log in.");
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center p-4" style={{ backgroundImage: "radial-gradient(ellipse at 50% 0%, #0f3460 0%, transparent 60%)" }}>
      <div className="w-full max-w-sm">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 mb-3">
            <div className="w-8 h-8 rounded-lg bg-cyan-500 flex items-center justify-center text-slate-950 font-black text-lg">T</div>
            <span className="text-xl font-bold text-slate-100 tracking-tight">TradeOS</span>
          </div>
          <p className="text-slate-400 text-sm">Professional Crypto Trading Platform</p>
        </div>

        <div className="bg-slate-800/80 border border-slate-700/50 rounded-2xl p-6 backdrop-blur shadow-2xl">
          <div className="flex gap-1 mb-6 bg-slate-900/60 rounded-lg p-1">
            {["login", "register"].map((m) => (
              <button key={m} onClick={() => setMode(m)}
                className={cl("flex-1 py-2 rounded-md text-sm font-semibold transition-all capitalize",
                  mode === m ? "bg-cyan-500 text-slate-950" : "text-slate-400 hover:text-slate-200")}>
                {m}
              </button>
            ))}
          </div>

          <form onSubmit={submit} className="space-y-3">
            {mode === "register" && (
              <input type="email" placeholder="Email" value={form.email}
                onChange={(e) => setForm({ ...form, email: e.target.value })}
                className="w-full bg-slate-900/60 border border-slate-700 rounded-lg px-3 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition-colors" required />
            )}
            <input type="text" placeholder="Username" value={form.username}
              onChange={(e) => setForm({ ...form, username: e.target.value })}
              className="w-full bg-slate-900/60 border border-slate-700 rounded-lg px-3 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition-colors" required />
            <input type="password" placeholder="Password" value={form.password}
              onChange={(e) => setForm({ ...form, password: e.target.value })}
              className="w-full bg-slate-900/60 border border-slate-700 rounded-lg px-3 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition-colors" required />

            {error && <p className={cl("text-xs rounded-lg p-2 font-mono", error.includes("Registered") ? "bg-emerald-900/40 text-emerald-400" : "bg-red-900/40 text-red-400")}>{error}</p>}

            <button type="submit" disabled={loading}
              className="w-full bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold py-2.5 rounded-lg transition-colors disabled:opacity-50">
              {loading ? "..." : mode === "login" ? "Sign In" : "Create Account"}
            </button>
          </form>

          {mode === "login" && (
            <p className="text-center text-xs text-slate-500 mt-4">
              Default admin: <span className="text-slate-300 font-mono">admin / Admin@1234</span>
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── API Keys Modal ───────────────────────────────────────────────────────────

function APIKeyModal({ onClose, onSaved, api }) {
  const [form, setForm] = useState({ exchange: "binance", api_key: "", api_secret: "", is_testnet: false, label: "" });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const submit = async (e) => {
    e.preventDefault();
    setError(""); setLoading(true);
    try {
      await api.post("/add-api", form);
      onSaved();
      onClose();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-800 border border-slate-700 rounded-2xl w-full max-w-md shadow-2xl">
        <div className="flex items-center justify-between p-5 border-b border-slate-700">
          <h2 className="text-lg font-bold text-slate-100">Connect Exchange API</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-200 text-xl">✕</button>
        </div>
        <form onSubmit={submit} className="p-5 space-y-4">
          <div className="flex gap-2">
            {["binance", "bybit"].map((ex) => (
              <button key={ex} type="button" onClick={() => setForm({ ...form, exchange: ex })}
                className={cl("flex-1 py-2.5 rounded-lg text-sm font-semibold capitalize transition-all border",
                  form.exchange === ex ? "bg-cyan-500/20 border-cyan-500 text-cyan-300" : "border-slate-600 text-slate-400 hover:border-slate-500")}>
                {ex}
              </button>
            ))}
          </div>
          <input placeholder="Label (e.g. Main Account)" value={form.label}
            onChange={(e) => setForm({ ...form, label: e.target.value })}
            className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500" />
          <input placeholder="API Key" value={form.api_key}
            onChange={(e) => setForm({ ...form, api_key: e.target.value })}
            className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-mono" required />
          <input type="password" placeholder="API Secret" value={form.api_secret}
            onChange={(e) => setForm({ ...form, api_secret: e.target.value })}
            className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-mono" required />
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={form.is_testnet} onChange={(e) => setForm({ ...form, is_testnet: e.target.checked })}
              className="w-4 h-4 rounded" />
            <span className="text-sm text-slate-300">Use Testnet</span>
          </label>
          {error && <p className="text-xs text-red-400 bg-red-900/30 rounded-lg p-2 font-mono">{error}</p>}
          <button type="submit" disabled={loading}
            className="w-full bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold py-2.5 rounded-lg transition-colors disabled:opacity-50">
            {loading ? "Verifying..." : "Connect Exchange"}
          </button>
        </form>
      </div>
    </div>
  );
}

// ─── Open Trade Modal ─────────────────────────────────────────────────────────

function OpenTradeModal({ onClose, onOpened, api, user, prices }) {
  const symbols = Object.keys(prices).length > 0 ? Object.keys(prices) : ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"];
  const [form, setForm] = useState({
    exchange: "binance", symbol: symbols[0],
    side: "buy", quantity: "", entry_price: "",
    leverage: "1", stop_loss: "", take_profit: "",
    mode: user?.paper_trading ? "paper" : "live", notes: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Auto-fill price
  useEffect(() => {
    const p = prices[form.symbol];
    if (p) setForm((f) => ({ ...f, entry_price: p.price?.toFixed(2) || "" }));
  }, [form.symbol, prices]);

  const submit = async (e) => {
    e.preventDefault();
    setError(""); setLoading(true);
    try {
      const body = {
        ...form,
        quantity: parseFloat(form.quantity),
        entry_price: parseFloat(form.entry_price),
        leverage: parseFloat(form.leverage),
        stop_loss: form.stop_loss ? parseFloat(form.stop_loss) : null,
        take_profit: form.take_profit ? parseFloat(form.take_profit) : null,
      };
      await api.post("/trades", body);
      onOpened();
      onClose();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const isBuy = form.side === "buy";

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-800 border border-slate-700 rounded-2xl w-full max-w-lg shadow-2xl max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between p-5 border-b border-slate-700 sticky top-0 bg-slate-800">
          <h2 className="text-lg font-bold text-slate-100">Open Trade</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-200 text-xl">✕</button>
        </div>
        <form onSubmit={submit} className="p-5 space-y-4">
          {/* Mode */}
          <div className="flex gap-2">
            {["paper", "live"].map((m) => (
              <button key={m} type="button" onClick={() => setForm({ ...form, mode: m })}
                className={cl("flex-1 py-2 rounded-lg text-sm font-semibold capitalize transition-all border",
                  form.mode === m
                    ? m === "paper" ? "bg-violet-500/20 border-violet-500 text-violet-300" : "bg-cyan-500/20 border-cyan-500 text-cyan-300"
                    : "border-slate-600 text-slate-400 hover:border-slate-500")}>
                {m === "paper" ? "📄 Paper" : "⚡ Live"}
              </button>
            ))}
          </div>

          {/* Exchange + Symbol */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-slate-400 block mb-1">Exchange</label>
              <select value={form.exchange} onChange={(e) => setForm({ ...form, exchange: e.target.value })}
                className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-cyan-500">
                <option value="binance">Binance</option>
                <option value="bybit">Bybit</option>
              </select>
            </div>
            <div>
              <label className="text-xs text-slate-400 block mb-1">Symbol</label>
              <select value={form.symbol} onChange={(e) => setForm({ ...form, symbol: e.target.value })}
                className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-cyan-500 font-mono">
                {symbols.map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
          </div>

          {/* Side */}
          <div className="flex gap-2">
            {["buy", "sell"].map((s) => (
              <button key={s} type="button" onClick={() => setForm({ ...form, side: s })}
                className={cl("flex-1 py-2.5 rounded-lg text-sm font-bold uppercase transition-all",
                  form.side === s
                    ? s === "buy" ? "bg-emerald-500 text-white" : "bg-red-500 text-white"
                    : "bg-slate-700 text-slate-400 hover:bg-slate-600")}>
                {s === "buy" ? "▲ Long / Buy" : "▼ Short / Sell"}
              </button>
            ))}
          </div>

          {/* Price + Qty */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-slate-400 block mb-1">Entry Price (USDT)</label>
              <input type="number" step="any" placeholder="0.00" value={form.entry_price}
                onChange={(e) => setForm({ ...form, entry_price: e.target.value })}
                className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-mono" required />
            </div>
            <div>
              <label className="text-xs text-slate-400 block mb-1">Quantity</label>
              <input type="number" step="any" placeholder="0.00" value={form.quantity}
                onChange={(e) => setForm({ ...form, quantity: e.target.value })}
                className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-mono" required />
            </div>
          </div>

          {/* Leverage */}
          <div>
            <label className="text-xs text-slate-400 block mb-1">Leverage: {form.leverage}x</label>
            <input type="range" min="1" max="100" step="1" value={form.leverage}
              onChange={(e) => setForm({ ...form, leverage: e.target.value })}
              className="w-full accent-cyan-500" />
            <div className="flex justify-between text-xs text-slate-500 mt-1">
              <span>1x</span><span>25x</span><span>50x</span><span>100x</span>
            </div>
          </div>

          {/* SL / TP */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-red-400 block mb-1">Stop Loss</label>
              <input type="number" step="any" placeholder="Optional" value={form.stop_loss}
                onChange={(e) => setForm({ ...form, stop_loss: e.target.value })}
                className="w-full bg-slate-900 border border-red-900/50 rounded-lg px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-red-500 font-mono" />
            </div>
            <div>
              <label className="text-xs text-emerald-400 block mb-1">Take Profit</label>
              <input type="number" step="any" placeholder="Optional" value={form.take_profit}
                onChange={(e) => setForm({ ...form, take_profit: e.target.value })}
                className="w-full bg-slate-900 border border-emerald-900/50 rounded-lg px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500 font-mono" />
            </div>
          </div>

          {error && <p className="text-xs text-red-400 bg-red-900/30 rounded-lg p-2 font-mono">{error}</p>}

          <button type="submit" disabled={loading}
            className={cl("w-full font-bold py-3 rounded-xl transition-colors text-sm disabled:opacity-50",
              isBuy ? "bg-emerald-500 hover:bg-emerald-400 text-white" : "bg-red-500 hover:bg-red-400 text-white")}>
            {loading ? "Placing Order..." : `${isBuy ? "Buy / Long" : "Sell / Short"} ${form.symbol}`}
          </button>
        </form>
      </div>
    </div>
  );
}

// ─── Admin Panel ──────────────────────────────────────────────────────────────

function AdminPanel({ api, addToast }) {
  const [users, setUsers] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    try {
      const [u, s] = await Promise.all([api.get("/admin/users"), api.get("/admin/stats")]);
      setUsers(u);
      setStats(s);
    } catch (e) {
      addToast("error", e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const updateUser = async (id, data) => {
    try {
      await api.patch(`/admin/users/${id}`, data);
      addToast("success", "User updated");
      load();
    } catch (e) {
      addToast("error", e.message);
    }
  };

  if (loading) return <Spinner />;

  const statusColor = { pending: "warning", approved: "success", rejected: "danger" };

  return (
    <div className="space-y-6">
      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard label="Total Users" value={stats.users.total} color="blue" />
          <StatCard label="Pending" value={stats.users.pending} color="violet" sub="Awaiting approval" />
          <StatCard label="Total Trades" value={stats.trades.total} color="default" />
          <StatCard label="Live Connections" value={stats.connections} color="green" />
        </div>
      )}

      {/* Users Table */}
      <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl overflow-hidden">
        <div className="p-4 border-b border-slate-700/50">
          <h3 className="font-bold text-slate-100">User Management</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-slate-400 font-mono uppercase border-b border-slate-700/50">
                <th className="px-4 py-3">User</th>
                <th className="px-4 py-3">Role</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3">Joined</th>
                <th className="px-4 py-3">Actions</th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.id} className="border-b border-slate-700/30 hover:bg-slate-700/20 transition-colors">
                  <td className="px-4 py-3">
                    <div className="font-semibold text-slate-200">{u.username}</div>
                    <div className="text-xs text-slate-500">{u.email}</div>
                  </td>
                  <td className="px-4 py-3"><Badge variant={u.role === "admin" ? "warning" : "default"}>{u.role}</Badge></td>
                  <td className="px-4 py-3"><Badge variant={statusColor[u.status]}>{u.status}</Badge></td>
                  <td className="px-4 py-3 text-slate-400 text-xs font-mono">{fmt.date(u.created_at)}</td>
                  <td className="px-4 py-3">
                    <div className="flex gap-2">
                      {u.status === "pending" && (
                        <>
                          <button onClick={() => updateUser(u.id, { status: "approved" })}
                            className="px-2 py-1 bg-emerald-600 hover:bg-emerald-500 text-white rounded text-xs font-semibold transition-colors">✓ Approve</button>
                          <button onClick={() => updateUser(u.id, { status: "rejected" })}
                            className="px-2 py-1 bg-red-600 hover:bg-red-500 text-white rounded text-xs font-semibold transition-colors">✗ Reject</button>
                        </>
                      )}
                      {u.status === "approved" && (
                        <button onClick={() => updateUser(u.id, { is_active: !u.is_active })}
                          className={cl("px-2 py-1 rounded text-xs font-semibold transition-colors", u.is_active ? "bg-amber-600 hover:bg-amber-500 text-white" : "bg-slate-600 hover:bg-slate-500 text-white")}>
                          {u.is_active ? "Suspend" : "Activate"}
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ─── Trades Table ─────────────────────────────────────────────────────────────

function TradesTable({ trades, onClose, onRefresh }) {
  const api = useApi();
  const [closingId, setClosingId] = useState(null);

  const closeTrade = async (trade) => {
    const price = prompt(`Close ${trade.symbol} at price:`, trade.current_price);
    if (!price) return;
    setClosingId(trade.id);
    try {
      await api.patch(`/trades/${trade.id}/close`, { exit_price: parseFloat(price) });
      onRefresh();
    } catch (e) {
      alert(e.message);
    } finally {
      setClosingId(null);
    }
  };

  if (!trades.length) {
    return (
      <div className="text-center py-16 text-slate-500">
        <div className="text-4xl mb-3">📊</div>
        <p className="font-mono">No trades yet. Open your first trade!</p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs text-slate-400 font-mono uppercase border-b border-slate-700/50">
            <th className="px-4 py-3">Symbol</th>
            <th className="px-4 py-3">Side</th>
            <th className="px-4 py-3">Mode</th>
            <th className="px-4 py-3">Entry</th>
            <th className="px-4 py-3">Current</th>
            <th className="px-4 py-3">PnL</th>
            <th className="px-4 py-3">Status</th>
            <th className="px-4 py-3">Opened</th>
            <th className="px-4 py-3">Action</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((t) => {
            const pnlPos = t.pnl >= 0;
            return (
              <tr key={t.id} className="border-b border-slate-700/30 hover:bg-slate-700/20 transition-colors">
                <td className="px-4 py-3">
                  <span className="font-bold text-slate-100 font-mono">{fmt.sym(t.symbol)}</span>
                  <span className="text-slate-500 font-mono text-xs">/USDT</span>
                  {t.leverage > 1 && <Badge variant="info">{t.leverage}x</Badge>}
                </td>
                <td className="px-4 py-3">
                  <Badge variant={t.side === "buy" ? "success" : "danger"}>{t.side === "buy" ? "▲ Long" : "▼ Short"}</Badge>
                </td>
                <td className="px-4 py-3">
                  <Badge variant={t.mode === "paper" ? "paper" : "info"}>{t.mode}</Badge>
                </td>
                <td className="px-4 py-3 font-mono text-slate-300">{fmt.usd(t.entry_price)}</td>
                <td className="px-4 py-3 font-mono text-slate-300">{fmt.usd(t.current_price || t.exit_price)}</td>
                <td className="px-4 py-3 font-mono">
                  <div className={pnlPos ? "text-emerald-400" : "text-red-400"}>
                    {fmt.usd(t.pnl)}
                  </div>
                  <div className={cl("text-xs", pnlPos ? "text-emerald-500" : "text-red-500")}>
                    {fmt.pct(t.pnl_percent)}
                  </div>
                </td>
                <td className="px-4 py-3">
                  <Badge variant={t.status === "open" ? "success" : t.status === "closed" ? "default" : "danger"}>
                    {t.status}
                  </Badge>
                </td>
                <td className="px-4 py-3 text-slate-500 text-xs font-mono">{fmt.date(t.opened_at)}</td>
                <td className="px-4 py-3">
                  {t.status === "open" && (
                    <button onClick={() => closeTrade(t)} disabled={closingId === t.id}
                      className="px-2 py-1 bg-red-600 hover:bg-red-500 text-white rounded text-xs font-semibold transition-colors disabled:opacity-50">
                      {closingId === t.id ? "..." : "Close"}
                    </button>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ─── Market Ticker ────────────────────────────────────────────────────────────

function MarketTicker({ prices }) {
  const entries = Object.entries(prices).slice(0, 8);
  if (!entries.length) return null;

  return (
    <div className="flex gap-4 overflow-x-auto pb-1 scrollbar-hide">
      {entries.map(([sym, data]) => (
        <div key={sym} className="flex-shrink-0 bg-slate-800/60 border border-slate-700/50 rounded-xl px-4 py-3 min-w-[140px]">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-mono font-bold text-slate-300">{fmt.sym(sym)}</span>
            <span className={cl("text-xs font-mono font-bold", data.change >= 0 ? "text-emerald-400" : "text-red-400")}>
              {fmt.pct(data.change)}
            </span>
          </div>
          <div className="text-sm font-mono font-bold text-slate-100">{fmt.usd(data.price)}</div>
          <div className="text-xs text-slate-500 font-mono mt-0.5">Vol: {(data.volume / 1e6).toFixed(1)}M</div>
        </div>
      ))}
    </div>
  );
}

// ─── Main Dashboard ───────────────────────────────────────────────────────────

export default function TradingDashboard() {
  const [token, setToken] = useState(localStorage.getItem("token"));
  const [user, setUser] = useState(null);
  const [trades, setTrades] = useState([]);
  const [portfolio, setPortfolio] = useState(null);
  const [prices, setPrices] = useState({});
  const [apiKeys, setApiKeys] = useState([]);
  const [tab, setTab] = useState("dashboard");
  const [showAPIModal, setShowAPIModal] = useState(false);
  const [showTradeModal, setShowTradeModal] = useState(false);
  const [loading, setLoading] = useState(true);
  const [toasts, setToasts] = useState([]);
  const [wsConnected, setWsConnected] = useState(false);

  const api = useApi();

  const addToast = useCallback((level, message) => {
    const id = Date.now();
    setToasts((t) => [...t, { id, level, message }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 5000);
  }, []);

  const removeToast = (id) => setToasts((t) => t.filter((x) => x.id !== id));

  // WS message handler
  const handleWsMessage = useCallback((msg) => {
    if (msg.type === "prices" && msg.data) {
      setPrices(msg.data);
      // Update current prices in open trades
      setTrades((prev) => prev.map((t) => {
        const p = msg.data[t.symbol];
        if (!p || t.status !== "open") return t;
        const pnl = t.side === "buy"
          ? (p.price - t.entry_price) * t.quantity * t.leverage
          : (t.entry_price - p.price) * t.quantity * t.leverage;
        const pnlPct = (pnl / (t.entry_price * t.quantity)) * 100;
        return { ...t, current_price: p.price, pnl: +pnl.toFixed(4), pnl_percent: +pnlPct.toFixed(4) };
      }));
    }
    if (msg.type === "notification") {
      addToast(msg.level, msg.message);
    }
    if (msg.type === "trade_update") {
      setTrades((prev) => prev.map((t) =>
        t.id === msg.trade_id ? { ...t, ...msg } : t
      ));
    }
  }, [addToast]);

  const { send } = useWebSocket(token, handleWsMessage);

  // Load data
  const loadData = useCallback(async () => {
    if (!token) return;
    try {
      const [u, t, p, k] = await Promise.all([
        api.get("/user"),
        api.get("/trades"),
        api.get("/portfolio"),
        api.get("/api-keys"),
      ]);
      setUser(u);
      setTrades(t);
      setPortfolio(p);
      setApiKeys(k);
    } catch (e) {
      if (e.message.includes("Invalid") || e.message.includes("401")) {
        logout();
      }
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => { loadData(); }, [loadData]);

  const logout = () => {
    localStorage.removeItem("token");
    setToken(null);
    setUser(null);
  };

  if (!token) return <AuthScreen onAuth={(t) => { setToken(t); window.location.reload(); }} />;
  if (loading) return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center">
      <div className="text-center">
        <div className="w-10 h-10 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
        <p className="text-slate-400 font-mono text-sm">Connecting to trading system...</p>
      </div>
    </div>
  );

  const isPending = user?.status === "pending";
  const isAdmin = user?.role === "admin";
  const totalPnl = portfolio?.total_pnl ?? 0;
  const openTrades = trades.filter((t) => t.status === "open");

  const tabs = [
    { id: "dashboard", label: "Dashboard" },
    { id: "trades", label: `Trades${openTrades.length > 0 ? ` (${openTrades.length})` : ""}` },
    { id: "history", label: "History" },
    { id: "settings", label: "Settings" },
    ...(isAdmin ? [{ id: "admin", label: "⚡ Admin" }] : []),
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100" style={{ fontFamily: "'JetBrains Mono', 'Fira Code', monospace" }}>
      <Toast toasts={toasts} remove={removeToast} />

      {/* Modals */}
      {showAPIModal && <APIKeyModal onClose={() => setShowAPIModal(false)} onSaved={loadData} api={api} />}
      {showTradeModal && user?.status === "approved" && (
        <OpenTradeModal onClose={() => setShowTradeModal(false)} onOpened={loadData} api={api} user={user} prices={prices} />
      )}

      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-950/90 backdrop-blur sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-md bg-cyan-500 flex items-center justify-center text-slate-950 font-black text-sm">T</div>
              <span className="font-bold text-slate-100 tracking-tight">TradeOS</span>
            </div>
            <nav className="hidden md:flex gap-1">
              {tabs.map((t) => (
                <button key={t.id} onClick={() => setTab(t.id)}
                  className={cl("px-3 py-1.5 rounded-lg text-xs font-semibold transition-all",
                    tab === t.id ? "bg-slate-700 text-slate-100" : "text-slate-400 hover:text-slate-200")}>
                  {t.label}
                </button>
              ))}
            </nav>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <div className={cl("w-1.5 h-1.5 rounded-full", Object.keys(prices).length > 0 ? "bg-emerald-500 animate-pulse" : "bg-red-500")} />
              <span className="text-xs text-slate-500">{Object.keys(prices).length > 0 ? "Live" : "Offline"}</span>
            </div>
            {user?.paper_trading && <Badge variant="paper">Paper Mode</Badge>}
            <div className="text-xs text-slate-400 font-mono">{user?.username}</div>
            <button onClick={logout} className="px-2 py-1 text-xs text-slate-400 hover:text-red-400 transition-colors">Logout</button>
          </div>
        </div>
      </header>

      {/* Pending Banner */}
      {isPending && (
        <div className="bg-amber-900/40 border-b border-amber-700/50 text-amber-300 text-sm text-center py-2.5 font-mono">
          ⏳ Your account is <strong>pending admin approval</strong>. You can browse but cannot trade yet.
        </div>
      )}

      <main className="max-w-7xl mx-auto px-4 py-6">
        {/* Market Ticker */}
        {Object.keys(prices).length > 0 && (
          <div className="mb-6">
            <MarketTicker prices={prices} />
          </div>
        )}

        {/* Dashboard Tab */}
        {tab === "dashboard" && (
          <div className="space-y-6">
            {/* Stats Row */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <StatCard label="Total PnL" value={fmt.usd(totalPnl)}
                color={totalPnl >= 0 ? "green" : "red"}
                sub={fmt.pct(portfolio?.total_pnl_percent)} />
              <StatCard label="Open Trades" value={openTrades.length} color="blue"
                sub={`${portfolio?.closed_trades ?? 0} closed`} />
              <StatCard label="Win Rate" value={fmt.pct(portfolio?.win_rate)} color="violet"
                sub={`${portfolio?.total_trades ?? 0} total trades`} />
              <StatCard label="Best Trade" value={fmt.usd(portfolio?.best_trade)} color="green"
                sub={portfolio?.worst_trade != null ? `Worst: ${fmt.usd(portfolio.worst_trade)}` : ""} />
            </div>

            {/* Open Trades */}
            {openTrades.length > 0 && (
              <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl overflow-hidden">
                <div className="flex items-center justify-between p-4 border-b border-slate-700/50">
                  <h2 className="font-bold text-slate-100">Open Positions</h2>
                  <span className="text-xs text-slate-400 font-mono">{openTrades.length} active</span>
                </div>
                <TradesTable trades={openTrades} onRefresh={loadData} />
              </div>
            )}

            {/* CTA if no trades */}
            {!isPending && trades.length === 0 && (
              <div className="text-center py-16 bg-slate-800/40 border border-slate-700/50 rounded-xl">
                <div className="text-5xl mb-4">🚀</div>
                <h3 className="text-lg font-bold text-slate-200 mb-2">Ready to trade?</h3>
                <p className="text-slate-400 text-sm mb-6">
                  {user?.paper_trading
                    ? "Paper trading mode is active. No real funds at risk."
                    : "Connect your exchange API to start live trading."}
                </p>
                <div className="flex gap-3 justify-center">
                  <button onClick={() => setShowTradeModal(true)}
                    className="px-5 py-2.5 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold rounded-xl transition-colors text-sm">
                    Open First Trade
                  </button>
                  {!apiKeys.length && (
                    <button onClick={() => setShowAPIModal(true)}
                      className="px-5 py-2.5 border border-slate-600 hover:border-slate-400 text-slate-300 font-semibold rounded-xl transition-colors text-sm">
                      Connect Exchange
                    </button>
                  )}
                </div>
              </div>
            )}

            {/* Quick Action Buttons */}
            {!isPending && (
              <div className="flex gap-3 flex-wrap">
                <button onClick={() => setShowTradeModal(true)}
                  className="px-4 py-2 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold rounded-xl transition-colors text-sm">
                  + New Trade
                </button>
                <button onClick={() => setShowAPIModal(true)}
                  className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 font-semibold rounded-xl transition-colors text-sm">
                  🔑 {apiKeys.length > 0 ? "Manage APIs" : "Connect Exchange"}
                </button>
                <button onClick={() => { send({ type: "subscribe" }); loadData(); }}
                  className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 font-semibold rounded-xl transition-colors text-sm">
                  ↻ Refresh
                </button>
              </div>
            )}
          </div>
        )}

        {/* Trades Tab */}
        {tab === "trades" && (
          <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b border-slate-700/50">
              <h2 className="font-bold text-slate-100">Open Trades</h2>
              {!isPending && (
                <button onClick={() => setShowTradeModal(true)}
                  className="px-3 py-1.5 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold rounded-lg transition-colors text-xs">
                  + New Trade
                </button>
              )}
            </div>
            <TradesTable trades={openTrades} onRefresh={loadData} />
          </div>
        )}

        {/* History Tab */}
        {tab === "history" && (
          <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl overflow-hidden">
            <div className="p-4 border-b border-slate-700/50">
              <h2 className="font-bold text-slate-100">Trade History</h2>
            </div>
            <TradesTable trades={trades.filter((t) => t.status === "closed")} onRefresh={loadData} />
          </div>
        )}

        {/* Settings Tab */}
        {tab === "settings" && (
          <div className="space-y-6 max-w-2xl">
            {/* User Info */}
            <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-5">
              <h3 className="font-bold text-slate-100 mb-4">Account</h3>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div><span className="text-slate-400">Username</span><div className="font-mono text-slate-200 mt-0.5">{user?.username}</div></div>
                <div><span className="text-slate-400">Email</span><div className="font-mono text-slate-200 mt-0.5">{user?.email}</div></div>
                <div><span className="text-slate-400">Role</span><div className="mt-0.5"><Badge variant={user?.role === "admin" ? "warning" : "default"}>{user?.role}</Badge></div></div>
                <div><span className="text-slate-400">Status</span><div className="mt-0.5"><Badge variant={user?.status === "approved" ? "success" : "warning"}>{user?.status}</Badge></div></div>
              </div>
            </div>

            {/* Trading Mode */}
            <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-5">
              <h3 className="font-bold text-slate-100 mb-2">Trading Mode</h3>
              <p className="text-slate-400 text-sm mb-4">Paper mode simulates trades without real funds.</p>
              <div className="flex gap-3">
                {[true, false].map((isPaper) => (
                  <button key={String(isPaper)} onClick={async () => {
                    try {
                      await api.patch("/user", { paper_trading: isPaper });
                      addToast("success", `Switched to ${isPaper ? "paper" : "live"} trading`);
                      loadData();
                    } catch (e) { addToast("error", e.message); }
                  }}
                    className={cl("flex-1 py-3 rounded-xl text-sm font-bold border transition-all",
                      user?.paper_trading === isPaper
                        ? isPaper ? "bg-violet-500/20 border-violet-500 text-violet-300" : "bg-cyan-500/20 border-cyan-500 text-cyan-300"
                        : "border-slate-600 text-slate-400 hover:border-slate-500")}>
                    {isPaper ? "📄 Paper Trading" : "⚡ Live Trading"}
                  </button>
                ))}
              </div>
            </div>

            {/* API Keys */}
            <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-bold text-slate-100">Exchange API Keys</h3>
                <button onClick={() => setShowAPIModal(true)}
                  className="px-3 py-1.5 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold rounded-lg transition-colors text-xs">
                  + Add Key
                </button>
              </div>
              {apiKeys.length === 0 ? (
                <p className="text-slate-500 text-sm font-mono">No API keys connected. Add one to enable live trading.</p>
              ) : (
                <div className="space-y-2">
                  {apiKeys.map((k) => (
                    <div key={k.id} className="flex items-center justify-between bg-slate-900/60 rounded-lg px-3 py-2.5">
                      <div>
                        <span className="font-semibold capitalize text-slate-200">{k.exchange}</span>
                        {k.label && <span className="text-slate-400 text-xs ml-2">— {k.label}</span>}
                        {k.is_testnet && <Badge variant="warning">testnet</Badge>}
                        <div className="font-mono text-xs text-slate-500 mt-0.5">{k.api_key_preview}</div>
                      </div>
                      <button onClick={async () => {
                        await api.del(`/api-keys/${k.id}`);
                        addToast("info", "API key removed");
                        loadData();
                      }} className="text-red-400 hover:text-red-300 text-xs transition-colors">Remove</button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Admin Tab */}
        {tab === "admin" && isAdmin && <AdminPanel api={api} addToast={addToast} />}
      </main>
    </div>
  );
}
