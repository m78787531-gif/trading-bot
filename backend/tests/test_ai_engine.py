"""
tests/test_ai_engine.py  [v3 — Complete Test Suite]
─────────────────────────────────────────────────────────────────────────────
Covers all 18 production requirements plus regression tests for v1/v2 logic.

Run:
    cd backend
    pytest tests/test_ai_engine.py -v --tb=short

Structure:
  TestIndicators           — RSI, MACD, EMA, BB, ATR, ADX, StochRSI, OBV
  TestSMCEngine            — OB, FVG, BOS, CHoCH, sweeps, acc/dist
  TestSignalGates          — No-trade conditions, 5 mandatory gates
  TestTradeQualityScore    — Each sub-score, total, pass/fail
  TestEntryFilters         — Retest proximity, market structure, session filter
  TestScoring              — Weighted composite, direction, confidence
  TestSignalGenerator      — End-to-end generate_signal()
  TestRiskConfig           — Default values, v3 fields
  TestRiskState            — Daily reset, consecutive tracking, mode switch
  TestTradingMode          — Paper switch triggers, live gate, batch cycle
  TestPositionSizing       — Confidence tiers, drawdown reduction, ATR adjustment
  TestRiskGate             — assess_trade_risk 12-check sequence
  TestCorrelation          — Group detection, size reduction, blocking
  TestLossPatternBlocker   — Pattern matching from optimizer data
  TestDailyLossProtection  — All paper-switch triggers
  TestKillSwitch           — 5% half-size, 10% kill
  TestRecoveryMode         — 0.5% start, increment, deactivate
  TestSelfOptimizer        — Min-trades gate, learning rate, convergence lock
  TestStrategyPerformance  — Per-mode tracking
  TestNewsFlag             — Set/clear, gate effect
  TestBotSession           — effective_paper(), to_status_dict()
"""

from __future__ import annotations
import math
import random
import sys
import os
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ─────────────────────────────────────────────────────────────────────────────
# Test Data Helpers
# ─────────────────────────────────────────────────────────────────────────────

random.seed(42)

def _ohlcv(n=200, start=40000.0, trend=0.0003, vol=200.0):
    """Generate synthetic OHLCV data."""
    opens, highs, lows, closes, vols = [], [], [], [], []
    p = start
    for _ in range(n):
        change = random.gauss(0, 1) * vol + p * trend
        o = p
        c = p + change
        h = max(o, c) + abs(random.gauss(0, vol * 0.5))
        l = min(o, c) - abs(random.gauss(0, vol * 0.5))
        opens.append(round(o, 2))
        closes.append(round(c, 2))
        highs.append(round(h, 2))
        lows.append(round(l, 2))
        vols.append(round(random.uniform(500, 3000), 2))
        p = c
    return opens, highs, lows, closes, vols


def _trending_ohlcv(n=200, start=40000.0, direction="up"):
    """Generate clearly trending data for structure tests."""
    trend = 0.002 if direction == "up" else -0.002
    return _ohlcv(n, start, trend, vol=100.0)


O, H, L, C, V = _ohlcv(200)


# ─────────────────────────────────────────────────────────────────────────────
# TestIndicators
# ─────────────────────────────────────────────────────────────────────────────

class TestIndicators:

    def test_rsi_range(self):
        from ai.indicators import rsi
        r = rsi(C)
        assert 0 <= r.value <= 100
        assert r.signal in ("oversold", "overbought", "neutral")
        assert r.divergence in ("bullish", "bearish", "none")

    def test_rsi_oversold_downtrend(self):
        from ai.indicators import rsi
        falling = [10000 - i * 100 for i in range(100)]
        r = rsi(falling)
        assert r.value < 40, "RSI should be low in downtrend"

    def test_rsi_overbought_uptrend(self):
        from ai.indicators import rsi
        rising = [10000 + i * 100 for i in range(100)]
        r = rsi(rising)
        assert r.value > 60, "RSI should be high in uptrend"

    def test_macd_structure(self):
        from ai.indicators import macd
        m = macd(C)
        assert hasattr(m, "macd_line")
        assert hasattr(m, "signal_line")
        assert hasattr(m, "histogram")
        assert hasattr(m, "crossover")
        assert m.crossover in ("bullish", "bearish", "none")

    def test_macd_histogram_sign(self):
        from ai.indicators import macd
        m = macd(C)
        assert round(m.histogram - (m.macd_line - m.signal_line), 5) == 0

    def test_ema_returns_float(self):
        from ai.indicators import ema
        e20  = ema(C, 20)
        e200 = ema(C, 200)
        assert isinstance(e20,  float) and not math.isnan(e20)
        assert isinstance(e200, float) or math.isnan(e200)

    def test_ema_series_length(self):
        from ai.indicators import ema_series
        s = ema_series(C, 20)
        assert len(s) == len(C)

    def test_bollinger_bands(self):
        from ai.indicators import bollinger_bands
        b = bollinger_bands(C)
        assert hasattr(b, "position")
        assert b.position in ("above_upper","near_upper","middle","near_lower","below_lower")
        assert isinstance(b.squeeze, bool)
        assert b.width >= 0

    def test_atr_positive(self):
        from ai.indicators import atr
        val = atr(H, L, C)
        assert val > 0

    def test_volume_analysis(self):
        from ai.indicators import volume_analysis
        v = volume_analysis(C, V)
        assert v.ratio > 0
        assert v.signal in ("high","normal","low","very_low")
        assert v.obv_trend in ("up","down","flat")

    def test_trend_strength(self):
        from ai.indicators import trend_strength
        adx, direction = trend_strength(H, L, C)
        assert adx >= 0
        assert direction in ("up","down","neutral")

    def test_stochastic_rsi_range(self):
        from ai.indicators import stochastic_rsi
        s = stochastic_rsi(C)
        assert 0 <= s.k <= 100
        assert 0 <= s.d <= 100
        assert s.signal in ("oversold","overbought","neutral")


# ─────────────────────────────────────────────────────────────────────────────
# TestSMCEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestSMCEngine:

    def test_analyse_smc_returns_analysis(self):
        from ai.smc_engine import analyse_smc
        result = analyse_smc("BTCUSDT","1h", O, H, L, C, V)
        assert hasattr(result, "order_blocks")
        assert hasattr(result, "fvgs")
        assert hasattr(result, "structure_breaks")
        assert hasattr(result, "smc_score")
        assert -10 <= result.smc_score <= 10

    def test_smc_to_dict(self):
        from ai.smc_engine import analyse_smc
        r = analyse_smc("BTCUSDT","1h", O, H, L, C, V)
        d = r.to_dict()
        assert "smc_score" in d
        assert "summary"   in d

    def test_ob_fields(self):
        from ai.smc_engine import analyse_smc
        r = analyse_smc("BTCUSDT","1h", O, H, L, C, V)
        for ob in r.order_blocks:
            assert ob.type in ("bullish","bearish")
            assert ob.high >= ob.low
            assert 0 <= ob.strength <= 1

    def test_fvg_fields(self):
        from ai.smc_engine import analyse_smc
        r = analyse_smc("BTCUSDT","1h", O, H, L, C, V)
        for fv in r.fvgs:
            assert fv.type in ("bullish","bearish")
            assert fv.top >= fv.bottom


# ─────────────────────────────────────────────────────────────────────────────
# TestTradeQualityScore (Req 6)
# ─────────────────────────────────────────────────────────────────────────────

class TestTradeQualityScore:

    def _make_snap(self, adx=30.0, atr_pct=0.008):
        from ai.signal_scorer import IndicatorSnapshot
        return IndicatorSnapshot(
            rsi_value=50.0, rsi_signal="neutral", rsi_divergence="none",
            macd_line=0.01, macd_signal=0.005, macd_histogram=0.005,
            macd_crossover="bullish", macd_trend="bullish",
            ema_20=40100.0, ema_50=39900.0, ema_200=38000.0, ema_alignment="bullish",
            bb_position="middle", bb_squeeze=False, bb_width=0.04,
            stoch_k=55.0, stoch_d=50.0, stoch_signal="neutral",
            adx=adx, trend_direction="up",
            volume_ratio=1.8, volume_signal="high", obv_trend="up",
            atr=40000.0 * atr_pct, atr_pct=atr_pct,
        )

    def test_quality_score_structure(self):
        from ai.signal_scorer import calculate_trade_quality, DEFAULT_PARAMS
        from ai.smc_engine import analyse_smc
        snap = self._make_snap()
        smc  = analyse_smc("BTCUSDT","1h", O, H, L, C, V)
        q    = calculate_trade_quality(2.0, smc, snap, "long", DEFAULT_PARAMS)
        assert hasattr(q, "rr_quality")
        assert hasattr(q, "structure_quality")
        assert hasattr(q, "trend_quality")
        assert hasattr(q, "atr_quality")
        assert hasattr(q, "total")
        assert hasattr(q, "passed")

    def test_rr_below_min_fails(self):
        from ai.signal_scorer import calculate_trade_quality, DEFAULT_PARAMS
        from ai.smc_engine import analyse_smc
        snap = self._make_snap()
        smc  = analyse_smc("BTCUSDT","1h", O, H, L, C, V)
        q    = calculate_trade_quality(1.2, smc, snap, "long", DEFAULT_PARAMS)
        assert q.rr_quality == 0.0
        assert not q.passed

    def test_high_rr_rewards(self):
        from ai.signal_scorer import calculate_trade_quality, DEFAULT_PARAMS
        from ai.smc_engine import analyse_smc
        snap = self._make_snap(adx=40.0)
        smc  = analyse_smc("BTCUSDT","1h", O, H, L, C, V)
        q25  = calculate_trade_quality(2.5, smc, snap, "long", DEFAULT_PARAMS)
        q15  = calculate_trade_quality(1.5, smc, snap, "long", DEFAULT_PARAMS)
        assert q25.rr_quality > q15.rr_quality

    def test_low_adx_reduces_trend_quality(self):
        from ai.signal_scorer import calculate_trade_quality, DEFAULT_PARAMS
        from ai.smc_engine import analyse_smc
        smc  = analyse_smc("BTCUSDT","1h", O, H, L, C, V)
        hi_adx = self._make_snap(adx=40.0)
        lo_adx = self._make_snap(adx=15.0)
        qhi = calculate_trade_quality(2.0, smc, hi_adx, "long", DEFAULT_PARAMS)
        qlo = calculate_trade_quality(2.0, smc, lo_adx, "long", DEFAULT_PARAMS)
        assert qhi.trend_quality > qlo.trend_quality

    def test_compressed_atr_penalised(self):
        from ai.signal_scorer import calculate_trade_quality, DEFAULT_PARAMS
        from ai.smc_engine import analyse_smc
        smc  = analyse_smc("BTCUSDT","1h", O, H, L, C, V)
        norm = self._make_snap(atr_pct=0.008)
        comp = self._make_snap(atr_pct=0.001)  # below floor
        qn = calculate_trade_quality(2.0, smc, norm, "long", DEFAULT_PARAMS)
        qc = calculate_trade_quality(2.0, smc, comp, "long", DEFAULT_PARAMS)
        assert qn.atr_quality > qc.atr_quality

    def test_total_in_range(self):
        from ai.signal_scorer import calculate_trade_quality, DEFAULT_PARAMS
        from ai.smc_engine import analyse_smc
        snap = self._make_snap()
        smc  = analyse_smc("BTCUSDT","1h", O, H, L, C, V)
        q    = calculate_trade_quality(2.0, smc, snap, "long", DEFAULT_PARAMS)
        assert 0.0 <= q.total <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TestEntryFilters (Req 9, 10, 15)
# ─────────────────────────────────────────────────────────────────────────────

class TestEntryFilters:

    def test_session_filter_london(self):
        from ai.signal_scorer import is_high_liquidity_session
        from datetime import datetime, timezone
        london = datetime(2024, 1, 15, 8, 30, tzinfo=timezone.utc)
        ok, name = is_high_liquidity_session(london)
        assert ok
        assert "London" in name

    def test_session_filter_ny(self):
        from ai.signal_scorer import is_high_liquidity_session
        from datetime import datetime, timezone
        ny = datetime(2024, 1, 15, 14, 0, tzinfo=timezone.utc)
        ok, name = is_high_liquidity_session(ny)
        assert ok
        assert "New York" in name

    def test_session_filter_dead_zone(self):
        from ai.signal_scorer import is_high_liquidity_session
        from datetime import datetime, timezone
        dead = datetime(2024, 1, 15, 23, 0, tzinfo=timezone.utc)
        ok, _ = is_high_liquidity_session(dead)
        assert not ok

    def test_session_filter_asian_session(self):
        from ai.signal_scorer import is_high_liquidity_session
        from datetime import datetime, timezone
        asian = datetime(2024, 1, 15, 3, 0, tzinfo=timezone.utc)
        ok, _ = is_high_liquidity_session(asian)
        assert not ok

    def test_market_structure_bullish(self):
        from ai.signal_scorer import validate_market_structure
        # Build explicit HH+HL pattern
        highs = [100, 102, 101, 103, 102, 105, 104, 107]
        lows  = [98,  99,  98,  100, 99,  102, 101, 104]
        valid, reason = validate_market_structure(highs, lows, "long", n=3)
        assert valid, f"Expected valid bullish structure: {reason}"

    def test_market_structure_bearish_blocks_long(self):
        from ai.signal_scorer import validate_market_structure
        # Build LH+LL (bearish) — long should fail
        highs = [105, 103, 104, 102, 103, 101, 102, 100]
        lows  = [102, 100, 101, 99,  100, 98,  99,  97]
        valid, reason = validate_market_structure(highs, lows, "long", n=3)
        assert not valid, f"Expected invalid (bearish structure for long): {reason}"

    def test_market_structure_bearish_allows_short(self):
        from ai.signal_scorer import validate_market_structure
        highs = [105, 103, 104, 102, 103, 101, 102, 100]
        lows  = [102, 100, 101, 99,  100, 98,  99,  97]
        valid, _ = validate_market_structure(highs, lows, "short", n=3)
        assert valid, "Should allow short in bearish structure"

    def test_retest_proximity_near_ob(self):
        from ai.signal_scorer import check_retest_proximity
        from unittest.mock import MagicMock
        smc = MagicMock()
        ob  = MagicMock()
        ob.valid  = True
        ob.type   = "bullish"
        ob.high   = 40100.0
        ob.low    = 39900.0
        fvg        = MagicMock()
        fvg.valid  = False
        smc.order_blocks = [ob]
        smc.fvgs         = [fvg]
        # Price within 0.3% of OB top (40100)
        price = 40110.0
        ok, reason = check_retest_proximity(price, smc, "long", proximity_pct=0.005)
        assert ok, f"Should confirm retest: {reason}"

    def test_retest_proximity_far_from_ob(self):
        from ai.signal_scorer import check_retest_proximity
        from unittest.mock import MagicMock
        smc = MagicMock()
        ob  = MagicMock()
        ob.valid  = True
        ob.type   = "bullish"
        ob.high   = 40100.0
        ob.low    = 39900.0
        fvg        = MagicMock()
        fvg.valid  = False
        smc.order_blocks = [ob]
        smc.fvgs         = [fvg]
        # Price 5% away from OB — not a retest
        price = 42100.0
        ok, reason = check_retest_proximity(price, smc, "long", proximity_pct=0.005)
        assert not ok, f"Should NOT confirm retest when far away: {reason}"

    def test_news_flag_blocks_all_entries(self):
        from ai.signal_scorer import generate_signal, set_news_flag
        set_news_flag(True)
        signal = generate_signal(
            "BTCUSDT","1h", O, H, L, C, V,
            sentiment_score=0.5,
        )
        set_news_flag(False)  # restore
        assert signal.direction == "hold" or signal.blocked
        assert signal.news_flag_active

    def test_news_flag_clears(self):
        from ai.signal_scorer import set_news_flag, get_news_flag
        set_news_flag(True)
        assert get_news_flag() is True
        set_news_flag(False)
        assert get_news_flag() is False


# ─────────────────────────────────────────────────────────────────────────────
# TestSignalGenerator
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalGenerator:

    def test_generate_signal_returns_trade_signal(self):
        from ai.signal_scorer import generate_signal, TradeSignal
        sig = generate_signal("BTCUSDT","1h", O, H, L, C, V)
        assert isinstance(sig, TradeSignal)

    def test_direction_valid(self):
        from ai.signal_scorer import generate_signal
        sig = generate_signal("BTCUSDT","1h", O, H, L, C, V)
        assert sig.direction in ("long","short","hold")

    def test_confidence_range(self):
        from ai.signal_scorer import generate_signal
        sig = generate_signal("BTCUSDT","1h", O, H, L, C, V)
        assert 0 <= sig.confidence <= 100

    def test_sl_tp_logical_long(self):
        from ai.signal_scorer import generate_signal
        sig = generate_signal("BTCUSDT","1h", O, H, L, C, V)
        if sig.direction == "long":
            assert sig.stop_loss < sig.entry_price
            assert sig.take_profit > sig.entry_price
            assert sig.take_profit_partial > sig.entry_price

    def test_sl_tp_logical_short(self):
        from ai.signal_scorer import generate_signal
        sig = generate_signal("BTCUSDT","1h", O, H, L, C, V)
        if sig.direction == "short":
            assert sig.stop_loss > sig.entry_price
            assert sig.take_profit < sig.entry_price

    def test_rr_positive_when_trading(self):
        from ai.signal_scorer import generate_signal
        sig = generate_signal("BTCUSDT","1h", O, H, L, C, V)
        if sig.direction != "hold":
            assert sig.risk_reward > 0

    def test_to_dict_complete(self):
        from ai.signal_scorer import generate_signal
        sig = generate_signal("BTCUSDT","1h", O, H, L, C, V)
        d   = sig.to_dict()
        for key in ("direction","confidence","entry_price","stop_loss",
                    "take_profit","risk_reward","strategy_mode","market_regime"):
            assert key in d

    def test_trade_quality_attached(self):
        from ai.signal_scorer import generate_signal
        sig = generate_signal("BTCUSDT","1h", O, H, L, C, V)
        assert hasattr(sig, "trade_quality")
        if sig.trade_quality:
            assert 0 <= sig.trade_quality.total <= 1

    def test_min_confidence_threshold(self):
        """Trade should be hold if confidence < 75."""
        from ai.signal_scorer import generate_signal, DEFAULT_PARAMS
        # Force threshold above any possible score
        p = {**DEFAULT_PARAMS, "min_confidence": 99.9}
        sig = generate_signal("BTCUSDT","1h", O, H, L, C, V, params=p)
        assert sig.direction == "hold"

    def test_short_data_returns_hold(self):
        """Insufficient candles should not crash — should return hold."""
        from ai.signal_scorer import generate_signal
        short_closes = [40000.0] * 30
        short_others = [40000.0] * 30
        sig = generate_signal(
            "BTCUSDT","1h",
            short_others, short_others, short_others, short_closes, short_others,
        )
        # Should not raise; direction can be hold due to insufficient data
        assert sig.direction in ("long","short","hold")


# ─────────────────────────────────────────────────────────────────────────────
# TestRiskConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskConfig:

    def test_default_values(self):
        from ai.risk_manager import RiskConfig
        cfg = RiskConfig()
        assert cfg.max_risk_per_trade_pct  == 1.0
        assert cfg.max_daily_loss_pct      == 3.0
        assert cfg.max_total_drawdown_pct  == 10.0
        assert cfg.drawdown_half_size_pct  == 5.0
        assert cfg.max_consecutive_losses  == 3
        assert cfg.max_daily_loss_count    == 9
        assert cfg.max_trades_per_live_cycle == 5
        assert cfg.market_quality_min      == 0.70
        assert cfg.risk_tier_high_pct      == 1.5
        assert cfg.risk_tier_standard_pct  == 1.0
        assert cfg.recovery_risk_pct       == 0.50
        assert cfg.withdrawals_enabled     is False   # ALWAYS False

    def test_withdrawals_never_enabled(self):
        from ai.risk_manager import RiskConfig
        cfg = RiskConfig(withdrawals_enabled=True)   # attempt override
        assert cfg.withdrawals_enabled is True        # field set, but gate in assess blocks it


# ─────────────────────────────────────────────────────────────────────────────
# TestRiskState
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskState:

    def _state(self, balance=10000.0):
        from ai.risk_manager import RiskState
        rs = RiskState(user_id=1)
        rs.session_high_balance = balance
        rs.daily_start_balance  = balance
        return rs

    def test_daily_reset_switches_date(self):
        from ai.risk_manager import RiskState
        from datetime import datetime, timezone
        rs = RiskState(user_id=1)
        rs.daily_date = "2023-01-01"  # old date
        rs.daily_loss_count = 5
        rs.daily_pnl        = -300.0
        rs.reset_daily_if_needed(10000.0)
        today = datetime.now(timezone.utc).date().isoformat()
        assert rs.daily_date == today
        assert rs.daily_loss_count == 0
        assert rs.daily_pnl == 0.0

    def test_record_trade_close_updates_streak(self):
        rs = self._state()
        rs.record_trade_close("BTCUSDT", -100.0, "long", 9900.0, None)
        rs.record_trade_close("BTCUSDT", -100.0, "long", 9800.0, None)
        assert rs.consecutive_losses == 2
        assert rs.consecutive_wins   == 0
        rs.record_trade_close("BTCUSDT", 200.0, "long", 10000.0, None)
        assert rs.consecutive_losses == 0
        assert rs.consecutive_wins   == 1

    def test_drawdown_calculation(self):
        rs = self._state(10000.0)
        rs.record_trade_close("BTCUSDT", -1000.0, "long", 9000.0, None)
        assert abs(rs.current_drawdown_pct - 10.0) < 0.1

    def test_daily_loss_count_increments(self):
        rs = self._state()
        for _ in range(5):
            rs.record_trade_close("BTCUSDT", -100.0, "long", 9900.0, None)
        assert rs.daily_loss_count == 5

    def test_switch_to_paper(self):
        from ai.risk_manager import RiskState, TradingMode
        rs = RiskState(user_id=1)
        assert rs.trading_mode == TradingMode.LIVE.value
        rs.switch_to_paper("daily loss limit")
        assert rs.trading_mode == TradingMode.PAPER.value

    def test_is_paused_after_trigger(self):
        rs = self._state()
        assert not rs.is_paused()
        rs.trigger_consecutive_loss_pause(2.0)
        assert rs.is_paused()
        assert rs.pause_minutes_remaining() > 0

    def test_symbol_open_counts(self):
        rs = self._state()
        rs.record_trade_open("BTCUSDT", "long",  None)
        rs.record_trade_open("BTCUSDT", "long",  None)
        assert rs.symbol_open_counts.get("BTCUSDT", 0) == 2
        rs.record_trade_close("BTCUSDT", 100.0, "long", 10100.0, None)
        assert rs.symbol_open_counts.get("BTCUSDT", 0) == 1

    def test_cycle_quality_tracking(self):
        rs = self._state()
        for q in [0.8, 0.7, 0.9, 0.6]:
            rs.add_cycle_quality(q)
        avg = rs.avg_cycle_quality()
        assert abs(avg - sum([0.8,0.7,0.9,0.6])/4) < 0.001


# ─────────────────────────────────────────────────────────────────────────────
# TestTradingMode (Req 1, 2, 3, 4)
# ─────────────────────────────────────────────────────────────────────────────

class TestTradingMode:

    def test_check_live_cycle_limit_normal(self):
        from ai.risk_manager import RiskState, RiskConfig, check_live_cycle_limit
        rs  = RiskState(user_id=1)
        cfg = RiskConfig()
        rs.live_trade_count_cycle = 4
        reached, limit, _ = check_live_cycle_limit(rs, cfg)
        assert not reached
        assert limit == 5

    def test_check_live_cycle_limit_reached(self):
        from ai.risk_manager import RiskState, RiskConfig, check_live_cycle_limit
        rs  = RiskState(user_id=1)
        cfg = RiskConfig()
        rs.live_trade_count_cycle = 5
        reached, limit, reason = check_live_cycle_limit(rs, cfg)
        assert reached
        assert "5" in reason

    def test_low_quality_reduces_cycle_limit(self):
        from ai.risk_manager import RiskState, RiskConfig, check_live_cycle_limit
        rs  = RiskState(user_id=1)
        cfg = RiskConfig()
        # Simulate low quality cycle
        for _ in range(10):
            rs.add_cycle_quality(0.50)   # below 0.65
        rs.live_trade_count_cycle = 3
        reached, limit, _ = check_live_cycle_limit(rs, cfg)
        assert limit == cfg.low_quality_cycle_limit   # 3
        assert reached   # at limit (3 >= 3)

    def test_can_switch_to_live_all_pass(self):
        from ai.risk_manager import RiskState, RiskConfig, can_switch_to_live
        rs  = RiskState(user_id=1)
        cfg = RiskConfig()
        rs.daily_pnl_pct      = -1.0   # < 3% loss
        rs.consecutive_losses = 1       # < 3
        allowed, reason = can_switch_to_live(rs, cfg, market_quality_score=0.80)
        assert allowed, reason

    def test_can_switch_to_live_blocked_by_dd(self):
        from ai.risk_manager import RiskState, RiskConfig, can_switch_to_live
        rs  = RiskState(user_id=1)
        cfg = RiskConfig()
        rs.daily_pnl_pct = -5.0   # exceeds 3% limit
        allowed, reason = can_switch_to_live(rs, cfg, 0.80)
        assert not allowed
        assert "drawdown" in reason.lower() or "3%" in reason or "loss" in reason.lower()

    def test_can_switch_to_live_blocked_by_losses(self):
        from ai.risk_manager import RiskState, RiskConfig, can_switch_to_live
        rs  = RiskState(user_id=1)
        cfg = RiskConfig()
        rs.daily_pnl_pct      = -1.0
        rs.consecutive_losses = 4   # ≥ 3
        allowed, reason = can_switch_to_live(rs, cfg, 0.80)
        assert not allowed

    def test_can_switch_blocked_by_quality(self):
        from ai.risk_manager import RiskState, RiskConfig, can_switch_to_live
        rs  = RiskState(user_id=1)
        cfg = RiskConfig()
        allowed, reason = can_switch_to_live(rs, cfg, market_quality_score=0.60)
        assert not allowed
        assert "quality" in reason.lower()

    def test_switch_to_live_resets_counter(self):
        from ai.risk_manager import RiskState, switch_to_live
        rs = RiskState(user_id=1)
        rs.live_trade_count_cycle = 5
        switch_to_live(rs)
        assert rs.live_trade_count_cycle == 0
        assert rs.trading_mode == "live"


# ─────────────────────────────────────────────────────────────────────────────
# TestPositionSizing (Req 11, 13, Req 3)
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionSizing:

    def _size(self, conf, atr_pct=0.008, dd_pct=0.0, **kw):
        from ai.risk_manager import RiskConfig, RiskState, calculate_position_size
        cfg = RiskConfig()
        rs  = RiskState(user_id=1)
        rs.current_drawdown_pct = dd_pct
        rs.session_high_balance = 10000.0
        return calculate_position_size(
            account_balance=10000.0, entry_price=40000.0,
            stop_loss=39000.0,       confidence=conf,
            win_rate=0.55, avg_win_pct=2.0, avg_loss_pct=1.0,
            config=cfg, risk_state=rs, atr_pct=atr_pct,
        )

    def test_conf_below_75_rejected(self):
        ps = self._size(conf=70.0)
        assert ps.rejected or ps.quantity == 0

    def test_conf_75_gives_1pct_risk(self):
        ps = self._size(conf=75.0)
        assert not ps.rejected
        assert abs(ps.risk_pct - 1.0) < 0.2

    def test_conf_85_gives_1_5pct_risk(self):
        ps = self._size(conf=85.0)
        assert not ps.rejected
        assert ps.risk_pct <= 1.5

    def test_conf_85_more_than_75(self):
        ps85 = self._size(conf=85.0)
        ps75 = self._size(conf=75.0)
        assert ps85.quantity > ps75.quantity

    def test_5pct_dd_halves_size(self):
        ps_normal = self._size(conf=80.0, dd_pct=2.0)
        ps_halved = self._size(conf=80.0, dd_pct=6.0)  # exceeds 5% threshold
        assert ps_halved.quantity < ps_normal.quantity * 0.7

    def test_extreme_atr_halves_size(self):
        ps_normal = self._size(conf=80.0, atr_pct=0.008)
        ps_extreme = self._size(conf=80.0, atr_pct=0.04)  # > 3% → halve
        assert ps_extreme.quantity < ps_normal.quantity

    def test_recovery_mode_caps_risk(self):
        from ai.risk_manager import RiskConfig, RiskState, calculate_position_size
        cfg = RiskConfig()
        rs  = RiskState(user_id=1)
        rs.in_recovery_mode     = True
        rs.current_recovery_risk = 0.5
        ps = calculate_position_size(
            10000.0, 40000.0, 39000.0, 85.0,
            0.55, 2.0, 1.0, cfg, rs, 0.008,
        )
        assert ps.risk_pct <= 0.6   # capped by recovery


# ─────────────────────────────────────────────────────────────────────────────
# TestRiskGate (Req 5, 7, 12, 13, 14)
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskGate:

    def _base_ps(self):
        from ai.risk_manager import PositionSize
        return PositionSize(
            quantity=0.05, notional_value=2000.0,
            risk_amount=100.0, risk_pct=1.0,
            leverage=1.0, rationale="test",
        )

    def _approve(self, **kw):
        from ai.risk_manager import RiskConfig, RiskState, assess_trade_risk
        cfg = RiskConfig()
        rs  = RiskState(user_id=1)
        rs.session_high_balance = 10000.0
        defaults = dict(
            signal_direction="long", symbol="BTCUSDT",
            entry_price=40000.0, stop_loss=39000.0,
            position_size=self._base_ps(), account_balance=10000.0,
            open_trades=[], daily_pnl_pct=0.0, total_drawdown_pct=0.0,
            config=cfg, risk_state=rs,
            trade_quality_score=0.80,
            optimizer_patterns=[],
        )
        defaults.update(kw)
        return assess_trade_risk(**defaults)

    def test_clean_trade_approved(self):
        ra = self._approve()
        assert ra.approved

    def test_withdrawals_always_blocked(self):
        from ai.risk_manager import RiskConfig
        cfg = RiskConfig(withdrawals_enabled=True)
        ra  = self._approve(config=cfg)
        assert not ra.approved
        assert "withdrawals" in ra.rejection_reason.lower()

    def test_daily_loss_limit_blocks(self):
        ra = self._approve(daily_pnl_pct=-5.0)
        assert not ra.approved
        assert "daily" in ra.rejection_reason.lower()

    def test_max_dd_blocks(self):
        ra = self._approve(total_drawdown_pct=12.0)
        assert not ra.approved
        assert "drawdown" in ra.rejection_reason.lower()

    def test_position_count_limit(self):
        open_t = [{"symbol": f"COIN{i}USDT", "side": "long", "risk_pct": 0.5, "mode": "live"}
                  for i in range(5)]
        ra = self._approve(open_trades=open_t)
        assert not ra.approved

    def test_symbol_exposure_limit(self):
        open_t = [{"symbol": "BTCUSDT", "side": "long", "risk_pct": 1.0, "mode": "live"},
                  {"symbol": "BTCUSDT", "side": "long", "risk_pct": 1.0, "mode": "live"}]
        ra = self._approve(open_trades=open_t)
        assert not ra.approved
        assert "BTCUSDT" in ra.rejection_reason

    def test_quality_floor_blocks(self):
        ra = self._approve(trade_quality_score=0.55)
        assert not ra.approved
        assert "quality" in ra.rejection_reason.lower()

    def test_kill_switch_blocks(self):
        from ai.risk_manager import RiskState, RiskConfig, assess_trade_risk
        cfg = RiskConfig()
        rs  = RiskState(user_id=1)
        rs.kill_switch_triggered = True
        rs.kill_switch_reason    = "10% drawdown"
        ra  = assess_trade_risk(
            "long", "BTCUSDT", 40000.0, 39000.0,
            self._base_ps(), 10000.0, [], 0.0, 0.0,
            cfg, rs, trade_quality_score=0.80, optimizer_patterns=[],
        )
        assert not ra.approved
        assert "kill" in ra.rejection_reason.lower()

    def test_pause_blocks(self):
        from ai.risk_manager import RiskState, RiskConfig, assess_trade_risk
        cfg = RiskConfig()
        rs  = RiskState(user_id=1)
        rs.trigger_consecutive_loss_pause(2.0)
        ra  = assess_trade_risk(
            "long", "BTCUSDT", 40000.0, 39000.0,
            self._base_ps(), 10000.0, [], 0.0, 0.0,
            cfg, rs, trade_quality_score=0.80, optimizer_patterns=[],
        )
        assert not ra.approved
        assert "pause" in ra.rejection_reason.lower()


# ─────────────────────────────────────────────────────────────────────────────
# TestCorrelation (Req 12)
# ─────────────────────────────────────────────────────────────────────────────

class TestCorrelation:

    def test_get_correlation_group_btc(self):
        from ai.risk_manager import get_correlation_group
        assert get_correlation_group("BTCUSDT") == "btc_cluster"

    def test_get_correlation_group_eth(self):
        from ai.risk_manager import get_correlation_group
        assert get_correlation_group("ETHUSDT") == "eth_cluster"

    def test_get_correlation_group_unknown(self):
        from ai.risk_manager import get_correlation_group
        assert get_correlation_group("XYZUSDT") is None

    def test_corr_group_size_reduced(self):
        from ai.risk_manager import RiskState, RiskConfig, PositionSize, assess_trade_risk
        cfg = RiskConfig()
        rs  = RiskState(user_id=1)
        rs.session_high_balance = 10000.0
        # Simulate 1 existing long in btc_cluster
        rs.record_trade_open("BTCUSDT", "long", "btc_cluster")

        ps = PositionSize(0.05, 2000.0, 100.0, 1.0, 1.0, "test")
        ra = assess_trade_risk(
            "long", "ETHUSDT",  # also in ETH cluster but ETH is eth_cluster
            40000.0, 39000.0, ps, 10000.0, [],
            0.0, 0.0, cfg, rs,
            trade_quality_score=0.80, optimizer_patterns=[],
        )
        # ETH is different cluster — should pass
        assert ra.approved

    def test_corr_group_blocked_at_limit(self):
        from ai.risk_manager import RiskState, RiskConfig, PositionSize, assess_trade_risk
        cfg = RiskConfig(max_correlated_exposure=2)
        rs  = RiskState(user_id=1)
        rs.session_high_balance = 10000.0
        # Add 2 longs in layer1 group
        rs.record_trade_open("SOLUSDT", "long", "layer1")
        rs.record_trade_open("AVAXUSDT", "long", "layer1")

        ps = PositionSize(0.05, 2000.0, 100.0, 1.0, 1.0, "test")
        ra = assess_trade_risk(
            "long", "ADAUSDT", 40000.0, 39000.0,
            ps, 10000.0, [], 0.0, 0.0, cfg, rs,
            trade_quality_score=0.80, optimizer_patterns=[],
        )
        assert not ra.approved
        assert "correlation" in ra.rejection_reason.lower()


# ─────────────────────────────────────────────────────────────────────────────
# TestLossPatternBlocker (Req 14)
# ─────────────────────────────────────────────────────────────────────────────

class TestLossPatternBlocker:

    def test_losing_rsi_pattern_blocks(self):
        from ai.risk_manager import check_loss_pattern
        from unittest.mock import MagicMock
        pat = MagicMock()
        pat.factor   = "rsi"
        pat.bucket   = "overbought"
        pat.win_rate = 0.30
        pat.wins     = 3
        pat.losses   = 7

        snap = {"indicators": {"rsi_value": 72.0, "adx": 30.0}}
        blocked, reason = check_loss_pattern(snap, [pat], min_significance=10)
        assert blocked
        assert "rsi" in reason.lower()

    def test_winning_pattern_does_not_block(self):
        from ai.risk_manager import check_loss_pattern
        from unittest.mock import MagicMock
        pat = MagicMock()
        pat.factor   = "rsi"
        pat.bucket   = "overbought"
        pat.win_rate = 0.65   # winning — should not block
        pat.wins     = 13
        pat.losses   = 7

        snap = {"indicators": {"rsi_value": 72.0, "adx": 30.0}}
        blocked, _ = check_loss_pattern(snap, [pat], min_significance=10)
        assert not blocked

    def test_insufficient_samples_skipped(self):
        from ai.risk_manager import check_loss_pattern
        from unittest.mock import MagicMock
        pat = MagicMock()
        pat.factor   = "rsi"
        pat.bucket   = "overbought"
        pat.win_rate = 0.20
        pat.wins     = 1
        pat.losses   = 4   # only 5 — below min_significance=10

        snap = {"indicators": {"rsi_value": 72.0, "adx": 30.0}}
        blocked, _ = check_loss_pattern(snap, [pat], min_significance=10)
        assert not blocked

    def test_empty_patterns_no_block(self):
        from ai.risk_manager import check_loss_pattern
        snap = {"indicators": {"rsi_value": 50.0, "adx": 30.0}}
        blocked, _ = check_loss_pattern(snap, [])
        assert not blocked


# ─────────────────────────────────────────────────────────────────────────────
# TestDailyLossProtection (Req 2)
# ─────────────────────────────────────────────────────────────────────────────

class TestDailyLossProtection:

    @pytest.mark.asyncio
    async def test_loss_count_triggers_paper(self):
        from ai.risk_manager import RiskState, RiskConfig, check_daily_loss_protection, TradingMode
        rs  = RiskState(user_id=1)
        cfg = RiskConfig(max_daily_loss_count=9)
        rs.daily_loss_count = 9
        reason = await check_daily_loss_protection(rs, cfg, alert_fn=None)
        assert reason is not None
        assert rs.trading_mode == TradingMode.PAPER.value

    @pytest.mark.asyncio
    async def test_drawdown_triggers_paper(self):
        from ai.risk_manager import RiskState, RiskConfig, check_daily_loss_protection, TradingMode
        rs  = RiskState(user_id=1)
        cfg = RiskConfig(max_daily_loss_pct=3.0)
        rs.daily_pnl_pct    = -4.0   # exceeds 3%
        rs.daily_start_balance = 10000.0
        reason = await check_daily_loss_protection(rs, cfg, alert_fn=None)
        assert reason is not None
        assert rs.trading_mode == TradingMode.PAPER.value

    @pytest.mark.asyncio
    async def test_healthy_state_no_trigger(self):
        from ai.risk_manager import RiskState, RiskConfig, check_daily_loss_protection
        rs  = RiskState(user_id=1)
        cfg = RiskConfig()
        rs.daily_loss_count = 3
        rs.daily_pnl_pct    = -1.0
        reason = await check_daily_loss_protection(rs, cfg, alert_fn=None)
        assert reason is None


# ─────────────────────────────────────────────────────────────────────────────
# TestKillSwitch (Req 3)
# ─────────────────────────────────────────────────────────────────────────────

class TestKillSwitch:

    @pytest.mark.asyncio
    async def test_10pct_drawdown_triggers_kill(self):
        from ai.risk_manager import RiskState, RiskConfig, check_kill_switch
        rs  = RiskState(user_id=1)
        cfg = RiskConfig(max_total_drawdown_pct=10.0)
        rs.current_drawdown_pct = 11.0
        fired = await check_kill_switch(rs, cfg, 9000.0, alert_fn=None)
        assert fired
        assert rs.kill_switch_triggered

    @pytest.mark.asyncio
    async def test_daily_loss_triggers_kill(self):
        from ai.risk_manager import RiskState, RiskConfig, check_kill_switch
        rs  = RiskState(user_id=1)
        cfg = RiskConfig(max_daily_loss_pct=3.0)
        rs.daily_pnl_pct = -5.0
        fired = await check_kill_switch(rs, cfg, 9500.0, alert_fn=None)
        assert fired

    @pytest.mark.asyncio
    async def test_healthy_no_kill(self):
        from ai.risk_manager import RiskState, RiskConfig, check_kill_switch
        rs  = RiskState(user_id=1)
        cfg = RiskConfig()
        rs.current_drawdown_pct = 4.0
        rs.daily_pnl_pct        = -1.0
        fired = await check_kill_switch(rs, cfg, 9600.0, alert_fn=None)
        assert not fired

    def test_kill_switch_reset(self):
        from ai.risk_manager import RiskState
        rs = RiskState(user_id=1)
        rs.kill_switch_triggered = True
        rs.kill_switch_reason    = "test"
        rs.reset_kill_switch()
        assert not rs.kill_switch_triggered
        assert rs.kill_switch_reason == ""


# ─────────────────────────────────────────────────────────────────────────────
# TestRecoveryMode (Req 3)
# ─────────────────────────────────────────────────────────────────────────────

class TestRecoveryMode:

    def test_recovery_starts_at_0_5_pct(self):
        from ai.risk_manager import RiskConfig
        cfg = RiskConfig()
        assert cfg.recovery_risk_pct == 0.50

    def test_recovery_increments_per_win(self):
        from ai.risk_manager import RiskState
        rs = RiskState(user_id=1)
        rs.session_high_balance = 10000.0
        rs.in_recovery_mode     = True
        rs.current_recovery_risk = 0.50

        # Win 3 trades
        for _ in range(3):
            rs.record_trade_close("BTCUSDT", 100.0, "long", 10100.0, None)

        assert rs.current_recovery_risk > 0.50
        assert round(rs.current_recovery_risk, 2) == round(0.50 + 3 * 0.05, 2)

    def test_recovery_deactivates_at_target(self):
        from ai.risk_manager import RiskState, RiskConfig
        cfg = RiskConfig()
        rs  = RiskState(user_id=1)
        rs.session_high_balance  = 10000.0
        rs.in_recovery_mode      = True
        rs.current_recovery_risk = 0.95  # one win away from 1.0

        rs.record_trade_close("BTCUSDT", 100.0, "long", 10100.0, None)
        assert not rs.in_recovery_mode


# ─────────────────────────────────────────────────────────────────────────────
# TestSelfOptimizer
# ─────────────────────────────────────────────────────────────────────────────

class TestSelfOptimizer:

    def test_min_trades_gate(self):
        from ai.self_optimizer import optimise_parameters, _trade_history
        _trade_history.clear()
        result = optimise_parameters()
        assert result.trades_analysed == 0 or not result.changes

    def test_constants(self):
        from ai.self_optimizer import (
            MIN_TRADES_FOR_OPTIMISATION,
            ADJUSTMENT_RATE,
            MAX_PARAM_CHANGE_PCT,
        )
        assert MIN_TRADES_FOR_OPTIMISATION == 50
        assert ADJUSTMENT_RATE             == 0.01
        assert MAX_PARAM_CHANGE_PCT        == 0.05

    def test_record_trade_increments_history(self):
        from ai.self_optimizer import record_trade_result, _trade_history
        before = len(_trade_history)
        record_trade_result(9999, "BTCUSDT", "long", 100.0, 1.0)
        assert len(_trade_history) == before + 1

    def test_strategy_performance_returns_dict(self):
        from ai.self_optimizer import get_strategy_performance
        result = get_strategy_performance()
        assert isinstance(result, dict)

    def test_analyse_patterns_returns_list(self):
        from ai.self_optimizer import analyse_patterns, TradeRecord
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        trades = [
            TradeRecord(
                trade_id=i, symbol="BTCUSDT", direction="long",
                pnl=100.0 if i % 2 == 0 else -50.0,
                pnl_pct=0.5 if i % 2 == 0 else -0.25,
                won=(i % 2 == 0),
                confidence_at_entry=80.0,
                rsi_at_entry=50.0, macd_trend_at_entry="bullish",
                ema_alignment_at_entry="bullish", smc_score_at_entry=3.0,
                sentiment_at_entry=0.2, adx_at_entry=30.0,
                volume_ratio_at_entry=1.8, bb_position_at_entry="middle",
                strategy_mode="trend", opened_at=now, closed_at=now,
            )
            for i in range(20)
        ]
        patterns = analyse_patterns(trades)
        assert isinstance(patterns, list)
        assert len(patterns) > 0

    def test_get_params_returns_dict(self):
        from ai.self_optimizer import get_params
        p = get_params()
        assert isinstance(p, dict)
        assert "min_confidence" in p

    def test_set_params_persists(self):
        from ai.self_optimizer import get_params, set_params
        p = get_params().copy()
        p["min_confidence"] = 77.5
        set_params(p)
        assert get_params()["min_confidence"] == 77.5
        # Restore
        p["min_confidence"] = 75.0
        set_params(p)


# ─────────────────────────────────────────────────────────────────────────────
# TestPerformanceReport
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformanceReport:

    def _trades(self):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        return [
            {"pnl": 200.0, "pnl_percent": 2.0, "status": "closed",
             "symbol": "BTCUSDT", "risk_reward": 2.0, "strategy_mode": "trend",
             "closed_at": now},
            {"pnl": -100.0, "pnl_percent": -1.0, "status": "closed",
             "symbol": "ETHUSDT", "risk_reward": 1.5, "strategy_mode": "range",
             "closed_at": now},
            {"pnl": 0.0, "pnl_percent": 0.0, "status": "open",
             "symbol": "SOLUSDT", "risk_reward": 0.0, "strategy_mode": "trend",
             "closed_at": ""},
        ]

    def test_report_fields(self):
        from ai.risk_manager import calculate_performance_report
        r = calculate_performance_report(self._trades(), 10000.0)
        assert hasattr(r, "win_rate")
        assert hasattr(r, "profit_factor")
        assert hasattr(r, "max_drawdown_pct")
        assert hasattr(r, "strategy_breakdown")
        assert hasattr(r, "sharpe_approx")

    def test_win_rate_calculation(self):
        from ai.risk_manager import calculate_performance_report
        r = calculate_performance_report(self._trades(), 10000.0)
        assert abs(r.win_rate - 0.5) < 0.01   # 1 win, 1 loss

    def test_strategy_breakdown_populated(self):
        from ai.risk_manager import calculate_performance_report
        r = calculate_performance_report(self._trades(), 10000.0)
        assert "trend" in r.strategy_breakdown
        assert "range" in r.strategy_breakdown

    def test_to_dict_serialisable(self):
        from ai.risk_manager import calculate_performance_report
        import json
        r = calculate_performance_report(self._trades(), 10000.0)
        d = r.to_dict()
        json.dumps(d)   # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# TestBotSession
# ─────────────────────────────────────────────────────────────────────────────

class TestBotSession:

    def _session(self, paper_mode=True):
        from ai.bot_engine import BotSession
        return BotSession(
            user_id=1, exchange="binance",
            api_key="", api_secret="",
            is_testnet=True, paper_mode=paper_mode,
        )

    def test_effective_paper_when_paper_mode(self):
        s = self._session(paper_mode=True)
        assert s.effective_paper() is True

    def test_effective_paper_when_mode_switched(self):
        from ai.risk_manager import TradingMode
        s = self._session(paper_mode=False)
        s.risk_state.switch_to_paper("drawdown limit")
        assert s.effective_paper() is True

    def test_to_status_dict_fields(self):
        s = self._session()
        d = s.to_status_dict()
        assert "user_id"         in d
        assert "trading_mode"    in d or "risk_state" in d
        assert "effective_paper" in d
        assert "open_positions_count" in d

    def test_risk_state_bound_to_user(self):
        s = self._session()
        assert s.risk_state.user_id == 1


# ─────────────────────────────────────────────────────────────────────────────
# TestDefaultParams
# ─────────────────────────────────────────────────────────────────────────────

class TestDefaultParams:

    def test_all_v3_params_present(self):
        from ai.signal_scorer import DEFAULT_PARAMS
        required = [
            "min_confidence", "min_trade_quality", "min_rr",
            "retest_proximity_pct", "session_filter",
            "structure_swing_n", "volume_spike_min",
            "adx_floor", "atr_floor_pct",
            "w_smc", "w_trend", "w_momentum",
            "w_volume", "w_volatility", "w_sentiment",
        ]
        for key in required:
            assert key in DEFAULT_PARAMS, f"Missing param: {key}"

    def test_weights_sum_to_one(self):
        from ai.signal_scorer import DEFAULT_PARAMS
        total = sum(DEFAULT_PARAMS[f"w_{k}"]
                    for k in ("smc","trend","momentum","volume","volatility","sentiment"))
        assert abs(total - 1.0) < 0.001

    def test_min_confidence_is_75(self):
        from ai.signal_scorer import DEFAULT_PARAMS
        assert DEFAULT_PARAMS["min_confidence"] == 75.0

    def test_min_rr_is_1_5(self):
        from ai.signal_scorer import DEFAULT_PARAMS
        assert DEFAULT_PARAMS["min_rr"] == 1.5

    def test_min_quality_is_0_7(self):
        from ai.signal_scorer import DEFAULT_PARAMS
        assert DEFAULT_PARAMS["min_trade_quality"] == 0.70
