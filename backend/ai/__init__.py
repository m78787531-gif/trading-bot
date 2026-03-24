"""
ai/__init__.py
─────────────────────────────────────────────────────────────────────────────
TradeOS AI Engine — public API surface.

Quick-start example:
    from ai import generate_signal, get_sentiment, start_bot

    # One-shot signal (no trade placed)
    signal = generate_signal(
        symbol="BTCUSDT", timeframe="1h",
        opens=[...], highs=[...], lows=[...], closes=[...], volumes=[...],
    )
    print(signal.direction, signal.confidence, signal.reasoning)

    # Start automated bot for a user
    session = start_bot(
        user_id=1, exchange="binance",
        api_key="xxx", api_secret="yyy",
        paper_mode=True,
        symbols=["BTCUSDT", "ETHUSDT"],
    )
"""

from ai.signal_scorer import generate_signal, TradeSignal, DEFAULT_PARAMS
from ai.indicators import rsi, macd, ema, bollinger_bands, atr, volume_analysis
from ai.smc_engine import analyse_smc, SMCAnalysis
from ai.sentiment import get_sentiment, get_sentiment_sync, SentimentResult
from ai.risk_manager import (
    RiskConfig, calculate_position_size, assess_trade_risk,
    trailing_stop, breakeven_stop, partial_profit_levels,
)
from ai.self_optimizer import (
    record_trade_result, optimise_parameters,
    get_performance_summary, get_params, set_params,
)
from ai.alerts import (
    alert_trade_opened, alert_trade_closed,
    alert_high_confidence_signal, alert_risk_warning,
    send_daily_report, send_weekly_digest,
)
from ai.bot_engine import (
    start_bot, stop_bot, get_session, list_active_sessions,
    run_cycle, fetch_ohlcv,
)

__all__ = [
    # Signal generation
    "generate_signal", "TradeSignal", "DEFAULT_PARAMS",
    # Indicators
    "rsi", "macd", "ema", "bollinger_bands", "atr", "volume_analysis",
    # SMC
    "analyse_smc", "SMCAnalysis",
    # Sentiment
    "get_sentiment", "get_sentiment_sync", "SentimentResult",
    # Risk
    "RiskConfig", "calculate_position_size", "assess_trade_risk",
    "trailing_stop", "breakeven_stop", "partial_profit_levels",
    # Self-optimizer
    "record_trade_result", "optimise_parameters",
    "get_performance_summary", "get_params", "set_params",
    # Alerts
    "alert_trade_opened", "alert_trade_closed",
    "alert_high_confidence_signal", "alert_risk_warning",
    "send_daily_report", "send_weekly_digest",
    # Bot engine
    "start_bot", "stop_bot", "get_session", "list_active_sessions",
    "run_cycle", "fetch_ohlcv",
]
