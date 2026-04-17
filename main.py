#!/usr/bin/env python3
# main.py — CLI entry point for the trading bot (terminal mode, no web UI)

import argparse
import logging
import signal
import sys
import time
import threading

import pandas as pd

import candle_analysis as ca
import indicators as ind
from config import PAIRS, DEFAULT_INTERVAL, INITIAL_BALANCE, SIGNAL_CONFIDENCE_THRESHOLD, SMA_SHORT, SMA_LONG
from data_fetcher import get_klines, get_ticker_24hr
from paper_trader import PaperTrader
from portfolio import Portfolio
from strategy import get_strategy
from websocket_stream import LiveCandleStream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trades.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crypto Algo Trading Bot")
    parser.add_argument("--pairs", nargs="+", default=PAIRS, help="Trading pairs")
    parser.add_argument("--strategy", default="master", choices=["candle", "trend", "rsi", "master"])
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="Candle interval (1m, 5m, 1h …)")
    parser.add_argument("--balance", type=float, default=INITIAL_BALANCE, help="Starting USDT balance")
    return parser.parse_args()


def run_bot(pairs: list[str], strategy_name: str, interval: str, balance: float) -> None:
    strategy = get_strategy(strategy_name)
    trader = PaperTrader(balance)
    portfolio = Portfolio(balance)
    candle_dfs: dict[str, pd.DataFrame] = {}
    streams: dict[str, LiveCandleStream] = {}
    lock = threading.Lock()

    # ── Load historical data ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  CRYPTO ALGO TRADING BOT  — Paper Trading Mode")
    print(f"  Strategy: {strategy_name.upper()} | Interval: {interval}")
    print(f"  Pairs: {', '.join(pairs)} | Balance: ${balance:,.2f}")
    print(f"{'='*60}\n")

    for symbol in pairs:
        try:
            print(f"Loading historical candles for {symbol}…")
            df = get_klines(symbol, interval, 500)
            candle_dfs[symbol] = df
            print(f"  ✓ {symbol}: {len(df)} candles loaded")
        except Exception as e:
            logger.error(f"Failed to load {symbol}: {e}")
            candle_dfs[symbol] = pd.DataFrame()

    # ── Candle close handler ──────────────────────────────────────────────────
    def on_candle_close(candle, symbol: str) -> None:
        with lock:
            df = candle_dfs.get(symbol)
            if df is None or df.empty:
                return

            new_row = pd.DataFrame([{
                "timestamp": pd.Timestamp(candle.open_time, unit="ms"),
                "open": candle.open, "high": candle.high,
                "low": candle.low, "close": candle.close,
                "volume": candle.volume, "num_trades": candle.num_trades,
                "is_up": candle.close >= candle.open,
                "body_size": abs(candle.close - candle.open),
                "upper_wick": candle.high - max(candle.open, candle.close),
                "lower_wick": min(candle.open, candle.close) - candle.low,
                "candle_range": candle.high - candle.low,
                "body_pct": abs(candle.close - candle.open) / (candle.high - candle.low + 1e-10),
            }])
            df = pd.concat([df, new_row], ignore_index=True).tail(500)
            candle_dfs[symbol] = df

            candle_info = ca.analyze_candles(df)
            computed = ind.compute_all(df, SMA_SHORT, SMA_LONG)
            signal_result = strategy.generate_signal(df, candle_info)

            signal = signal_result["signal"]
            confidence = signal_result["confidence"]
            price = candle.close

            marker = "🟢" if candle.is_up else "🔴"
            print(
                f"{marker} {symbol} | {price:>12,.4f} | "
                f"Pattern: {candle_info['pattern']:<22} | "
                f"RSI: {computed['rsi']:>5.1f} | "
                f"Trend: {candle_info['trend']:<10} | "
                f"Signal: {signal:<4} ({confidence}%)"
            )

            if signal == "BUY" and confidence >= SIGNAL_CONFIDENCE_THRESHOLD:
                trade = trader.buy(symbol, price)
                if trade:
                    portfolio.record_trade(trade.to_dict())
                    print(f"  ✅ BUY  {symbol} @ {price:.4f} | reason: {signal_result['reason'][:60]}")

            elif signal == "SELL" and confidence >= SIGNAL_CONFIDENCE_THRESHOLD:
                trade = trader.sell(symbol, price, reason="signal")
                if trade:
                    portfolio.record_trade(trade.to_dict())
                    print(f"  ❌ SELL {symbol} @ {price:.4f} | pnl: ${trade.pnl:+.2f}")

            # Check SL/TP
            prices = {s: df["close"].iloc[-1] for s, df in candle_dfs.items() if not df.empty}
            for sym in list(trader._positions.keys()):
                p = prices.get(sym)
                if p:
                    t = trader.check_stop_loss(sym, p) or trader.check_take_profit(sym, p)
                    if t:
                        portfolio.record_trade(t.to_dict())

            # Balance update
            prices_now = {s: float(df["close"].iloc[-1]) for s, df in candle_dfs.items() if not df.empty}
            total = trader.get_total_value(prices_now)
            portfolio.snapshot_equity(total)

    # ── Start streams ─────────────────────────────────────────────────────────
    for symbol in pairs:
        stream = LiveCandleStream(
            symbol=symbol,
            interval=interval,
            on_candle_close=lambda c, s=symbol: on_candle_close(c, s),
        )
        stream.start()
        streams[symbol] = stream
        print(f"  📡 WebSocket stream started: {symbol}")

    print(f"\nBot running — watching {len(pairs)} pair(s). Press Ctrl+C to stop.\n")

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    stop = threading.Event()

    def handle_signal(*_):
        stop.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while not stop.is_set():
            time.sleep(1)
    finally:
        print("\nShutting down…")
        for stream in streams.values():
            stream.stop()
        portfolio.print_summary()
        portfolio.export_csv("trade_history.csv")
        print("Trade history saved to trade_history.csv")


if __name__ == "__main__":
    args = parse_args()
    run_bot(args.pairs, args.strategy, args.interval, args.balance)
