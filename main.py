#!/usr/bin/env python3
# main.py — Terminal CLI entry point

import argparse
import logging
import signal
import sys
import threading
import time

import candle_analysis as ca
import indicators as ind
from candle_store import CandleStore
from config import PAIRS, INITIAL_BALANCE, SIGNAL_CONFIDENCE_THRESHOLD
from paper_trader import PaperTrader
from portfolio import Portfolio
from price_fallback import get_prices_coingecko
from strategy import get_strategy
from ws_stream import CryptoWebSocket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("trades.log")],
)
logger = logging.getLogger(__name__)


def run(pairs: list[str], strategy_name: str, balance: float) -> None:
    store = CandleStore()
    trader = PaperTrader(balance)
    portfolio = Portfolio(balance)
    strat = get_strategy(strategy_name)
    stop_event = threading.Event()

    print(f"\n{'='*55}")
    print(f"  CRYPTO ALGO BOT  |  Strategy: {strategy_name.upper()}  |  ${balance:,.0f}")
    print(f"  Pairs: {', '.join(pairs)}")
    print(f"{'='*55}\n")

    # Seed initial prices
    try:
        prices = get_prices_coingecko(pairs)
        for sym, p in prices.items():
            if p > 0:
                store.set_price(sym, p)
                print(f"  {sym}: ${p:,.2f} (CoinGecko)")
    except Exception:
        pass

    def on_candle_close(symbol: str, candle: dict) -> None:
        store.add_candle(symbol, candle)
        count = store.get_count(symbol)

        if not store.is_ready(symbol):
            print(f"  Warming {symbol}: {count}/26 candles")
            return

        df = store.get_dataframe(symbol)
        ci = ca.analyze_candles(df)
        comp = ind.compute_all(df)
        sig = strat.generate_signal(df, ci, comp)

        price = candle["close"]
        marker = "+" if candle["close"] >= candle["open"] else "-"
        trend_sym = {"UPTREND": "^", "DOWNTREND": "v", "SIDEWAYS": "-"}[ci["trend"]]
        print(
            f"  [{marker}] {symbol} ${price:>12,.2f} | "
            f"{ci['pattern']:<22} | RSI:{comp['rsi']:>5.1f} | "
            f"Trend:{ci['trend']:<10} {trend_sym} | "
            f"{sig['signal']:<4} ({sig['confidence']}%)"
        )

        if sig["signal"] == "BUY" and sig["confidence"] >= SIGNAL_CONFIDENCE_THRESHOLD:
            t = trader.buy(symbol, price)
            if t:
                portfolio.record_trade(t.to_dict())
                print(f"  >> BUY  {symbol} @ {price:.4f}")
        elif sig["signal"] == "SELL" and sig["confidence"] >= SIGNAL_CONFIDENCE_THRESHOLD:
            t = trader.sell(symbol, price)
            if t:
                portfolio.record_trade(t.to_dict())
                print(f"  >> SELL {symbol} @ {price:.4f} | PnL: ${t.pnl:+.2f}")

        all_prices = store.get_all_prices()
        for sym in list(trader._positions.keys()):
            p = all_prices.get(sym, 0)
            if p:
                t = trader.check_stop_loss(sym, p) or trader.check_take_profit(sym, p)
                if t:
                    portfolio.record_trade(t.to_dict())
        portfolio.snapshot_equity(trader.get_total_value(all_prices))

    def on_price_update(symbol: str, price: float) -> None:
        store.set_price(symbol, price)

    ws = CryptoWebSocket(on_candle_close=on_candle_close, on_price_update=on_price_update)
    ws.start()
    print("  WebSocket stream started. Waiting for candles...\n")

    def handle_exit(*_):
        stop_event.set()

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    try:
        while not stop_event.is_set():
            time.sleep(1)
    finally:
        ws.stop()
        portfolio.print_summary()
        portfolio.export_csv("trade_history.csv")
        print("Saved: trade_history.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", nargs="+", default=PAIRS)
    p.add_argument("--strategy", default="master", choices=["candle", "trend", "rsi", "master"])
    p.add_argument("--balance", type=float, default=INITIAL_BALANCE)
    args = p.parse_args()
    run(args.pairs, args.strategy, args.balance)
