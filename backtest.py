#!/usr/bin/env python3
# backtest.py — Backtest any strategy on historical candle data

import argparse
import math
import sys
from datetime import datetime, timedelta

import pandas as pd

import candle_analysis as ca
import indicators as ind
from config import INITIAL_BALANCE, STOP_LOSS_PCT, TAKE_PROFIT_PCT, SMA_SHORT, SMA_LONG, SIGNAL_CONFIDENCE_THRESHOLD
from data_fetcher import get_klines
from strategy import get_strategy

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest crypto strategies")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1h")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--strategy", default="master", choices=["candle", "trend", "rsi", "master"])
    p.add_argument("--balance", type=float, default=INITIAL_BALANCE)
    return p.parse_args()


def run_backtest(symbol: str, interval: str, days: int, strategy_name: str, balance: float) -> dict:
    strategy = get_strategy(strategy_name)

    # Map days → candle limit (approximate)
    interval_minutes = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
                        "1h": 60, "4h": 240, "1d": 1440}.get(interval, 60)
    limit = min(1000, days * 24 * 60 // interval_minutes)

    print(f"Fetching {limit} {interval} candles for {symbol}…")
    df = get_klines(symbol, interval, limit=limit)
    print(f"Loaded {len(df)} candles ({df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]})")

    # Paper trading state
    cash = balance
    position = None
    trades: list[dict] = []
    equity_curve = [balance]

    for i in range(50, len(df)):  # need warmup period for indicators
        slice_df = df.iloc[:i + 1].copy()
        current = float(slice_df.iloc[-1]["close"])
        ts = slice_df.iloc[-1]["timestamp"]

        # Check SL/TP on open position
        if position:
            if current <= position["stop_loss"]:
                proceeds = position["qty"] * current
                pnl = proceeds - position["cost"]
                cash += proceeds
                trades.append({
                    "side": "SELL", "price": current, "pnl": pnl,
                    "reason": "STOP_LOSS", "timestamp": ts,
                })
                position = None
            elif current >= position["take_profit"]:
                proceeds = position["qty"] * current
                pnl = proceeds - position["cost"]
                cash += proceeds
                trades.append({
                    "side": "SELL", "price": current, "pnl": pnl,
                    "reason": "TAKE_PROFIT", "timestamp": ts,
                })
                position = None

        candle_info = ca.analyze_candles(slice_df)
        signal_result = strategy.generate_signal(slice_df, candle_info)
        sig = signal_result["signal"]
        conf = signal_result["confidence"]

        if sig == "BUY" and conf >= SIGNAL_CONFIDENCE_THRESHOLD and not position and cash > 10:
            spend = cash * 0.30
            qty = spend / current
            position = {
                "entry": current,
                "qty": qty,
                "cost": spend,
                "stop_loss": current * (1 - STOP_LOSS_PCT / 100),
                "take_profit": current * (1 + TAKE_PROFIT_PCT / 100),
            }
            cash -= spend
            trades.append({"side": "BUY", "price": current, "pnl": 0, "reason": "SIGNAL", "timestamp": ts})

        elif sig == "SELL" and conf >= SIGNAL_CONFIDENCE_THRESHOLD and position:
            proceeds = position["qty"] * current
            pnl = proceeds - position["cost"]
            cash += proceeds
            trades.append({
                "side": "SELL", "price": current, "pnl": pnl,
                "reason": "SIGNAL", "timestamp": ts,
            })
            position = None

        # Equity snapshot
        total = cash + (position["qty"] * current if position else 0)
        equity_curve.append(total)

    # Close open position at end
    final_price = float(df.iloc[-1]["close"])
    if position:
        proceeds = position["qty"] * final_price
        pnl = proceeds - position["cost"]
        cash += proceeds
        trades.append({"side": "SELL", "price": final_price, "pnl": pnl, "reason": "END", "timestamp": df.iloc[-1]["timestamp"]})
        position = None
        equity_curve[-1] = cash

    final_equity = cash
    sells = [t for t in trades if t["side"] == "SELL"]
    wins = [t for t in sells if t["pnl"] > 0]
    total_pnl = sum(t["pnl"] for t in sells)
    win_rate = len(wins) / len(sells) * 100 if sells else 0.0

    # Max drawdown
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100 if peak else 0
        max_dd = max(max_dd, dd)

    # Sharpe ratio (simplified)
    returns = [(equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
               for i in range(1, len(equity_curve)) if equity_curve[i - 1] > 0]
    if returns:
        mean_r = sum(returns) / len(returns)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / len(returns)) if len(returns) > 1 else 1e-9
        sharpe = (mean_r / std_r) * math.sqrt(252) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "symbol": symbol,
        "interval": interval,
        "strategy": strategy_name,
        "total_candles": len(df),
        "total_trades": len(sells),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "return_pct": (total_pnl / balance) * 100,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "initial_balance": balance,
        "final_equity": final_equity,
        "wins": len(wins),
        "losses": len(sells) - len(wins),
        "equity_curve": equity_curve,
    }


def print_results(results: dict) -> None:
    if HAS_RICH:
        _print_rich(results)
    else:
        _print_plain(results)


def _print_plain(r: dict) -> None:
    print("\n" + "=" * 55)
    print(f"  BACKTEST RESULTS — {r['symbol']} {r['interval']} ({r['strategy'].upper()})")
    print("=" * 55)
    print(f"  Total Candles    : {r['total_candles']}")
    print(f"  Total Trades     : {r['total_trades']} ({r['wins']}W / {r['losses']}L)")
    print(f"  Win Rate         : {r['win_rate']:.1f}%")
    print(f"  Total P&L        : ${r['total_pnl']:+.2f}")
    print(f"  Return           : {r['return_pct']:+.2f}%")
    print(f"  Max Drawdown     : {r['max_drawdown']:.2f}%")
    print(f"  Sharpe Ratio     : {r['sharpe']:.2f}")
    print(f"  Final Equity     : ${r['final_equity']:.2f}")
    print("=" * 55)

    # ASCII equity chart
    curve = r["equity_curve"]
    if len(curve) > 1:
        n = min(60, len(curve))
        step = len(curve) // n
        sampled = [curve[i * step] for i in range(n)]
        min_v, max_v = min(sampled), max(sampled)
        height = 8
        print("\n  Equity Curve:")
        for row in range(height, -1, -1):
            threshold = min_v + (max_v - min_v) * row / height
            line = "  "
            for v in sampled:
                line += "█" if v >= threshold else " "
            if row == height:
                print(f"  ${max_v:>9.0f} |{line[2:]}")
            elif row == 0:
                print(f"  ${min_v:>9.0f} |{line[2:]}")
            else:
                print(f"             |{line[2:]}")
        print()


def _print_rich(r: dict) -> None:
    console = Console()

    pnl_color = "green" if r["total_pnl"] >= 0 else "red"
    title = f"[bold]BACKTEST — {r['symbol']} {r['interval']} | Strategy: {r['strategy'].upper()}[/bold]"

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold dim")
    table.add_column("Metric", style="dim", width=22)
    table.add_column("Value", justify="right")

    table.add_row("Total Candles", str(r["total_candles"]))
    table.add_row("Total Trades", f"{r['total_trades']} ({r['wins']}W / {r['losses']}L)")
    table.add_row("Win Rate", f"[{'green' if r['win_rate'] >= 50 else 'red'}]{r['win_rate']:.1f}%[/]")
    table.add_row("Total P&L", f"[{pnl_color}]${r['total_pnl']:+.2f}[/]")
    table.add_row("Return", f"[{pnl_color}]{r['return_pct']:+.2f}%[/]")
    table.add_row("Max Drawdown", f"[{'red' if r['max_drawdown'] > 10 else 'yellow'}]{r['max_drawdown']:.2f}%[/]")
    table.add_row("Sharpe Ratio", f"{r['sharpe']:.2f}")
    table.add_row("Initial Balance", f"${r['initial_balance']:,.2f}")
    table.add_row("Final Equity", f"[{pnl_color}]${r['final_equity']:,.2f}[/]")

    console.print(Panel(table, title=title, border_style="blue"))


if __name__ == "__main__":
    args = parse_args()
    results = run_backtest(args.symbol, args.interval, args.days, args.strategy, args.balance)
    print_results(results)
