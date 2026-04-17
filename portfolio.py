# portfolio.py — Portfolio tracking, stats, and trade history export

import csv
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class Portfolio:
    """Tracks all trades and computes portfolio statistics."""

    def __init__(self, initial_balance: float = 10_000.0):
        self._initial_balance = initial_balance
        self._trades: list[dict] = []
        self._equity_curve: list[dict] = []      # balance snapshots over time

    def record_trade(self, trade_dict: dict) -> None:
        """Record a completed trade."""
        self._trades.append(trade_dict)
        logger.info(f"Portfolio recorded: {trade_dict['side']} {trade_dict['symbol']} pnl={trade_dict.get('pnl', 0):.2f}")

    def snapshot_equity(self, total_value: float) -> None:
        """Save an equity curve data point."""
        self._equity_curve.append({
            "timestamp": datetime.utcnow().isoformat(),
            "value": round(total_value, 2),
        })

    def get_trade_history(self) -> list[dict]:
        return list(reversed(self._trades))

    def get_recent_trades(self, n: int = 10) -> list[dict]:
        return list(reversed(self._trades[-n:]))

    def get_equity_curve(self) -> list[dict]:
        return list(self._equity_curve)

    def get_stats(self) -> dict:
        sells = [t for t in self._trades if t.get("side") == "SELL"]
        buys = [t for t in self._trades if t.get("side") == "BUY"]
        wins = [t for t in sells if t.get("pnl", 0) > 0]
        losses = [t for t in sells if t.get("pnl", 0) <= 0]
        total_pnl = sum(t.get("pnl", 0) for t in sells)
        win_rate = len(wins) / len(sells) * 100 if sells else 0.0

        # Max drawdown from equity curve
        max_drawdown = self._compute_max_drawdown()

        return {
            "total_trades": len(sells),
            "buy_trades": len(buys),
            "sell_trades": len(sells),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "max_drawdown": round(max_drawdown, 2),
            "best_trade": round(max((t.get("pnl", 0) for t in sells), default=0.0), 2),
            "worst_trade": round(min((t.get("pnl", 0) for t in sells), default=0.0), 2),
            "return_pct": round((total_pnl / self._initial_balance) * 100, 2),
        }

    def _compute_max_drawdown(self) -> float:
        if len(self._equity_curve) < 2:
            return 0.0
        values = [e["value"] for e in self._equity_curve]
        peak = values[0]
        max_dd = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100 if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd

    def export_csv(self, filename: str = "trade_history.csv") -> None:
        """Export trade history to CSV file."""
        if not self._trades:
            logger.warning("No trades to export.")
            return
        keys = ["timestamp", "symbol", "side", "price", "quantity", "cost_usdt", "pnl"]
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for trade in self._trades:
                writer.writerow({k: trade.get(k, "") for k in keys})
        logger.info(f"Trade history exported to {filename}")

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        stats = self.get_stats()
        print("\n" + "=" * 50)
        print("  PORTFOLIO SUMMARY")
        print("=" * 50)
        for key, val in stats.items():
            label = key.replace("_", " ").title()
            suffix = "%" if "pct" in key or "rate" in key else ""
            prefix = "$" if "pnl" in key or "trade" in key else ""
            print(f"  {label:<25} {prefix}{val}{suffix}")
        print("=" * 50 + "\n")
