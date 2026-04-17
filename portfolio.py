# portfolio.py — Trade history and portfolio statistics

import csv
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Portfolio:
    def __init__(self, initial_balance: float = 10_000.0):
        self._initial = initial_balance
        self._trades: list[dict] = []
        self._equity: list[dict] = []

    def record_trade(self, trade: dict) -> None:
        self._trades.append(trade)

    def snapshot_equity(self, total_value: float) -> None:
        self._equity.append({"timestamp": datetime.utcnow().isoformat(),
                              "value": round(total_value, 2)})

    def get_trade_history(self) -> list[dict]:
        return list(reversed(self._trades))

    def get_recent_trades(self, n: int = 10) -> list[dict]:
        return list(reversed(self._trades[-n:]))

    def get_equity_curve(self) -> list[dict]:
        return list(self._equity)

    def get_stats(self) -> dict:
        sells = [t for t in self._trades if t.get("side") == "SELL"]
        wins = [t for t in sells if t.get("pnl", 0) > 0]
        total_pnl = sum(t.get("pnl", 0) for t in sells)
        wr = len(wins) / len(sells) * 100 if sells else 0.0
        return {
            "total_trades": len(sells),
            "winning_trades": len(wins),
            "losing_trades": len(sells) - len(wins),
            "win_rate": round(wr, 1),
            "total_pnl": round(total_pnl, 2),
            "return_pct": round(total_pnl / self._initial * 100, 2),
            "best_trade": round(max((t.get("pnl", 0) for t in sells), default=0.0), 2),
            "worst_trade": round(min((t.get("pnl", 0) for t in sells), default=0.0), 2),
            "max_drawdown": self._max_drawdown(),
        }

    def _max_drawdown(self) -> float:
        if len(self._equity) < 2:
            return 0.0
        peak, max_dd = self._equity[0]["value"], 0.0
        for e in self._equity:
            v = e["value"]
            peak = max(peak, v)
            max_dd = max(max_dd, (peak - v) / peak * 100 if peak else 0)
        return round(max_dd, 2)

    def export_csv(self, filename: str = "trade_history.csv") -> None:
        if not self._trades:
            return
        keys = ["timestamp", "symbol", "side", "price", "quantity", "cost_usdt", "pnl", "reason"]
        with open(filename, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for t in self._trades:
                w.writerow({k: t.get(k, "") for k in keys})
        logger.info(f"Exported {len(self._trades)} trades to {filename}")

    def print_summary(self) -> None:
        s = self.get_stats()
        print("\n" + "=" * 50)
        print("  PORTFOLIO SUMMARY")
        print("=" * 50)
        for k, v in s.items():
            print(f"  {k.replace('_',' ').title():<25} {v}")
        print("=" * 50)
