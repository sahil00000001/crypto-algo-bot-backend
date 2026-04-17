# paper_trader.py — Simulated trading (no real money)

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from config import MAX_POSITION_PCT, STOP_LOSS_PCT, TAKE_PROFIT_PCT

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    entry_price: float
    quantity: float
    cost_usdt: float
    stop_loss: float
    take_profit: float
    opened_at: datetime = field(default_factory=datetime.utcnow)

    def current_value(self, price: float) -> float:
        return self.quantity * price

    def unrealized_pnl(self, price: float) -> float:
        return self.current_value(price) - self.cost_usdt

    def unrealized_pnl_pct(self, price: float) -> float:
        return (self.unrealized_pnl(price) / self.cost_usdt * 100) if self.cost_usdt else 0.0


@dataclass
class Trade:
    symbol: str
    side: str
    price: float
    quantity: float
    cost_usdt: float
    pnl: float = 0.0
    reason: str = "signal"
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "side": self.side,
            "price": self.price, "quantity": round(self.quantity, 6),
            "cost_usdt": round(self.cost_usdt, 2),
            "pnl": round(self.pnl, 2), "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


class PaperTrader:
    def __init__(self, initial_balance: float = 10000.0):
        self._balance = initial_balance
        self._initial = initial_balance
        self._positions: dict[str, Position] = {}
        self._trades: list[Trade] = []
        self._peak_equity = initial_balance
        self._max_dd_pct = 0.0

    def buy(self, symbol: str, price: float, pct_of_balance: float = MAX_POSITION_PCT) -> Optional[Trade]:
        if symbol in self._positions or price <= 0:
            return None
        spend = min(self._balance * (pct_of_balance / 100), self._balance)
        if spend < 1.0:
            return None
        qty = spend / price
        self._balance -= spend
        self._positions[symbol] = Position(
            symbol=symbol, entry_price=price, quantity=qty, cost_usdt=spend,
            stop_loss=price * (1 - STOP_LOSS_PCT / 100),
            take_profit=price * (1 + TAKE_PROFIT_PCT / 100),
        )
        t = Trade(symbol=symbol, side="BUY", price=price, quantity=qty, cost_usdt=spend)
        self._trades.append(t)
        logger.info(f"BUY {symbol} @ {price:.4f} | qty={qty:.6f} | ${spend:.2f}")
        return t

    def sell(self, symbol: str, price: float, reason: str = "signal") -> Optional[Trade]:
        pos = self._positions.pop(symbol, None)
        if not pos:
            return None
        proceeds = pos.quantity * price
        pnl = proceeds - pos.cost_usdt
        self._balance += proceeds
        t = Trade(symbol=symbol, side="SELL", price=price,
                  quantity=pos.quantity, cost_usdt=proceeds, pnl=pnl, reason=reason)
        self._trades.append(t)
        logger.info(f"SELL {symbol} @ {price:.4f} | pnl={pnl:+.2f} | reason={reason}")
        return t

    def check_stop_loss(self, symbol: str, price: float) -> Optional[Trade]:
        pos = self._positions.get(symbol)
        if pos and price <= pos.stop_loss:
            return self.sell(symbol, price, "stop_loss")
        return None

    def check_take_profit(self, symbol: str, price: float) -> Optional[Trade]:
        pos = self._positions.get(symbol)
        if pos and price >= pos.take_profit:
            return self.sell(symbol, price, "take_profit")
        return None

    def get_positions(self) -> dict:
        return {s: {
            "symbol": p.symbol, "entry_price": p.entry_price,
            "quantity": round(p.quantity, 6), "cost_usdt": round(p.cost_usdt, 2),
            "stop_loss": round(p.stop_loss, 4), "take_profit": round(p.take_profit, 4),
            "opened_at": p.opened_at.isoformat(),
        } for s, p in self._positions.items()}

    def get_positions_with_pnl(self, prices: dict[str, float]) -> list[dict]:
        result = []
        for s, p in self._positions.items():
            price = prices.get(s, p.entry_price)
            d = self.get_positions()[s]
            d.update({
                "current_price": price,
                "current_value": round(p.current_value(price), 2),
                "unrealized_pnl": round(p.unrealized_pnl(price), 2),
                "unrealized_pnl_pct": round(p.unrealized_pnl_pct(price), 2),
            })
            result.append(d)
        return result

    def get_balance(self) -> float:
        return round(self._balance, 2)

    def get_total_value(self, prices: dict[str, float]) -> float:
        pos_val = sum(p.quantity * prices.get(s, p.entry_price)
                      for s, p in self._positions.items())
        total = round(self._balance + pos_val, 2)
        # Track peak equity for max drawdown
        if total > self._peak_equity:
            self._peak_equity = total
        elif self._peak_equity > 0:
            dd = (self._peak_equity - total) / self._peak_equity * 100
            if dd > self._max_dd_pct:
                self._max_dd_pct = dd
        return total

    def get_trades(self) -> list[dict]:
        return [t.to_dict() for t in self._trades]

    def get_stats(self) -> dict:
        sells = [t for t in self._trades if t.side == "SELL"]
        wins = [t for t in sells if t.pnl > 0]
        total_pnl = sum(t.pnl for t in sells)
        wr = len(wins) / len(sells) * 100 if sells else 0.0
        return {
            "initial_balance": self._initial,
            "current_balance": round(self._balance, 2),
            "total_trades": len(sells),
            "winning_trades": len(wins),
            "losing_trades": len(sells) - len(wins),
            "win_rate": round(wr, 1),
            "total_pnl": round(total_pnl, 2),
            "best_trade": round(max((t.pnl for t in sells), default=0.0), 2),
            "worst_trade": round(min((t.pnl for t in sells), default=0.0), 2),
            "return_pct": round(total_pnl / self._initial * 100, 2),
            "max_drawdown": round(self._max_dd_pct, 2),
        }
