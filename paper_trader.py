# paper_trader.py — Simulated trading engine (no real money)

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
    quantity: float           # units of crypto
    cost_usdt: float          # USDT spent
    stop_loss: float
    take_profit: float
    opened_at: datetime = field(default_factory=datetime.utcnow)

    def current_value(self, price: float) -> float:
        return self.quantity * price

    def unrealized_pnl(self, price: float) -> float:
        return self.current_value(price) - self.cost_usdt

    def unrealized_pnl_pct(self, price: float) -> float:
        return (self.unrealized_pnl(price) / self.cost_usdt) * 100 if self.cost_usdt else 0.0


@dataclass
class Trade:
    symbol: str
    side: str                 # "BUY" or "SELL"
    price: float
    quantity: float
    cost_usdt: float
    pnl: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "price": self.price,
            "quantity": round(self.quantity, 6),
            "cost_usdt": round(self.cost_usdt, 2),
            "pnl": round(self.pnl, 2),
            "timestamp": self.timestamp.isoformat(),
        }


class PaperTrader:
    def __init__(self, initial_balance: float = 10000.0):
        self._balance: float = initial_balance
        self._initial_balance: float = initial_balance
        self._positions: dict[str, Position] = {}
        self._trades: list[Trade] = []

    # ── Core Trading Methods ──────────────────────────────────────────────────

    def buy(self, symbol: str, price: float, amount_usdt: Optional[float] = None) -> Optional[Trade]:
        """Open a long position. amount_usdt defaults to MAX_POSITION_PCT of balance."""
        if symbol in self._positions:
            logger.info(f"Already holding {symbol}, skipping buy.")
            return None

        if price <= 0:
            return None

        max_spend = self._balance * (MAX_POSITION_PCT / 100)
        spend = min(amount_usdt or max_spend, max_spend, self._balance)

        if spend < 1.0:
            logger.warning(f"Insufficient balance to buy {symbol}: ${self._balance:.2f}")
            return None

        quantity = spend / price
        stop = price * (1 - STOP_LOSS_PCT / 100)
        take = price * (1 + TAKE_PROFIT_PCT / 100)

        self._balance -= spend
        self._positions[symbol] = Position(
            symbol=symbol,
            entry_price=price,
            quantity=quantity,
            cost_usdt=spend,
            stop_loss=stop,
            take_profit=take,
        )

        trade = Trade(symbol=symbol, side="BUY", price=price, quantity=quantity, cost_usdt=spend)
        self._trades.append(trade)
        logger.info(f"BUY {symbol} @ {price:.4f} | qty={quantity:.6f} | ${spend:.2f}")
        return trade

    def sell(self, symbol: str, price: float, reason: str = "signal") -> Optional[Trade]:
        """Close an existing position."""
        pos = self._positions.pop(symbol, None)
        if pos is None:
            return None

        proceeds = pos.quantity * price
        pnl = proceeds - pos.cost_usdt
        self._balance += proceeds

        trade = Trade(
            symbol=symbol,
            side="SELL",
            price=price,
            quantity=pos.quantity,
            cost_usdt=proceeds,
            pnl=pnl,
        )
        self._trades.append(trade)
        logger.info(
            f"SELL {symbol} @ {price:.4f} | pnl={pnl:+.2f} | reason={reason} | balance=${self._balance:.2f}"
        )
        return trade

    def check_stop_loss(self, symbol: str, current_price: float) -> Optional[Trade]:
        """Auto-sell if stop-loss level is breached."""
        pos = self._positions.get(symbol)
        if pos and current_price <= pos.stop_loss:
            logger.info(f"STOP-LOSS triggered for {symbol} @ {current_price:.4f}")
            return self.sell(symbol, current_price, reason="stop_loss")
        return None

    def check_take_profit(self, symbol: str, current_price: float) -> Optional[Trade]:
        """Auto-sell if take-profit level is reached."""
        pos = self._positions.get(symbol)
        if pos and current_price >= pos.take_profit:
            logger.info(f"TAKE-PROFIT triggered for {symbol} @ {current_price:.4f}")
            return self.sell(symbol, current_price, reason="take_profit")
        return None

    # ── Query Methods ─────────────────────────────────────────────────────────

    def get_position(self, symbol: str) -> Optional[dict]:
        pos = self._positions.get(symbol)
        if pos is None:
            return None
        return {
            "symbol": pos.symbol,
            "entry_price": pos.entry_price,
            "quantity": round(pos.quantity, 6),
            "cost_usdt": round(pos.cost_usdt, 2),
            "stop_loss": round(pos.stop_loss, 4),
            "take_profit": round(pos.take_profit, 4),
            "opened_at": pos.opened_at.isoformat(),
        }

    def get_position_with_pnl(self, symbol: str, current_price: float) -> Optional[dict]:
        pos = self._positions.get(symbol)
        if pos is None:
            return None
        base = self.get_position(symbol)
        base["current_price"] = current_price
        base["current_value"] = round(pos.current_value(current_price), 2)
        base["unrealized_pnl"] = round(pos.unrealized_pnl(current_price), 2)
        base["unrealized_pnl_pct"] = round(pos.unrealized_pnl_pct(current_price), 2)
        return base

    def get_all_positions(self, current_prices: dict[str, float]) -> list[dict]:
        return [
            self.get_position_with_pnl(sym, current_prices.get(sym, pos.entry_price))
            for sym, pos in self._positions.items()
        ]

    def get_balance(self) -> float:
        return round(self._balance, 2)

    def get_total_value(self, current_prices: dict[str, float]) -> float:
        positions_value = sum(
            pos.quantity * current_prices.get(sym, pos.entry_price)
            for sym, pos in self._positions.items()
        )
        return round(self._balance + positions_value, 2)

    def get_trades(self) -> list[dict]:
        return [t.to_dict() for t in self._trades]

    def get_stats(self) -> dict:
        sells = [t for t in self._trades if t.side == "SELL"]
        wins = [t for t in sells if t.pnl > 0]
        losses = [t for t in sells if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in sells)
        win_rate = len(wins) / len(sells) * 100 if sells else 0.0

        return {
            "initial_balance": self._initial_balance,
            "current_balance": round(self._balance, 2),
            "total_trades": len(sells),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "best_trade": round(max((t.pnl for t in sells), default=0.0), 2),
            "worst_trade": round(min((t.pnl for t in sells), default=0.0), 2),
            "return_pct": round((total_pnl / self._initial_balance) * 100, 2),
        }
