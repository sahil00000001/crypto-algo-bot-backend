# candle_store.py — Thread-safe in-memory candle storage built from WebSocket

import threading
from typing import Optional

import numpy as np
import pandas as pd

from config import MAX_CANDLES_STORED, MIN_CANDLES_FOR_SIGNALS


class CandleStore:
    """
    Stores confirmed candles per symbol in memory.
    Built entirely from WebSocket — no REST calls.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._candles: dict[str, list[dict]] = {}        # confirmed candles
        self._current: dict[str, Optional[dict]] = {}    # forming candle
        self._prices: dict[str, float] = {}              # latest price

    # ── Write ──────────────────────────────────────────────────────────────────

    def add_candle(self, symbol: str, candle: dict) -> None:
        """Append a confirmed (closed) candle."""
        with self._lock:
            if symbol not in self._candles:
                self._candles[symbol] = []
            self._candles[symbol].append(candle)
            # Cap storage
            if len(self._candles[symbol]) > MAX_CANDLES_STORED:
                self._candles[symbol] = self._candles[symbol][-MAX_CANDLES_STORED:]
            self._prices[symbol] = candle["close"]

    def update_current(self, symbol: str, candle: dict) -> None:
        """Update the currently forming (unconfirmed) candle."""
        with self._lock:
            self._current[symbol] = candle
            self._prices[symbol] = candle["close"]

    def bulk_load_from_df(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Seed the store with historical candles from a DataFrame (e.g. Kraken REST).
        Each row must have: timestamp (datetime), open, high, low, close, volume.
        Existing candles for the symbol are replaced.
        """
        if df.empty:
            return
        rows = []
        for _, row in df.iterrows():
            ts = row["timestamp"]
            # Store timestamps as ms integers (consistent with WebSocket candles)
            ts_ms = int(ts.timestamp() * 1000) if hasattr(ts, "timestamp") else int(ts)
            rows.append({
                "timestamp": ts_ms,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })
        with self._lock:
            self._candles[symbol] = rows[-MAX_CANDLES_STORED:]
            if rows:
                self._prices[symbol] = rows[-1]["close"]

    def set_price(self, symbol: str, price: float) -> None:
        with self._lock:
            self._prices[symbol] = price

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_candles(self, symbol: str) -> list[dict]:
        with self._lock:
            return list(self._candles.get(symbol, []))

    def get_count(self, symbol: str) -> int:
        with self._lock:
            return len(self._candles.get(symbol, []))

    def is_ready(self, symbol: str) -> bool:
        return self.get_count(symbol) >= MIN_CANDLES_FOR_SIGNALS

    def get_current_price(self, symbol: str) -> float:
        with self._lock:
            return self._prices.get(symbol, 0.0)

    def get_all_prices(self) -> dict[str, float]:
        with self._lock:
            return dict(self._prices)

    def get_warming_status(self) -> dict[str, dict]:
        with self._lock:
            result = {}
            for sym, candles in self._candles.items():
                count = len(candles)
                pct = min(count / MIN_CANDLES_FOR_SIGNALS * 100, 100)
                result[sym] = {
                    "count": count,
                    "ready": count >= MIN_CANDLES_FOR_SIGNALS,
                    "pct": round(pct, 1),
                    "needed": MIN_CANDLES_FOR_SIGNALS,
                }
            # Add pairs not yet seen
            from config import PAIRS
            for sym in PAIRS:
                if sym not in result:
                    result[sym] = {"count": 0, "ready": False, "pct": 0.0, "needed": MIN_CANDLES_FOR_SIGNALS}
            return result

    def get_dataframe(self, symbol: str) -> pd.DataFrame:
        """
        Returns a pandas DataFrame with OHLCV + computed candle metrics.
        Returns empty DataFrame if no candles yet.
        """
        candles = self.get_candles(symbol)
        if not candles:
            return _empty_df()

        df = pd.DataFrame(candles)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        # Computed columns
        df["is_up"] = df["close"] >= df["open"]
        df["body_size"] = (df["close"] - df["open"]).abs()
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["candle_range"] = df["high"] - df["low"]
        df["body_pct"] = np.where(
            df["candle_range"] > 0,
            df["body_size"] / df["candle_range"],
            0.0,
        )
        df.reset_index(drop=True, inplace=True)
        return df


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "is_up", "body_size", "upper_wick", "lower_wick", "candle_range", "body_pct",
    ])
