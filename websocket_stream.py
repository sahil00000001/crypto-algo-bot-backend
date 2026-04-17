# websocket_stream.py — Live WebSocket candle streaming from Binance

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime
from typing import Callable, Optional

import websocket

from config import BINANCE_WS_BASE

logger = logging.getLogger(__name__)


class Candle:
    """Represents a single OHLCV candle."""
    __slots__ = ("symbol", "interval", "open_time", "open", "high", "low", "close",
                 "volume", "is_closed", "num_trades")

    def __init__(self, k: dict):
        self.symbol: str = k["s"]
        self.interval: str = k["i"]
        self.open_time: int = k["t"]
        self.open: float = float(k["o"])
        self.high: float = float(k["h"])
        self.low: float = float(k["l"])
        self.close: float = float(k["c"])
        self.volume: float = float(k["v"])
        self.is_closed: bool = k["x"]
        self.num_trades: int = k.get("n", 0)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "open_time": self.open_time,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "is_closed": self.is_closed,
            "num_trades": self.num_trades,
            "is_up": self.close >= self.open,
        }


class LiveCandleStream:
    """
    Connects to a Binance WebSocket kline stream and maintains a rolling
    buffer of closed candles.  Auto-reconnects on disconnect.
    """

    def __init__(
        self,
        symbol: str,
        interval: str,
        max_candles: int = 500,
        on_candle_close: Optional[Callable[[Candle], None]] = None,
        on_candle_update: Optional[Callable[[Candle], None]] = None,
        on_price_update: Optional[Callable[[str, float], None]] = None,
    ):
        self.symbol = symbol.lower()
        self.interval = interval
        self._max_candles = max_candles
        self._closed_candles: deque[Candle] = deque(maxlen=max_candles)
        self._current_candle: Optional[Candle] = None
        self._lock = threading.Lock()

        self._on_candle_close = on_candle_close
        self._on_candle_update = on_candle_update
        self._on_price_update = on_price_update

        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._connected = False

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start streaming in a background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"LiveCandleStream started: {self.symbol}@kline_{self.interval}")

    def stop(self) -> None:
        """Stop the stream."""
        self._stop_event.set()
        if self._ws:
            self._ws.close()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(f"LiveCandleStream stopped: {self.symbol}")

    def get_closed_candles(self) -> list[Candle]:
        with self._lock:
            return list(self._closed_candles)

    def get_current_candle(self) -> Optional[Candle]:
        with self._lock:
            return self._current_candle

    def is_connected(self) -> bool:
        return self._connected

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        retry_delay = 1
        while not self._stop_event.is_set():
            try:
                url = f"{BINANCE_WS_BASE}/{self.symbol}@kline_{self.interval}"
                self._ws = websocket.WebSocketApp(
                    url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open,
                )
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
                retry_delay = min(retry_delay * 2, 60)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            if not self._stop_event.is_set():
                logger.info(f"Reconnecting in {retry_delay}s…")
                time.sleep(retry_delay)

    def _on_open(self, ws) -> None:
        self._connected = True
        logger.info(f"WS connected: {self.symbol}@kline_{self.interval}")

    def _on_close(self, ws, code, msg) -> None:
        self._connected = False
        logger.warning(f"WS closed ({code}): {msg}")

    def _on_error(self, ws, error) -> None:
        logger.error(f"WS error: {error}")

    def _on_message(self, ws, raw: str) -> None:
        try:
            data = json.loads(raw)
            if data.get("e") != "kline":
                return
            candle = Candle(data["k"])

            with self._lock:
                self._current_candle = candle
                if candle.is_closed:
                    self._closed_candles.append(candle)

            if self._on_price_update:
                self._on_price_update(self.symbol.upper(), candle.close)

            if candle.is_closed and self._on_candle_close:
                self._on_candle_close(candle)
            elif not candle.is_closed and self._on_candle_update:
                self._on_candle_update(candle)

        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.debug(f"Parse error: {e}")
