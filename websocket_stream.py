# websocket_stream.py — Live WebSocket candle streaming via Bybit (no geo-restriction)

import json
import logging
import threading
import time
from collections import deque
from typing import Callable, Optional

import websocket

from config import BYBIT_WS_SPOT, INTERVAL_MAP

logger = logging.getLogger(__name__)


class Candle:
    __slots__ = ("symbol", "interval", "open_time", "open", "high", "low",
                 "close", "volume", "is_closed", "num_trades")

    def __init__(self, symbol: str, interval: str, d: dict):
        self.symbol: str = symbol
        self.interval: str = interval
        self.open_time: int = int(d["start"])
        self.open: float = float(d["open"])
        self.high: float = float(d["high"])
        self.low: float = float(d["low"])
        self.close: float = float(d["close"])
        self.volume: float = float(d["volume"])
        self.is_closed: bool = bool(d.get("confirm", False))
        self.num_trades: int = 0

    @property
    def is_up(self) -> bool:
        return self.close >= self.open

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
            "is_up": self.is_up,
        }


class LiveCandleStream:
    """
    Connects to Bybit WebSocket spot stream for live kline data.
    Auto-reconnects. Thread-safe.
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
        self.symbol = symbol.upper()
        self.interval = interval
        self._bybit_interval = INTERVAL_MAP.get(interval, interval)
        self._topic = f"kline.{self._bybit_interval}.{self.symbol}"

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

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"LiveCandleStream started: {self._topic}")

    def stop(self) -> None:
        self._stop_event.set()
        if self._ws:
            self._ws.close()
        if self._thread:
            self._thread.join(timeout=5)

    def get_closed_candles(self) -> list[Candle]:
        with self._lock:
            return list(self._closed_candles)

    def get_current_candle(self) -> Optional[Candle]:
        with self._lock:
            return self._current_candle

    def is_connected(self) -> bool:
        return self._connected

    def _run_loop(self) -> None:
        retry_delay = 1
        while not self._stop_event.is_set():
            try:
                self._ws = websocket.WebSocketApp(
                    BYBIT_WS_SPOT,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
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
        subscribe = json.dumps({"op": "subscribe", "args": [self._topic]})
        ws.send(subscribe)
        logger.info(f"WS connected, subscribed to {self._topic}")

    def _on_close(self, ws, code, msg) -> None:
        self._connected = False
        logger.warning(f"WS closed ({code})")

    def _on_error(self, ws, error) -> None:
        logger.error(f"WS error: {error}")

    def _on_message(self, ws, raw: str) -> None:
        try:
            data = json.loads(raw)

            # Heartbeat / subscription ack
            if "op" in data or "topic" not in data:
                return

            topic = data.get("topic", "")
            if not topic.startswith("kline."):
                return

            for item in data.get("data", []):
                candle = Candle(self.symbol, self.interval, item)

                with self._lock:
                    self._current_candle = candle
                    if candle.is_closed:
                        self._closed_candles.append(candle)

                if self._on_price_update:
                    self._on_price_update(self.symbol, candle.close)

                if candle.is_closed and self._on_candle_close:
                    self._on_candle_close(candle)
                elif not candle.is_closed and self._on_candle_update:
                    self._on_candle_update(candle)

        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.debug(f"Parse error: {e}")
