# ws_stream.py — WebSocket-ONLY market data. ZERO REST calls to exchanges.
# Bybit primary → Binance fallback. Both work from cloud servers.

import json
import logging
import threading
import time
from typing import Callable, Optional

import websocket

from config import (PAIRS, INTERVAL, BINANCE_INTERVAL,
                    WS_PING_INTERVAL, WS_RECONNECT_DELAY, PRIMARY_SOURCE)

logger = logging.getLogger(__name__)

BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/spot"
BINANCE_WS_URL = (
    "wss://stream.binance.com:9443/stream?streams="
    + "/".join(f"{p.lower()}@kline_{BINANCE_INTERVAL}" for p in PAIRS)
)


def _make_candle(ts: int, o: float, h: float, l: float, c: float,
                 v: float, confirmed: bool, source: str) -> dict:
    return {
        "timestamp": ts,
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "is_confirmed": confirmed,
        "source": source,
    }


class CryptoWebSocket:
    """
    Single WebSocket connection streaming klines for all configured pairs.
    Bybit primary (subscribes all pairs in one connection).
    Falls back to Binance combined stream if Bybit fails.
    Never makes any REST calls.
    """

    def __init__(
        self,
        on_candle_close: Callable[[str, dict], None],
        on_price_update: Callable[[str, float], None],
    ):
        self._on_candle_close = on_candle_close
        self._on_price_update = on_price_update

        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._ping_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._source: str = PRIMARY_SOURCE   # "bybit" or "binance"
        self._connected: bool = False
        self._bybit_failed: bool = False

    # ── Public ────────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._ws:
            self._ws.close()

    def is_connected(self) -> bool:
        return self._connected

    def current_source(self) -> str:
        return self._source

    # ── Connection Loop ───────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        retry_delay = WS_RECONNECT_DELAY
        bybit_fail_count = 0

        while not self._stop_event.is_set():
            if bybit_fail_count >= 3:
                self._source = "binance"
                self._bybit_failed = True

            url = BYBIT_WS_URL if self._source == "bybit" else BINANCE_WS_URL
            logger.info(f"Connecting to {self._source} WebSocket…")

            try:
                self._ws = websocket.WebSocketApp(
                    url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever()
                retry_delay = min(retry_delay * 2, 60)
                if self._source == "bybit":
                    bybit_fail_count += 1
            except Exception as e:
                logger.error(f"WebSocket loop error: {e}")

            if not self._stop_event.is_set():
                logger.info(f"Reconnecting in {retry_delay}s…")
                time.sleep(retry_delay)

    # ── Bybit Handlers ────────────────────────────────────────────────────────

    def _on_open(self, ws) -> None:
        self._connected = True
        logger.info(f"WS connected ({self._source})")

        if self._source == "bybit":
            args = [f"kline.{INTERVAL}.{pair}" for pair in PAIRS]
            ws.send(json.dumps({"op": "subscribe", "args": args}))
            logger.info(f"Subscribed Bybit: {args}")
            # Start ping thread
            self._ping_thread = threading.Thread(
                target=self._ping_loop, args=(ws,), daemon=True
            )
            self._ping_thread.start()

    def _on_close(self, ws, code, msg) -> None:
        self._connected = False
        logger.warning(f"WS closed ({self._source}) code={code}")

    def _on_error(self, ws, error) -> None:
        logger.error(f"WS error ({self._source}): {error}")

    def _on_message(self, ws, raw: str) -> None:
        try:
            data = json.loads(raw)

            # Route to correct parser
            if self._source == "bybit":
                self._parse_bybit(data)
            else:
                self._parse_binance(data)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug(f"Parse error: {e}")

    def _ping_loop(self, ws) -> None:
        while self._connected and not self._stop_event.is_set():
            time.sleep(WS_PING_INTERVAL)
            if self._connected:
                try:
                    ws.send(json.dumps({"op": "ping"}))
                except Exception:
                    break

    # ── Bybit Parser ──────────────────────────────────────────────────────────

    def _parse_bybit(self, data: dict) -> None:
        # Ignore pong / subscription ack
        if "op" in data:
            return

        topic = data.get("topic", "")
        if not topic.startswith("kline."):
            return

        # topic = "kline.5.BTCUSDT"
        parts = topic.split(".")
        if len(parts) < 3:
            return
        symbol = parts[2].upper()

        for item in data.get("data", []):
            candle = _make_candle(
                ts=int(item["start"]),
                o=float(item["open"]),
                h=float(item["high"]),
                l=float(item["low"]),
                c=float(item["close"]),
                v=float(item["volume"]),
                confirmed=bool(item.get("confirm", False)),
                source="bybit",
            )
            self._on_price_update(symbol, candle["close"])
            if candle["is_confirmed"]:
                self._on_candle_close(symbol, candle)

    # ── Binance Parser ────────────────────────────────────────────────────────

    def _parse_binance(self, data: dict) -> None:
        # Combined stream wraps payload in {"stream":..., "data":{...}}
        payload = data.get("data", data)

        if payload.get("e") != "kline":
            return

        k = payload["k"]
        symbol = payload["s"].upper()
        candle = _make_candle(
            ts=int(k["t"]),
            o=float(k["o"]),
            h=float(k["h"]),
            l=float(k["l"]),
            c=float(k["c"]),
            v=float(k["v"]),
            confirmed=bool(k.get("x", False)),
            source="binance",
        )
        self._on_price_update(symbol, candle["close"])
        if candle["is_confirmed"]:
            self._on_candle_close(symbol, candle)
