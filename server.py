# server.py — FastAPI server: REST endpoints + WebSocket broadcast for web dashboard

import asyncio
import json
import logging
import threading
import time
from typing import Any

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import candle_analysis as ca
import indicators as ind
from config import PAIRS, DEFAULT_INTERVAL, INITIAL_BALANCE, SERVER_HOST, SERVER_PORT, SMA_SHORT, SMA_LONG, SIGNAL_CONFIDENCE_THRESHOLD
from data_fetcher import get_klines, get_ticker_24hr, get_multiple_prices
from paper_trader import PaperTrader
from portfolio import Portfolio
from strategy import get_strategy
from websocket_stream import LiveCandleStream

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Crypto Algo Bot", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ──────────────────────────────────────────────────────────────

trader = PaperTrader(INITIAL_BALANCE)
portfolio = Portfolio(INITIAL_BALANCE)
master_strategy = get_strategy("master")

# Per-symbol data
symbol_data: dict[str, dict] = {sym: {} for sym in PAIRS}
candle_dfs: dict[str, pd.DataFrame] = {}
streams: dict[str, LiveCandleStream] = {}

# WebSocket clients connected to /ws/stream
_ws_clients: list[WebSocket] = []
_ws_lock = asyncio.Lock()


# ── WebSocket Broadcast ───────────────────────────────────────────────────────

async def broadcast(payload: dict) -> None:
    message = json.dumps(payload)
    disconnected = []
    async with _ws_lock:
        for client in _ws_clients:
            try:
                await client.send_text(message)
            except Exception:
                disconnected.append(client)
        for c in disconnected:
            _ws_clients.remove(c)


def broadcast_sync(payload: dict) -> None:
    """Thread-safe broadcast from synchronous context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(broadcast(payload), loop)
    except RuntimeError:
        pass


# ── Candle Processing ─────────────────────────────────────────────────────────

def process_closed_candle(symbol: str, candle) -> None:
    """Called on every closed WebSocket candle. Runs analysis and strategy."""
    df = candle_dfs.get(symbol)
    if df is None or len(df) < 5:
        return

    # Append the new closed candle
    new_row = pd.DataFrame([{
        "timestamp": pd.Timestamp(candle.open_time, unit="ms"),
        "open": candle.open, "high": candle.high,
        "low": candle.low, "close": candle.close,
        "volume": candle.volume, "num_trades": candle.num_trades,
        "is_up": candle.close >= candle.open,
        "body_size": abs(candle.close - candle.open),
        "upper_wick": candle.high - max(candle.open, candle.close),
        "lower_wick": min(candle.open, candle.close) - candle.low,
        "candle_range": candle.high - candle.low,
        "body_pct": abs(candle.close - candle.open) / (candle.high - candle.low + 1e-10),
    }])
    df = pd.concat([df, new_row], ignore_index=True).tail(500)
    candle_dfs[symbol] = df

    # Analysis
    candle_info = ca.analyze_candles(df)
    computed = ind.compute_all(df, SMA_SHORT, SMA_LONG)
    signal_result = master_strategy.generate_signal(df, candle_info)

    signal = signal_result["signal"]
    confidence = signal_result["confidence"]

    # Paper trading
    current_price = candle.close
    auto_trade = None
    if signal == "BUY" and confidence >= SIGNAL_CONFIDENCE_THRESHOLD:
        trade = trader.buy(symbol, current_price)
        if trade:
            portfolio.record_trade(trade.to_dict())
            auto_trade = trade.to_dict()
    elif signal == "SELL" and confidence >= SIGNAL_CONFIDENCE_THRESHOLD:
        trade = trader.sell(symbol, current_price, reason="signal")
        if trade:
            portfolio.record_trade(trade.to_dict())
            auto_trade = trade.to_dict()

    # Check SL/TP for ALL symbols
    prices = {s: candle_dfs[s]["close"].iloc[-1] for s in candle_dfs if len(candle_dfs[s]) > 0}
    for sym in list(trader._positions.keys()):
        p = prices.get(sym)
        if p:
            t = trader.check_stop_loss(sym, p) or trader.check_take_profit(sym, p)
            if t:
                portfolio.record_trade(t.to_dict())

    # Build snapshot
    snapshot = build_snapshot(symbol, current_price, candle_info, computed, signal_result)
    if auto_trade:
        snapshot["trade"] = auto_trade
    broadcast_sync({"type": "update", "data": snapshot})


def build_snapshot(
    symbol: str,
    price: float,
    candle_info: dict,
    computed: dict,
    signal_result: dict,
) -> dict:
    prices = {s: candle_dfs[s]["close"].iloc[-1] for s in candle_dfs if len(candle_dfs[s]) > 0}
    positions = trader.get_all_positions(prices)
    trader_stats = trader.get_stats()
    total_value = trader.get_total_value(prices)
    portfolio.snapshot_equity(total_value)

    candles_list = []
    df = candle_dfs.get(symbol)
    if df is not None:
        tail = df.tail(100)
        candles_list = [
            {
                "time": int(row["timestamp"].timestamp()),
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }
            for _, row in tail.iterrows()
        ]

    return {
        "symbol": symbol,
        "price": price,
        "candle_analysis": candle_info,
        "indicators": computed,
        "signal": signal_result,
        "positions": positions,
        "balance": trader.get_balance(),
        "total_value": total_value,
        "stats": trader_stats,
        "recent_trades": portfolio.get_recent_trades(10),
        "candles": candles_list,
        "equity_curve": portfolio.get_equity_curve()[-50:],
    }


# ── Background Init ───────────────────────────────────────────────────────────

def init_background() -> None:
    """Load historical data and start WebSocket streams."""
    logger.info("Loading historical candle data…")
    for symbol in PAIRS:
        try:
            df = get_klines(symbol, DEFAULT_INTERVAL, 500)
            candle_dfs[symbol] = df
            logger.info(f"Loaded {len(df)} candles for {symbol}")
        except Exception as e:
            logger.error(f"Failed to load {symbol}: {e}")
            candle_dfs[symbol] = pd.DataFrame()

    logger.info("Starting WebSocket streams…")
    for symbol in PAIRS:
        stream = LiveCandleStream(
            symbol=symbol,
            interval=DEFAULT_INTERVAL,
            on_candle_close=lambda c, s=symbol: process_closed_candle(s, c),
        )
        stream.start()
        streams[symbol] = stream
        logger.info(f"Stream started: {symbol}")

    # Periodic price broadcast for the ticker bar
    def ticker_loop():
        while True:
            try:
                prices = get_multiple_prices(PAIRS)
                broadcast_sync({"type": "prices", "data": prices})
            except Exception:
                pass
            time.sleep(3)

    threading.Thread(target=ticker_loop, daemon=True).start()


@app.on_event("startup")
async def startup_event():
    threading.Thread(target=init_background, daemon=True).start()


@app.on_event("shutdown")
async def shutdown_event():
    for stream in streams.values():
        stream.stop()


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    return {
        "pairs": PAIRS,
        "interval": DEFAULT_INTERVAL,
        "streams": {s: streams[s].is_connected() for s in streams},
    }


@app.get("/api/candles/{symbol}")
async def get_candles(symbol: str, limit: int = 100):
    df = candle_dfs.get(symbol.upper())
    if df is None or df.empty:
        return {"candles": []}
    tail = df.tail(limit)
    return {
        "candles": [
            {
                "time": int(row["timestamp"].timestamp()),
                "open": row["open"], "high": row["high"],
                "low": row["low"], "close": row["close"],
                "volume": row["volume"],
            }
            for _, row in tail.iterrows()
        ]
    }


@app.get("/api/snapshot/{symbol}")
async def get_snapshot(symbol: str):
    sym = symbol.upper()
    df = candle_dfs.get(sym)
    if df is None or df.empty:
        return {"error": "No data"}
    price = float(df["close"].iloc[-1])
    candle_info = ca.analyze_candles(df)
    computed = ind.compute_all(df, SMA_SHORT, SMA_LONG)
    signal_result = master_strategy.generate_signal(df, candle_info)
    return build_snapshot(sym, price, candle_info, computed, signal_result)


@app.get("/api/portfolio")
async def get_portfolio():
    prices = {s: float(candle_dfs[s]["close"].iloc[-1]) for s in candle_dfs if len(candle_dfs.get(s, [])) > 0}
    return {
        "balance": trader.get_balance(),
        "total_value": trader.get_total_value(prices),
        "positions": trader.get_all_positions(prices),
        "stats": trader.get_stats(),
        "recent_trades": portfolio.get_recent_trades(20),
        "equity_curve": portfolio.get_equity_curve(),
    }


@app.get("/api/ticker")
async def get_ticker():
    result = {}
    for symbol in PAIRS:
        try:
            result[symbol] = get_ticker_24hr(symbol)
        except Exception:
            result[symbol] = {}
    return result


# ── WebSocket Endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    async with _ws_lock:
        _ws_clients.append(websocket)
    logger.info("WS client connected")
    try:
        # Send initial state for all pairs
        for sym in PAIRS:
            df = candle_dfs.get(sym)
            if df is not None and not df.empty:
                price = float(df["close"].iloc[-1])
                candle_info = ca.analyze_candles(df)
                computed = ind.compute_all(df, SMA_SHORT, SMA_LONG)
                signal_result = master_strategy.generate_signal(df, candle_info)
                snap = build_snapshot(sym, price, candle_info, computed, signal_result)
                await websocket.send_text(json.dumps({"type": "init", "data": snap}))
        # Keep alive
        while True:
            await asyncio.sleep(30)
            await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        pass
    finally:
        async with _ws_lock:
            if websocket in _ws_clients:
                _ws_clients.remove(websocket)
        logger.info("WS client disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)
