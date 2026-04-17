# server.py — FastAPI server with Bybit WebSocket + CoinGecko REST

import asyncio
import json
import logging
import threading
import time

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import candle_analysis as ca
import indicators as ind
from config import (PAIRS, DEFAULT_INTERVAL, INITIAL_BALANCE,
                    SERVER_HOST, SERVER_PORT, SMA_SHORT, SMA_LONG,
                    SIGNAL_CONFIDENCE_THRESHOLD)
from data_fetcher import get_klines, get_multiple_prices, get_ticker_24hr, make_candle_row, _empty_df
from paper_trader import PaperTrader
from portfolio import Portfolio
from strategy import get_strategy
from websocket_stream import LiveCandleStream

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Crypto Algo Bot", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── Global State ──────────────────────────────────────────────────────────────

trader = PaperTrader(INITIAL_BALANCE)
portfolio = Portfolio(INITIAL_BALANCE)
master_strategy = get_strategy("master")

candle_dfs: dict[str, pd.DataFrame] = {sym: _empty_df() for sym in PAIRS}
live_prices: dict[str, float] = {sym: 0.0 for sym in PAIRS}
streams: dict[str, LiveCandleStream] = {}

_ws_clients: list[WebSocket] = []
_ws_lock = asyncio.Lock()
_main_loop: asyncio.AbstractEventLoop | None = None


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
    if _main_loop and _main_loop.is_running():
        asyncio.run_coroutine_threadsafe(broadcast(payload), _main_loop)


# ── Candle Processing ─────────────────────────────────────────────────────────

def process_closed_candle(symbol: str, candle) -> None:
    global live_prices
    live_prices[symbol] = candle.close

    df = candle_dfs.get(symbol, _empty_df())
    new_row = pd.DataFrame([make_candle_row(candle)])
    df = pd.concat([df, new_row], ignore_index=True).tail(500)
    candle_dfs[symbol] = df

    # Need at least 5 candles for any analysis
    if len(df) < 5:
        broadcast_sync({"type": "prices", "data": dict(live_prices)})
        return

    candle_info = ca.analyze_candles(df)
    computed = ind.compute_all(df, SMA_SHORT, SMA_LONG)
    signal_result = master_strategy.generate_signal(df, candle_info)

    signal = signal_result["signal"]
    confidence = signal_result["confidence"]
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

    for sym in list(trader._positions.keys()):
        p = live_prices.get(sym, 0)
        if p:
            t = trader.check_stop_loss(sym, p) or trader.check_take_profit(sym, p)
            if t:
                portfolio.record_trade(t.to_dict())

    snap = build_snapshot(symbol, current_price, candle_info, computed, signal_result)
    if auto_trade:
        snap["trade"] = auto_trade
    broadcast_sync({"type": "update", "data": snap})


def on_candle_update(symbol: str, candle) -> None:
    """Broadcast live price on every tick (not just closed candles)."""
    live_prices[symbol] = candle.close
    broadcast_sync({"type": "prices", "data": dict(live_prices)})


def build_snapshot(symbol: str, price: float, candle_info: dict,
                   computed: dict, signal_result: dict) -> dict:
    positions = trader.get_all_positions(live_prices)
    total_value = trader.get_total_value(live_prices)
    portfolio.snapshot_equity(total_value)

    df = candle_dfs.get(symbol, _empty_df())
    candles_list = []
    if not df.empty:
        for _, row in df.tail(100).iterrows():
            candles_list.append({
                "time": int(row["timestamp"].timestamp()),
                "open": row["open"], "high": row["high"],
                "low": row["low"], "close": row["close"],
                "volume": row["volume"],
            })

    return {
        "symbol": symbol,
        "price": price,
        "candle_analysis": candle_info,
        "indicators": computed,
        "signal": signal_result,
        "positions": positions,
        "balance": trader.get_balance(),
        "total_value": total_value,
        "stats": trader.get_stats(),
        "recent_trades": portfolio.get_recent_trades(10),
        "candles": candles_list,
        "equity_curve": portfolio.get_equity_curve()[-50:],
        "candle_count": len(df),
    }


# ── Background Init ───────────────────────────────────────────────────────────

def init_background() -> None:
    logger.info("Starting WebSocket streams (no REST init — builds from live data)…")
    for symbol in PAIRS:
        stream = LiveCandleStream(
            symbol=symbol,
            interval=DEFAULT_INTERVAL,
            on_candle_close=lambda c, s=symbol: process_closed_candle(s, c),
            on_candle_update=lambda c, s=symbol: on_candle_update(s, c),
        )
        stream.start()
        streams[symbol] = stream
        logger.info(f"Stream started: {symbol}")

    # Broadcast live prices every 3s using WebSocket-derived data
    def price_broadcast_loop():
        while True:
            if any(p > 0 for p in live_prices.values()):
                broadcast_sync({"type": "prices", "data": dict(live_prices)})
            time.sleep(3)

    threading.Thread(target=price_broadcast_loop, daemon=True).start()


@app.on_event("startup")
async def startup_event():
    global _main_loop
    _main_loop = asyncio.get_event_loop()
    threading.Thread(target=init_background, daemon=True).start()


@app.on_event("shutdown")
async def shutdown_event():
    for stream in streams.values():
        stream.stop()


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "running", "pairs": PAIRS, "candles": {s: len(candle_dfs.get(s, [])) for s in PAIRS}}


@app.get("/api/status")
async def get_status():
    return {
        "pairs": PAIRS,
        "interval": DEFAULT_INTERVAL,
        "streams": {s: streams[s].is_connected() for s in streams},
        "candle_counts": {s: len(candle_dfs.get(s, [])) for s in PAIRS},
        "live_prices": live_prices,
    }


@app.get("/api/candles/{symbol}")
async def get_candles(symbol: str, limit: int = 100):
    df = candle_dfs.get(symbol.upper(), _empty_df())
    if df.empty:
        return {"candles": [], "count": 0}
    tail = df.tail(limit)
    return {
        "candles": [
            {"time": int(row["timestamp"].timestamp()),
             "open": row["open"], "high": row["high"],
             "low": row["low"], "close": row["close"], "volume": row["volume"]}
            for _, row in tail.iterrows()
        ],
        "count": len(df),
    }


@app.get("/api/snapshot/{symbol}")
async def get_snapshot(symbol: str):
    sym = symbol.upper()
    df = candle_dfs.get(sym, _empty_df())
    price = live_prices.get(sym, 0)
    if df.empty or len(df) < 5:
        return {"symbol": sym, "price": price, "candle_count": len(df),
                "message": f"Warming up… {len(df)} candles collected so far"}
    candle_info = ca.analyze_candles(df)
    computed = ind.compute_all(df, SMA_SHORT, SMA_LONG)
    signal_result = master_strategy.generate_signal(df, candle_info)
    return build_snapshot(sym, price, candle_info, computed, signal_result)


@app.get("/api/portfolio")
async def get_portfolio():
    return {
        "balance": trader.get_balance(),
        "total_value": trader.get_total_value(live_prices),
        "positions": trader.get_all_positions(live_prices),
        "stats": trader.get_stats(),
        "recent_trades": portfolio.get_recent_trades(20),
        "equity_curve": portfolio.get_equity_curve(),
    }


@app.get("/api/ticker")
async def get_ticker():
    return {sym: {"symbol": sym, "price": live_prices.get(sym, 0)} for sym in PAIRS}


# ── WebSocket Endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    async with _ws_lock:
        _ws_clients.append(websocket)
    logger.info("WS client connected")
    try:
        # Send current state immediately
        for sym in PAIRS:
            df = candle_dfs.get(sym, _empty_df())
            price = live_prices.get(sym, 0)
            if len(df) >= 5:
                candle_info = ca.analyze_candles(df)
                computed = ind.compute_all(df, SMA_SHORT, SMA_LONG)
                signal_result = master_strategy.generate_signal(df, candle_info)
                snap = build_snapshot(sym, price, candle_info, computed, signal_result)
            else:
                snap = {"symbol": sym, "price": price, "candle_count": len(df),
                        "candles": [], "message": "Warming up…"}
            await websocket.send_text(json.dumps({"type": "init", "data": snap}))

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
