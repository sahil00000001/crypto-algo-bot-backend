# server.py — FastAPI + WebSocket dashboard server
# Uses WS-only architecture: no REST calls to exchanges.

import asyncio
import json
import logging
import threading
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import candle_analysis as ca
import indicators as ind
from candle_store import CandleStore
from config import PAIRS, INITIAL_BALANCE, SMA_SHORT, SMA_LONG, SIGNAL_CONFIDENCE_THRESHOLD, SERVER_HOST, SERVER_PORT
from data_fetcher import get_klines
from paper_trader import PaperTrader
from portfolio import Portfolio
from price_fallback import get_prices_coingecko
from strategy import get_strategy
from ws_stream import CryptoWebSocket

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Crypto Algo Bot", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── Singletons ────────────────────────────────────────────────────────────────

store = CandleStore()
trader = PaperTrader(INITIAL_BALANCE)
portfolio = Portfolio(INITIAL_BALANCE)
strategy = get_strategy("master")

ws_feed: CryptoWebSocket | None = None
_ws_clients: list[WebSocket] = []
_ws_lock = asyncio.Lock()
_main_loop: asyncio.AbstractEventLoop | None = None

# ── Broadcast ─────────────────────────────────────────────────────────────────

async def broadcast(payload: dict) -> None:
    msg = json.dumps(payload)
    dead = []
    async with _ws_lock:
        for c in _ws_clients:
            try:
                await c.send_text(msg)
            except Exception:
                dead.append(c)
        for c in dead:
            _ws_clients.remove(c)


def broadcast_sync(payload: dict) -> None:
    if _main_loop and _main_loop.is_running():
        asyncio.run_coroutine_threadsafe(broadcast(payload), _main_loop)


# ── Core Candle Handler ───────────────────────────────────────────────────────

def on_candle_close(symbol: str, candle: dict) -> None:
    store.add_candle(symbol, candle)
    prices = store.get_all_prices()

    if not store.is_ready(symbol):
        status = store.get_warming_status()
        broadcast_sync({"type": "warming", "data": status, "prices": prices})
        return

    df = store.get_dataframe(symbol)
    candle_info = ca.analyze_candles(df)
    computed = ind.compute_all(df)
    signal_result = strategy.generate_signal(df, candle_info, computed)

    sig = signal_result["signal"]
    conf = signal_result["confidence"]
    price = candle["close"]

    # Execute paper trades
    auto_trade = None
    if sig == "BUY" and conf >= SIGNAL_CONFIDENCE_THRESHOLD:
        t = trader.buy(symbol, price)
        if t:
            portfolio.record_trade(t.to_dict())
            auto_trade = t.to_dict()
    elif sig == "SELL" and conf >= SIGNAL_CONFIDENCE_THRESHOLD:
        t = trader.sell(symbol, price)
        if t:
            portfolio.record_trade(t.to_dict())
            auto_trade = t.to_dict()

    # Check SL/TP for all positions
    all_prices = store.get_all_prices()
    for sym in list(trader._positions.keys()):
        p = all_prices.get(sym, 0)
        if p:
            t = trader.check_stop_loss(sym, p) or trader.check_take_profit(sym, p)
            if t:
                portfolio.record_trade(t.to_dict())

    total_value = trader.get_total_value(all_prices)
    portfolio.snapshot_equity(total_value)

    snap = _build_snapshot(symbol, price, candle_info, computed, signal_result)
    if auto_trade:
        snap["trade"] = auto_trade
    broadcast_sync({"type": "update", "data": snap})


def on_price_update(symbol: str, price: float) -> None:
    store.set_price(symbol, price)
    broadcast_sync({"type": "prices", "data": store.get_all_prices()})


# ── Snapshot Builder ──────────────────────────────────────────────────────────

def _build_snapshot(symbol: str, price: float, candle_info: dict,
                    computed: dict, signal_result: dict) -> dict:
    prices = store.get_all_prices()
    df = store.get_dataframe(symbol)

    candles_list = []
    if not df.empty:
        for _, row in df.tail(100).iterrows():
            candles_list.append({
                "time": int(row["timestamp"].timestamp()),
                "open": float(row["open"]), "high": float(row["high"]),
                "low": float(row["low"]), "close": float(row["close"]),
                "volume": float(row["volume"]),
            })

    return {
        "symbol": symbol,
        "price": price,
        "candle_count": store.get_count(symbol),
        "warming": store.get_warming_status(),
        "candle_analysis": candle_info,
        "indicators": computed,
        "signal": signal_result,
        "positions": trader.get_positions_with_pnl(prices),
        "balance": trader.get_balance(),
        "total_value": trader.get_total_value(prices),
        "stats": trader.get_stats(),
        "recent_trades": portfolio.get_recent_trades(10),
        "candles": candles_list,
        "equity_curve": portfolio.get_equity_curve()[-50:],
        "ws_source": ws_feed.current_source() if ws_feed else "unknown",
    }


# ── Startup ───────────────────────────────────────────────────────────────────

def init_background() -> None:
    global ws_feed

    # Seed prices from CoinGecko (non-blocked)
    try:
        prices = get_prices_coingecko(PAIRS)
        for sym, price in prices.items():
            if price > 0:
                store.set_price(sym, price)
        logger.info(f"Initial prices from CoinGecko: {prices}")
    except Exception as e:
        logger.warning(f"CoinGecko seed failed (non-fatal): {e}")

    # Bootstrap chart with Kraken historical candles (not geo-blocked from US cloud)
    for sym in PAIRS:
        try:
            df = get_klines(sym, interval="5", limit=300)
            if not df.empty:
                store.bulk_load_from_df(sym, df)
                logger.info(f"Kraken bootstrap: loaded {len(df)} candles for {sym}")
                broadcast_sync({"type": "kraken_bootstrap", "symbol": sym, "count": len(df)})
        except Exception as e:
            logger.warning(f"Kraken bootstrap failed for {sym} (non-fatal): {e}")

    # Start WebSocket
    ws_feed = CryptoWebSocket(
        on_candle_close=on_candle_close,
        on_price_update=on_price_update,
    )
    ws_feed.start()
    logger.info("WebSocket stream started")

    # Periodic price broadcast from store
    def price_loop():
        while True:
            time.sleep(3)
            prices = store.get_all_prices()
            if any(v > 0 for v in prices.values()):
                broadcast_sync({"type": "prices", "data": prices})

    threading.Thread(target=price_loop, daemon=True).start()


@app.on_event("startup")
async def startup():
    global _main_loop
    _main_loop = asyncio.get_event_loop()
    threading.Thread(target=init_background, daemon=True).start()


@app.on_event("shutdown")
async def shutdown():
    if ws_feed:
        ws_feed.stop()


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "running",
        "pairs": PAIRS,
        "candle_counts": {s: store.get_count(s) for s in PAIRS},
        "prices": store.get_all_prices(),
        "ws_connected": ws_feed.is_connected() if ws_feed else False,
        "ws_source": ws_feed.current_source() if ws_feed else "none",
    }


@app.get("/api/status")
async def status():
    return {
        "warming": store.get_warming_status(),
        "prices": store.get_all_prices(),
        "ws_connected": ws_feed.is_connected() if ws_feed else False,
        "ws_source": ws_feed.current_source() if ws_feed else "none",
    }


@app.get("/api/snapshot/{symbol}")
async def snapshot(symbol: str):
    sym = symbol.upper()
    df = store.get_dataframe(sym)
    price = store.get_current_price(sym)
    count = store.get_count(sym)

    if count < 5:
        return {"symbol": sym, "price": price, "candle_count": count,
                "warming": store.get_warming_status(),
                "message": f"Warming up… {count}/26 candles"}

    candle_info = ca.analyze_candles(df)
    computed = ind.compute_all(df)
    signal_result = strategy.generate_signal(df, candle_info, computed)
    return _build_snapshot(sym, price, candle_info, computed, signal_result)


@app.get("/api/candles/{symbol}")
async def candles(symbol: str, limit: int = 100):
    df = store.get_dataframe(symbol.upper())
    if df.empty:
        return {"candles": [], "count": 0}
    return {
        "candles": [
            {"time": int(r["timestamp"].timestamp()),
             "open": r["open"], "high": r["high"],
             "low": r["low"], "close": r["close"], "volume": r["volume"]}
            for _, r in df.tail(limit).iterrows()
        ],
        "count": len(df),
    }


@app.get("/api/portfolio")
async def get_portfolio():
    prices = store.get_all_prices()
    return {
        "balance": trader.get_balance(),
        "total_value": trader.get_total_value(prices),
        "positions": trader.get_positions_with_pnl(prices),
        "stats": trader.get_stats(),
        "recent_trades": portfolio.get_recent_trades(20),
        "equity_curve": portfolio.get_equity_curve(),
    }


# ── WebSocket Endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    async with _ws_lock:
        _ws_clients.append(websocket)
    logger.info("Dashboard client connected")

    try:
        # Send immediate state
        prices = store.get_all_prices()
        warming = store.get_warming_status()
        for sym in PAIRS:
            count = store.get_count(sym)
            price = store.get_current_price(sym)
            if count >= 5:
                df = store.get_dataframe(sym)
                ci = ca.analyze_candles(df)
                comp = ind.compute_all(df)
                sig = strategy.generate_signal(df, ci, comp)
                data = _build_snapshot(sym, price, ci, comp, sig)
            else:
                data = {"symbol": sym, "price": price, "candle_count": count,
                        "warming": warming, "candles": []}
            await websocket.send_text(json.dumps({"type": "init", "data": data}))

        while True:
            await asyncio.sleep(30)
            await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        pass
    finally:
        async with _ws_lock:
            if websocket in _ws_clients:
                _ws_clients.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)
