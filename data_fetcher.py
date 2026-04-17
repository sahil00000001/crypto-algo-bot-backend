# data_fetcher.py — CoinGecko for prices (unrestricted), Bybit WebSocket for candles

import time
import logging

import numpy as np
import pandas as pd
import requests

from config import MAX_RETRIES, REQUEST_DELAY

logger = logging.getLogger(__name__)

# CoinGecko — completely unrestricted, no API key, works everywhere
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

SYMBOL_TO_ID = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "SOLUSDT": "solana",
}

SYMBOL_TO_VS = {
    "BTCUSDT": "usd",
    "ETHUSDT": "usd",
    "SOLUSDT": "usd",
}


def _get_cg(endpoint: str, params: dict) -> dict | list:
    url = f"{COINGECKO_BASE}/{endpoint}"
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=10,
                                headers={"Accept": "application/json"})
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp.json()
        except Exception as e:
            logger.warning(f"CoinGecko request failed (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 * (attempt + 1))
    raise ConnectionError(f"CoinGecko {endpoint} failed after {MAX_RETRIES} retries")


def get_current_price(symbol: str) -> float:
    cg_id = SYMBOL_TO_ID.get(symbol, "bitcoin")
    data = _get_cg("simple/price", {"ids": cg_id, "vs_currencies": "usd"})
    return float(data[cg_id]["usd"])


def get_multiple_prices(symbols: list[str]) -> dict[str, float]:
    ids = ",".join(SYMBOL_TO_ID.get(s, "bitcoin") for s in symbols)
    try:
        data = _get_cg("simple/price", {"ids": ids, "vs_currencies": "usd"})
        return {
            sym: float(data.get(SYMBOL_TO_ID.get(sym, "bitcoin"), {}).get("usd", 0))
            for sym in symbols
        }
    except Exception as e:
        logger.warning(f"get_multiple_prices failed: {e}")
        return {s: 0.0 for s in symbols}


def get_ticker_24hr(symbol: str) -> dict:
    cg_id = SYMBOL_TO_ID.get(symbol, "bitcoin")
    try:
        data = _get_cg(f"coins/{cg_id}", {
            "localization": "false", "tickers": "false",
            "market_data": "true", "community_data": "false",
        })
        md = data["market_data"]
        price = float(md["current_price"]["usd"])
        change_pct = float(md["price_change_percentage_24h"] or 0)
        return {
            "symbol": symbol,
            "price": price,
            "price_change": price * change_pct / 100,
            "price_change_pct": change_pct,
            "high_24h": float(md["high_24h"]["usd"]),
            "low_24h": float(md["low_24h"]["usd"]),
            "volume_24h": float(md["total_volume"]["usd"]),
            "quote_volume_24h": float(md["total_volume"]["usd"]),
        }
    except Exception:
        price = get_current_price(symbol)
        return {"symbol": symbol, "price": price, "price_change_pct": 0,
                "high_24h": price, "low_24h": price, "volume_24h": 0}


def get_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """
    Returns empty DataFrame — historical klines are built up live via WebSocket.
    CoinGecko free tier only provides daily OHLC, not minute-level data.
    """
    logger.info(f"Skipping REST historical load for {symbol} — will build from WebSocket")
    return _empty_df()


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "timestamp", "open", "high", "low", "close", "volume", "num_trades",
        "is_up", "body_size", "upper_wick", "lower_wick", "candle_range", "body_pct",
    ])


def make_candle_row(c) -> dict:
    """Convert a WebSocket Candle object into a DataFrame row dict."""
    body = abs(c.close - c.open)
    rng = c.high - c.low
    return {
        "timestamp": pd.Timestamp(c.open_time, unit="ms"),
        "open": c.open, "high": c.high, "low": c.low,
        "close": c.close, "volume": c.volume, "num_trades": 0,
        "is_up": c.close >= c.open,
        "body_size": body,
        "upper_wick": c.high - max(c.open, c.close),
        "lower_wick": min(c.open, c.close) - c.low,
        "candle_range": rng,
        "body_pct": body / rng if rng > 0 else 0.0,
    }
