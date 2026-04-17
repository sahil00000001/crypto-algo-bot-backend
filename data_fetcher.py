# data_fetcher.py — Bybit public REST API (no API key, no geo-restriction)

import time
import logging
from typing import Optional

import numpy as np
import pandas as pd
import requests

from config import BYBIT_BASE_URL, REQUEST_DELAY, MAX_RETRIES, INTERVAL_MAP

logger = logging.getLogger(__name__)


def _get(endpoint: str, params: dict) -> dict:
    url = f"{BYBIT_BASE_URL}/{endpoint}"
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("retCode", 0) != 0:
                raise ValueError(f"Bybit error: {data.get('retMsg')}")
            time.sleep(REQUEST_DELAY)
            return data["result"]
        except Exception as e:
            logger.warning(f"Request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(1 * (attempt + 1))
    raise ConnectionError(f"Failed to fetch {endpoint} after {MAX_RETRIES} retries")


def get_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """
    Fetch OHLCV candle data from Bybit.
    Returns DataFrame with computed candle metrics.
    """
    bybit_interval = INTERVAL_MAP.get(interval, interval)
    result = _get("kline", {
        "category": "spot",
        "symbol": symbol,
        "interval": bybit_interval,
        "limit": min(limit, 1000),
    })

    # Bybit returns newest-first → reverse to oldest-first
    rows = list(reversed(result["list"]))

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])

    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
    df.drop(columns=["turnover"], inplace=True)
    df["num_trades"] = 0

    # Computed candle metrics
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


def get_current_price(symbol: str) -> float:
    result = _get("tickers", {"category": "spot", "symbol": symbol})
    return float(result["list"][0]["lastPrice"])


def get_ticker_24hr(symbol: str) -> dict:
    result = _get("tickers", {"category": "spot", "symbol": symbol})
    t = result["list"][0]
    return {
        "symbol": t["symbol"],
        "price": float(t["lastPrice"]),
        "price_change": float(t["price24hPcnt"]) * float(t["lastPrice"]),
        "price_change_pct": float(t["price24hPcnt"]) * 100,
        "high_24h": float(t["highPrice24h"]),
        "low_24h": float(t["lowPrice24h"]),
        "volume_24h": float(t["volume24h"]),
        "quote_volume_24h": float(t["turnover24h"]),
    }


def get_order_book(symbol: str, limit: int = 20) -> dict:
    result = _get("orderbook", {"category": "spot", "symbol": symbol, "limit": limit})
    return {
        "bids": [[float(p), float(q)] for p, q in result["b"]],
        "asks": [[float(p), float(q)] for p, q in result["a"]],
    }


def get_multiple_prices(symbols: list[str]) -> dict[str, float]:
    prices = {}
    for symbol in symbols:
        try:
            prices[symbol] = get_current_price(symbol)
        except Exception:
            prices[symbol] = 0.0
    return prices
