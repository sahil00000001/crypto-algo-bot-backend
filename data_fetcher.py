# data_fetcher.py — REST API calls to Binance public endpoints (no API key required)

import time
import logging
from typing import Optional

import numpy as np
import pandas as pd
import requests

from config import BINANCE_BASE_URL, REQUEST_DELAY, MAX_RETRIES

logger = logging.getLogger(__name__)


def _get(endpoint: str, params: dict) -> dict | list:
    """Make a GET request to Binance REST API with retry logic."""
    url = f"{BINANCE_BASE_URL}/{endpoint}"
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(1 * (attempt + 1))
    raise ConnectionError(f"Failed to fetch {endpoint} after {MAX_RETRIES} retries")


def get_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """
    Fetch OHLCV candle data from Binance.
    Returns a DataFrame with computed candle metrics.
    """
    raw: list = _get("klines", {"symbol": symbol, "interval": interval, "limit": limit})

    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    # Convert types
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df["num_trades"] = df["num_trades"].astype(int)

    # Computed candle metrics
    df["is_up"] = df["close"] >= df["open"]
    df["body_size"] = (df["close"] - df["open"]).abs()
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["candle_range"] = df["high"] - df["low"]
    df["body_pct"] = np.where(
        df["candle_range"] > 0,
        df["body_size"] / df["candle_range"],
        0.0
    )

    # Drop unused columns
    df.drop(columns=["close_time", "quote_volume", "taker_buy_base", "taker_buy_quote", "ignore"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def get_current_price(symbol: str) -> float:
    """Fetch current price for a symbol."""
    data: dict = _get("ticker/price", {"symbol": symbol})
    return float(data["price"])


def get_ticker_24hr(symbol: str) -> dict:
    """Fetch 24-hour statistics for a symbol."""
    data: dict = _get("ticker/24hr", {"symbol": symbol})
    return {
        "symbol": data["symbol"],
        "price": float(data["lastPrice"]),
        "price_change": float(data["priceChange"]),
        "price_change_pct": float(data["priceChangePercent"]),
        "high_24h": float(data["highPrice"]),
        "low_24h": float(data["lowPrice"]),
        "volume_24h": float(data["volume"]),
        "quote_volume_24h": float(data["quoteVolume"]),
    }


def get_order_book(symbol: str, limit: int = 20) -> dict:
    """Fetch current order book (bids and asks)."""
    data: dict = _get("depth", {"symbol": symbol, "limit": limit})
    return {
        "bids": [[float(p), float(q)] for p, q in data["bids"]],
        "asks": [[float(p), float(q)] for p, q in data["asks"]],
    }


def get_recent_trades(symbol: str, limit: int = 50) -> list[dict]:
    """Fetch recent trades for a symbol."""
    trades: list = _get("trades", {"symbol": symbol, "limit": limit})
    return [
        {
            "id": t["id"],
            "price": float(t["price"]),
            "qty": float(t["qty"]),
            "time": t["time"],
            "is_buyer_maker": t["isBuyerMaker"],
        }
        for t in trades
    ]


def get_multiple_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch current prices for multiple symbols."""
    prices: list = _get("ticker/price", {})
    price_map = {item["symbol"]: float(item["price"]) for item in prices}
    return {s: price_map.get(s, 0.0) for s in symbols}
