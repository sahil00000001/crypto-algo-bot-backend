# data_fetcher.py — Kraken REST for historical candles (NOT geo-blocked from cloud)
# Kraken is a US-registered exchange with no IP restrictions on public data.

import logging
import time

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

KRAKEN_BASE = "https://api.kraken.com/0/public"

# Kraken pair names for each symbol
KRAKEN_PAIRS: dict[str, str] = {
    "BTCUSDT": "XBTUSD",
    "ETHUSDT": "ETHUSD",
    "SOLUSDT": "SOLUSD",
}

# Bybit interval → Kraken interval (minutes)
KRAKEN_INTERVALS: dict[str, int] = {
    "1": 1, "3": 5, "5": 5, "15": 15,
    "30": 30, "60": 60, "1h": 60, "240": 240, "D": 1440,
}


def get_klines(symbol: str, interval: str = "5", limit: int = 300) -> pd.DataFrame:
    """
    Fetch historical OHLCV candles from Kraken public REST API.
    Kraken is NOT geo-blocked from US cloud servers (AWS/HuggingFace).
    Returns a DataFrame pre-populated with all computed candle metrics.
    """
    pair = KRAKEN_PAIRS.get(symbol, "XBTUSD")
    kraken_interval = KRAKEN_INTERVALS.get(str(interval), 5)

    for attempt in range(3):
        try:
            resp = requests.get(
                f"{KRAKEN_BASE}/OHLC",
                params={"pair": pair, "interval": kraken_interval},
                timeout=15,
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("error"):
                raise ValueError(f"Kraken API error: {data['error']}")

            result = data["result"]
            # Kraken puts candles under the pair's internal key (skip "last")
            pair_key = next(k for k in result if k != "last")
            rows = result[pair_key]

            df = pd.DataFrame(rows, columns=[
                "timestamp", "open", "high", "low", "close", "vwap", "volume", "count"
            ])
            df = df.tail(limit).copy()

            # Cast types
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            # Kraken timestamps are in seconds (not ms)
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
            df.drop(columns=["vwap", "count"], inplace=True)

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
            logger.info(f"Kraken: {len(df)} candles loaded for {symbol} ({pair} {kraken_interval}m)")
            return df

        except Exception as e:
            logger.warning(f"Kraken fetch attempt {attempt + 1}/3 failed for {symbol}: {e}")
            if attempt < 2:
                time.sleep(2 * (attempt + 1))

    logger.error(f"Kraken: all retries failed for {symbol}, starting empty")
    return pd.DataFrame()
