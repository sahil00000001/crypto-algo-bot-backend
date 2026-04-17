# indicators.py — Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands, VWAP, ATR

import numpy as np
import pandas as pd


def SMA(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=1).mean()


def EMA(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def RSI(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (0–100)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def MACD(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD indicator.
    Returns: (macd_line, signal_line, histogram)
    """
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def BollingerBands(
    series: pd.Series, period: int = 20, std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.
    Returns: (upper, middle, lower)
    """
    middle = SMA(series, period)
    rolling_std = series.rolling(window=period, min_periods=1).std()
    upper = middle + std * rolling_std
    lower = middle - std * rolling_std
    return upper, middle, lower


def VWAP(df: pd.DataFrame) -> pd.Series:
    """
    Volume Weighted Average Price.
    Requires columns: high, low, close, volume
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical_price * df["volume"]
    cum_tp_vol = tp_vol.cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def ATR(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range — measures volatility for stop-loss calculation.
    Requires columns: high, low, close
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=1).mean()


def compute_all(df: pd.DataFrame, sma_short: int = 7, sma_long: int = 25) -> dict:
    """
    Compute all indicators for a given DataFrame and return the latest values.
    Returns a flat dict of the most recent computed indicator values.
    """
    closes = df["close"]

    sma_s = SMA(closes, sma_short)
    sma_l = SMA(closes, sma_long)
    ema_s = EMA(closes, sma_short)
    ema_l = EMA(closes, sma_long)
    rsi = RSI(closes)
    macd_line, signal_line, histogram = MACD(closes)
    bb_upper, bb_mid, bb_lower = BollingerBands(closes)
    vwap = VWAP(df)
    atr = ATR(df)

    return {
        f"sma{sma_short}": round(float(sma_s.iloc[-1]), 6),
        f"sma{sma_long}": round(float(sma_l.iloc[-1]), 6),
        f"ema{sma_short}": round(float(ema_s.iloc[-1]), 6),
        f"ema{sma_long}": round(float(ema_l.iloc[-1]), 6),
        "rsi": round(float(rsi.iloc[-1]), 2),
        "macd": round(float(macd_line.iloc[-1]), 6),
        "macd_signal": round(float(signal_line.iloc[-1]), 6),
        "macd_hist": round(float(histogram.iloc[-1]), 6),
        "bb_upper": round(float(bb_upper.iloc[-1]), 6),
        "bb_mid": round(float(bb_mid.iloc[-1]), 6),
        "bb_lower": round(float(bb_lower.iloc[-1]), 6),
        "vwap": round(float(vwap.iloc[-1]), 6),
        "atr": round(float(atr.iloc[-1]), 6),
        # Crossover flags
        "sma_cross_up": bool(sma_s.iloc[-1] > sma_l.iloc[-1] and sma_s.iloc[-2] <= sma_l.iloc[-2]),
        "sma_cross_down": bool(sma_s.iloc[-1] < sma_l.iloc[-1] and sma_s.iloc[-2] >= sma_l.iloc[-2]),
        "macd_cross_up": bool(macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]),
        "macd_cross_down": bool(macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]),
    }
