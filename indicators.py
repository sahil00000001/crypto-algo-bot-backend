# indicators.py — Technical indicators (gracefully handle insufficient data)

import numpy as np
import pandas as pd
from typing import Optional

from config import SMA_SHORT, SMA_LONG


def SMA(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=1).mean()

def EMA(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def RSI(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def MACD(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
         ) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_f = EMA(series, fast)
    ema_s = EMA(series, slow)
    macd_line = ema_f - ema_s
    sig_line = EMA(macd_line, signal)
    return macd_line, sig_line, macd_line - sig_line

def BollingerBands(series: pd.Series, period: int = 20, std: float = 2.0
                   ) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = SMA(series, period)
    sd = series.rolling(window=period, min_periods=1).std().fillna(0)
    return mid + std * sd, mid, mid - std * sd

def ATR(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=1).mean()


def compute_all(df: pd.DataFrame) -> dict:
    """Compute all indicators and return latest values as a flat dict."""
    if df.empty or len(df) < 2:
        return _empty_indicators()

    closes = df["close"]
    sma_s = SMA(closes, SMA_SHORT)
    sma_l = SMA(closes, SMA_LONG)
    rsi = RSI(closes)
    macd_line, sig_line, hist = MACD(closes)
    bb_u, bb_m, bb_l = BollingerBands(closes)
    atr = ATR(df)

    def _f(s: pd.Series) -> float:
        v = s.iloc[-1]
        return round(float(v), 6) if pd.notna(v) else 0.0

    return {
        f"sma{SMA_SHORT}": _f(sma_s),
        f"sma{SMA_LONG}": _f(sma_l),
        "rsi": _f(rsi),
        "macd": _f(macd_line),
        "macd_signal": _f(sig_line),
        "macd_hist": _f(hist),
        "bb_upper": _f(bb_u),
        "bb_mid": _f(bb_m),
        "bb_lower": _f(bb_l),
        "atr": _f(atr),
        "sma_cross_up": bool(
            len(sma_s) >= 2
            and sma_s.iloc[-1] > sma_l.iloc[-1]
            and sma_s.iloc[-2] <= sma_l.iloc[-2]
        ),
        "sma_cross_down": bool(
            len(sma_s) >= 2
            and sma_s.iloc[-1] < sma_l.iloc[-1]
            and sma_s.iloc[-2] >= sma_l.iloc[-2]
        ),
    }


def _empty_indicators() -> dict:
    return {
        f"sma{SMA_SHORT}": 0.0, f"sma{SMA_LONG}": 0.0,
        "rsi": 50.0, "macd": 0.0, "macd_signal": 0.0, "macd_hist": 0.0,
        "bb_upper": 0.0, "bb_mid": 0.0, "bb_lower": 0.0, "atr": 0.0,
        "sma_cross_up": False, "sma_cross_down": False,
    }
