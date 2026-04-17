# candle_analysis.py — Candlestick pattern detection and candle statistics

import pandas as pd
import numpy as np
from typing import NamedTuple


class CandleRow(NamedTuple):
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_up: bool
    body_size: float
    upper_wick: float
    lower_wick: float
    candle_range: float
    body_pct: float


def _row(candle: pd.Series) -> CandleRow:
    return CandleRow(
        open=candle["open"],
        high=candle["high"],
        low=candle["low"],
        close=candle["close"],
        volume=candle["volume"],
        is_up=bool(candle["is_up"]),
        body_size=float(candle["body_size"]),
        upper_wick=float(candle["upper_wick"]),
        lower_wick=float(candle["lower_wick"]),
        candle_range=float(candle["candle_range"]),
        body_pct=float(candle["body_pct"]),
    )


# ─── Single Candle Patterns ───────────────────────────────────────────────────

def is_doji(candle: pd.Series) -> bool:
    """Body is very small compared to range (body_pct < 10%). Signals indecision."""
    c = _row(candle)
    return c.candle_range > 0 and c.body_pct < 0.10


def is_hammer(candle: pd.Series) -> bool:
    """Small body at top, long lower wick (≥2x body). Bullish reversal."""
    c = _row(candle)
    if c.candle_range == 0:
        return False
    has_small_body = c.body_pct < 0.4
    long_lower_wick = c.lower_wick >= 2 * c.body_size
    small_upper_wick = c.upper_wick <= 0.1 * c.candle_range
    return has_small_body and long_lower_wick and small_upper_wick


def is_inverted_hammer(candle: pd.Series) -> bool:
    """Small body at bottom, long upper wick. Potential bullish reversal."""
    c = _row(candle)
    if c.candle_range == 0:
        return False
    has_small_body = c.body_pct < 0.4
    long_upper_wick = c.upper_wick >= 2 * c.body_size
    small_lower_wick = c.lower_wick <= 0.1 * c.candle_range
    return has_small_body and long_upper_wick and small_lower_wick


def is_shooting_star(candle: pd.Series) -> bool:
    """Small body at bottom after uptrend, long upper wick. Bearish signal."""
    return is_inverted_hammer(candle) and not candle["is_up"]


def is_marubozu(candle: pd.Series) -> bool:
    """Very large body (body_pct > 90%), no/tiny wicks. Strong momentum candle."""
    c = _row(candle)
    return c.body_pct > 0.90


def is_spinning_top(candle: pd.Series) -> bool:
    """Small body, long wicks both sides. Indecision."""
    c = _row(candle)
    if c.candle_range == 0:
        return False
    small_body = c.body_pct < 0.35
    significant_wicks = (
        c.upper_wick >= 0.25 * c.candle_range
        and c.lower_wick >= 0.25 * c.candle_range
    )
    return small_body and significant_wicks


# ─── Multi-Candle Patterns ────────────────────────────────────────────────────

def is_bullish_engulfing(candles: pd.DataFrame) -> bool:
    """Red candle followed by bigger green candle that fully covers it. BUY signal."""
    if len(candles) < 2:
        return False
    prev, curr = _row(candles.iloc[-2]), _row(candles.iloc[-1])
    return (
        not prev.is_up
        and curr.is_up
        and curr.open <= prev.close
        and curr.close >= prev.open
        and curr.body_size > prev.body_size
    )


def is_bearish_engulfing(candles: pd.DataFrame) -> bool:
    """Green candle followed by bigger red candle. SELL signal."""
    if len(candles) < 2:
        return False
    prev, curr = _row(candles.iloc[-2]), _row(candles.iloc[-1])
    return (
        prev.is_up
        and not curr.is_up
        and curr.open >= prev.close
        and curr.close <= prev.open
        and curr.body_size > prev.body_size
    )


def is_morning_star(candles: pd.DataFrame) -> bool:
    """Big red → small body → big green. Strong BUY signal."""
    if len(candles) < 3:
        return False
    c1, c2, c3 = _row(candles.iloc[-3]), _row(candles.iloc[-2]), _row(candles.iloc[-1])
    big_red = not c1.is_up and c1.body_pct > 0.6
    small_middle = c2.body_pct < 0.35
    big_green = c3.is_up and c3.body_pct > 0.6
    gap_down = c2.close < c1.close
    recovery = c3.close > (c1.open + c1.close) / 2
    return big_red and small_middle and big_green and gap_down and recovery


def is_evening_star(candles: pd.DataFrame) -> bool:
    """Big green → small body → big red. Strong SELL signal."""
    if len(candles) < 3:
        return False
    c1, c2, c3 = _row(candles.iloc[-3]), _row(candles.iloc[-2]), _row(candles.iloc[-1])
    big_green = c1.is_up and c1.body_pct > 0.6
    small_middle = c2.body_pct < 0.35
    big_red = not c3.is_up and c3.body_pct > 0.6
    gap_up = c2.close > c1.close
    rejection = c3.close < (c1.open + c1.close) / 2
    return big_green and small_middle and big_red and gap_up and rejection


def is_three_white_soldiers(candles: pd.DataFrame) -> bool:
    """3 consecutive green candles, each closing higher. BUY signal."""
    if len(candles) < 3:
        return False
    last3 = [_row(candles.iloc[-(i+1)]) for i in range(2, -1, -1)]
    return all(c.is_up and c.body_pct > 0.5 for c in last3) and (
        last3[1].close > last3[0].close and last3[2].close > last3[1].close
    )


def is_three_black_crows(candles: pd.DataFrame) -> bool:
    """3 consecutive red candles, each closing lower. SELL signal."""
    if len(candles) < 3:
        return False
    last3 = [_row(candles.iloc[-(i+1)]) for i in range(2, -1, -1)]
    return all(not c.is_up and c.body_pct > 0.5 for c in last3) and (
        last3[1].close < last3[0].close and last3[2].close < last3[1].close
    )


def is_tweezer_top(candles: pd.DataFrame) -> bool:
    """2 candles with nearly same high. SELL signal."""
    if len(candles) < 2:
        return False
    prev, curr = _row(candles.iloc[-2]), _row(candles.iloc[-1])
    similar_high = abs(prev.high - curr.high) / max(curr.high, 1) < 0.001
    return similar_high and prev.is_up and not curr.is_up


def is_tweezer_bottom(candles: pd.DataFrame) -> bool:
    """2 candles with nearly same low. BUY signal."""
    if len(candles) < 2:
        return False
    prev, curr = _row(candles.iloc[-2]), _row(candles.iloc[-1])
    similar_low = abs(prev.low - curr.low) / max(curr.low, 1) < 0.001
    return similar_low and not prev.is_up and curr.is_up


# ─── Candle Statistics & Trend ────────────────────────────────────────────────

def up_candle_count(candles: pd.DataFrame, period: int) -> int:
    """Count of green (up) candles in last N candles."""
    return int(candles["is_up"].iloc[-period:].sum())


def down_candle_count(candles: pd.DataFrame, period: int) -> int:
    """Count of red (down) candles in last N candles."""
    return period - up_candle_count(candles, period)


def up_down_ratio(candles: pd.DataFrame, period: int) -> float:
    """Ratio of up vs down candles. >1 = bullish, <1 = bearish."""
    up = up_candle_count(candles, period)
    down = down_candle_count(candles, period)
    return up / down if down > 0 else float(up)


def avg_candle_body(candles: pd.DataFrame, period: int) -> float:
    """Average body size over last N candles (measures momentum)."""
    return float(candles["body_size"].iloc[-period:].mean())


def candle_average(candles: pd.DataFrame, period: int) -> dict:
    """Average metrics over last N candles."""
    sliced = candles.iloc[-period:]
    return {
        "avg_body": float(sliced["body_size"].mean()),
        "avg_volume": float(sliced["volume"].mean()),
        "avg_upper_wick": float(sliced["upper_wick"].mean()),
        "avg_lower_wick": float(sliced["lower_wick"].mean()),
        "avg_range": float(sliced["candle_range"].mean()),
    }


def consecutive_up(candles: pd.DataFrame) -> int:
    """How many consecutive green candles from the latest."""
    count = 0
    for i in range(len(candles) - 1, -1, -1):
        if candles.iloc[i]["is_up"]:
            count += 1
        else:
            break
    return count


def consecutive_down(candles: pd.DataFrame) -> int:
    """How many consecutive red candles from the latest."""
    count = 0
    for i in range(len(candles) - 1, -1, -1):
        if not candles.iloc[i]["is_up"]:
            count += 1
        else:
            break
    return count


def is_uptrend(candles: pd.DataFrame, period: int = 20) -> bool:
    """True if SMA of closes is rising AND up_candle_count > down_candle_count."""
    if len(candles) < period:
        return False
    closes = candles["close"].iloc[-period:]
    sma_rising = closes.iloc[-1] > closes.mean()
    majority_up = up_candle_count(candles, period) > down_candle_count(candles, period)
    return sma_rising and majority_up


def is_downtrend(candles: pd.DataFrame, period: int = 20) -> bool:
    """True if SMA of closes is falling AND down > up."""
    if len(candles) < period:
        return False
    closes = candles["close"].iloc[-period:]
    sma_falling = closes.iloc[-1] < closes.mean()
    majority_down = down_candle_count(candles, period) > up_candle_count(candles, period)
    return sma_falling and majority_down


# ─── Main Analysis Function ───────────────────────────────────────────────────

def analyze_candles(df: pd.DataFrame) -> dict:
    """
    Run full candle analysis on DataFrame.
    Returns pattern, signal, strength, and statistics.
    """
    if len(df) < 3:
        return {
            "pattern": "none", "signal": "HOLD", "strength": 0,
            "up_candles_last_10": 0, "down_candles_last_10": 0,
            "up_down_ratio": 1.0, "consecutive": 0,
            "avg_body": 0.0, "trend": "SIDEWAYS"
        }

    period = min(10, len(df))
    last_candle = df.iloc[-1]

    # Determine trend
    if is_uptrend(df, min(20, len(df))):
        trend = "UPTREND"
    elif is_downtrend(df, min(20, len(df))):
        trend = "DOWNTREND"
    else:
        trend = "SIDEWAYS"

    # Pattern detection (priority order)
    pattern = "none"
    signal = "HOLD"
    strength = 0

    # --- SELL patterns (check first as they override) ---
    if is_three_black_crows(df):
        pattern, signal, strength = "Three Black Crows", "SELL", 5
    elif is_evening_star(df):
        pattern, signal, strength = "Evening Star", "SELL", 5
    elif is_bearish_engulfing(df):
        pattern, signal, strength = "Bearish Engulfing", "SELL", 4
    elif is_tweezer_top(df):
        pattern, signal, strength = "Tweezer Top", "SELL", 3
    elif is_shooting_star(last_candle):
        pattern, signal, strength = "Shooting Star", "SELL", 3

    # --- BUY patterns ---
    elif is_three_white_soldiers(df):
        pattern, signal, strength = "Three White Soldiers", "BUY", 5
    elif is_morning_star(df):
        pattern, signal, strength = "Morning Star", "BUY", 5
    elif is_bullish_engulfing(df):
        pattern, signal, strength = "Bullish Engulfing", "BUY", 4
    elif is_tweezer_bottom(df):
        pattern, signal, strength = "Tweezer Bottom", "BUY", 3
    elif is_hammer(last_candle):
        pattern, signal, strength = "Hammer", "BUY", 3

    # --- Indecision patterns ---
    elif is_doji(last_candle):
        pattern, signal, strength = "Doji", "HOLD", 1
    elif is_spinning_top(last_candle):
        pattern, signal, strength = "Spinning Top", "HOLD", 1
    elif is_marubozu(last_candle):
        direction = "BUY" if last_candle["is_up"] else "SELL"
        pattern, signal, strength = "Marubozu", direction, 4
    elif is_inverted_hammer(last_candle):
        pattern, signal, strength = "Inverted Hammer", "BUY", 2

    # Statistics
    up_count = up_candle_count(df, period)
    down_count = down_candle_count(df, period)
    ratio = up_down_ratio(df, period)
    consec = consecutive_up(df) if df.iloc[-1]["is_up"] else -consecutive_down(df)

    return {
        "pattern": pattern,
        "signal": signal,
        "strength": strength,
        "up_candles_last_10": up_count,
        "down_candles_last_10": down_count,
        "up_down_ratio": round(ratio, 2),
        "consecutive": consec,
        "avg_body": round(avg_candle_body(df, period), 6),
        "trend": trend,
    }
