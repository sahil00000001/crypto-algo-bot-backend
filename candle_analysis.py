# candle_analysis.py — Candlestick pattern detection and candle statistics

import pandas as pd
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def _r(row: pd.Series) -> dict:
    return {k: float(row[k]) if k not in ("is_up",) else bool(row[k])
            for k in ("open", "high", "low", "close", "volume",
                      "body_size", "upper_wick", "lower_wick", "candle_range", "body_pct", "is_up")}


# ── Single Candle Patterns ────────────────────────────────────────────────────

def is_doji(c: dict) -> bool:
    return c["candle_range"] > 0 and c["body_pct"] < 0.10

def is_hammer(c: dict) -> bool:
    return (c["body_pct"] < 0.4
            and c["lower_wick"] >= 2 * max(c["body_size"], 1e-9)
            and c["upper_wick"] <= 0.15 * c["candle_range"])

def is_inverted_hammer(c: dict) -> bool:
    return (c["body_pct"] < 0.4
            and c["upper_wick"] >= 2 * max(c["body_size"], 1e-9)
            and c["lower_wick"] <= 0.15 * c["candle_range"])

def is_shooting_star(c: dict) -> bool:
    return is_inverted_hammer(c) and not c["is_up"]

def is_marubozu(c: dict) -> bool:
    return c["body_pct"] > 0.90

def is_spinning_top(c: dict) -> bool:
    return (c["body_pct"] < 0.35
            and c["upper_wick"] >= 0.25 * c["candle_range"]
            and c["lower_wick"] >= 0.25 * c["candle_range"])


# ── Multi-Candle Patterns ─────────────────────────────────────────────────────

def is_bullish_engulfing(c1: dict, c2: dict) -> bool:
    return (not c1["is_up"] and c2["is_up"]
            and c2["open"] <= c1["close"]
            and c2["close"] >= c1["open"]
            and c2["body_size"] > c1["body_size"])

def is_bearish_engulfing(c1: dict, c2: dict) -> bool:
    return (c1["is_up"] and not c2["is_up"]
            and c2["open"] >= c1["close"]
            and c2["close"] <= c1["open"]
            and c2["body_size"] > c1["body_size"])

def is_morning_star(c1: dict, c2: dict, c3: dict) -> bool:
    return (not c1["is_up"] and c1["body_pct"] > 0.6
            and c2["body_pct"] < 0.35
            and c3["is_up"] and c3["body_pct"] > 0.6
            and c3["close"] > (c1["open"] + c1["close"]) / 2)

def is_evening_star(c1: dict, c2: dict, c3: dict) -> bool:
    return (c1["is_up"] and c1["body_pct"] > 0.6
            and c2["body_pct"] < 0.35
            and not c3["is_up"] and c3["body_pct"] > 0.6
            and c3["close"] < (c1["open"] + c1["close"]) / 2)

def is_three_white_soldiers(c1: dict, c2: dict, c3: dict) -> bool:
    return (c1["is_up"] and c2["is_up"] and c3["is_up"]
            and c1["body_pct"] > 0.5 and c2["body_pct"] > 0.5 and c3["body_pct"] > 0.5
            and c2["close"] > c1["close"] and c3["close"] > c2["close"])

def is_three_black_crows(c1: dict, c2: dict, c3: dict) -> bool:
    return (not c1["is_up"] and not c2["is_up"] and not c3["is_up"]
            and c1["body_pct"] > 0.5 and c2["body_pct"] > 0.5 and c3["body_pct"] > 0.5
            and c2["close"] < c1["close"] and c3["close"] < c2["close"])

def is_tweezer_top(c1: dict, c2: dict) -> bool:
    return (c1["is_up"] and not c2["is_up"]
            and abs(c1["high"] - c2["high"]) / max(c1["high"], 1e-9) < 0.001)

def is_tweezer_bottom(c1: dict, c2: dict) -> bool:
    return (not c1["is_up"] and c2["is_up"]
            and abs(c1["low"] - c2["low"]) / max(c1["low"], 1e-9) < 0.001)


# ── Candle Statistics ─────────────────────────────────────────────────────────

def up_candle_count(df: pd.DataFrame, period: int = 10) -> int:
    return int(df["is_up"].iloc[-period:].sum())

def down_candle_count(df: pd.DataFrame, period: int = 10) -> int:
    n = min(period, len(df))
    return n - up_candle_count(df, n)

def up_down_ratio(df: pd.DataFrame, period: int = 10) -> float:
    up = up_candle_count(df, period)
    dn = down_candle_count(df, period)
    return up / max(dn, 1)

def avg_body_size(df: pd.DataFrame, period: int = 10) -> float:
    return float(df["body_size"].iloc[-period:].mean())

def avg_volume(df: pd.DataFrame, period: int = 10) -> float:
    return float(df["volume"].iloc[-period:].mean())

def consecutive_up(df: pd.DataFrame) -> int:
    count = 0
    for i in range(len(df) - 1, -1, -1):
        if df.iloc[i]["is_up"]:
            count += 1
        else:
            break
    return count

def consecutive_down(df: pd.DataFrame) -> int:
    count = 0
    for i in range(len(df) - 1, -1, -1):
        if not df.iloc[i]["is_up"]:
            count += 1
        else:
            break
    return count

def is_uptrend(df: pd.DataFrame, period: int = 10) -> bool:
    n = min(period, len(df))
    closes = df["close"].iloc[-n:]
    rising = float(closes.iloc[-1]) > float(closes.mean())
    return rising and up_candle_count(df, n) > down_candle_count(df, n)

def is_downtrend(df: pd.DataFrame, period: int = 10) -> bool:
    n = min(period, len(df))
    closes = df["close"].iloc[-n:]
    falling = float(closes.iloc[-1]) < float(closes.mean())
    return falling and down_candle_count(df, n) > up_candle_count(df, n)

def candle_momentum(df: pd.DataFrame, period: int = 5) -> float:
    n = min(period, len(df))
    return float((df["close"] - df["open"]).iloc[-n:].sum())


# ── Main Analysis ─────────────────────────────────────────────────────────────

def analyze_candles(df: pd.DataFrame) -> dict:
    if len(df) < 3:
        return {
            "pattern": "none", "pattern_signal": "HOLD", "pattern_strength": 0,
            "up_candles_10": 0, "down_candles_10": 0, "up_down_ratio": 1.0,
            "consecutive_up": 0, "consecutive_down": 0,
            "avg_body": 0.0, "avg_volume": 0.0, "momentum": 0.0,
            "trend": "SIDEWAYS",
        }

    period = min(10, len(df))
    c1 = _r(df.iloc[-1])
    c2 = _r(df.iloc[-2]) if len(df) >= 2 else c1
    c3 = _r(df.iloc[-3]) if len(df) >= 3 else c2

    pattern = "none"
    sig = "HOLD"
    strength = 0

    # 5-star SELL
    if is_three_black_crows(c3, c2, c1):
        pattern, sig, strength = "three_black_crows", "SELL", 5
    elif is_evening_star(c3, c2, c1):
        pattern, sig, strength = "evening_star", "SELL", 5
    # 4-star SELL
    elif is_bearish_engulfing(c2, c1):
        pattern, sig, strength = "bearish_engulfing", "SELL", 4
    elif is_marubozu(c1) and not c1["is_up"]:
        pattern, sig, strength = "bearish_marubozu", "SELL", 4
    # 3-star SELL
    elif is_tweezer_top(c2, c1):
        pattern, sig, strength = "tweezer_top", "SELL", 3
    elif is_shooting_star(c1):
        pattern, sig, strength = "shooting_star", "SELL", 3
    # 5-star BUY
    elif is_three_white_soldiers(c3, c2, c1):
        pattern, sig, strength = "three_white_soldiers", "BUY", 5
    elif is_morning_star(c3, c2, c1):
        pattern, sig, strength = "morning_star", "BUY", 5
    # 4-star BUY
    elif is_bullish_engulfing(c2, c1):
        pattern, sig, strength = "bullish_engulfing", "BUY", 4
    elif is_marubozu(c1) and c1["is_up"]:
        pattern, sig, strength = "bullish_marubozu", "BUY", 4
    # 3-star BUY
    elif is_tweezer_bottom(c2, c1):
        pattern, sig, strength = "tweezer_bottom", "BUY", 3
    elif is_hammer(c1):
        pattern, sig, strength = "hammer", "BUY", 3
    elif is_inverted_hammer(c1):
        pattern, sig, strength = "inverted_hammer", "BUY", 2
    # Indecision
    elif is_doji(c1):
        pattern, sig, strength = "doji", "HOLD", 1
    elif is_spinning_top(c1):
        pattern, sig, strength = "spinning_top", "HOLD", 1

    trend = (
        "UPTREND" if is_uptrend(df, period)
        else "DOWNTREND" if is_downtrend(df, period)
        else "SIDEWAYS"
    )

    con_up = consecutive_up(df)
    con_dn = consecutive_down(df)

    return {
        "pattern": pattern,
        "pattern_signal": sig,
        "pattern_strength": strength,
        "up_candles_10": up_candle_count(df, period),
        "down_candles_10": down_candle_count(df, period),
        "up_down_ratio": round(up_down_ratio(df, period), 2),
        "consecutive_up": con_up,
        "consecutive_down": con_dn,
        "avg_body": round(avg_body_size(df, period), 6),
        "avg_volume": round(avg_volume(df, period), 2),
        "momentum": round(candle_momentum(df, 5), 6),
        "trend": trend,
    }
