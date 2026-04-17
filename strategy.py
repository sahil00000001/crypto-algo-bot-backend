# strategy.py — Trading strategies combining candle patterns and indicators

from abc import ABC, abstractmethod
import pandas as pd

from candle_analysis import analyze_candles
import indicators as ind
from config import SMA_SHORT, SMA_LONG, RSI_OVERBOUGHT, RSI_OVERSOLD


class Strategy(ABC):
    """Base class for all trading strategies."""

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, candle_info: dict) -> dict:
        """
        Returns:
            {
                "signal": "BUY" | "SELL" | "HOLD",
                "confidence": 0–100,
                "reason": str,
            }
        """
        raise NotImplementedError


# ─── Strategy 1: Candle Pattern Only ─────────────────────────────────────────

class CandlePatternStrategy(Strategy):
    """Trades purely on candlestick patterns."""

    _BULLISH = {"Bullish Engulfing", "Morning Star", "Hammer", "Three White Soldiers", "Tweezer Bottom", "Inverted Hammer"}
    _BEARISH = {"Bearish Engulfing", "Evening Star", "Shooting Star", "Three Black Crows", "Tweezer Top"}

    def generate_signal(self, df: pd.DataFrame, candle_info: dict) -> dict:
        pattern = candle_info.get("pattern", "none")
        strength = candle_info.get("strength", 0)

        if pattern in self._BULLISH:
            confidence = min(40 + strength * 12, 95)
            return {"signal": "BUY", "confidence": confidence, "reason": f"Candle pattern: {pattern}"}

        if pattern in self._BEARISH:
            confidence = min(40 + strength * 12, 95)
            return {"signal": "SELL", "confidence": confidence, "reason": f"Candle pattern: {pattern}"}

        return {"signal": "HOLD", "confidence": 0, "reason": "No significant candle pattern"}


# ─── Strategy 2: Trend Follow ─────────────────────────────────────────────────

class TrendFollowStrategy(Strategy):
    """Uses candle averages + SMA crossover."""

    def generate_signal(self, df: pd.DataFrame, candle_info: dict) -> dict:
        computed = ind.compute_all(df, SMA_SHORT, SMA_LONG)
        sma_s = computed[f"sma{SMA_SHORT}"]
        sma_l = computed[f"sma{SMA_LONG}"]
        ratio = candle_info.get("up_down_ratio", 1.0)
        consec = candle_info.get("consecutive", 0)

        if sma_s > sma_l and ratio > 1.2 and consec >= 2:
            confidence = min(50 + int((ratio - 1) * 30) + consec * 5, 90)
            return {
                "signal": "BUY",
                "confidence": confidence,
                "reason": f"SMA{SMA_SHORT}>{SMA_LONG}, up/down ratio={ratio:.2f}, {consec} green candles",
            }

        if sma_s < sma_l and ratio < 0.8 and consec <= -2:
            confidence = min(50 + int((1 - ratio) * 30) + abs(consec) * 5, 90)
            return {
                "signal": "SELL",
                "confidence": confidence,
                "reason": f"SMA{SMA_SHORT}<{SMA_LONG}, up/down ratio={ratio:.2f}, {abs(consec)} red candles",
            }

        return {"signal": "HOLD", "confidence": 0, "reason": "No clear trend signal"}


# ─── Strategy 3: RSI + Candle ─────────────────────────────────────────────────

class RSI_Candle_Strategy(Strategy):
    """Combines RSI extremes with candle patterns for confirmation."""

    _BULLISH = {"Bullish Engulfing", "Morning Star", "Hammer", "Three White Soldiers", "Tweezer Bottom"}
    _BEARISH = {"Bearish Engulfing", "Evening Star", "Shooting Star", "Three Black Crows", "Tweezer Top"}

    def generate_signal(self, df: pd.DataFrame, candle_info: dict) -> dict:
        computed = ind.compute_all(df, SMA_SHORT, SMA_LONG)
        rsi = computed["rsi"]
        pattern = candle_info.get("pattern", "none")
        signal_from_candle = candle_info.get("signal", "HOLD")

        if rsi < RSI_OVERSOLD and (pattern in self._BULLISH or signal_from_candle == "BUY"):
            confidence = min(60 + int((RSI_OVERSOLD - rsi) * 1.5), 95)
            return {
                "signal": "BUY",
                "confidence": confidence,
                "reason": f"RSI oversold ({rsi:.1f}) + bullish pattern ({pattern})",
            }

        if rsi > RSI_OVERBOUGHT and (pattern in self._BEARISH or signal_from_candle == "SELL"):
            confidence = min(60 + int((rsi - RSI_OVERBOUGHT) * 1.5), 95)
            return {
                "signal": "SELL",
                "confidence": confidence,
                "reason": f"RSI overbought ({rsi:.1f}) + bearish pattern ({pattern})",
            }

        return {"signal": "HOLD", "confidence": 0, "reason": f"RSI neutral ({rsi:.1f}), no confirmation"}


# ─── Strategy 4: Master (weighted voting) ────────────────────────────────────

class MasterStrategy(Strategy):
    """
    Combines all strategies with weighted voting.
    Weights: CandlePattern=3, TrendFollow=2, RSI_Candle=2
    """

    def __init__(self):
        self._strategies: list[tuple[Strategy, int]] = [
            (CandlePatternStrategy(), 3),
            (TrendFollowStrategy(), 2),
            (RSI_Candle_Strategy(), 2),
        ]

    def generate_signal(self, df: pd.DataFrame, candle_info: dict) -> dict:
        buy_votes = sell_votes = 0
        total_confidence = []
        reasons: list[str] = []

        for strategy, weight in self._strategies:
            result = strategy.generate_signal(df, candle_info)
            sig = result["signal"]
            conf = result["confidence"]
            reason = result["reason"]

            if sig == "BUY":
                buy_votes += weight
                total_confidence.append(conf)
                reasons.append(f"[{strategy.__class__.__name__}] {reason}")
            elif sig == "SELL":
                sell_votes += weight
                total_confidence.append(conf)
                reasons.append(f"[{strategy.__class__.__name__}] {reason}")

        avg_conf = int(sum(total_confidence) / len(total_confidence)) if total_confidence else 0

        if buy_votes > sell_votes and buy_votes >= 2:
            return {"signal": "BUY", "confidence": avg_conf, "reason": " | ".join(reasons[:2])}
        if sell_votes > buy_votes and sell_votes >= 2:
            return {"signal": "SELL", "confidence": avg_conf, "reason": " | ".join(reasons[:2])}

        return {"signal": "HOLD", "confidence": 0, "reason": "No consensus across strategies"}


STRATEGIES: dict[str, type[Strategy]] = {
    "candle": CandlePatternStrategy,
    "trend": TrendFollowStrategy,
    "rsi": RSI_Candle_Strategy,
    "master": MasterStrategy,
}


def get_strategy(name: str) -> Strategy:
    cls = STRATEGIES.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGIES)}")
    return cls()
