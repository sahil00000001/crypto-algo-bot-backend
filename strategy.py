# strategy.py — Trading strategies

from abc import ABC, abstractmethod
import pandas as pd
from config import SMA_SHORT, SMA_LONG, RSI_OVERBOUGHT, RSI_OVERSOLD, SIGNAL_CONFIDENCE_THRESHOLD


class Strategy(ABC):
    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, candle_info: dict, indicators: dict) -> dict:
        raise NotImplementedError


class CandlePatternStrategy(Strategy):
    _BUY = {"bullish_engulfing", "morning_star", "hammer", "three_white_soldiers",
             "tweezer_bottom", "inverted_hammer", "bullish_marubozu"}
    _SELL = {"bearish_engulfing", "evening_star", "shooting_star", "three_black_crows",
              "tweezer_top", "bearish_marubozu"}

    def generate_signal(self, df, candle_info, indicators) -> dict:
        p = candle_info.get("pattern", "none")
        s = candle_info.get("pattern_strength", 0)
        if p in self._BUY:
            return {"signal": "BUY", "confidence": min(s * 20, 95),
                    "reason": f"Pattern: {p}"}
        if p in self._SELL:
            return {"signal": "SELL", "confidence": min(s * 20, 95),
                    "reason": f"Pattern: {p}"}
        return {"signal": "HOLD", "confidence": 0, "reason": "No pattern"}


class TrendFollowStrategy(Strategy):
    def generate_signal(self, df, candle_info, indicators) -> dict:
        sma_s = indicators.get(f"sma{SMA_SHORT}", 0)
        sma_l = indicators.get(f"sma{SMA_LONG}", 0)
        ratio = candle_info.get("up_down_ratio", 1.0)
        con_up = candle_info.get("consecutive_up", 0)
        con_dn = candle_info.get("consecutive_down", 0)

        if sma_s > sma_l and ratio > 1.3 and con_up >= 2:
            conf = min(50 + int((ratio - 1) * 25) + con_up * 5, 90)
            return {"signal": "BUY", "confidence": conf,
                    "reason": f"SMA{SMA_SHORT}>{SMA_LONG}, ratio={ratio:.2f}, {con_up} green"}
        if sma_s < sma_l and ratio < 0.7 and con_dn >= 2:
            conf = min(50 + int((1 - ratio) * 25) + con_dn * 5, 90)
            return {"signal": "SELL", "confidence": conf,
                    "reason": f"SMA{SMA_SHORT}<{SMA_LONG}, ratio={ratio:.2f}, {con_dn} red"}
        return {"signal": "HOLD", "confidence": 0, "reason": "No trend"}


class RSICandleStrategy(Strategy):
    _BUY_PATTERNS = {"bullish_engulfing", "morning_star", "hammer", "three_white_soldiers"}
    _SELL_PATTERNS = {"bearish_engulfing", "evening_star", "shooting_star", "three_black_crows"}

    def generate_signal(self, df, candle_info, indicators) -> dict:
        rsi = indicators.get("rsi", 50)
        pattern = candle_info.get("pattern", "none")
        con_up = candle_info.get("consecutive_up", 0)
        con_dn = candle_info.get("consecutive_down", 0)

        if rsi < RSI_OVERSOLD and (pattern in self._BUY_PATTERNS or con_up >= 2):
            conf = min(60 + int((RSI_OVERSOLD - rsi) * 1.5), 95)
            return {"signal": "BUY", "confidence": conf,
                    "reason": f"RSI oversold ({rsi:.1f}) + {pattern}"}
        if rsi > RSI_OVERBOUGHT and (pattern in self._SELL_PATTERNS or con_dn >= 2):
            conf = min(60 + int((rsi - RSI_OVERBOUGHT) * 1.5), 95)
            return {"signal": "SELL", "confidence": conf,
                    "reason": f"RSI overbought ({rsi:.1f}) + {pattern}"}
        return {"signal": "HOLD", "confidence": 0, "reason": f"RSI neutral ({rsi:.1f})"}


class MasterStrategy(Strategy):
    def __init__(self):
        self._strats: list[tuple[Strategy, int]] = [
            (CandlePatternStrategy(), 3),
            (TrendFollowStrategy(), 2),
            (RSICandleStrategy(), 2),
        ]

    def generate_signal(self, df, candle_info, indicators) -> dict:
        buy_w = sell_w = 0
        confs: list[float] = []
        reasons: list[str] = []

        for strat, w in self._strats:
            r = strat.generate_signal(df, candle_info, indicators)
            if r["signal"] == "BUY":
                buy_w += w; confs.append(r["confidence"])
                reasons.append(r["reason"])
            elif r["signal"] == "SELL":
                sell_w += w; confs.append(r["confidence"])
                reasons.append(r["reason"])

        avg_conf = int(sum(confs) / len(confs)) if confs else 0

        if buy_w > sell_w and buy_w >= 2 and avg_conf >= SIGNAL_CONFIDENCE_THRESHOLD:
            return {"signal": "BUY", "confidence": avg_conf, "reason": " | ".join(reasons[:2])}
        if sell_w > buy_w and sell_w >= 2 and avg_conf >= SIGNAL_CONFIDENCE_THRESHOLD:
            return {"signal": "SELL", "confidence": avg_conf, "reason": " | ".join(reasons[:2])}
        return {"signal": "HOLD", "confidence": avg_conf or 0, "reason": "No consensus"}


STRATEGIES: dict[str, type[Strategy]] = {
    "candle": CandlePatternStrategy,
    "trend": TrendFollowStrategy,
    "rsi": RSICandleStrategy,
    "master": MasterStrategy,
}

def get_strategy(name: str) -> Strategy:
    cls = STRATEGIES.get(name.lower())
    if not cls:
        raise ValueError(f"Unknown strategy '{name}'. Options: {list(STRATEGIES)}")
    return cls()
