# config.py — Central configuration for the crypto algo trading bot

PAIRS: list[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_INTERVAL: str = "5m"
INITIAL_BALANCE: float = 10000.0
MAX_POSITION_PCT: float = 30.0
STOP_LOSS_PCT: float = 2.0
TAKE_PROFIT_PCT: float = 4.0

SMA_SHORT: int = 7
SMA_LONG: int = 25
RSI_PERIOD: int = 14
RSI_OVERBOUGHT: int = 70
RSI_OVERSOLD: int = 30

CANDLE_LOOKBACK: int = 3
MIN_BODY_PCT: float = 0.3

# Bybit REST API (no geo-restriction, no API key needed)
BYBIT_BASE_URL: str = "https://api.bybit.com/v5/market"

# Bybit WebSocket
BYBIT_WS_SPOT: str = "wss://stream.bybit.com/v5/public/spot"

# Interval map: human-readable → Bybit format
INTERVAL_MAP: dict[str, str] = {
    "1m": "1",  "3m": "3",  "5m": "5",  "15m": "15",
    "30m": "30", "1h": "60", "4h": "240", "1d": "D",
}

# Strategy signal confidence threshold
SIGNAL_CONFIDENCE_THRESHOLD: int = 60

# Server config
SERVER_HOST: str = "0.0.0.0"
SERVER_PORT: int = 7860

# Logging
LOG_FILE: str = "trades.log"

# Rate limiting
REQUEST_DELAY: float = 0.1
MAX_RETRIES: int = 3
