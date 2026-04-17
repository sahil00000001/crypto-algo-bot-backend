# config.py — Central configuration for the crypto algo trading bot

PAIRS: list[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_INTERVAL: str = "5m"
INITIAL_BALANCE: float = 10000.0        # USDT
MAX_POSITION_PCT: float = 30.0          # max 30% of balance per trade
STOP_LOSS_PCT: float = 2.0              # sell if price drops 2%
TAKE_PROFIT_PCT: float = 4.0            # sell if price rises 4%

# SMA periods
SMA_SHORT: int = 7
SMA_LONG: int = 25

# RSI config
RSI_PERIOD: int = 14
RSI_OVERBOUGHT: int = 70
RSI_OVERSOLD: int = 30

# Candle analysis config
CANDLE_LOOKBACK: int = 3               # how many candles to look back for patterns
MIN_BODY_PCT: float = 0.3              # minimum body size as % of candle range

# Binance REST API
BINANCE_BASE_URL: str = "https://api.binance.com/api/v3"
BINANCE_WS_BASE: str = "wss://stream.binance.com:9443/ws"

# Strategy signal confidence threshold
SIGNAL_CONFIDENCE_THRESHOLD: int = 60  # minimum confidence to place a trade

# Server config (for web dashboard)
SERVER_HOST: str = "0.0.0.0"
SERVER_PORT: int = 8000

# Logging
LOG_FILE: str = "trades.log"

# Rate limiting
REQUEST_DELAY: float = 0.1  # seconds between REST requests
MAX_RETRIES: int = 3
