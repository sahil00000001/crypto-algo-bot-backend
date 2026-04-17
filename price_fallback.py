# price_fallback.py — CoinGecko REST for initial prices (NOT blocked from cloud)

import logging
import time
from typing import Optional

import requests

from config import COINGECKO_IDS, COINGECKO_CACHE_TTL

logger = logging.getLogger(__name__)

_cache: dict[str, float] = {}
_last_fetch: float = 0.0


def get_prices_coingecko(symbols: list[str]) -> dict[str, float]:
    """
    Fetch current USD prices via CoinGecko (unrestricted from all cloud servers).
    Caches results for COINGECKO_CACHE_TTL seconds to respect 30 req/min limit.
    """
    global _cache, _last_fetch

    now = time.time()
    if _cache and (now - _last_fetch) < COINGECKO_CACHE_TTL:
        return {s: _cache.get(s, 0.0) for s in symbols}

    ids = ",".join(COINGECKO_IDS.get(s, "") for s in symbols if COINGECKO_IDS.get(s))
    if not ids:
        return {s: 0.0 for s in symbols}

    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": ids, "vs_currencies": "usd"},
            timeout=10,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

        result = {}
        for sym in symbols:
            cg_id = COINGECKO_IDS.get(sym)
            if cg_id and cg_id in data:
                result[sym] = float(data[cg_id]["usd"])
            else:
                result[sym] = 0.0

        _cache = result
        _last_fetch = now
        logger.info(f"CoinGecko prices fetched: { {k: v for k,v in result.items()} }")
        return result

    except Exception as e:
        logger.warning(f"CoinGecko fetch failed: {e}")
        return {s: _cache.get(s, 0.0) for s in symbols}
