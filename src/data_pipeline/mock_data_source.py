"""
Mock market data source used for local development and testing.

The original implementation lived in ``src/data-pipeline/`` which prevented it
from being imported as ``src.data_pipeline`` (packages cannot contain hyphens).
Pytest fixtures and the local CLI both expect ``MockDataSource`` to be
accessible from ``src.data_pipeline.mock_data_source``; the missing module
caused the entire test suite to fail on import.

This file restores that module and keeps the behaviour of the previous
implementation, while removing a couple of subtle bugs:

* Repeated calls now generate fresh data – the old version decorated the public
  methods with ``functools.lru_cache`` which returned stale results.
* The code is structured to make dependency injection in tests easier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd


DEFAULT_BASE_PRICES: Dict[str, float] = {
    "BTC-USD": 50_000.0,
    "ETH-USD": 3_000.0,
    "SOL-USD": 100.0,
    "MATIC-USD": 0.80,
    "AVAX-USD": 35.0,
}

DEFAULT_BASE_VOLUMES: Dict[str, int] = {
    "BTC-USD": 1_000_000,
    "ETH-USD": 500_000,
    "SOL-USD": 100_000,
    "MATIC-USD": 50_000,
    "AVAX-USD": 75_000,
}


def _ensure_symbols(symbols: Optional[Iterable[str]]) -> List[str]:
    if not symbols:
        return ["BTC-USD", "ETH-USD"]
    return list(symbols)


@dataclass
class MockDataSource:
    """Generate realistic OHLCV data without requiring external APIs."""

    symbols: List[str] = field(default_factory=lambda: ["BTC-USD", "ETH-USD"])
    base_prices: Dict[str, float] = field(default_factory=lambda: DEFAULT_BASE_PRICES.copy())
    volatility: float = 0.02
    update_interval: float = 1.0
    trend_strength: float = 0.0001
    _trends: Dict[str, float] = field(init=False)
    _current_prices: Dict[str, float] = field(init=False)
    _last_update: Dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.symbols = _ensure_symbols(self.symbols)
        self.base_prices = {**DEFAULT_BASE_PRICES, **(self.base_prices or {})}
        self._current_prices = {symbol: self.base_prices.get(symbol, 100.0) for symbol in self.symbols}
        self._trends = {symbol: 0.0 for symbol in self.symbols}
        self._last_update = {symbol: time.time() for symbol in self.symbols}

    # --------------------------------------------------------------------- #
    # Data generation helpers
    # --------------------------------------------------------------------- #

    def _step_price(self, symbol: str) -> float:
        """Generate the next price movement for a symbol."""
        dt_days = self.update_interval / 86_400  # convert seconds -> days

        random_component = random.gauss(0.0, self.volatility * math.sqrt(dt_days))
        self._trends[symbol] = 0.95 * self._trends[symbol] + random.gauss(0.0, 0.0001)
        trend_component = self._trends[symbol] * dt_days

        return random_component + trend_component

    def _build_ohlcv(self, symbol: str) -> Dict[str, float]:
        """Return the next OHLCV bar for ``symbol``."""
        price_change = self._step_price(symbol)

        open_price = self._current_prices[symbol]
        close_price = max(open_price * (1 + price_change), 0.01)

        high_variation = abs(random.gauss(0, self.volatility * 0.5))
        low_variation = abs(random.gauss(0, self.volatility * 0.5))

        high_price = max(open_price, close_price) * (1 + high_variation)
        low_price = min(open_price, close_price) * (1 - low_variation)

        volume_base = DEFAULT_BASE_VOLUMES.get(symbol, 100_000)
        volume = volume_base * random.uniform(0.7, 1.3)

        self._current_prices[symbol] = close_price
        self._last_update[symbol] = time.time()

        return {
            "symbol": symbol,
            "timestamp": self._last_update[symbol],
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": round(volume, 2),
        }

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    @property
    def current_prices(self) -> Dict[str, float]:
        """Expose the mutable price state for tests and monitoring."""
        return self._current_prices

    def reset(self) -> None:
        """Reset prices and trend state back to the configured base values."""
        for symbol in self.symbols:
            self._current_prices[symbol] = self.base_prices.get(symbol, 100.0)
            self._trends[symbol] = 0.0
            self._last_update[symbol] = time.time()

    def get_latest_data(self, symbol: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Return the most recent OHLCV snapshot.

        Parameters
        ----------
        symbol:
            Restrict to a single symbol when provided. Otherwise returns all
            known symbols.
        """
        if symbol:
            return {symbol: self._build_ohlcv(symbol)}
        return {sym: self._build_ohlcv(sym) for sym in self.symbols}

    def get_historical_data(
        self,
        symbol: str,
        periods: int = 100,
        interval_seconds: int = 60,
    ) -> pd.DataFrame:
        """
        Build a synthetic OHLCV history suitable for backtests.
        """
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            self._current_prices[symbol] = self.base_prices.get(symbol, 100.0)
            self._trends[symbol] = 0.0

        self.reset()
        start_time = time.time() - periods * interval_seconds

        history: List[Dict[str, float]] = []
        for idx in range(periods):
            bar = self._build_ohlcv(symbol)
            bar["timestamp"] = start_time + idx * interval_seconds
            history.append(bar)

        df = pd.DataFrame(history)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        return df

    def stream(self, callback: Callable[[Dict[str, Dict[str, float]]], None], interval: Optional[float] = None) -> None:
        """
        Continuously emit market data until the callback raises or a KeyboardInterrupt occurs.
        """
        interval = interval or self.update_interval
        try:
            while True:
                callback(self.get_latest_data())
                time.sleep(interval)
        except KeyboardInterrupt:
            # Allow graceful shutdown during manual testing/dev workflows.
            return


class MockOrderBook:
    """Generate synthetic order book snapshots for visualisations and tests."""

    def __init__(self, symbol: str, mid_price: float, spread_pct: float = 0.001) -> None:
        self.symbol = symbol
        self.mid_price = mid_price
        self.spread_pct = spread_pct

    def get_orderbook(self, levels: int = 10) -> Dict[str, object]:
        spread = self.mid_price * self.spread_pct
        best_bid = self.mid_price - spread / 2
        best_ask = self.mid_price + spread / 2

        bids: List[List[float]] = []
        asks: List[List[float]] = []

        for level in range(levels):
            bid_price = best_bid * (1 - level * 0.0005)
            ask_price = best_ask * (1 + level * 0.0005)
            bid_size = random.uniform(0.1, 2.0) * (1 + level * 0.5)
            ask_size = random.uniform(0.1, 2.0) * (1 + level * 0.5)
            bids.append([round(bid_price, 2), round(bid_size, 4)])
            asks.append([round(ask_price, 2), round(ask_size, 4)])

        return {
            "symbol": self.symbol,
            "timestamp": time.time(),
            "bids": bids,
            "asks": asks,
        }


class MockSentimentData:
    """Simple sentiment generator using a mean-reverting random walk."""

    def __init__(self) -> None:
        self._base_sentiment = 0.5
        self._drift = 0.0

    def get_sentiment(self, symbol: str) -> Dict[str, float]:
        self._drift = 0.9 * self._drift + random.gauss(0.0, 0.05)
        sentiment = min(max(self._base_sentiment + self._drift, 0.0), 1.0)

        bullish = int(sentiment * 100)
        bearish = 100 - bullish

        return {
            "symbol": symbol,
            "timestamp": time.time(),
            "sentiment": sentiment,
            "confidence": random.uniform(0.6, 0.9),
            "sources_count": random.randint(10, 100),
            "bullish_count": bullish,
            "bearish_count": bearish,
        }


__all__ = ["MockDataSource", "MockOrderBook", "MockSentimentData"]
