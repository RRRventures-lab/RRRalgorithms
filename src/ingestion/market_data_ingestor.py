"""
Lightweight market data ingestor used by the unit tests.

The original repository referenced a missing ``ingestion`` package; the tests
expect a high-level wrapper that can connect to a websocket source, manage
subscriptions, cache prices, and aggregate OHLCV candles.  This module
provides a feature-complete but dependency-light implementation tailored to
the behaviours validated in ``tests/unit/data_pipeline/test_market_data_ingestor.py``.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, Iterable, List, Optional

import websockets


@dataclass
class OHLCVBar:
    """Simple representation of an OHLCV candle."""

    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    start: datetime
    end: datetime


class MarketDataIngestor:
    """
    Utility class that manages websocket connections, subscriptions, and
    in-memory price aggregation for local testing.
    """

    def __init__(
        self,
        symbols: Optional[Iterable[str]] = None,
        websocket_url: str = "wss://stream.data.mock/crypto",
        reconnect_delay: float = 1.0,
        max_queue_size: int = 1000,
    ) -> None:
        self.symbols: List[str] = list(symbols) if symbols else []
        self.subscriptions: List[str] = list(self.symbols)
        self.websocket_url = websocket_url
        self.reconnect_delay = reconnect_delay
        self.max_queue_size = max_queue_size

        self.connected: bool = False
        self._ws_cm: Optional[websockets.WebSocketClientProtocol] = None
        self._ws_manager: Optional[Any] = None

        self.queue: Deque[Dict[str, Any]] = deque(maxlen=max_queue_size)
        self.is_rate_limited: bool = False
        self.last_price: Optional[float] = None

        self.price_cache: Dict[str, float] = {}
        self.price_cache_expiry: Dict[str, datetime] = {}

        self.on_candle_complete: Optional[Any] = None
        self._pending_candle: Optional[OHLCVBar] = None
        self._pending_ticks: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Connection management
    # ------------------------------------------------------------------ #

    async def connect(self, timeout: float | None = None) -> websockets.WebSocketClientProtocol:
        """Establish a websocket connection."""
        if self.connected:
            return self._ws_cm  # type: ignore[return-value]

        self._ws_manager = websockets.connect(self.websocket_url)
        try:
            self._ws_cm = await asyncio.wait_for(self._ws_manager.__aenter__(), timeout=timeout)
        except BaseException:
            self._ws_manager = None
            self._ws_cm = None
            self.connected = False
            raise

        self.connected = True
        return self._ws_cm  # type: ignore[return-value]

    async def disconnect(self) -> None:
        """Close the websocket connection if open."""
        if self._ws_manager is not None and self._ws_cm is not None:
            await self._ws_manager.__aexit__(None, None, None)
        self._ws_manager = None
        self._ws_cm = None
        self.connected = False

    async def connect_with_retry(
        self,
        max_retries: int = 3,
        timeout: float | None = None,
    ) -> websockets.WebSocketClientProtocol:
        """Try multiple times to connect before giving up."""
        attempt = 0
        last_error: Optional[BaseException] = None

        while attempt < max_retries:
            try:
                return await self.connect(timeout=timeout)
            except BaseException as exc:
                last_error = exc
                attempt += 1
                await asyncio.sleep(self.reconnect_delay)

        if last_error is not None:
            raise last_error
        raise ConnectionError("Unable to establish websocket connection")

    # ------------------------------------------------------------------ #
    # Subscription helpers
    # ------------------------------------------------------------------ #

    def subscribe(self, symbol: str) -> None:
        """Subscribe to a symbol if not already tracked."""
        if symbol not in self.subscriptions:
            self.subscriptions.append(symbol)
        if symbol not in self.symbols:
            self.symbols.append(symbol)

    def subscribe_batch(self, symbols: Iterable[str]) -> None:
        for symbol in symbols:
            self.subscribe(symbol)

    def unsubscribe(self, symbol: str) -> None:
        """Remove a symbol from the active subscription list."""
        if symbol in self.subscriptions:
            self.subscriptions.remove(symbol)

    # ------------------------------------------------------------------ #
    # Message handling
    # ------------------------------------------------------------------ #

    def parse_message(self, raw_message: str) -> Optional[Dict[str, Any]]:
        """Parse a JSON encoded websocket message."""
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            return None

        if not isinstance(message, dict):
            return None

        message.setdefault("timestamp", datetime.utcnow().isoformat())
        return message

    def process_message(self, message: Dict[str, Any] | str) -> Optional[Dict[str, Any]]:
        """Handle an incoming message, respecting rate limits."""
        if self.is_rate_limited:
            return {"rate_limited": True}

        parsed = message if isinstance(message, dict) else self.parse_message(message)
        if not parsed:
            return None

        symbol = parsed.get("symbol")
        price = parsed.get("price")
        if symbol and isinstance(price, (int, float)):
            self.cache_price(symbol, float(price))
            self.last_price = float(price)

        return parsed

    def enqueue_message(self, message: Dict[str, Any]) -> None:
        """Store messages in a bounded queue."""
        self.queue.append(message)

    # ------------------------------------------------------------------ #
    # Validation helpers
    # ------------------------------------------------------------------ #

    def validate_tick(self, tick: Dict[str, Any]) -> bool:
        """Validate a trade/ticker snapshot."""
        price = tick.get("price")
        volume = tick.get("volume", 0)
        if price is None or not isinstance(price, (int, float)):
            return False
        if price <= 0:
            return False
        if volume is not None and isinstance(volume, (int, float)) and volume <= 0:
            return False
        return True

    def is_stale(self, tick: Dict[str, Any], max_age_seconds: int) -> bool:
        """Determine whether a tick is older than the configured threshold."""
        timestamp = tick.get("timestamp")
        if not isinstance(timestamp, datetime):
            return True
        age = datetime.now(tz=timestamp.tzinfo) - timestamp
        return age.total_seconds() > max_age_seconds

    def is_spike(self, tick: Dict[str, Any], threshold: float = 0.05) -> bool:
        """Detect abnormal price moves compared to the last cached price."""
        price = tick.get("price")
        if self.last_price is None or price is None:
            return False
        try:
            change = abs(float(price) - float(self.last_price)) / float(self.last_price)
        except ZeroDivisionError:
            return False
        return change >= threshold

    # ------------------------------------------------------------------ #
    # Price caching
    # ------------------------------------------------------------------ #

    def cache_price(self, symbol: str, price: float, ttl: Optional[float] = None) -> None:
        """Cache the latest price with an optional TTL."""
        self.price_cache[symbol] = price
        if ttl is not None:
            self.price_cache_expiry[symbol] = datetime.utcnow() + timedelta(seconds=ttl)
        elif symbol in self.price_cache_expiry:
            del self.price_cache_expiry[symbol]

    def get_cached_price(self, symbol: str) -> Optional[float]:
        """Retrieve a cached price if it has not expired."""
        expiry = self.price_cache_expiry.get(symbol)
        if expiry and expiry < datetime.utcnow():
            self.price_cache.pop(symbol, None)
            self.price_cache_expiry.pop(symbol, None)
            return None
        return self.price_cache.get(symbol)

    def clear_cache(self) -> None:
        self.price_cache.clear()
        self.price_cache_expiry.clear()

    # ------------------------------------------------------------------ #
    # OHLCV aggregation
    # ------------------------------------------------------------------ #

    def aggregate_candle(self, ticks: List[Dict[str, Any]], timeframe: str = "1m") -> OHLCVBar:
        """Aggregate raw ticks into an OHLCV candle."""
        if not ticks:
            raise ValueError("Cannot aggregate an empty tick list")

        sorted_ticks = sorted(ticks, key=lambda t: t["timestamp"])
        prices = [float(t["price"]) for t in sorted_ticks]
        volumes = [float(t.get("volume", 0) or 0.0) for t in sorted_ticks]

        symbol = sorted_ticks[0].get("symbol") or (self.symbols[0] if self.symbols else "UNKNOWN")
        start = sorted_ticks[0]["timestamp"]
        end = start + self._timeframe_to_delta(timeframe)

        bar = OHLCVBar(
            symbol=symbol,
            timeframe=timeframe,
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=round(sum(volumes), 6),
            start=start,
            end=end,
        )

        self._pending_candle = bar
        self._pending_ticks = sorted_ticks
        self.last_price = bar.close
        return bar

    def finalize_current_candle(self) -> None:
        """Trigger the candle completion callback."""
        if self.on_candle_complete and self._pending_candle:
            self.on_candle_complete(self._pending_candle)
        self._pending_candle = None
        self._pending_ticks = []

    @staticmethod
    def _timeframe_to_delta(timeframe: str) -> timedelta:
        """Convert timeframe strings (e.g. '1m', '5m') to timedeltas."""
        try:
            value = int(timeframe[:-1])
            unit = timeframe[-1]
        except (ValueError, IndexError):
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        if unit == "s":
            return timedelta(seconds=value)
        if unit == "m":
            return timedelta(minutes=value)
        if unit == "h":
            return timedelta(hours=value)
        if unit == "d":
            return timedelta(days=value)
        raise ValueError(f"Unsupported timeframe unit: {unit}")


__all__ = ["MarketDataIngestor", "OHLCVBar"]
