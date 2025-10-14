from .models import (
from datetime import datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from functools import wraps
from typing import Optional, List, Dict, Any
import hashlib
import json
import logging
import os
import requests
import time


"""
Polygon.io REST API client with rate limiting, caching, and error handling.
"""


    Aggregate,
    Trade,
    Quote,
    TickerDetails,
    MarketStatus,
    AggregatesResponse,
    LastTradeResponse,
    LastQuoteResponse,
)

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, requests_per_second: int = 5):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0

    def wait_if_needed(self):
        """Wait if we're exceeding rate limit."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()


class SimpleCache:
    """Simple in-memory cache with TTL."""

    def __init__(self, ttl_seconds: int = 300):
        self.cache: Dict[str, tuple[Any, float]] = {}
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: Any):
        """Set value in cache with current timestamp."""
        self.cache[key] = (value, time.time())

    def clear(self):
        """Clear all cache."""
        self.cache.clear()


def with_retry(max_retries: int = 3, backoff_factor: float = 2.0):
    """Decorator for automatic retry with exponential backoff."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        sleep_time = backoff_factor ** attempt
                        logger.warning(
                            f"Request failed (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {sleep_time}s..."
                        )
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"Request failed after {max_retries} attempts: {e}")

            raise last_exception

        return wrapper

    return decorator


class PolygonRESTClient:
    """
    Polygon.io REST API client.

    Features:
    - Rate limiting (respects free tier: 5 req/sec)
    - Response caching (Redis or in-memory)
    - Automatic retry with exponential backoff
    - Comprehensive error handling
    - Request logging and metrics
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit: int = 5,  # requests per second
        cache_ttl: int = 300,  # 5 minutes
        enable_cache: bool = True,
    ):
        """
        Initialize Polygon REST client.

        Args:
            api_key: Polygon API key (or set POLYGON_API_KEY env var)
            rate_limit: Maximum requests per second
            cache_ttl: Cache TTL in seconds
            enable_cache: Whether to enable caching
        """
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Polygon API key required. Set POLYGON_API_KEY env var or pass api_key parameter."
            )

        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

        self.rate_limiter = RateLimiter(requests_per_second=rate_limit)
        self.cache = SimpleCache(ttl_seconds=cache_ttl) if enable_cache else None

        self.request_count = 0
        self.error_count = 0

        logger.info(f"Polygon REST client initialized (rate limit: {rate_limit} req/s)")

    def _make_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key from endpoint and params."""
        key_str = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    @with_retry(max_retries=3)
    def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Make API request with rate limiting, caching, and retry logic.

        Args:
            endpoint: API endpoint (e.g., "/v2/aggs/ticker/X:BTCUSD/range/1/hour/...")
            params: Query parameters
            use_cache: Whether to use cache for this request

        Returns:
            JSON response as dict

        Raises:
            requests.exceptions.RequestException: On request failure
        """
        params = params or {}
        url = f"{self.BASE_URL}{endpoint}"

        # Check cache
        if use_cache and self.cache:
            cache_key = self._make_cache_key(endpoint, params)
            cached_response = self.cache.get(cache_key)
            if cached_response is not None:
                logger.debug(f"Cache hit for {endpoint}")
                return cached_response

        # Rate limiting
        self.rate_limiter.wait_if_needed()

        # Make request
        logger.debug(f"GET {url} with params {params}")
        self.request_count += 1

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Cache successful response
            if use_cache and self.cache and response.status_code == 200:
                cache_key = self._make_cache_key(endpoint, params)
                self.cache.set(cache_key, data)

            return data

        except requests.exceptions.HTTPError as e:
            self.error_count += 1
            logger.error(f"HTTP error for {url}: {e}")
            if e.response.status_code == 429:
                logger.warning("Rate limit exceeded. Consider upgrading your plan.")
            raise
        except requests.exceptions.RequestException as e:
            self.error_count += 1
            logger.error(f"Request error for {url}: {e}")
            raise

    # ========================================================================
    # AGGREGATES (OHLCV BARS)
    # ========================================================================

    @lru_cache(maxsize=128)

    def get_aggregates(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,  # "minute", "hour", "day", "week", "month"
        from_date: str,  # YYYY-MM-DD
        to_date: str,  # YYYY-MM-DD
        limit: int = 50000,
        adjusted: bool = True,
    ) -> List[Aggregate]:
        """
        Get aggregate bars (OHLCV) for a ticker.

        Example:
            # Get 1-minute bars for BTC
            bars = client.get_aggregates(
                ticker="X:BTCUSD",
                multiplier=1,
                timespan="minute",
                from_date="2024-01-01",
                to_date="2024-01-02"
            )

        Args:
            ticker: Ticker symbol (e.g., "X:BTCUSD" for crypto)
            multiplier: Size of timespan multiplier (e.g., 1, 5, 15)
            timespan: Size of time window (minute, hour, day, week, month)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            limit: Max number of results (default 50000, API limit)
            adjusted: Whether to adjust for splits (stocks only)

        Returns:
            List of Aggregate objects
        """
        endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": limit}

        data = self._request(endpoint, params)

        if data.get("status") != "OK":
            logger.warning(f"Aggregates request returned status: {data.get('status')}")
            return []

        # Add ticker to response and each result before parsing
        data["ticker"] = ticker
        if "results" in data:
            for result in data["results"]:
                result["ticker"] = ticker

        response = AggregatesResponse(**data)
        return response.results

    # ========================================================================
    # TRADES
    # ========================================================================

    @lru_cache(maxsize=128)

    def get_trades(
        self,
        ticker: str,
        timestamp: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Trade]:
        """
        Get individual trades for a ticker.

        Args:
            ticker: Ticker symbol
            timestamp: Timestamp for trades (default: now)
            limit: Max number of trades

        Returns:
            List of Trade objects
        """
        timestamp = timestamp or datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d")

        endpoint = f"/v3/trades/{ticker}"
        params = {"timestamp": timestamp_str, "limit": limit}

        data = self._request(endpoint, params)
        return [Trade(**trade) for trade in data.get("results", [])]

    @lru_cache(maxsize=128)

    def get_last_trade(self, ticker: str) -> Trade:
        """
        Get the most recent trade for a ticker.

        Example:
            trade = client.get_last_trade("X:BTCUSD")
            print(f"Last BTC trade: ${trade.price} at {trade.datetime}")

        Args:
            ticker: Ticker symbol

        Returns:
            Trade object
        """
        endpoint = f"/v2/last/trade/{ticker}"
        data = self._request(endpoint, use_cache=False)  # Don't cache last price

        response = LastTradeResponse(**data)
        response.last.ticker = ticker  # Add ticker to trade object
        return response.last

    # ========================================================================
    # QUOTES
    # ========================================================================

    @lru_cache(maxsize=128)

    def get_last_quote(self, ticker: str) -> Quote:
        """
        Get the most recent quote (bid/ask) for a ticker.

        Example:
            quote = client.get_last_quote("X:BTCUSD")
            print(f"BTC bid: ${quote.bid_price}, ask: ${quote.ask_price}")
            print(f"Spread: ${quote.spread}")

        Args:
            ticker: Ticker symbol

        Returns:
            Quote object
        """
        endpoint = f"/v2/last/nbbo/{ticker}"
        data = self._request(endpoint, use_cache=False)

        response = LastQuoteResponse(**data)
        response.last.ticker = ticker
        return response.last

    # ========================================================================
    # REFERENCE DATA
    # ========================================================================

    @lru_cache(maxsize=128)

    def get_ticker_details(self, ticker: str) -> TickerDetails:
        """
        Get details about a ticker/symbol.

        Args:
            ticker: Ticker symbol

        Returns:
            TickerDetails object
        """
        endpoint = f"/v3/reference/tickers/{ticker}"
        data = self._request(endpoint)

        return TickerDetails(**data.get("results", {}))

    def list_crypto_tickers(self, active: bool = True) -> List[TickerDetails]:
        """
        List all available cryptocurrency tickers.

        Args:
            active: Only return active tickers

        Returns:
            List of TickerDetails
        """
        endpoint = "/v3/reference/tickers"
        params = {"market": "crypto", "active": str(active).lower(), "limit": 1000}

        data = self._request(endpoint, params)
        return [TickerDetails(**t) for t in data.get("results", [])]

    @lru_cache(maxsize=128)

    def get_market_status(self) -> MarketStatus:
        """
        Get current market status (open/closed).

        Returns:
            MarketStatus object
        """
        endpoint = "/v1/marketstatus/now"
        data = self._request(endpoint, use_cache=False)

        return MarketStatus(**data)

    # ========================================================================
    # CONVENIENCE METHODS
    # ========================================================================

    @lru_cache(maxsize=128)

    def get_latest_price(self, ticker: str) -> Decimal:
        """
        Get the latest price for a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Latest trade price
        """
        trade = self.get_last_trade(ticker)
        return trade.price

    @lru_cache(maxsize=128)

    def get_daily_bars(
        self, ticker: str, days_back: int = 30
    ) -> List[Aggregate]:
        """
        Get daily bars for the last N days.

        Args:
            ticker: Ticker symbol
            days_back: Number of days to look back

        Returns:
            List of daily Aggregate objects
        """
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)

        return self.get_aggregates(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_date=from_date.strftime("%Y-%m-%d"),
            to_date=to_date.strftime("%Y-%m-%d"),
        )

    # ========================================================================
    # STATS
    # ========================================================================

    @lru_cache(maxsize=128)

    def get_stats(self) -> Dict[str, int]:
        """Get client usage statistics."""
        return {
            "total_requests": self.request_count,
            "errors": self.error_count,
            "success_rate": (
                (self.request_count - self.error_count) / self.request_count
                if self.request_count > 0
                else 0.0
            ),
        }

    def clear_cache(self):
        """Clear response cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Cache cleared")
