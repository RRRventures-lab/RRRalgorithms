from config import config
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Any
import aiohttp
import asyncio
import json
import numpy as np
import pandas as pd
import sqlite3
import time


"""
Professional data collectors using real APIs: Polygon.io, SQLite, Perplexity AI.

This module provides production-grade data collection with:
- Polygon.io for real-time and historical crypto/stock data
- Local SQLite for caching and storage
- Perplexity AI for sentiment and news analysis
- Proper rate limiting and error handling
"""




class LocalDatabase:
    """SQLite database for caching hypothesis testing data."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize local database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or config.get_local_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_tables()

    def _init_tables(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # OHLCV data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                source TEXT,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                UNIQUE(symbol, timestamp, source)
            )
        """)

        # Sentiment data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                sentiment_score REAL,
                sentiment_text TEXT,
                confidence REAL,
                source TEXT,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                UNIQUE(symbol, timestamp, source)
            )
        """)

        # Hypothesis results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hypothesis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hypothesis_id TEXT NOT NULL,
                test_date INTEGER NOT NULL,
                decision TEXT,
                confidence REAL,
                sharpe_ratio REAL,
                win_rate REAL,
                total_return REAL,
                max_drawdown REAL,
                total_trades INTEGER,
                p_value REAL,
                execution_time REAL,
                results_json TEXT,
                created_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timestamp ON ohlcv_data(symbol, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_timestamp ON sentiment_data(symbol, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hypothesis_id ON hypothesis_results(hypothesis_id)")

        conn.commit()
        conn.close()

        print(f"[LocalDB] Initialized database at {self.db_path}")

    def store_ohlcv(
        self,
        symbol: str,
        data: pd.DataFrame,
        source: str = "polygon"
    ) -> int:
        """
        Store OHLCV data in database.

        Args:
            symbol: Trading symbol
            data: DataFrame with columns [timestamp, open, high, low, close, volume]
            source: Data source name

        Returns:
            Number of rows inserted
        """
        conn = sqlite3.connect(str(self.db_path))

        # Prepare data for insertion
        data = data.copy()
        data['symbol'] = symbol
        data['source'] = source

        # Convert timestamp to unix timestamp
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp']).astype(int) // 10**9

        # Insert with conflict resolution
        rows_inserted = 0
        for _, row in data.iterrows():
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO ohlcv_data
                    (symbol, timestamp, open, high, low, close, volume, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['symbol'],
                    row['timestamp'],
                    row.get('open'),
                    row.get('high'),
                    row.get('low'),
                    row.get('close'),
                    row.get('volume'),
                    row['source']
                ))
                rows_inserted += 1
            except Exception as e:
                print(f"[LocalDB] Error inserting row: {e}")

        conn.commit()
        conn.close()

        return rows_inserted

    @lru_cache(maxsize=128)

    def get_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        source: str = "polygon"
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve OHLCV data from database.

        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            source: Data source name

        Returns:
            DataFrame with OHLCV data or None
        """
        conn = sqlite3.connect(str(self.db_path))

        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())

        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_data
            WHERE symbol = ? AND source = ?
            AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
        """

        df = pd.read_sql_query(query, conn, params=(symbol, source, start_ts, end_ts))
        conn.close()

        if df.empty:
            return None

        # Convert timestamp back to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        return df

    def store_hypothesis_result(self, result: Dict[str, Any]) -> int:
        """
        Store hypothesis test result.

        Args:
            result: Dictionary with test results

        Returns:
            Row ID
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO hypothesis_results
            (hypothesis_id, test_date, decision, confidence, sharpe_ratio, win_rate,
             total_return, max_drawdown, total_trades, p_value, execution_time, results_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.get('hypothesis_id'),
            int(time.time()),
            result.get('decision'),
            result.get('confidence'),
            result.get('sharpe_ratio'),
            result.get('win_rate'),
            result.get('total_return'),
            result.get('max_drawdown'),
            result.get('total_trades'),
            result.get('p_value'),
            result.get('execution_time'),
            json.dumps(result)
        ))

        row_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return row_id


class PolygonDataCollector:
    """
    Collect data from Polygon.io API (Currencies Starter plan).

    Features:
    - Real-time crypto prices
    - Historical aggregates
    - High-frequency data (100 req/sec)
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: Optional[str] = None, db: Optional[LocalDatabase] = None):
        """
        Initialize Polygon collector.

        Args:
            api_key: Polygon API key (default: from config)
            db: Local database for caching
        """
        self.api_key = api_key or config.polygon_api_key
        self.rate_limit = 1.0 / config.polygon_rate_limit  # Convert to delay between requests
        self.last_request_time = 0.0
        self.db = db or LocalDatabase()

        if not self.api_key:
            raise ValueError("Polygon API key not configured")

    async def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    async def get_crypto_aggregates(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timespan: str = "hour",  # minute, hour, day, week, month
        multiplier: int = 1,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get aggregated crypto bars.

        Args:
            symbol: Crypto ticker (e.g., "X:BTCUSD", "X:ETHUSD")
            start_date: Start date
            end_date: End date
            timespan: Bar timespan
            multiplier: Timespan multiplier
            use_cache: Use local database cache

        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        if use_cache:
            cached_data = self.db.get_ohlcv(symbol, start_date, end_date, source="polygon")
            if cached_data is not None and len(cached_data) > 0:
                print(f"[Polygon] Using cached data for {symbol}: {len(cached_data)} bars")
                return cached_data

        # Fetch from API
        print(f"[Polygon] Fetching {symbol} from {start_date.date()} to {end_date.date()}...")

        # Format dates for Polygon API
        from_date = start_date.strftime("%Y-%m-%d")
        to_date = end_date.strftime("%Y-%m-%d")

        url = f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"

        params = {
            "apiKey": self.api_key,
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000  # Max limit
        }

        all_results = []

        async with aiohttp.ClientSession() as session:
            await self._rate_limit()

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    if data.get("status") == "OK" and "results" in data:
                        all_results = data["results"]
                        print(f"[Polygon] Fetched {len(all_results)} bars")
                    else:
                        print(f"[Polygon] No data returned: {data.get('status')}")
                        return pd.DataFrame()
                else:
                    error_text = await response.text()
                    print(f"[Polygon] Error {response.status}: {error_text}")
                    return pd.DataFrame()

        if not all_results:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_results)

        # Rename columns to standard format
        df = df.rename(columns={
            't': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vw': 'vwap',
            'n': 'transactions'
        })

        # Convert timestamp from milliseconds to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Store in cache
        if use_cache:
            rows = self.db.store_ohlcv(symbol, df, source="polygon")
            print(f"[Polygon] Cached {rows} bars to database")

        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]


class PerplexityCollector:
    """
    Collect sentiment and news data using Perplexity AI.

    Features:
    - Real-time news search
    - Sentiment analysis
    - Market intelligence
    """

    BASE_URL = "https://api.perplexity.ai"

    def __init__(self, api_key: Optional[str] = None, db: Optional[LocalDatabase] = None):
        """
        Initialize Perplexity collector.

        Args:
            api_key: Perplexity API key (default: from config)
            db: Local database for caching
        """
        self.api_key = api_key or config.perplexity_api_key
        self.db = db or LocalDatabase()

        if not self.api_key:
            raise ValueError("Perplexity API key not configured")

    async def get_market_sentiment(
        self,
        symbol: str,
        date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get market sentiment for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTC", "ETH")
            date: Date for sentiment (default: today)

        Returns:
            Dict with sentiment data
        """
        if date is None:
            date = datetime.now()

        # Create search query
        query = f"Latest market sentiment and news for {symbol} cryptocurrency on {date.strftime('%Y-%m-%d')}. Include price predictions and analyst opinions."

        url = f"{self.BASE_URL}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "sonar",  # Updated: New Sonar model (March 2025) - lightweight grounded search
            "messages": [
                {
                    "role": "system",
                    "content": "You are a financial analyst providing concise market sentiment analysis. Respond with sentiment (bullish/bearish/neutral) and a confidence score (0-1)."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "max_tokens": 500,
            "temperature": 0.2,
            "top_p": 0.9,
            "return_citations": True,
            "return_images": False,
            "search_recency_filter": "day"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()

                    # Extract sentiment from response
                    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    citations = data.get("citations", [])

                    # Enhanced sentiment scoring
                    sentiment_score = self._score_sentiment(text)
                    confidence = self._calculate_confidence(text, citations)

                    return {
                        "symbol": symbol,
                        "timestamp": date,
                        "sentiment_score": sentiment_score,
                        "sentiment_text": text,
                        "citations": citations,
                        "confidence": confidence
                    }
                else:
                    error_text = await response.text()
                    print(f"[Perplexity] Error {response.status}: {error_text}")
                    return {
                        "symbol": symbol,
                        "timestamp": date,
                        "sentiment_score": 0.0,
                        "sentiment_text": f"Error: {error_text}",
                        "confidence": 0.0
                    }

    def _score_sentiment(self, text: str) -> float:
        """
        Enhanced sentiment scoring with weighted keywords and intensity modifiers.

        Args:
            text: Text to analyze

        Returns:
            Sentiment score (-1 to 1)
        """
        text_lower = text.lower()

        # Strong bullish indicators (weight: 3)
        strong_bullish = ['surge', 'soar', 'rally', 'breakout', 'explosion', 'skyrocket',
                         'all-time high', 'ath', 'moon', 'parabolic', 'massive gain']

        # Moderate bullish indicators (weight: 2)
        moderate_bullish = ['bullish', 'positive', 'rise', 'gain', 'increase', 'growth',
                           'momentum', 'uptrend', 'strength', 'accumulation', 'buying pressure']

        # Mild bullish indicators (weight: 1)
        mild_bullish = ['optimistic', 'hopeful', 'recovery', 'stabilize', 'support',
                       'bid', 'demand', 'adoption', 'upgrade', 'improvement']

        # Strong bearish indicators (weight: 3)
        strong_bearish = ['crash', 'collapse', 'plunge', 'plummet', 'capitulation',
                         'panic', 'bloodbath', 'massacre', 'dump', 'free fall']

        # Moderate bearish indicators (weight: 2)
        moderate_bearish = ['bearish', 'negative', 'drop', 'fall', 'decline', 'loss',
                           'downtrend', 'weakness', 'distribution', 'selling pressure']

        # Mild bearish indicators (weight: 1)
        mild_bearish = ['concern', 'worry', 'cautious', 'resistance', 'pressure',
                       'ask', 'supply', 'correction', 'pullback', 'uncertainty']

        # Intensity modifiers
        intensifiers = ['very', 'extremely', 'highly', 'significantly', 'massively',
                       'strongly', 'heavily', 'major', 'substantial']

        diminishers = ['slightly', 'somewhat', 'moderately', 'minor', 'small', 'little']

        # Negation words
        negations = ['not', 'no', 'never', 'neither', 'nor', 'none', 'hardly', 'barely']

        # Calculate weighted sentiment
        bullish_score = 0.0
        bearish_score = 0.0

        # Check for intensifiers/diminishers context
        has_intensifier = any(word in text_lower for word in intensifiers)
        has_diminisher = any(word in text_lower for word in diminishers)
        has_negation = any(word in text_lower for word in negations)

        # Apply intensity multiplier
        intensity_multiplier = 1.5 if has_intensifier else (0.5 if has_diminisher else 1.0)

        # Score bullish terms
        for word in strong_bullish:
            if word in text_lower:
                bullish_score += 3.0 * intensity_multiplier

        for word in moderate_bullish:
            if word in text_lower:
                bullish_score += 2.0 * intensity_multiplier

        for word in mild_bullish:
            if word in text_lower:
                bullish_score += 1.0 * intensity_multiplier

        # Score bearish terms
        for word in strong_bearish:
            if word in text_lower:
                bearish_score += 3.0 * intensity_multiplier

        for word in moderate_bearish:
            if word in text_lower:
                bearish_score += 2.0 * intensity_multiplier

        for word in mild_bearish:
            if word in text_lower:
                bearish_score += 1.0 * intensity_multiplier

        # Handle negation (flip sentiment if negation detected)
        if has_negation:
            bullish_score, bearish_score = bearish_score, bullish_score

        # Calculate net sentiment
        total_score = bullish_score + bearish_score
        if total_score == 0:
            return 0.0

        sentiment = (bullish_score - bearish_score) / total_score

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, sentiment))

    def _calculate_confidence(self, text: str, citations: list) -> float:
        """
        Calculate confidence score for sentiment analysis.

        Factors:
        - Text length (more content = higher confidence)
        - Number of citations (more sources = higher confidence)
        - Presence of definitive language
        - Absence of uncertainty markers

        Args:
            text: Analyzed text
            citations: List of source citations

        Returns:
            Confidence score (0 to 1)
        """
        confidence = 0.5  # Base confidence

        # Factor 1: Text length (more content = more confidence)
        text_length = len(text)
        if text_length > 500:
            confidence += 0.15
        elif text_length > 300:
            confidence += 0.10
        elif text_length > 100:
            confidence += 0.05

        # Factor 2: Number of citations (more sources = more confidence)
        citation_count = len(citations) if citations else 0
        if citation_count >= 5:
            confidence += 0.15
        elif citation_count >= 3:
            confidence += 0.10
        elif citation_count >= 1:
            confidence += 0.05

        # Factor 3: Definitive language (increases confidence)
        definitive_words = ['clear', 'obvious', 'definite', 'certain', 'confirmed',
                           'proven', 'established', 'demonstrated', 'shows', 'indicates']
        definitive_count = sum(1 for word in definitive_words if word in text.lower())
        confidence += min(0.10, definitive_count * 0.02)

        # Factor 4: Uncertainty markers (decreases confidence)
        uncertainty_words = ['maybe', 'perhaps', 'possibly', 'uncertain', 'unclear',
                            'might', 'could', 'may', 'potentially', 'unclear']
        uncertainty_count = sum(1 for word in uncertainty_words if word in text.lower())
        confidence -= min(0.15, uncertainty_count * 0.03)

        # Factor 5: Data and statistics (increases confidence)
        data_indicators = ['data', 'statistics', 'analysis', 'report', 'study',
                          'research', 'survey', 'poll', 'metric', 'indicator']
        data_count = sum(1 for word in data_indicators if word in text.lower())
        confidence += min(0.10, data_count * 0.02)

        # Clamp to [0.1, 0.95] (never 0 or 1)
        return max(0.1, min(0.95, confidence))


class CoinbaseDataCollector:
    """
    Collect real-time data from Coinbase Advanced Trade API.

    Features:
    - Order book snapshots (level 2)
    - Recent trades
    - 24h stats and tickers
    - Order book imbalance calculation
    - Bid-ask spread analysis
    """

    BASE_URL = "https://api.coinbase.com/api/v3/brokerage"

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, db: Optional[LocalDatabase] = None):
        """
        Initialize Coinbase collector.

        Args:
            api_key: Coinbase API key (default: from config)
            api_secret: Coinbase API secret (default: from config)
            db: Local database for caching
        """
        self.api_key = api_key or config.coinbase_api_key
        self.api_secret = api_secret or config.coinbase_api_secret
        self.db = db or LocalDatabase()

        if not self.api_key or not self.api_secret:
            print("[Coinbase] Warning: API credentials not configured")

    async def get_order_book(self, product_id: str = "BTC-USD", level: int = 2) -> Dict[str, Any]:
        """
        Get order book snapshot from Coinbase.

        Args:
            product_id: Trading pair (e.g., "BTC-USD", "ETH-USD")
            level: Depth level (2 = top 50 levels)

        Returns:
            Dict with bids, asks, and metadata
        """
        # Use public API endpoint (no auth needed for market data)
        url = f"https://api.exchange.coinbase.com/products/{product_id}/book?level={level}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    # Parse bids and asks
                    bids = [[float(price), float(size)] for price, size, _ in data.get('bids', [])]
                    asks = [[float(price), float(size)] for price, size, _ in data.get('asks', [])]

                    return {
                        'product_id': product_id,
                        'timestamp': datetime.now(),
                        'bids': bids,  # [[price, size], ...]
                        'asks': asks,
                        'sequence': data.get('sequence'),
                        'time': data.get('time')
                    }
                else:
                    error = await response.text()
                    print(f"[Coinbase] Order book error {response.status}: {error}")
                    return {'bids': [], 'asks': [], 'error': error}

    def calculate_order_book_imbalance(self, order_book: Dict[str, Any], depth_pct: float = 0.01) -> Dict[str, float]:
        """
        Calculate order book imbalance metrics.

        Args:
            order_book: Order book from get_order_book()
            depth_pct: Depth percentage (0.01 = within 1% of mid price)

        Returns:
            Dict with imbalance metrics
        """
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])

        if not bids or not asks:
            return {'imbalance_ratio': 0.5, 'bid_volume': 0, 'ask_volume': 0}

        # Calculate mid price
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2

        # Calculate depth within percentage
        min_price = mid_price * (1 - depth_pct)
        max_price = mid_price * (1 + depth_pct)

        # Sum volumes within range
        bid_volume = sum(size for price, size in bids if price >= min_price)
        ask_volume = sum(size for price, size in asks if price <= max_price)

        # Calculate imbalance (0 = all asks, 1 = all bids)
        total_volume = bid_volume + ask_volume
        imbalance_ratio = bid_volume / total_volume if total_volume > 0 else 0.5

        # Calculate spread
        spread_bps = ((best_ask - best_bid) / mid_price) * 10000  # basis points

        return {
            'imbalance_ratio': imbalance_ratio,  # 0-1, >0.5 = bullish
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'total_volume': total_volume,
            'spread_bps': spread_bps,
            'mid_price': mid_price,
            'best_bid': best_bid,
            'best_ask': best_ask
        }

    async def get_recent_trades(self, product_id: str = "BTC-USD", limit: int = 100) -> List[Dict]:
        """
        Get recent trades from Coinbase.

        Args:
            product_id: Trading pair
            limit: Number of trades to fetch

        Returns:
            List of trade dicts
        """
        url = f"https://api.exchange.coinbase.com/products/{product_id}/trades?limit={limit}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    trades = await response.json()
                    return [{
                        'time': t.get('time'),
                        'trade_id': t.get('trade_id'),
                        'price': float(t.get('price', 0)),
                        'size': float(t.get('size', 0)),
                        'side': t.get('side')
                    } for t in trades]
                else:
                    print(f"[Coinbase] Trades error: {response.status}")
                    return []

    async def get_24h_stats(self, product_id: str = "BTC-USD") -> Dict[str, float]:
        """
        Get 24-hour statistics.

        Args:
            product_id: Trading pair

        Returns:
            Dict with 24h stats
        """
        url = f"https://api.exchange.coinbase.com/products/{product_id}/stats"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    stats = await response.json()
                    return {
                        'open': float(stats.get('open', 0)),
                        'high': float(stats.get('high', 0)),
                        'low': float(stats.get('low', 0)),
                        'last': float(stats.get('last', 0)),
                        'volume': float(stats.get('volume', 0)),
                        'volume_30day': float(stats.get('volume_30day', 0))
                    }
                else:
                    return {}


class ProfessionalDataCollector:
    """Unified interface for all professional data collectors."""

    def __init__(self):
        """Initialize all collectors."""
        self.db = LocalDatabase()
        self.polygon = PolygonDataCollector(db=self.db)
        self.coinbase = CoinbaseDataCollector(db=self.db)
        self.perplexity = PerplexityCollector(db=self.db)

        print("[Professional] Initialized all data collectors")

    async def collect_crypto_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        include_sentiment: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect comprehensive crypto data.

        Args:
            symbol: Crypto symbol (e.g., "BTC", "ETH")
            start_date: Start date
            end_date: End date
            include_sentiment: Include sentiment data

        Returns:
            Dict with 'price' and optionally 'sentiment' DataFrames
        """
        results = {}

        # Convert symbol to Polygon format
        polygon_symbol = f"X:{symbol}USD"

        # Collect price data
        print(f"[Professional] Collecting price data for {symbol}...")
        price_data = await self.polygon.get_crypto_aggregates(
            polygon_symbol,
            start_date,
            end_date,
            timespan="hour",
            multiplier=1,
            use_cache=True
        )

        results['price'] = price_data

        # Collect sentiment data if requested
        if include_sentiment and config.enable_sentiment:
            print(f"[Professional] Collecting sentiment data for {symbol}...")
            # Sample sentiment at daily intervals
            sentiment_data = []
            current_date = start_date

            while current_date <= end_date:
                sentiment = await self.perplexity.get_market_sentiment(symbol, current_date)
                sentiment_data.append(sentiment)
                current_date += timedelta(days=7)  # Weekly sentiment

            if sentiment_data:
                results['sentiment'] = pd.DataFrame(sentiment_data)

        return results

    @lru_cache(maxsize=128)

    def get_database(self) -> LocalDatabase:
        """Get local database instance."""
        return self.db


# Global collector instance
professional_collector = ProfessionalDataCollector()


if __name__ == "__main__":
    # Test professional data collectors
    async def test():
        print("=" * 70)
        print("Testing Professional Data Collectors")
        print("=" * 70)

        # Test Polygon
        print("\n1. Testing Polygon.io...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        data = await professional_collector.collect_crypto_data(
            "BTC",
            start_date,
            end_date,
            include_sentiment=False
        )

        if not data['price'].empty:
            print(f"✅ Collected {len(data['price'])} BTC price bars")
            print(data['price'].head())
        else:
            print("❌ No price data collected")

        # Test Perplexity (commented out to save API calls)
        # print("\n2. Testing Perplexity AI...")
        # sentiment = await professional_collector.perplexity.get_market_sentiment("BTC")
        # print(f"Sentiment Score: {sentiment['sentiment_score']:.2f}")
        # print(f"Analysis: {sentiment['sentiment_text'][:200]}...")

        print("\n" + "=" * 70)

    asyncio.run(test())
