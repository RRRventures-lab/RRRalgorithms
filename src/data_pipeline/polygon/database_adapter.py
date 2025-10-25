"""
Database adapter for Polygon.io data pipeline.
Adapts the PolygonWebSocketClient to work with our SQLite database.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class PolygonDatabaseAdapter:
    """
    Adapter class to bridge Polygon WebSocket client with our SQLite database.

    The WebSocket client expects methods like:
    - insert_crypto_trade()
    - insert_crypto_quote()
    - insert_crypto_aggregate()

    This adapter translates those calls to our database schema.
    """

    def __init__(self, db_client):
        """
        Initialize adapter with database client.

        Args:
            db_client: DatabaseClient instance (SQLiteClient or compatible)
        """
        self.db = db_client
        logger.info("Polygon database adapter initialized")

    def insert_crypto_trade(self, trade_data: Dict[str, Any]):
        """
        Insert crypto trade data into trades_data table.

        Polygon format:
            {
                "ticker": "X:BTCUSD",
                "event_time": "2024-01-01T12:00:00",
                "price": 50000.0,
                "size": 0.5,
                "exchange_id": 1,
                "conditions": [1, 12],
                "trade_id": "12345"
            }

        Maps to trades_data table:
            - symbol TEXT
            - timestamp INTEGER
            - price REAL
            - size REAL
            - side TEXT
            - exchange_id INTEGER
            - trade_id TEXT
            - conditions TEXT (JSON)
        """
        try:
            # Extract ticker and convert to our symbol format
            ticker = trade_data.get("ticker", "")
            symbol = self._convert_ticker_to_symbol(ticker)

            # Parse event_time to Unix timestamp
            event_time_str = trade_data.get("event_time", "")
            if isinstance(event_time_str, str):
                event_time = datetime.fromisoformat(event_time_str.replace('Z', '+00:00'))
                timestamp = int(event_time.timestamp())
            else:
                timestamp = int(datetime.utcnow().timestamp())

            # Prepare data for database
            db_row = {
                "symbol": symbol,
                "timestamp": timestamp,
                "price": float(trade_data.get("price", 0)),
                "size": float(trade_data.get("size", 0)),
                "side": "unknown",  # Polygon doesn't provide side for crypto trades
                "exchange_id": trade_data.get("exchange_id"),
                "trade_id": trade_data.get("trade_id"),
                "conditions": str(trade_data.get("conditions", []))  # Convert to JSON string
            }

            # Insert into database (sync wrapper for async)
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            loop.run_until_complete(self.db.insert("trades_data", db_row))
            logger.debug(f"Inserted trade: {symbol} @ ${db_row['price']}")

        except Exception as e:
            logger.error(f"Error inserting crypto trade: {e}")
            raise

    def insert_crypto_quote(self, quote_data: Dict[str, Any]):
        """
        Insert crypto quote (bid/ask) data into quotes table.

        Polygon format:
            {
                "ticker": "X:BTCUSD",
                "event_time": "2024-01-01T12:00:00",
                "bid_price": 49990.0,
                "bid_size": 1.2,
                "ask_price": 50010.0,
                "ask_size": 0.8,
                "spread": 20.0,
                "exchange_id": 1
            }

        Maps to quotes table:
            - symbol TEXT
            - timestamp INTEGER
            - bid_price REAL
            - bid_size REAL
            - ask_price REAL
            - ask_size REAL
            - exchange_id INTEGER
        """
        try:
            # Extract ticker and convert to our symbol format
            ticker = quote_data.get("ticker", "")
            symbol = self._convert_ticker_to_symbol(ticker)

            # Parse event_time to Unix timestamp
            event_time_str = quote_data.get("event_time", "")
            if isinstance(event_time_str, str):
                event_time = datetime.fromisoformat(event_time_str.replace('Z', '+00:00'))
                timestamp = int(event_time.timestamp())
            else:
                timestamp = int(datetime.utcnow().timestamp())

            # Prepare data for database
            db_row = {
                "symbol": symbol,
                "timestamp": timestamp,
                "bid_price": float(quote_data.get("bid_price", 0)),
                "bid_size": float(quote_data.get("bid_size", 0)),
                "ask_price": float(quote_data.get("ask_price", 0)),
                "ask_size": float(quote_data.get("ask_size", 0)),
                "exchange_id": quote_data.get("exchange_id")
            }

            # Insert into database (sync wrapper for async)
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            loop.run_until_complete(self.db.insert("quotes", db_row))
            logger.debug(f"Inserted quote: {symbol} bid=${db_row['bid_price']} ask=${db_row['ask_price']}")

        except Exception as e:
            logger.error(f"Error inserting crypto quote: {e}")
            raise

    def insert_crypto_aggregate(self, agg_data: Dict[str, Any]):
        """
        Insert crypto aggregate (OHLCV bar) data into market_data table.

        Polygon format:
            {
                "ticker": "X:BTCUSD",
                "event_time": "2024-01-01T12:00:00",
                "open": 49000.0,
                "high": 50500.0,
                "low": 48800.0,
                "close": 50000.0,
                "volume": 125.5,
                "vwap": 49950.0,
                "trade_count": 1234
            }

        Maps to market_data table:
            - symbol TEXT
            - timestamp INTEGER
            - open REAL
            - high REAL
            - low REAL
            - close REAL
            - volume REAL
            - vwap REAL
            - trade_count INTEGER
        """
        try:
            # Extract ticker and convert to our symbol format
            ticker = agg_data.get("ticker", "")
            symbol = self._convert_ticker_to_symbol(ticker)

            # Parse event_time to Unix timestamp
            event_time_str = agg_data.get("event_time", "")
            if isinstance(event_time_str, str):
                event_time = datetime.fromisoformat(event_time_str.replace('Z', '+00:00'))
                timestamp = int(event_time.timestamp())
            else:
                timestamp = int(datetime.utcnow().timestamp())

            # Prepare data for database
            db_row = {
                "symbol": symbol,
                "timestamp": timestamp,
                "open": float(agg_data.get("open", 0)),
                "high": float(agg_data.get("high", 0)),
                "low": float(agg_data.get("low", 0)),
                "close": float(agg_data.get("close", 0)),
                "volume": float(agg_data.get("volume", 0)),
                "vwap": agg_data.get("vwap"),  # Optional
                "trade_count": agg_data.get("trade_count")  # Optional
            }

            # Insert into database (sync wrapper for async)
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            loop.run_until_complete(self.db.insert("market_data", db_row))
            logger.debug(f"Inserted aggregate: {symbol} close=${db_row['close']} vol={db_row['volume']}")

        except Exception as e:
            logger.error(f"Error inserting crypto aggregate: {e}")
            raise

    @staticmethod
    def _convert_ticker_to_symbol(ticker: str) -> str:
        """
        Convert Polygon ticker format to our symbol format.

        Polygon: "X:BTCUSD"
        Ours: "BTC-USD"

        Args:
            ticker: Polygon ticker (e.g., "X:BTCUSD")

        Returns:
            Our symbol format (e.g., "BTC-USD")
        """
        if not ticker:
            return ""

        # Remove "X:" prefix if present
        if ticker.startswith("X:"):
            ticker = ticker[2:]

        # Extract base and quote currency
        # Most crypto tickers are like "BTCUSD", "ETHUSD"
        # We need to convert to "BTC-USD", "ETH-USD"

        # Common quote currencies
        quote_currencies = ["USD", "USDT", "USDC", "EUR", "GBP", "JPY"]

        for quote in quote_currencies:
            if ticker.endswith(quote):
                base = ticker[:-len(quote)]
                return f"{base}-{quote}"

        # If no match, return as-is with dash in middle
        if len(ticker) >= 6:
            mid = len(ticker) // 2
            return f"{ticker[:mid]}-{ticker[mid:]}"

        return ticker
