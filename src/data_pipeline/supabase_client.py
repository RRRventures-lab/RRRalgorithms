from datetime import datetime, timedelta
from dotenv import load_dotenv
from functools import lru_cache
from typing import Dict, List, Optional, Callable, Any
import logging
import os


"""
Supabase Client for RRRalgorithms Trading System
================================================

This module provides a high-level interface to interact with Supabase database.

Features:
- Type-safe data insertion
- Real-time subscriptions
- Query helpers
- Error handling
- Connection pooling

Usage:
    from src.database import SQLiteClient as DatabaseClient

    client = get_db()

    # Insert OHLCV data
    client.insert_crypto_aggregate({
        "ticker": "X:BTCUSD",
        "event_time": "2025-10-11T14:00:00Z",
        "open": 67000,
        "high": 67500,
        "low": 66800,
        "close": 67200,
        "volume": 150.5
    })

    # Query latest prices
    prices = client.get_latest_prices("X:BTCUSD", limit=10)

    # Subscribe to real-time updates
    client.subscribe_to_trades(callback_function)
"""


# Import Supabase client
try:
    from src.database import get_db, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("⚠️  Warning: supabase-py not installed. Run: pip install supabase")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupabaseClient:
    """
    High-level client for interacting with Supabase database.

    This client provides methods for:
    - Inserting market data (OHLCV, trades, quotes)
    - Querying historical data
    - Managing orders and positions
    - Storing ML model data
    - Real-time subscriptions
    """

    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize Supabase client.

        Args:
            env_file: Path to .env file (optional, defaults to config/api-keys/.env)
        """
        if not SUPABASE_AVAILABLE:
            raise ImportError("supabase-py is required. Install: pip install supabase")

        # Load environment variables
        if env_file is None:
            # Try to find .env in project root
            project_root = os.getenv("PROJECT_ROOT", "/Volumes/Lexar/RRRVentures/RRRalgorithms")
            env_file = os.path.join(project_root, "config/api-keys/.env")

        if os.path.exists(env_file):
            load_dotenv(env_file)
            logger.info(f"Loaded environment from {env_file}")
        else:
            logger.warning(f"Environment file not found: {env_file}")

        # Get credentials
        self.url = os.getenv("DATABASE_PATH")
        self.key = os.getenv("DATABASE_TYPE")  # Use service key for backend

        if not self.url or not self.key:
            raise ValueError(
                "DATABASE_PATH and DATABASE_TYPE must be set in environment. "
                "Please check config/api-keys/.env"
            )

        # Create client
        self.client: Client = get_db()
        logger.info("Supabase client initialized")

    # =========================================================================
    # Market Data - OHLCV Aggregates
    # =========================================================================

    def insert_crypto_aggregate(self, data: Dict[str, Any]) -> Dict:
        """
        Insert OHLCV bar data.

        Args:
            data: Dictionary with keys:
                - ticker (str): e.g., "X:BTCUSD"
                - event_time (str or datetime): ISO format timestamp
                - open (float)
                - high (float)
                - low (float)
                - close (float)
                - volume (float)
                - vwap (float, optional)
                - trade_count (int, optional)

        Returns:
            Dict with inserted data
        """
        try:
            result = self.client.table("crypto_aggregates").insert(data).execute()
            logger.info(f"Inserted aggregate for {data.get('ticker')} at {data.get('event_time')}")
            return result.data
        except Exception as e:
            logger.error(f"Error inserting crypto aggregate: {e}")
            raise

    def insert_crypto_aggregates_bulk(self, data_list: List[Dict[str, Any]]) -> Dict:
        """
        Insert multiple OHLCV bars at once.

        Args:
            data_list: List of aggregate dictionaries

        Returns:
            Dict with inserted data
        """
        try:
            result = self.client.table("crypto_aggregates").insert(data_list).execute()
            logger.info(f"Inserted {len(data_list)} aggregates")
            return result.data
        except Exception as e:
            logger.error(f"Error inserting crypto aggregates bulk: {e}")
            raise

    @lru_cache(maxsize=128)

    def get_latest_prices(self, ticker: str, limit: int = 10) -> List[Dict]:
        """
        Get latest OHLCV bars for a ticker.

        Args:
            ticker: Ticker symbol (e.g., "X:BTCUSD")
            limit: Number of bars to retrieve

        Returns:
            List of OHLCV bars (most recent first)
        """
        try:
            result = (
                self.client
                .table("crypto_aggregates")
                .select("*")
                .eq("ticker", ticker)
                .order("event_time", desc=True)
                .limit(limit)
                .execute()
            )
            return result.data
        except Exception as e:
            logger.error(f"Error getting latest prices for {ticker}: {e}")
            raise

    @lru_cache(maxsize=128)

    def get_price_history(
        self,
        ticker: str,
        start_time: datetime,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get price history for a time range.

        Args:
            ticker: Ticker symbol
            start_time: Start of time range
            end_time: End of time range (defaults to now)

        Returns:
            List of OHLCV bars
        """
        if end_time is None:
            end_time = datetime.now()

        try:
            result = (
                self.client
                .table("crypto_aggregates")
                .select("*")
                .eq("ticker", ticker)
                .gte("event_time", start_time.isoformat())
                .lte("event_time", end_time.isoformat())
                .order("event_time", desc=False)
                .execute()
            )
            return result.data
        except Exception as e:
            logger.error(f"Error getting price history for {ticker}: {e}")
            raise

    # =========================================================================
    # Market Data - Trades
    # =========================================================================

    def insert_crypto_trade(self, data: Dict[str, Any]) -> Dict:
        """
        Insert individual trade data.

        Args:
            data: Dictionary with trade data

        Returns:
            Dict with inserted data
        """
        try:
            result = self.client.table("crypto_trades").insert(data).execute()
            return result.data
        except Exception as e:
            logger.error(f"Error inserting crypto trade: {e}")
            raise

    def insert_crypto_trades_bulk(self, data_list: List[Dict[str, Any]]) -> Dict:
        """
        Insert multiple trades at once.

        Args:
            data_list: List of trade dictionaries

        Returns:
            Dict with inserted data
        """
        try:
            result = self.client.table("crypto_trades").insert(data_list).execute()
            logger.info(f"Inserted {len(data_list)} trades")
            return result.data
        except Exception as e:
            logger.error(f"Error inserting crypto trades bulk: {e}")
            raise

    # =========================================================================
    # Market Data - Quotes
    # =========================================================================

    def insert_crypto_quote(self, data: Dict[str, Any]) -> Dict:
        """
        Insert bid/ask quote data.

        Args:
            data: Dictionary with quote data

        Returns:
            Dict with inserted data
        """
        try:
            result = self.client.table("crypto_quotes").insert(data).execute()
            return result.data
        except Exception as e:
            logger.error(f"Error inserting crypto quote: {e}")
            raise

    # =========================================================================
    # Sentiment Data
    # =========================================================================

    def insert_market_sentiment(self, data: Dict[str, Any]) -> Dict:
        """
        Insert sentiment analysis data.

        Args:
            data: Dictionary with keys:
                - asset (str): Asset symbol
                - source (str): Data source (e.g., 'perplexity', 'twitter')
                - sentiment_label (str): 'bullish', 'neutral', 'bearish'
                - sentiment_score (float): -1.0 to 1.0
                - confidence (float): 0.0 to 1.0
                - text (str, optional): Source text
                - metadata (dict, optional): Additional data

        Returns:
            Dict with inserted data
        """
        try:
            result = self.client.table("market_sentiment").insert(data).execute()
            logger.info(f"Inserted sentiment for {data.get('asset')} from {data.get('source')}")
            return result.data
        except Exception as e:
            logger.error(f"Error inserting market sentiment: {e}")
            raise

    @lru_cache(maxsize=128)

    def get_recent_sentiment(self, asset: str, hours: int = 24) -> List[Dict]:
        """
        Get recent sentiment data for an asset.

        Args:
            asset: Asset symbol
            hours: Number of hours to look back

        Returns:
            List of sentiment records
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        try:
            result = (
                self.client
                .table("market_sentiment")
                .select("*")
                .eq("asset", asset)
                .gte("event_time", cutoff_time.isoformat())
                .order("event_time", desc=True)
                .execute()
            )
            return result.data
        except Exception as e:
            logger.error(f"Error getting recent sentiment for {asset}: {e}")
            raise

    # =========================================================================
    # Trading Orders
    # =========================================================================

    def insert_order(self, data: Dict[str, Any]) -> Dict:
        """
        Insert trading order.

        Args:
            data: Order data dictionary

        Returns:
            Dict with inserted data
        """
        try:
            result = self.client.table("orders").insert(data).execute()
            logger.info(f"Inserted order {data.get('order_id')}")
            return result.data
        except Exception as e:
            logger.error(f"Error inserting order: {e}")
            raise

    def update_order_status(self, order_id: str, status: str, **kwargs) -> Dict:
        """
        Update order status.

        Args:
            order_id: Order ID
            status: New status ('filled', 'cancelled', etc.)
            **kwargs: Additional fields to update

        Returns:
            Dict with updated data
        """
        update_data = {"status": status, **kwargs}

        try:
            result = (
                self.client
                .table("orders")
                .update(update_data)
                .eq("order_id", order_id)
                .execute()
            )
            logger.info(f"Updated order {order_id} status to {status}")
            return result.data
        except Exception as e:
            logger.error(f"Error updating order status: {e}")
            raise

    # =========================================================================
    # Positions
    # =========================================================================

    @lru_cache(maxsize=128)

    def get_open_positions(self) -> List[Dict]:
        """
        Get all open positions.

        Returns:
            List of open position records
        """
        try:
            result = (
                self.client
                .table("positions")
                .select("*")
                .eq("status", "open")
                .execute()
            )
            return result.data
        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
            raise

    def upsert_position(self, data: Dict[str, Any]) -> Dict:
        """
        Insert or update position.

        Args:
            data: Position data dictionary

        Returns:
            Dict with upserted data
        """
        try:
            result = self.client.table("positions").upsert(data).execute()
            return result.data
        except Exception as e:
            logger.error(f"Error upserting position: {e}")
            raise

    # =========================================================================
    # ML Models
    # =========================================================================

    def register_ml_model(self, data: Dict[str, Any]) -> Dict:
        """
        Register a new ML model.

        Args:
            data: Model metadata dictionary

        Returns:
            Dict with inserted data
        """
        try:
            result = self.client.table("ml_models").insert(data).execute()
            logger.info(f"Registered model {data.get('model_name')} v{data.get('model_version')}")
            return result.data
        except Exception as e:
            logger.error(f"Error registering ML model: {e}")
            raise

    def insert_model_prediction(self, data: Dict[str, Any]) -> Dict:
        """
        Store model prediction.

        Args:
            data: Prediction data dictionary

        Returns:
            Dict with inserted data
        """
        try:
            result = self.client.table("model_predictions").insert(data).execute()
            return result.data
        except Exception as e:
            logger.error(f"Error inserting model prediction: {e}")
            raise

    # =========================================================================
    # System Logs
    # =========================================================================

    def log_system_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        component: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Log a system event.

        Args:
            event_type: Event type (e.g., 'startup', 'error', 'trade')
            severity: 'info', 'warning', 'error', 'critical'
            message: Event message
            component: Component name (optional)
            metadata: Additional metadata (optional)

        Returns:
            Dict with inserted data
        """
        data = {
            "event_type": event_type,
            "severity": severity,
            "message": message,
            "component": component,
            "metadata": metadata or {}
        }

        try:
            result = self.client.table("system_events").insert(data).execute()
            return result.data
        except Exception as e:
            logger.error(f"Error logging system event: {e}")
            raise

    # =========================================================================
    # Real-time Subscriptions
    # =========================================================================

    def subscribe_to_trades(self, callback: Callable[[Dict], None]):
        """
        Subscribe to real-time trade insertions.

        Args:
            callback: Function to call when new trade is inserted

        Returns:
            Subscription object
        """
        try:
            subscription = (
                self.client
                .table("crypto_trades")
                .on("INSERT", callback)
                .subscribe()
            )
            logger.info("Subscribed to crypto_trades")
            return subscription
        except Exception as e:
            logger.error(f"Error subscribing to trades: {e}")
            raise

    def subscribe_to_orders(self, callback: Callable[[Dict], None]):
        """
        Subscribe to order updates.

        Args:
            callback: Function to call when order is updated

        Returns:
            Subscription object
        """
        try:
            subscription = (
                self.client
                .table("orders")
                .on("*", callback)  # Subscribe to all events
                .subscribe()
            )
            logger.info("Subscribed to orders")
            return subscription
        except Exception as e:
            logger.error(f"Error subscribing to orders: {e}")
            raise

    def subscribe_to_signals(self, callback: Callable[[Dict], None]):
        """
        Subscribe to trading signals.

        Args:
            callback: Function to call when new signal is generated

        Returns:
            Subscription object
        """
        try:
            subscription = (
                self.client
                .table("trading_signals")
                .on("INSERT", callback)
                .subscribe()
            )
            logger.info("Subscribed to trading_signals")
            return subscription
        except Exception as e:
            logger.error(f"Error subscribing to signals: {e}")
            raise


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Initialize client
    client = get_db()

    # Example 1: Insert OHLCV data
    print("\n" + "=" * 70)
    print("Example 1: Inserting OHLCV data")
    print("=" * 70)

    aggregate_data = {
        "ticker": "X:BTCUSD",
        "event_time": datetime.now().isoformat(),
        "open": 67000.00,
        "high": 67500.00,
        "low": 66800.00,
        "close": 67200.00,
        "volume": 150.5,
        "vwap": 67100.00,
        "trade_count": 1250
    }

    result = client.insert_crypto_aggregate(aggregate_data)
    print(f"✅ Inserted: {result}")

    # Example 2: Query latest prices
    print("\n" + "=" * 70)
    print("Example 2: Querying latest prices")
    print("=" * 70)

    prices = client.get_latest_prices("X:BTCUSD", limit=5)
    print(f"✅ Found {len(prices)} price records")
    for price in prices:
        print(f"  {price['event_time']}: ${price['close']}")

    # Example 3: Log system event
    print("\n" + "=" * 70)
    print("Example 3: Logging system event")
    print("=" * 70)

    client.log_system_event(
        event_type="test",
        severity="info",
        message="Supabase client test completed successfully",
        component="supabase_client",
        metadata={"test_run": True}
    )
    print("✅ Event logged")

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
