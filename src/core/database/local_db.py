from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from src.core.exceptions import DataValidationError
from src.core.validation import validate_market_data
from typing import Any, Dict, List, Optional, Tuple
import json
import sqlite3
import threading
import time


"""
SQLite database manager for local development.
Lightweight alternative to PostgreSQL/Supabase for laptop development.
"""




class LocalDatabase:
    """
    SQLite database manager optimized for local development.
    Thread-safe with connection pooling.
    """
    
    def __init__(self, db_path: str = "data/local.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-local storage for connections
        self._local = threading.local()

        # Thread safety lock for critical operations
        self._lock = threading.RLock()

        # Initialize database schema
        self._initialize_schema()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable foreign keys
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            # Enable WAL mode for better concurrency
            self._local.connection.execute("PRAGMA journal_mode = WAL")
        
        return self._local.connection
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions with thread safety."""
        with self._lock:
            conn = self._get_connection()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    
    def execute(self, query: str, params: Optional[Tuple] = None) -> sqlite3.Cursor:
        """
        Execute a query.
        
        Args:
            query: SQL query
            params: Query parameters
        
        Returns:
            Cursor with results
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        conn.commit()
        return cursor
    
    def fetch_one(self, query: str, params: Optional[Tuple] = None) -> Optional[Dict[str, Any]]:
        """Fetch one row as dictionary."""
        cursor = self.execute(query, params)
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def fetch_all(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """Fetch all rows as list of dictionaries."""
        cursor = self.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def _initialize_schema(self):
        """Initialize database schema for all tables."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Market data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp REAL NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        """)
        
        # Create index for fast queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp 
            ON market_data(symbol, timestamp DESC)
        """)
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                timestamp REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                executed_quantity REAL DEFAULT 0,
                executed_price REAL,
                commission REAL DEFAULT 0,
                pnl REAL,
                strategy TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp 
            ON trades(symbol, timestamp DESC)
        """)
        
        # Positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL UNIQUE,
                quantity REAL NOT NULL DEFAULT 0,
                average_price REAL NOT NULL DEFAULT 0,
                current_price REAL,
                unrealized_pnl REAL DEFAULT 0,
                realized_pnl REAL DEFAULT 0,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Portfolio metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                total_value REAL NOT NULL,
                cash REAL NOT NULL,
                positions_value REAL NOT NULL,
                daily_pnl REAL DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                sharpe_ratio REAL,
                sortino_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_portfolio_metrics_timestamp 
            ON portfolio_metrics(timestamp DESC)
        """)
        
        # Predictions table (for ML models)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp REAL NOT NULL,
                horizon INTEGER NOT NULL,
                predicted_price REAL NOT NULL,
                predicted_direction TEXT,
                confidence REAL,
                model_version TEXT,
                features TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_symbol_timestamp 
            ON predictions(symbol, timestamp DESC)
        """)
        
        # Risk metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                var_95 REAL,
                var_99 REAL,
                portfolio_volatility REAL,
                beta REAL,
                leverage REAL,
                margin_used REAL,
                risk_score REAL,
                warnings TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # System logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                level TEXT NOT NULL,
                service TEXT NOT NULL,
                message TEXT NOT NULL,
                details TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp 
            ON system_logs(timestamp DESC)
        """)
        
        # Additional performance indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_market_data_timestamp 
            ON market_data(timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp 
            ON trades(timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
            ON predictions(timestamp DESC)
        """)

        # System flags table for control signals
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_flags (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                timestamp REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
    
    def insert_market_data(self, symbol: str, timestamp: float, ohlcv: Dict[str, float]):
        """
        Insert market data (OHLCV) with validation.
        
        Args:
            symbol: Trading symbol
            timestamp: Unix timestamp
            ohlcv: OHLCV data dictionary
            
        Raises:
            DataValidationError: If validation fails
        """
        # Validate input data
        validated = validate_market_data(symbol, timestamp, ohlcv)
        
        query = """
            INSERT OR REPLACE INTO market_data 
            (symbol, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        self.execute(query, (
            validated.symbol,
            validated.timestamp,
            validated.ohlcv.open,
            validated.ohlcv.high,
            validated.ohlcv.low,
            validated.ohlcv.close,
            validated.ohlcv.volume
        ))
    
    @lru_cache(maxsize=128)
    
    def get_market_data(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent market data for a symbol."""
        query = """
            SELECT * FROM market_data 
            WHERE symbol = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        return self.fetch_all(query, (symbol, limit))
    
    def insert_trade(self, trade: Dict[str, Any]) -> int:
        """Insert a new trade."""
        query = """
            INSERT INTO trades 
            (symbol, side, order_type, quantity, price, timestamp, status, strategy, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor = self.execute(query, (
            trade['symbol'], trade['side'], trade['order_type'],
            trade['quantity'], trade['price'], trade['timestamp'],
            trade.get('status', 'pending'),
            trade.get('strategy'), trade.get('notes')
        ))
        return cursor.lastrowid
    
    def update_trade(self, trade_id: int, updates: Dict[str, Any]):
        """Update trade status and execution details."""
        # Whitelist allowed columns to prevent SQL injection
        ALLOWED_COLUMNS = {
            'status', 'executed_quantity', 'executed_price', 
            'commission', 'pnl', 'strategy', 'notes', 'updated_at'
        }
        
        # Validate all columns are allowed
        invalid_cols = set(updates.keys()) - ALLOWED_COLUMNS
        if invalid_cols:
            raise ValueError(f"Invalid columns for update: {invalid_cols}")
        
        updates['updated_at'] = datetime.now().isoformat()
        
        set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
        query = f"UPDATE trades SET {set_clause} WHERE id = ?"
        
        params = list(updates.values()) + [trade_id]
        self.execute(query, tuple(params))
    
    @lru_cache(maxsize=128)
    
    def get_trades(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades."""
        if symbol:
            query = "SELECT * FROM trades WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?"
            return self.fetch_all(query, (symbol, limit))
        else:
            query = "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?"
            return self.fetch_all(query, (limit,))
    
    def upsert_position(self, symbol: str, quantity: float, average_price: float,
                       current_price: Optional[float] = None):
        """Update or insert position."""
        unrealized_pnl = None
        if current_price:
            unrealized_pnl = (current_price - average_price) * quantity
        
        query = """
            INSERT INTO positions (symbol, quantity, average_price, current_price, unrealized_pnl)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                quantity = excluded.quantity,
                average_price = excluded.average_price,
                current_price = excluded.current_price,
                unrealized_pnl = excluded.unrealized_pnl,
                updated_at = CURRENT_TIMESTAMP
        """
        self.execute(query, (symbol, quantity, average_price, current_price, unrealized_pnl))
    
    @lru_cache(maxsize=128)
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all current positions."""
        query = "SELECT * FROM positions WHERE quantity != 0 ORDER BY symbol"
        return self.fetch_all(query)
    
    def insert_portfolio_metrics(self, metrics: Dict[str, Any]):
        """Insert portfolio performance metrics."""
        query = """
            INSERT INTO portfolio_metrics 
            (timestamp, total_value, cash, positions_value, daily_pnl, total_pnl,
             sharpe_ratio, sortino_ratio, max_drawdown, win_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.execute(query, (
            metrics['timestamp'], metrics['total_value'],
            metrics['cash'], metrics['positions_value'],
            metrics.get('daily_pnl', 0), metrics.get('total_pnl', 0),
            metrics.get('sharpe_ratio'), metrics.get('sortino_ratio'),
            metrics.get('max_drawdown'), metrics.get('win_rate')
        ))
    
    @lru_cache(maxsize=128)
    
    def get_latest_portfolio_metrics(self) -> Optional[Dict[str, Any]]:
        """Get most recent portfolio metrics."""
        query = "SELECT * FROM portfolio_metrics ORDER BY timestamp DESC LIMIT 1"
        return self.fetch_one(query)
    
    def insert_prediction(self, prediction: Dict[str, Any]):
        """Insert ML model prediction."""
        query = """
            INSERT INTO predictions 
            (symbol, timestamp, horizon, predicted_price, predicted_direction,
             confidence, model_version, features)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.execute(query, (
            prediction['symbol'], prediction['timestamp'],
            prediction['horizon'], prediction['predicted_price'],
            prediction.get('predicted_direction'),
            prediction.get('confidence'),
            prediction.get('model_version'),
            json.dumps(prediction.get('features', {}))
        ))
    
    def log(self, level: str, service: str, message: str, details: Optional[Dict] = None):
        """Insert system log."""
        query = """
            INSERT INTO system_logs (timestamp, level, service, message, details)
            VALUES (?, ?, ?, ?, ?)
        """
        self.execute(query, (
            datetime.now().timestamp(), level, service, message,
            json.dumps(details) if details else None
        ))
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')


# Global database instance
_global_db: Optional[LocalDatabase] = None


@lru_cache(maxsize=128)


def get_db(db_path: Optional[str] = None) -> LocalDatabase:
    """
    Get global database instance.
    
    Args:
        db_path: Optional path to database file
    
    Returns:
        LocalDatabase instance
    """
    global _global_db
    
    if _global_db is None:
        if db_path is None:
            from src.core.config.loader import config_get
            db_path = config_get('database.path', 'data/local.db')
        _global_db = LocalDatabase(db_path)
    
    return _global_db


if __name__ == "__main__":
    # Test the database
    db = LocalDatabase("data/test.db")
    print("Database initialized successfully")
    print(f"Database path: {db.db_path}")
    
    # Test inserting market data
    import time
    db.insert_market_data("BTC-USD", time.time(), {
        'open': 50000, 'high': 51000, 'low': 49500,
        'close': 50500, 'volume': 1000000
    })
    
    # Test fetching
    data = db.get_market_data("BTC-USD", limit=10)
    print(f"Fetched {len(data)} market data points")
    
    db.close()

