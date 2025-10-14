"""
Base database client interface.
Defines the contract that all database implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime


class DatabaseClient(ABC):
    """Abstract base class for database clients."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish database connection."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connection."""
        pass
    
    @abstractmethod
    async def execute(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> Any:
        """Execute a query and return result."""
        pass
    
    @abstractmethod
    async def execute_many(
        self,
        query: str,
        params_list: List[tuple]
    ) -> None:
        """Execute same query with multiple parameter sets."""
        pass
    
    @abstractmethod
    async def fetch_one(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> Optional[Dict[str, Any]]:
        """Fetch single row as dictionary."""
        pass
    
    @abstractmethod
    async def fetch_all(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """Fetch all rows as list of dictionaries."""
        pass
    
    @abstractmethod
    async def insert(
        self,
        table: str,
        data: Dict[str, Any]
    ) -> int:
        """Insert single row, return inserted ID."""
        pass
    
    @abstractmethod
    async def insert_many(
        self,
        table: str,
        data_list: List[Dict[str, Any]]
    ) -> None:
        """Insert multiple rows."""
        pass
    
    @abstractmethod
    async def update(
        self,
        table: str,
        data: Dict[str, Any],
        where: Dict[str, Any]
    ) -> int:
        """Update rows matching where clause, return count."""
        pass
    
    @abstractmethod
    async def delete(
        self,
        table: str,
        where: Dict[str, Any]
    ) -> int:
        """Delete rows matching where clause, return count."""
        pass
    
    @abstractmethod
    async def table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        pass
    
    @abstractmethod
    async def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """Get table schema information."""
        pass
    
    # High-level convenience methods
    
    async def get_market_data(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Fetch market data for a symbol."""
        query = "SELECT * FROM market_data WHERE symbol = ?"
        params = [symbol]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(int(start_time.timestamp()))
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(int(end_time.timestamp()))
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        return await self.fetch_all(query, tuple(params))
    
    async def save_market_data(
        self,
        symbol: str,
        timestamp: int,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        vwap: Optional[float] = None,
        trade_count: Optional[int] = None
    ) -> int:
        """Save single market data point."""
        data = {
            'symbol': symbol,
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'vwap': vwap,
            'trade_count': trade_count
        }
        return await self.insert('market_data', data)
    
    async def get_active_positions(self) -> List[Dict[str, Any]]:
        """Get all active positions."""
        return await self.fetch_all(
            "SELECT * FROM positions WHERE quantity != 0 ORDER BY symbol"
        )
    
    async def save_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: int,
        status: str = 'filled',
        **kwargs
    ) -> int:
        """Save a trade record."""
        data = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'timestamp': timestamp,
            'status': status,
            **kwargs
        }
        return await self.insert('trades', data)
    
    async def get_recent_trades(
        self,
        limit: int = 100,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent trades."""
        query = "SELECT * FROM trades"
        params = []
        
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        return await self.fetch_all(query, tuple(params) if params else None)
    
    async def get_portfolio_snapshot(
        self,
        timestamp: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Get portfolio snapshot at specific time or latest."""
        if timestamp:
            query = """
                SELECT * FROM portfolio_snapshots 
                WHERE timestamp <= ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """
            return await self.fetch_one(query, (timestamp,))
        else:
            query = """
                SELECT * FROM portfolio_snapshots 
                ORDER BY timestamp DESC 
                LIMIT 1
            """
            return await self.fetch_one(query)
    
    async def save_system_event(
        self,
        component: str,
        event_type: str,
        severity: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> int:
        """Save system event."""
        import json
        import time
        
        data = {
            'component': component,
            'event_type': event_type,
            'severity': severity,
            'message': message,
            'details': json.dumps(details) if details else None,
            'timestamp': int(time.time())
        }
        return await self.insert('system_events', data)

