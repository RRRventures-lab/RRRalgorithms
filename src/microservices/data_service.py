from datetime import datetime
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from functools import lru_cache
from pydantic import BaseModel
from src.core.memory_cache import get_cache
from src.core.redis_cache import get_redis_cache
from src.data_pipeline.websocket_pipeline import WebSocketDataSource, MarketData
from typing import Dict, List, Optional, Any
import asyncio
import json
import logging


"""
Data Service
============

Microservice for real-time market data processing and distribution.
Handles WebSocket connections, data validation, and caching.

Author: RRR Ventures
Date: 2025-10-12
"""




class DataService:
    """
    Data service for real-time market data processing.
    
    Features:
    - WebSocket data streaming
    - Data validation and normalization
    - Real-time caching
    - WebSocket client management
    - Performance monitoring
    """
    
    def __init__(self, port: int = 8001):
        """
        Initialize data service.
        
        Args:
            port: Service port
        """
        self.port = port
        self.app = FastAPI(title="Data Service", version="1.0.0")
        
        # Data sources
        self.websocket_source = None
        self.redis_cache = None
        self.memory_cache = None
        
        # WebSocket clients
        self.active_connections: List[WebSocket] = []
        
        # Performance metrics
        self.metrics = {
            'total_data_points': 0,
            'active_connections': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_processing_time': 0.0
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Service health check."""
            return {
                "status": "healthy",
                "service": "data-service",
                "timestamp": datetime.now().isoformat(),
                "metrics": self.metrics
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self.metrics
        
        @self.app.get("/data/latest/{symbol}")
        async def get_latest_data(symbol: str):
            """Get latest market data for a symbol."""
            try:
                # Try memory cache first
                if self.memory_cache:
                    cached_data = self.memory_cache.get(f"market_data:{symbol}")
                    if cached_data:
                        self.metrics['cache_hits'] += 1
                        return cached_data
                
                # Try Redis cache
                if self.redis_cache:
                    cached_data = await self.redis_cache.get_cached_market_data(symbol)
                    if cached_data:
                        self.metrics['cache_hits'] += 1
                        return cached_data
                
                # Get from WebSocket source
                if self.websocket_source:
                    data = self.websocket_source.get_latest_data(symbol)
                    if data and symbol in data:
                        self.metrics['cache_misses'] += 1
                        return data[symbol]
                
                self.metrics['cache_misses'] += 1
                raise HTTPException(status_code=404, detail="Data not found")
                
            except Exception as e:
                self.logger.error(f"Error getting latest data for {symbol}: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/data/symbols")
        async def get_available_symbols():
            """Get list of available symbols."""
            if self.websocket_source:
                return {"symbols": self.websocket_source.symbols}
            return {"symbols": []}
        
        @self.app.websocket("/ws/data")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time data streaming."""
            await websocket.accept()
            self.active_connections.append(websocket)
            self.metrics['active_connections'] = len(self.active_connections)
            
            try:
                while True:
                    # Send heartbeat
                    await websocket.send_text(json.dumps({
                        "type": "heartbeat",
                        "timestamp": datetime.now().isoformat()
                    }))
                    await asyncio.sleep(30)
                    
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
                self.metrics['active_connections'] = len(self.active_connections)
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)
                    self.metrics['active_connections'] = len(self.active_connections)
    
    async def initialize(self) -> None:
        """Initialize data service components."""
        try:
            # Initialize WebSocket data source
            self.websocket_source = WebSocketDataSource(
                symbols=['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD'],
                exchanges=['polygon']
            )
            
            # Initialize caches
            self.redis_cache = await get_redis_cache()
            self.memory_cache = get_cache()
            
            # Setup data callback
            self.websocket_source.add_data_callback(self._on_market_data)
            
            self.logger.info("Data service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data service: {e}")
            raise
    
    async def _on_market_data(self, market_data: MarketData) -> None:
        """Handle incoming market data."""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Convert to dictionary
            data_dict = {
                'symbol': market_data.symbol,
                'timestamp': market_data.timestamp,
                'price': market_data.price,
                'volume': market_data.volume,
                'high': market_data.high,
                'low': market_data.low,
                'open': market_data.open,
                'close': market_data.close,
                'source': market_data.source
            }
            
            # Cache in memory
            if self.memory_cache:
                self.memory_cache.set(
                    f"market_data:{market_data.symbol}",
                    data_dict,
                    ttl=60
                )
            
            # Cache in Redis
            if self.redis_cache:
                await self.redis_cache.cache_market_data(
                    market_data.symbol,
                    data_dict,
                    ttl=300
                )
            
            # Broadcast to WebSocket clients
            await self._broadcast_to_clients(data_dict)
            
            # Update metrics
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_metrics(processing_time)
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
    
    async def _broadcast_to_clients(self, data: Dict[str, Any]) -> None:
        """Broadcast data to all connected WebSocket clients."""
        if not self.active_connections:
            return
        
        message = json.dumps({
            "type": "market_data",
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
        
        # Send to all connected clients
        disconnected = []
        for websocket in self.active_connections:
            try:
                await websocket.send_text(message)
            except Exception as e:
                self.logger.warning(f"Failed to send to client: {e}")
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            self.active_connections.remove(websocket)
        
        self.metrics['active_connections'] = len(self.active_connections)
    
    def _update_metrics(self, processing_time: float) -> None:
        """Update performance metrics."""
        self.metrics['total_data_points'] += 1
        
        # Update average processing time
        if self.metrics['total_data_points'] == 1:
            self.metrics['avg_processing_time'] = processing_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics['avg_processing_time'] = (
                alpha * processing_time + 
                (1 - alpha) * self.metrics['avg_processing_time']
            )
    
    async def start(self) -> None:
        """Start the data service."""
        self.logger.info(f"Starting Data Service on port {self.port}")
        
        # Initialize components
        await self.initialize()
        
        # Start WebSocket data source
        if self.websocket_source:
            await self.websocket_source.start()
        
        # Start FastAPI server
        import uvicorn
        config = uvicorn.Config(
            app=self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    @lru_cache(maxsize=128)
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            'service': 'data-service',
            'port': self.port,
            'active_connections': len(self.active_connections),
            'metrics': self.metrics
        }


# Global data service instance
_data_service: Optional[DataService] = None


@lru_cache(maxsize=128)


def get_data_service() -> DataService:
    """Get the global data service instance."""
    global _data_service
    
    if _data_service is None:
        _data_service = DataService()
    
    return _data_service


if __name__ == "__main__":
    # Run data service
    service = DataService()
    asyncio.run(service.start())