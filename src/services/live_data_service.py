#!/usr/bin/env python3
"""
Live Market Data Ingestion Service
===================================

Continuously ingests real-time cryptocurrency market data from Polygon.io
and stores it in the local SQLite database.

Features:
- WebSocket streaming of trades, quotes, and 1-minute aggregates
- Auto-reconnection on connection loss
- Graceful shutdown handling
- Statistics logging

Usage:
    python src/services/live_data_service.py

Environment Variables:
    POLYGON_API_KEY: Required Polygon.io API key
    DATABASE_PATH: Optional path to SQLite database (default: data/db/trading.db)
"""

import asyncio
import signal
import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.database.client_factory import get_db
from src.data_pipeline.polygon.websocket_client import PolygonWebSocketClient
from src.data_pipeline.polygon.database_adapter import PolygonDatabaseAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/live_data_service.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


class LiveDataService:
    """
    Service that manages the live market data ingestion pipeline.
    """

    def __init__(self):
        self.running = False
        self.db_client = None
        self.ws_client = None
        self.db_adapter = None

        # Statistics
        self.start_time = None
        self.total_messages = 0
        self.last_stats_print = None

    async def start(self):
        """Initialize and start the data ingestion service."""
        logger.info("=" * 70)
        logger.info("Live Market Data Ingestion Service")
        logger.info("=" * 70)

        # Initialize database
        logger.info("Initializing database connection...")
        try:
            self.db_client = get_db()
            await self.db_client.connect()
            logger.info("✓ Database connected")
        except Exception as e:
            logger.error(f"✗ Failed to connect to database: {e}")
            return False

        # Create database adapter
        self.db_adapter = PolygonDatabaseAdapter(self.db_client)
        logger.info("✓ Database adapter created")

        # Initialize WebSocket client
        logger.info("Initializing Polygon WebSocket client...")
        try:
            api_key = os.getenv("POLYGON_API_KEY")
            if not api_key:
                logger.error("✗ POLYGON_API_KEY environment variable not set")
                return False

            # Create WebSocket client with our database adapter
            self.ws_client = PolygonWebSocketClient(
                api_key=api_key,
                supabase_client=self.db_adapter,  # Our adapter has the same interface
                pairs=[
                    "X:BTCUSD",
                    "X:ETHUSD",
                    "X:SOLUSD",
                    "X:MATICUSD",
                    "X:AVAXUSD",
                ],
                enable_trades=True,
                enable_quotes=True,
                enable_aggregates=True
            )
            logger.info("✓ WebSocket client initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize WebSocket client: {e}")
            return False

        # Start the service
        self.running = True
        self.start_time = datetime.utcnow()
        self.last_stats_print = self.start_time

        logger.info("=" * 70)
        logger.info("Service started successfully!")
        logger.info("Streaming market data for: BTC, ETH, SOL, MATIC, AVAX")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 70)

        # Run WebSocket client
        try:
            await self.ws_client.run()
        except Exception as e:
            logger.error(f"Error in WebSocket client: {e}")
            self.running = False

        return True

    async def stop(self):
        """Stop the service gracefully."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("Stopping live data service...")
        logger.info("=" * 70)

        self.running = False

        # Stop WebSocket client
        if self.ws_client:
            self.ws_client.running = False
            logger.info("✓ WebSocket client stopped")

        # Print final statistics
        self.print_statistics(final=True)

        # Disconnect database
        if self.db_client:
            await self.db_client.disconnect()
            logger.info("✓ Database disconnected")

        logger.info("=" * 70)
        logger.info("Service stopped successfully")
        logger.info("=" * 70)

    def print_statistics(self, final=False):
        """Print service statistics."""
        if not self.ws_client or not self.start_time:
            return

        now = datetime.utcnow()
        uptime = (now - self.start_time).total_seconds()

        stats = [
            "",
            "=" * 70,
            "Service Statistics" + (" (FINAL)" if final else ""),
            "=" * 70,
            f"Uptime: {int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s",
            f"Total Messages: {self.ws_client.message_count:,}",
            f"Trades: {self.ws_client.trade_count:,}",
            f"Quotes: {self.ws_client.quote_count:,}",
            f"Aggregates: {self.ws_client.aggregate_count:,}",
            f"Errors: {self.ws_client.error_count:,}",
        ]

        if uptime > 0:
            stats.append(f"Messages/sec: {self.ws_client.message_count / uptime:.2f}")

        if self.ws_client.last_message_time:
            time_since_last = (now - self.ws_client.last_message_time).total_seconds()
            stats.append(f"Last message: {time_since_last:.1f}s ago")

        stats.append("=" * 70)

        for line in stats:
            logger.info(line)

    async def run_stats_printer(self):
        """Background task to print statistics every 60 seconds."""
        while self.running:
            await asyncio.sleep(60)
            if self.running:
                self.print_statistics()


async def main():
    """Main entry point."""
    # Create service
    service = LiveDataService()

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        asyncio.create_task(service.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start service
    try:
        # Run service and stats printer concurrently
        await asyncio.gather(
            service.start(),
            service.run_stats_printer()
        )
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await service.stop()


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Run the service
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Exiting...")
        sys.exit(0)
