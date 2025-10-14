from data_pipeline.polygon_db_writer import PolygonDatabaseWriter
from data_pipeline.polygon_websocket import PolygonWebSocketClient, CryptoTrade, CryptoQuote, CryptoAggregate
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, Any
import asyncio
import logging
import os
import signal
import sys

#!/usr/bin/env python3
"""
Start Polygon.io WebSocket streaming with database persistence.
Production-ready script with monitoring and graceful shutdown.
"""


# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PolygonStreamManager:
    """
    Manages Polygon.io WebSocket streaming with database persistence.
    Implements data validation, buffering, and monitoring.
    """

    def __init__(self, symbols: list = None):
        """Initialize the stream manager."""
        self.symbols = symbols or ['BTC-USD', 'ETH-USD', 'SOL-USD', 'MATIC-USD']
        self.websocket_client = PolygonWebSocketClient()
        self.db_writer = PolygonDatabaseWriter()

        # Data buffers for batch processing
        self.trade_buffer = []
        self.quote_buffer = []
        self.aggregate_buffer = []

        # Buffer settings
        self.buffer_size = 100
        self.flush_interval = 5  # seconds
        self.last_flush = datetime.now(timezone.utc)

        # Statistics
        self.stats = {
            'start_time': datetime.now(timezone.utc),
            'trades_processed': 0,
            'quotes_processed': 0,
            'aggregates_processed': 0,
            'buffer_flushes': 0,
            'errors': 0
        }

        self.is_running = False

    async def initialize(self):
        """Initialize database and register callbacks."""
        # Initialize database
        db_initialized = await self.db_writer.initialize()
        if not db_initialized:
            logger.error("Failed to initialize database")
            return False

        # Register WebSocket callbacks
        self.websocket_client.register_callback('trade', self.handle_trade)
        self.websocket_client.register_callback('quote', self.handle_quote)
        self.websocket_client.register_callback('aggregate', self.handle_aggregate)

        logger.info(f"Stream manager initialized for symbols: {self.symbols}")
        return True

    async def handle_trade(self, trade: CryptoTrade):
        """Handle incoming trade data."""
        try:
            # Convert to dict for database insertion
            trade_dict = asdict(trade)

            # Validate data
            if self.validate_trade(trade_dict):
                self.trade_buffer.append(trade_dict)
                self.stats['trades_processed'] += 1

                # Check if buffer should be flushed
                if len(self.trade_buffer) >= self.buffer_size:
                    await self.flush_trade_buffer()

            # Log every 100th trade for monitoring
            if self.stats['trades_processed'] % 100 == 0:
                logger.info(f"Processed {self.stats['trades_processed']} trades")

        except Exception as e:
            logger.error(f"Error handling trade: {e}")
            self.stats['errors'] += 1

    async def handle_quote(self, quote: CryptoQuote):
        """Handle incoming quote data."""
        try:
            # Convert to dict for database insertion
            quote_dict = asdict(quote)

            # Validate data
            if self.validate_quote(quote_dict):
                self.quote_buffer.append(quote_dict)
                self.stats['quotes_processed'] += 1

                # Check if buffer should be flushed
                if len(self.quote_buffer) >= self.buffer_size:
                    await self.flush_quote_buffer()

        except Exception as e:
            logger.error(f"Error handling quote: {e}")
            self.stats['errors'] += 1

    async def handle_aggregate(self, aggregate: CryptoAggregate):
        """Handle incoming aggregate bar data."""
        try:
            # Convert to dict for database insertion
            aggregate_dict = asdict(aggregate)

            # Validate data
            if self.validate_aggregate(aggregate_dict):
                self.aggregate_buffer.append(aggregate_dict)
                self.stats['aggregates_processed'] += 1

                # Always flush aggregates immediately (they're less frequent)
                await self.flush_aggregate_buffer()

                # Log aggregate for monitoring
                logger.info(
                    f"📊 {aggregate.symbol} Bar - "
                    f"O:${aggregate.open:.2f} H:${aggregate.high:.2f} "
                    f"L:${aggregate.low:.2f} C:${aggregate.close:.2f} "
                    f"V:{aggregate.volume:.2f}"
                )

        except Exception as e:
            logger.error(f"Error handling aggregate: {e}")
            self.stats['errors'] += 1

    def validate_trade(self, trade: Dict[str, Any]) -> bool:
        """Validate trade data quality."""
        # Check required fields
        if not trade.get('symbol') or not trade.get('price') or not trade.get('timestamp'):
            logger.warning(f"Trade missing required fields: {trade}")
            return False

        # Check price validity
        if trade['price'] <= 0 or trade['price'] > 1000000:
            logger.warning(f"Invalid trade price: {trade['price']} for {trade['symbol']}")
            return False

        # Check size validity
        if trade['size'] < 0:
            logger.warning(f"Invalid trade size: {trade['size']} for {trade['symbol']}")
            return False

        return True

    def validate_quote(self, quote: Dict[str, Any]) -> bool:
        """Validate quote data quality."""
        # Check required fields
        if not quote.get('symbol') or not quote.get('bid_price') or not quote.get('ask_price'):
            logger.warning(f"Quote missing required fields: {quote}")
            return False

        # Check price validity
        if quote['bid_price'] <= 0 or quote['ask_price'] <= 0:
            logger.warning(f"Invalid quote prices for {quote['symbol']}")
            return False

        # Check spread validity (bid should be less than ask)
        if quote['bid_price'] >= quote['ask_price']:
            logger.warning(f"Invalid spread for {quote['symbol']}: Bid >= Ask")
            return False

        return True

    def validate_aggregate(self, aggregate: Dict[str, Any]) -> bool:
        """Validate aggregate bar data quality."""
        # Check required fields
        if not all(key in aggregate for key in ['symbol', 'open', 'high', 'low', 'close', 'volume']):
            logger.warning(f"Aggregate missing required fields: {aggregate}")
            return False

        # Check OHLC relationships
        if not (aggregate['low'] <= aggregate['open'] <= aggregate['high']):
            logger.warning(f"Invalid OHLC relationship for open in {aggregate['symbol']}")
            return False

        if not (aggregate['low'] <= aggregate['close'] <= aggregate['high']):
            logger.warning(f"Invalid OHLC relationship for close in {aggregate['symbol']}")
            return False

        if aggregate['high'] < aggregate['low']:
            logger.warning(f"High < Low for {aggregate['symbol']}")
            return False

        # Check volume validity
        if aggregate['volume'] < 0:
            logger.warning(f"Invalid volume for {aggregate['symbol']}: {aggregate['volume']}")
            return False

        return True

    async def flush_trade_buffer(self):
        """Flush trade buffer to database."""
        if self.trade_buffer:
            await self.db_writer.insert_trades(self.trade_buffer)
            logger.debug(f"Flushed {len(self.trade_buffer)} trades to database")
            self.trade_buffer.clear()
            self.stats['buffer_flushes'] += 1

    async def flush_quote_buffer(self):
        """Flush quote buffer to database."""
        if self.quote_buffer:
            await self.db_writer.insert_quotes(self.quote_buffer)
            logger.debug(f"Flushed {len(self.quote_buffer)} quotes to database")
            self.quote_buffer.clear()
            self.stats['buffer_flushes'] += 1

    async def flush_aggregate_buffer(self):
        """Flush aggregate buffer to database."""
        if self.aggregate_buffer:
            await self.db_writer.insert_aggregates(self.aggregate_buffer)
            logger.debug(f"Flushed {len(self.aggregate_buffer)} aggregates to database")
            self.aggregate_buffer.clear()
            self.stats['buffer_flushes'] += 1

    async def periodic_flush(self):
        """Periodically flush all buffers."""
        while self.is_running:
            await asyncio.sleep(self.flush_interval)

            # Flush all buffers
            await self.flush_trade_buffer()
            await self.flush_quote_buffer()
            await self.flush_aggregate_buffer()

            # Log statistics
            self.log_statistics()

    def log_statistics(self):
        """Log current statistics."""
        runtime = (datetime.now(timezone.utc) - self.stats['start_time']).total_seconds()
        if runtime > 0:
            trades_per_sec = self.stats['trades_processed'] / runtime
            quotes_per_sec = self.stats['quotes_processed'] / runtime

            logger.info(
                f"📈 Statistics - Runtime: {runtime:.0f}s | "
                f"Trades: {self.stats['trades_processed']} ({trades_per_sec:.1f}/s) | "
                f"Quotes: {self.stats['quotes_processed']} ({quotes_per_sec:.1f}/s) | "
                f"Aggregates: {self.stats['aggregates_processed']} | "
                f"Errors: {self.stats['errors']}"
            )

    async def run(self):
        """Run the stream manager."""
        self.is_running = True

        # Initialize components
        if not await self.initialize():
            logger.error("Failed to initialize stream manager")
            return

        # Subscribe to market data
        await self.websocket_client.subscribe(
            self.symbols,
            channels=['XT', 'XQ', 'XA']  # Trades, Quotes, Aggregates
        )

        # Start periodic flush task
        flush_task = asyncio.create_task(self.periodic_flush())

        # Start WebSocket client
        logger.info(f"Starting Polygon.io WebSocket stream for: {', '.join(self.symbols)}")
        logger.info("Press Ctrl+C to stop...")

        try:
            await self.websocket_client.run()
        finally:
            # Clean up
            self.is_running = False
            flush_task.cancel()

            # Final flush
            await self.flush_trade_buffer()
            await self.flush_quote_buffer()
            await self.flush_aggregate_buffer()

            # Get final database statistics
            db_stats = await self.db_writer.get_statistics()

            # Log final statistics
            logger.info("=== Final Statistics ===")
            logger.info(f"Trades processed: {self.stats['trades_processed']}")
            logger.info(f"Quotes processed: {self.stats['quotes_processed']}")
            logger.info(f"Aggregates processed: {self.stats['aggregates_processed']}")
            logger.info(f"Buffer flushes: {self.stats['buffer_flushes']}")
            logger.info(f"Errors: {self.stats['errors']}")
            logger.info(f"Database - Total trades: {db_stats.get('total_trades', 0)}")
            logger.info(f"Database - Total quotes: {db_stats.get('total_quotes', 0)}")
            logger.info(f"Database - Total aggregates: {db_stats.get('total_aggregates', 0)}")

            # Cleanup database connection
            await self.db_writer.cleanup()

async def main():
    """Main entry point."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Start Polygon.io WebSocket streaming')
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['BTC-USD', 'ETH-USD'],
        help='Cryptocurrency symbols to stream (default: BTC-USD ETH-USD)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (30 seconds only)'
    )

    args = parser.parse_args()

    # Create stream manager
    manager = PolygonStreamManager(symbols=args.symbols)

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("\nReceived shutdown signal...")
        manager.websocket_client.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run in test mode if specified
    if args.test:
        logger.info("Running in test mode (30 seconds)...")
        task = asyncio.create_task(manager.run())
        await asyncio.sleep(30)
        manager.websocket_client.stop()
        await task
    else:
        # Run continuously
        await manager.run()

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Run the stream manager
    asyncio.run(main())