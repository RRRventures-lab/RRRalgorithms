from data_pipeline.polygon_websocket import PolygonWebSocketClient, CryptoTrade, CryptoQuote, CryptoAggregate
from datetime import datetime, timezone
from pathlib import Path
import asyncio
import json
import os
import signal
import sys

#!/usr/bin/env python3
"""
Test Polygon.io WebSocket connection without database dependency.
Logs data to console and JSON files for verification.
"""


# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Create data directory
DATA_DIR = Path("data/polygon_test")
DATA_DIR.mkdir(parents=True, exist_ok=True)

class TestPolygonStream:
    """Test Polygon streaming without database."""

    def __init__(self):
        self.client = PolygonWebSocketClient()
        self.data_log = []
        self.start_time = datetime.now(timezone.utc)

        # Statistics
        self.stats = {
            'trades': 0,
            'quotes': 0,
            'aggregates': 0
        }

    async def handle_trade(self, trade: CryptoTrade):
        """Log trade data."""
        self.stats['trades'] += 1
        trade_data = {
            'type': 'trade',
            'symbol': trade.symbol,
            'price': trade.price,
            'size': trade.size,
            'timestamp': trade.timestamp,
            'received_at': datetime.now(timezone.utc).isoformat()
        }
        self.data_log.append(trade_data)
        print(f"💰 TRADE: {trade.symbol} @ ${trade.price:.2f} x {trade.size:.8f}")

    async def handle_quote(self, quote: CryptoQuote):
        """Log quote data."""
        self.stats['quotes'] += 1
        quote_data = {
            'type': 'quote',
            'symbol': quote.symbol,
            'bid': quote.bid_price,
            'ask': quote.ask_price,
            'spread': quote.ask_price - quote.bid_price,
            'timestamp': quote.timestamp,
            'received_at': datetime.now(timezone.utc).isoformat()
        }
        self.data_log.append(quote_data)

        # Only log every 10th quote to avoid spam
        if self.stats['quotes'] % 10 == 0:
            print(f"📊 QUOTE: {quote.symbol} Bid: ${quote.bid_price:.2f} Ask: ${quote.ask_price:.2f} Spread: ${quote.ask_price - quote.bid_price:.2f}")

    async def handle_aggregate(self, aggregate: CryptoAggregate):
        """Log aggregate data."""
        self.stats['aggregates'] += 1
        agg_data = {
            'type': 'aggregate',
            'symbol': aggregate.symbol,
            'open': aggregate.open,
            'high': aggregate.high,
            'low': aggregate.low,
            'close': aggregate.close,
            'volume': aggregate.volume,
            'timestamp': aggregate.timestamp,
            'received_at': datetime.now(timezone.utc).isoformat()
        }
        self.data_log.append(agg_data)
        print(f"📈 BAR: {aggregate.symbol} O:${aggregate.open:.2f} H:${aggregate.high:.2f} L:${aggregate.low:.2f} C:${aggregate.close:.2f} V:{aggregate.volume:.4f}")

    def save_results(self):
        """Save test results to file."""
        runtime = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        # Summary
        summary = {
            'test_date': self.start_time.isoformat(),
            'runtime_seconds': runtime,
            'statistics': self.stats,
            'trades_per_second': self.stats['trades'] / runtime if runtime > 0 else 0,
            'quotes_per_second': self.stats['quotes'] / runtime if runtime > 0 else 0,
            'total_messages': sum(self.stats.values()),
            'status': 'SUCCESS' if sum(self.stats.values()) > 0 else 'FAILED'
        }

        # Save summary
        summary_file = DATA_DIR / f"test_summary_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save raw data (first 100 records)
        if self.data_log:
            data_file = DATA_DIR / f"test_data_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(data_file, 'w') as f:
                json.dump(self.data_log[:100], f, indent=2)

        print("\n" + "="*60)
        print("📋 TEST SUMMARY")
        print("="*60)
        print(f"Runtime: {runtime:.1f} seconds")
        print(f"Trades received: {self.stats['trades']} ({self.stats['trades'] / runtime if runtime > 0 else 0:.1f}/sec)")
        print(f"Quotes received: {self.stats['quotes']} ({self.stats['quotes'] / runtime if runtime > 0 else 0:.1f}/sec)")
        print(f"Aggregates received: {self.stats['aggregates']}")
        print(f"Total messages: {sum(self.stats.values())}")
        print(f"Status: {'✅ SUCCESS - Connection working!' if sum(self.stats.values()) > 0 else '❌ FAILED - No data received'}")
        print(f"\nResults saved to: {summary_file}")

        return summary

    async def run_test(self, duration: int = 30):
        """Run the test for specified duration."""
        print("="*60)
        print("🚀 POLYGON.IO WEBSOCKET CONNECTION TEST")
        print("="*60)
        print(f"Testing connection for {duration} seconds...")
        print(f"Subscribing to: BTC-USD, ETH-USD")
        print("="*60 + "\n")

        # Register callbacks
        self.client.register_callback('trade', self.handle_trade)
        self.client.register_callback('quote', self.handle_quote)
        self.client.register_callback('aggregate', self.handle_aggregate)

        # Subscribe to crypto symbols
        await self.client.subscribe(
            symbols=['BTC-USD', 'ETH-USD'],
            channels=['XT', 'XQ', 'XA']  # Trades, Quotes, Aggregates
        )

        # Run for specified duration
        client_task = asyncio.create_task(self.client.run())

        try:
            await asyncio.sleep(duration)
        finally:
            # Stop the client
            self.client.stop()

            # Wait for client to finish
            try:
                await asyncio.wait_for(client_task, timeout=5)
            except asyncio.TimeoutError:
                client_task.cancel()

            # Save and display results
            self.save_results()

async def main():
    """Run the test."""
    tester = TestPolygonStream()

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\n\nInterrupted! Saving results...")
        tester.client.stop()

    signal.signal(signal.SIGINT, signal_handler)

    # Run test
    await tester.run_test(duration=30)

if __name__ == "__main__":
    asyncio.run(main())