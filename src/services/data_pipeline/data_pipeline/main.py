from data_pipeline.perplexity.sentiment_analyzer import PerplexitySentimentAnalyzer
from data_pipeline.polygon.websocket_client import PolygonWebSocketClient
from data_pipeline.quality.validator import DataQualityValidator
from src.database import SQLiteClient as DatabaseClient
from datetime import datetime
from functools import lru_cache
from typing import List, Optional
import argparse
import asyncio
import logging
import signal
import sys


"""
Main Data Pipeline Orchestrator
================================

This is the main entry point for running the complete data pipeline system.

Components:
1. Polygon.io WebSocket streaming (real-time trades, quotes, aggregates)
2. Perplexity AI sentiment analysis (scheduled every 15 minutes)
3. Data quality validation (scheduled every 5 minutes)

All components run concurrently using asyncio.

Usage:
    # Run all components
    python src/data_pipeline/main.py

    # Run specific components
    python src/data_pipeline/main.py --no-websocket  # Skip WebSocket
    python src/data_pipeline/main.py --no-sentiment  # Skip sentiment
    python src/data_pipeline/main.py --no-quality    # Skip quality checks

    # Custom configuration
    python src/data_pipeline/main.py --tickers X:BTCUSD X:ETHUSD

    # Backfill historical data first
    python src/data_pipeline/backfill/historical.py --months 6
"""



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


class DataPipelineOrchestrator:
    """
    Main orchestrator for the data pipeline system.

    Manages and coordinates:
    - Real-time WebSocket streaming
    - Sentiment analysis
    - Data quality monitoring
    """

    def __init__(
        self,
        supabase_client: SupabaseClient,
        tickers: Optional[List[str]] = None,
        enable_websocket: bool = True,
        enable_sentiment: bool = True,
        enable_quality: bool = True,
        sentiment_interval: int = 900,  # 15 minutes
        quality_interval: int = 300,    # 5 minutes
    ):
        """
        Initialize the data pipeline orchestrator.

        Args:
            supabase_client: SupabaseClient instance
            tickers: List of tickers to track
            enable_websocket: Enable WebSocket streaming
            enable_sentiment: Enable sentiment analysis
            enable_quality: Enable quality monitoring
            sentiment_interval: Sentiment update interval (seconds)
            quality_interval: Quality check interval (seconds)
        """
        self.supabase_client = supabase_client
        self.tickers = tickers

        self.enable_websocket = enable_websocket
        self.enable_sentiment = enable_sentiment
        self.enable_quality = enable_quality

        # Initialize components
        self.websocket_client = None
        self.sentiment_analyzer = None
        self.quality_validator = None

        if enable_websocket:
            self.websocket_client = PolygonWebSocketClient(
                supabase_client=supabase_client,
                pairs=tickers,
            )
            logger.info("WebSocket client initialized")

        if enable_sentiment:
            # Convert ticker format for sentiment (X:BTCUSD -> BTC)
            sentiment_assets = None
            if tickers:
                sentiment_assets = [
                    ticker.replace("X:", "").replace("USD", "")
                    for ticker in tickers
                ]

            self.sentiment_analyzer = PerplexitySentimentAnalyzer(
                supabase_client=supabase_client,
                assets=sentiment_assets,
                update_interval=sentiment_interval,
            )
            logger.info("Sentiment analyzer initialized")

        if enable_quality:
            self.quality_validator = DataQualityValidator(
                supabase_client=supabase_client,
                tickers=tickers,
                check_interval=quality_interval,
            )
            logger.info("Quality validator initialized")

        self.running = False
        self.tasks = []

        logger.info("Data pipeline orchestrator initialized")

    async def start(self):
        """Start all enabled pipeline components."""
        self.running = True

        logger.info("=" * 70)
        logger.info("STARTING DATA PIPELINE")
        logger.info("=" * 70)
        logger.info(f"WebSocket streaming: {self.enable_websocket}")
        logger.info(f"Sentiment analysis: {self.enable_sentiment}")
        logger.info(f"Quality monitoring: {self.enable_quality}")
        logger.info(f"Monitored tickers: {self.tickers or 'All'}")
        logger.info("=" * 70)

        # Log startup event
        self.supabase_client.log_system_event(
            event_type="pipeline_start",
            severity="info",
            message="Data pipeline started",
            component="orchestrator",
            metadata={
                "websocket": self.enable_websocket,
                "sentiment": self.enable_sentiment,
                "quality": self.enable_quality,
                "tickers": self.tickers,
            }
        )

        # Start all components concurrently
        if self.websocket_client:
            task = asyncio.create_task(self.websocket_client.run())
            self.tasks.append(("websocket", task))
            logger.info("WebSocket streaming started")

        if self.sentiment_analyzer:
            task = asyncio.create_task(self.sentiment_analyzer.run_scheduled())
            self.tasks.append(("sentiment", task))
            logger.info("Sentiment analysis started")

        if self.quality_validator:
            task = asyncio.create_task(self.quality_validator.run_monitoring())
            self.tasks.append(("quality", task))
            logger.info("Quality monitoring started")

        logger.info(f"All {len(self.tasks)} components running")

        # Wait for all tasks
        try:
            await asyncio.gather(*[task for _, task in self.tasks])
        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
            await self.stop()

    async def stop(self):
        """Stop all pipeline components gracefully."""
        logger.info("Stopping data pipeline...")
        self.running = False

        # Stop all components
        if self.websocket_client:
            await self.websocket_client.stop()

        if self.sentiment_analyzer:
            await self.sentiment_analyzer.stop()

        if self.quality_validator:
            await self.quality_validator.stop()

        # Cancel all tasks
        for name, task in self.tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"{name} task cancelled")

        # Log shutdown event
        self.supabase_client.log_system_event(
            event_type="pipeline_stop",
            severity="info",
            message="Data pipeline stopped",
            component="orchestrator",
        )

        logger.info("Data pipeline stopped")

    @lru_cache(maxsize=128)

    def get_stats(self) -> dict:
        """Get statistics from all components."""
        stats = {
            "running": self.running,
            "start_time": datetime.now().isoformat(),
        }

        if self.websocket_client:
            stats["websocket"] = self.websocket_client.get_stats()

        if self.sentiment_analyzer:
            stats["sentiment"] = self.sentiment_analyzer.get_stats()

        if self.quality_validator:
            stats["quality"] = self.quality_validator.get_stats()

        return stats


# =============================================================================
# Signal Handlers
# =============================================================================

orchestrator = None


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    logger.info(f"\nReceived signal {signum}")
    if orchestrator:
        asyncio.create_task(orchestrator.stop())
    sys.exit(0)


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    """Main entry point for data pipeline."""
    global orchestrator

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run the cryptocurrency data pipeline system"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Specific tickers to track (e.g., X:BTCUSD X:ETHUSD)"
    )
    parser.add_argument(
        "--no-websocket",
        action="store_true",
        help="Disable WebSocket streaming"
    )
    parser.add_argument(
        "--no-sentiment",
        action="store_true",
        help="Disable sentiment analysis"
    )
    parser.add_argument(
        "--no-quality",
        action="store_true",
        help="Disable quality monitoring"
    )
    parser.add_argument(
        "--sentiment-interval",
        type=int,
        default=900,
        help="Sentiment analysis interval in seconds (default: 900 = 15 min)"
    )
    parser.add_argument(
        "--quality-interval",
        type=int,
        default=300,
        help="Quality check interval in seconds (default: 300 = 5 min)"
    )

    args = parser.parse_args()

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Initialize Supabase client
        logger.info("Initializing Supabase connection...")
        supabase = get_db()
        logger.info("Supabase connection established")

        # Initialize orchestrator
        orchestrator = DataPipelineOrchestrator(
            supabase_client=supabase,
            tickers=args.tickers,
            enable_websocket=not args.no_websocket,
            enable_sentiment=not args.no_sentiment,
            enable_quality=not args.no_quality,
            sentiment_interval=args.sentiment_interval,
            quality_interval=args.quality_interval,
        )

        # Start the pipeline
        await orchestrator.start()

    except KeyboardInterrupt:
        logger.info("\nReceived keyboard interrupt")
        if orchestrator:
            await orchestrator.stop()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        if orchestrator:
            await orchestrator.stop()
        sys.exit(1)


if __name__ == "__main__":
    """
    Run the data pipeline.

    Example:
        # Run everything
        python src/data_pipeline/main.py

        # Track only BTC and ETH
        python src/data_pipeline/main.py --tickers X:BTCUSD X:ETHUSD

        # Run only WebSocket streaming (no sentiment or quality)
        python src/data_pipeline/main.py --no-sentiment --no-quality

        # Custom intervals
        python src/data_pipeline/main.py --sentiment-interval 1800 --quality-interval 600
    """
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  RRRalgorithms - Cryptocurrency Data Pipeline                ║
    ║  Real-time Streaming | Sentiment Analysis | Quality Control  ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    asyncio.run(main())
