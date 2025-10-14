#!/usr/bin/env python3
"""
RRRalgorithms Unified Entry Point
Single command to start entire trading system.
Replaces Docker Compose with native Python processes.
"""

import asyncio
import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from database import get_db
from core.config import load_config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/system/trading_system.log')
    ]
)
logger = logging.getLogger(__name__)


class TradingSystem:
    """Unified trading system manager."""
    
    def __init__(self, config, mode='paper'):
        """
        Initialize trading system.
        
        Args:
            config: Configuration dictionary
            mode: Trading mode ('paper', 'live', 'backtest')
        """
        self.config = config
        self.mode = mode
        self.running = False
        self.tasks = []
        
        # Components (initialized later)
        self.db_client = None
        self.data_pipeline = None
        self.trading_engine = None
        self.risk_manager = None
        self.dashboard = None
        
        logger.info(f"TradingSystem initialized in {mode} mode")
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing components...")
        
        # Initialize database
        self.db_client = get_db()
        await self.db_client.connect()
        logger.info("✅ Database connected")
        
        # Import and initialize data pipeline
        try:
            from data_pipeline.main import DataPipeline
            self.data_pipeline = DataPipeline(self.config, self.db_client)
            logger.info("✅ Data pipeline initialized")
        except ImportError as e:
            logger.warning(f"Data pipeline not available: {e}")
        
        # Import and initialize trading engine
        try:
            # Try new consolidated structure
            from trading.engine.main import TradingEngine
            self.trading_engine = TradingEngine(
                self.config,
                self.db_client,
                mode=self.mode
            )
            logger.info("✅ Trading engine initialized")
        except ImportError:
            try:
                # Fallback to old structure
                from services.trading_engine.engine.main import TradingEngine
                self.trading_engine = TradingEngine(
                    self.config,
                    self.db_client,
                    mode=self.mode
                )
                logger.info("✅ Trading engine initialized (legacy)")
            except ImportError as e:
                logger.warning(f"Trading engine not available: {e}")
        
        # Import and initialize risk management
        try:
            from trading.risk.main import RiskManager
            self.risk_manager = RiskManager(self.config, self.db_client)
            logger.info("✅ Risk manager initialized")
        except ImportError:
            try:
                from services.risk_management.risk.main import RiskManager
                self.risk_manager = RiskManager(self.config, self.db_client)
                logger.info("✅ Risk manager initialized (legacy)")
            except ImportError as e:
                logger.warning(f"Risk manager not available: {e}")
        
        logger.info("All components initialized successfully")
    
    async def start_data_pipeline(self):
        """Start data pipeline."""
        if not self.data_pipeline:
            logger.warning("Data pipeline not available, skipping")
            return
        
        logger.info("Starting data pipeline...")
        try:
            await self.data_pipeline.run()
        except Exception as e:
            logger.error(f"Data pipeline error: {e}", exc_info=True)
    
    async def start_trading_engine(self):
        """Start trading engine."""
        if not self.trading_engine:
            logger.warning("Trading engine not available, skipping")
            return
        
        logger.info(f"Starting trading engine in {self.mode} mode...")
        try:
            await self.trading_engine.run()
        except Exception as e:
            logger.error(f"Trading engine error: {e}", exc_info=True)
    
    async def start_risk_manager(self):
        """Start risk manager."""
        if not self.risk_manager:
            logger.warning("Risk manager not available, skipping")
            return
        
        logger.info("Starting risk manager...")
        try:
            await self.risk_manager.run()
        except Exception as e:
            logger.error(f"Risk manager error: {e}", exc_info=True)
    
    async def start_dashboard(self):
        """Start monitoring dashboard."""
        logger.info("Dashboard will be started separately with: streamlit run src/dashboards/mobile_dashboard.py")
        # Dashboard runs in separate process via Streamlit
    
    async def run(self):
        """Run all components concurrently."""
        self.running = True
        logger.info("="*60)
        logger.info("🚀 RRRalgorithms Trading System Starting")
        logger.info("="*60)
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Time: {datetime.now()}")
        logger.info("="*60)
        
        # Initialize all components
        await self.initialize()
        
        # Create tasks for each component
        self.tasks = []
        
        if self.data_pipeline:
            self.tasks.append(
                asyncio.create_task(self.start_data_pipeline())
            )
        
        if self.trading_engine:
            self.tasks.append(
                asyncio.create_task(self.start_trading_engine())
            )
        
        if self.risk_manager:
            self.tasks.append(
                asyncio.create_task(self.start_risk_manager())
            )
        
        logger.info(f"Started {len(self.tasks)} components")
        
        try:
            # Run all tasks concurrently
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            logger.info("System shutdown requested")
        except Exception as e:
            logger.error(f"System error: {e}", exc_info=True)
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down trading system...")
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Close database connection
        if self.db_client:
            await self.db_client.disconnect()
            logger.info("✅ Database disconnected")
        
        logger.info("="*60)
        logger.info("Trading system shutdown complete")
        logger.info("="*60)


def setup_signal_handlers(system: TradingSystem):
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(system.shutdown())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='RRRalgorithms Unified Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Paper trading
  python src/main_unified.py --mode paper

  # Live trading (after validation)
  python src/main_unified.py --mode live

  # Backtesting
  python src/main_unified.py --mode backtest

  # With custom config
  python src/main_unified.py --mode paper --config config/trading_config.yml
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['paper', 'live', 'backtest'],
        default='paper',
        help='Trading mode (default: paper)'
    )
    
    parser.add_argument(
        '--config',
        default='config/trading_config.yml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Start dashboard (opens in browser)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(args.log_level)
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        # Use minimal default config
        config = {
            'trading_mode': args.mode,
            'database': {'type': 'sqlite', 'path': 'data/db/trading.db'},
        }
    
    # Create and run system
    system = TradingSystem(config, mode=args.mode)
    setup_signal_handlers(system)
    
    # Start dashboard if requested
    if args.dashboard:
        import subprocess
        logger.info("Starting dashboard...")
        dashboard_process = subprocess.Popen([
            'streamlit', 'run',
            'src/dashboards/mobile_dashboard.py',
            '--server.port', '8501',
            '--server.headless', 'true'
        ])
        logger.info("Dashboard started at http://localhost:8501")
    
    # Run system
    try:
        await system.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

