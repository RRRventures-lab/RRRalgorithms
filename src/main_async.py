from pathlib import Path
from src.core.async_trading_engine import AsyncTradingEngine
from src.core.config.loader import get_config, config_get
from src.core.constants import TradingConstants, RiskConstants
from src.core.database.local_db import get_db, LocalDatabase
from src.data_pipeline.mock_data_source import MockDataSource
from src.monitoring.local_monitor import LocalMonitor
from src.neural_network.mock_predictor import EnsemblePredictor
from typing import Optional, Dict, Any
import asyncio
import signal
import sys

#!/usr/bin/env python3
"""
RRRalgorithms - Async Entry Point for High-Performance Trading
Async version of the main trading system with parallel processing.

Usage:
    python -m src.main_async
"""


# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))



class AsyncTradingSystem:
    """
    High-performance async trading system controller.
    Manages all services with parallel processing for maximum efficiency.
    """
    
    def __init__(self) -> None:
        """Initialize async trading system."""
        self.config = get_config()
        self.db: LocalDatabase = get_db()
        self.monitor: LocalMonitor = LocalMonitor()
        
        self.running: bool = False
        self.services: Dict[str, Any] = {}
        self.engine: Optional[AsyncTradingEngine] = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        self.monitor.log('INFO', 'Shutdown signal received, stopping services...')
        # Get the running event loop safely
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.stop())
        except RuntimeError:
            # No event loop running, set flag for later
            self.running = False
    
    async def initialize(self) -> None:
        """Initialize all services."""
        self.monitor.log('INFO', 'Initializing RRRalgorithms Async Trading System...')
        
        # Check configuration
        env = self.config.environment
        self.monitor.log('INFO', f'Environment: {env}')
        self.monitor.log('INFO', f'Database: {config_get("database.type", "sqlite")}')
        
        # Initialize database
        self.monitor.log('INFO', 'Initializing database...')
        # Database is auto-initialized by get_db()
        self.monitor.log('SUCCESS', '✓ Database ready')
        
        # Initialize services based on config
        enabled_services = config_get('services.enabled_services', [])
        self.monitor.log('INFO', f'Enabled services: {", ".join(enabled_services)}')
        
        if 'data_pipeline' in enabled_services:
            await self._init_data_pipeline()
        
        if 'trading_engine' in enabled_services:
            await self._init_trading_engine()
        
        if 'risk_management' in enabled_services:
            await self._init_risk_management()
        
        self.monitor.log('SUCCESS', '✓ All services initialized')
    
    async def _init_data_pipeline(self) -> None:
        """Initialize data pipeline service."""
        mode: str = config_get('data_pipeline.mode', 'mock')
        
        if mode == 'mock':
            symbols = config_get('data_pipeline.mock.symbols', ['BTC-USD', 'ETH-USD'])
            volatility = config_get('data_pipeline.mock.volatility', 0.02)
            
            self.services['data_pipeline'] = MockDataSource(
                symbols=symbols,
                volatility=volatility
            )
            self.monitor.log('SUCCESS', f'✓ Data pipeline initialized (mock mode, {len(symbols)} symbols)')
        else:
            self.monitor.log('WARNING', f'Data mode "{mode}" not implemented yet, using mock')
            self.services['data_pipeline'] = MockDataSource()
        
        self.monitor.update_service_status('data_pipeline', 'running')
    
    async def _init_trading_engine(self) -> None:
        """Initialize trading engine service."""
        mode: str = config_get('trading.mode', 'paper')
        initial_capital: float = config_get('trading.initial_capital', 10000)
        
        # For now, just mark as initialized
        # Full trading engine implementation would go here
        self.services['trading_engine'] = {
            'mode': mode,
            'capital': initial_capital,
            'positions': {}
        }
        
        self.monitor.log('SUCCESS', f'✓ Trading engine initialized ({mode} mode, ${initial_capital:,.0f})')
        self.monitor.update_service_status('trading_engine', 'running')
    
    async def _init_risk_management(self) -> None:
        """Initialize risk management service."""
        max_position_size: float = config_get(
            'trading.max_position_size',
            TradingConstants.MAX_POSITION_SIZE_PCT
        )
        max_daily_loss: float = config_get(
            'trading.max_daily_loss',
            RiskConstants.MAX_DAILY_LOSS_PCT
        )
        
        self.services['risk_management'] = {
            'max_position_size': max_position_size,
            'max_daily_loss': max_daily_loss
        }
        
        self.monitor.log('SUCCESS', '✓ Risk management initialized')
        self.monitor.update_service_status('risk_management', 'running')
    
    async def start(self) -> None:
        """Start the async trading system."""
        self.monitor.log('INFO', 'Starting async trading system...')
        self.running = True
        
        # Initialize ML predictor
        nn_mode = config_get('neural_network.mode', 'mock')
        if nn_mode == 'mock':
            predictor = EnsemblePredictor()
        else:
            self.monitor.log('WARNING', f'Neural network mode "{nn_mode}" not implemented, using mock')
            predictor = EnsemblePredictor()
        
        # Get symbols from data pipeline
        symbols = config_get('data_pipeline.mock.symbols', ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD'])
        data_source = self.services['data_pipeline']
        
        # Create async trading engine
        update_interval = config_get(
            'data_pipeline.mock.update_interval',
            TradingConstants.DEFAULT_UPDATE_INTERVAL_SEC
        )
        max_concurrency = config_get('trading.max_concurrency', 10)
        
        self.engine = AsyncTradingEngine(
            symbols=symbols,
            data_source=data_source,
            predictor=predictor,
            db=self.db,
            monitor=self.monitor,
            update_interval=update_interval,
            max_concurrency=max_concurrency
        )
        
        # Start the async trading engine
        self.monitor.log('SUCCESS', '✓ Async trading system started')
        self.monitor.log('INFO', 'Press Ctrl+C to stop\n')
        
        try:
            await self.engine.start()
        except KeyboardInterrupt:
            self.monitor.log('INFO', 'Keyboard interrupt received')
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the async trading system."""
        self.monitor.log('INFO', 'Stopping async trading system...')
        self.running = False
        
        # Stop async trading engine
        if self.engine:
            await self.engine.stop()
            self.engine = None
        
        # Stop all services
        for service_name in self.services:
            self.monitor.update_service_status(service_name, 'stopped')
            self.monitor.log('INFO', f'✓ Stopped {service_name}')
        
        # Close database
        self.db.close()
        
        self.monitor.log('SUCCESS', '✓ Async trading system stopped gracefully')
    
    def status(self) -> None:
        """Display system status."""
        self.monitor.display()


async def main() -> None:
    """Main async entry point."""
    # Create system
    system = AsyncTradingSystem()
    
    try:
        # Initialize and start
        await system.initialize()
        await system.start()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await system.stop()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())