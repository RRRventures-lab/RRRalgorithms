from pathlib import Path
from src.core.config.loader import get_config, config_get
from src.core.constants import TradingConstants, RiskConstants
from src.core.database.local_db import get_db, LocalDatabase
from src.data_pipeline.mock_data_source import MockDataSource
from src.monitoring.local_monitor import LocalMonitor
from src.neural_network.mock_predictor import MockPredictor, EnsemblePredictor
from typing import Optional, Dict, Any
import argparse
import asyncio
import signal
import sys
import time

#!/usr/bin/env python3
"""
RRRalgorithms - Unified Entry Point for Local Development
Single entry point to run the entire trading system or individual services.
"""


# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))



class TradingSystem:
    """
    Main trading system controller.
    Manages all services in a single process for efficient local development.
    """
    
    def __init__(self) -> None:
        """Initialize trading system."""
        self.config = get_config()
        self.db: LocalDatabase = get_db()
        self.monitor: LocalMonitor = LocalMonitor()
        
        self.running: bool = False
        self.services: Dict[str, Any] = {}
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        self.monitor.log('INFO', 'Shutdown signal received, stopping services...')
        self.stop()
        sys.exit(0)
    
    def initialize(self) -> None:
        """Initialize all services."""
        self.monitor.log('INFO', 'Initializing RRRalgorithms Trading System...')
        
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
            self._init_data_pipeline()
        
        if 'trading_engine' in enabled_services:
            self._init_trading_engine()
        
        if 'risk_management' in enabled_services:
            self._init_risk_management()
        
        self.monitor.log('SUCCESS', '✓ All services initialized')
    
    def _init_data_pipeline(self) -> None:
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
    
    def _init_trading_engine(self) -> None:
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
    
    def _init_risk_management(self) -> None:
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
    
    def start(self) -> None:
        """Start the trading system."""
        self.monitor.log('INFO', 'Starting trading system...')
        self.running = True
        
        # Initialize ML predictor
        nn_mode = config_get('neural_network.mode', 'mock')
        if nn_mode == 'mock':
            predictor = EnsemblePredictor()
        else:
            self.monitor.log('WARNING', f'Neural network mode "{nn_mode}" not implemented, using mock')
            predictor = EnsemblePredictor()
        
        # Main trading loop
        self.monitor.log('SUCCESS', '✓ Trading system started')
        self.monitor.log('INFO', 'Press Ctrl+C to stop\n')
        
        try:
            self._run_trading_loop(predictor)
        except KeyboardInterrupt:
            self.monitor.log('INFO', 'Keyboard interrupt received')
        finally:
            self.stop()
    
    def _run_trading_loop(self, predictor: EnsemblePredictor) -> None:
        """Main trading loop."""
        iteration: int = 0
        initial_capital: float = config_get('trading.initial_capital', 10000.0)
        portfolio_value: float = initial_capital
        
        # Update initial portfolio state
        self.monitor.update_portfolio(
            value=portfolio_value,
            cash=portfolio_value,
            pnl=0.0,
            daily_pnl=0.0
        )
        
        while self.running:
            iteration += 1
            
            # Get latest market data
            if 'data_pipeline' in self.services:
                data_source = self.services['data_pipeline']
                market_data = data_source.get_latest_data()
                
                # Process each symbol
                for symbol, ohlcv in market_data.items():
                    current_price = ohlcv['close']
                    
                    # Get prediction
                    prediction = predictor.predict(symbol, current_price)
                    
                    # Log prediction (every 10 iterations to avoid spam)
                    if iteration % 10 == 0:
                        self.monitor.log(
                            'INFO',
                            f"{symbol}: ${current_price:,.2f} → "
                            f"${prediction['predicted_price']:,.2f} "
                            f"({prediction['direction']}, "
                            f"confidence={prediction['confidence']:.1%})"
                        )
                    
                    # Store in database
                    self.db.insert_market_data(symbol, ohlcv['timestamp'], {
                        'open': ohlcv['open'],
                        'high': ohlcv['high'],
                        'low': ohlcv['low'],
                        'close': ohlcv['close'],
                        'volume': ohlcv['volume']
                    })
                    
                    # Store prediction
                    self.db.insert_prediction({
                        'symbol': symbol,
                        'timestamp': prediction['timestamp'],
                        'horizon': 1,
                        'predicted_price': prediction['predicted_price'],
                        'predicted_direction': prediction['direction'],
                        'confidence': prediction['confidence'],
                        'model_version': prediction.get('model_type', 'unknown')
                    })
                
                # Simulate portfolio changes (random walk for demo)
                import random
                # Set seed for reproducible testing in development mode
                if self.config.environment == 'development':
                    random.seed(42)
                daily_return = random.gauss(0.0005, 0.01)  # 0.05% daily return, 1% vol
                portfolio_value *= (1 + daily_return)
                pnl = portfolio_value - initial_capital
                
                # Update monitor (every 5 iterations)
                if iteration % 5 == 0:
                    self.monitor.update_portfolio(
                        value=portfolio_value,
                        cash=portfolio_value * 0.5,  # Assume 50% cash
                        pnl=pnl,
                        daily_pnl=pnl  # Simplified
                    )
            
            # Sleep based on config
            update_interval: float = config_get(
                'data_pipeline.mock.update_interval',
                TradingConstants.DEFAULT_UPDATE_INTERVAL_SEC
            )
            time.sleep(update_interval)
        
    def stop(self) -> None:
        """Stop the trading system."""
        self.monitor.log('INFO', 'Stopping services...')
        self.running = False
        
        # Stop all services
        for service_name in self.services:
            self.monitor.update_service_status(service_name, 'stopped')
            self.monitor.log('INFO', f'✓ Stopped {service_name}')
        
        # Close database
        self.db.close()
        
        self.monitor.log('SUCCESS', '✓ Trading system stopped gracefully')
    
    def status(self) -> None:
        """Display system status."""
        self.monitor.display()


def run_service(service_name: str) -> None:
    """Run a specific service independently."""
    print(f"Starting {service_name} service...")
    
    if service_name == "data_pipeline":
        from src.data_pipeline.mock_data_source import MockDataSource
        data_source = MockDataSource()
        print("Data pipeline running. Press Ctrl+C to stop.")
        try:
            data_source.stream(lambda data: print(f"Update: {data}"))
        except KeyboardInterrupt:
            print("\nStopped.")
    
    elif service_name == "monitor":
        monitor = LocalMonitor()
        monitor.update_service_status("data_pipeline", "running")
        monitor.update_service_status("trading_engine", "running")
        monitor.display_live()
    
    else:
        print(f"Unknown service: {service_name}")
        print("Available services: data_pipeline, trading_engine, risk_management, monitor")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='RRRalgorithms Trading System - Local Development',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full trading system
  python -m src.main
  
  # Run specific service
  python -m src.main --service data_pipeline
  
  # Show status only
  python -m src.main --status
  
  # Initialize database
  python -m src.main --init-db
        """
    )
    
    parser.add_argument(
        '--service',
        type=str,
        help='Run specific service (data_pipeline, trading_engine, monitor)'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show system status and exit'
    )
    
    parser.add_argument(
        '--init-db',
        action='store_true',
        help='Initialize database and exit'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file (overrides ENVIRONMENT variable)'
    )
    
    args = parser.parse_args()
    
    # Handle init-db
    if args.init_db:
        print("Initializing database...")
        from scripts.setup.init_local_db import init_database
        init_database(with_sample_data=True)
        return
    
    # Create system
    system = TradingSystem()
    
    # Handle status
    if args.status:
        system.initialize()
        system.status()
        return
    
    # Handle specific service
    if args.service:
        run_service(args.service)
        return
    
    # Run full system
    system.initialize()
    system.start()


if __name__ == "__main__":
    main()

