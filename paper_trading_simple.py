#!/usr/bin/env python3
"""
Simplified Paper Trading System
Runs a basic paper trading loop for testing deployment
"""

import asyncio
import json
import logging
import os
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/trading/paper_trading.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


class SimplePaperTrader:
    """Simple paper trading implementation for testing."""
    
    def __init__(self):
        self.running = False
        self.positions: Dict[str, float] = {}
        self.portfolio_value = 100000.0  # Start with $100k
        self.cash = 100000.0
        self.trades_executed = 0
        self.start_time = time.time()
        
        # Trading symbols
        self.symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
        
        # Mock prices
        self.prices = {
            'BTC-USD': 45000.0,
            'ETH-USD': 2800.0,
            'SOL-USD': 100.0
        }
        
        # Performance tracking
        self.performance = {
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'trades': [],
            'portfolio_history': []
        }
        
        logger.info("SimplePaperTrader initialized")
        logger.info(f"Starting portfolio value: ${self.portfolio_value:,.2f}")
    
    async def start(self):
        """Start paper trading."""
        logger.info("Starting paper trading...")
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Main trading loop
        try:
            while self.running:
                await self._trading_iteration()
                await asyncio.sleep(5)  # Trade every 5 seconds for testing
                
        except asyncio.CancelledError:
            logger.info("Trading cancelled")
        except Exception as e:
            logger.error(f"Trading error: {e}")
        finally:
            await self.shutdown()
    
    async def _trading_iteration(self):
        """Single trading iteration."""
        try:
            # Update mock prices (random walk)
            self._update_prices()
            
            # Calculate portfolio value
            self._calculate_portfolio_value()
            
            # Make trading decision
            await self._make_trading_decision()
            
            # Log status
            if self.trades_executed % 10 == 0:
                self._log_status()
            
            # Save performance
            if self.trades_executed % 50 == 0:
                self._save_performance()
                
        except Exception as e:
            logger.error(f"Trading iteration error: {e}")
    
    def _update_prices(self):
        """Update mock prices with random walk."""
        for symbol in self.symbols:
            # Random walk: +/- 0.5%
            change = random.uniform(-0.005, 0.005)
            self.prices[symbol] *= (1 + change)
    
    def _calculate_portfolio_value(self):
        """Calculate current portfolio value."""
        positions_value = sum(
            qty * self.prices.get(symbol, 0)
            for symbol, qty in self.positions.items()
        )
        self.portfolio_value = self.cash + positions_value
        
        # Track portfolio history
        self.performance['portfolio_history'].append({
            'timestamp': datetime.now().isoformat(),
            'value': self.portfolio_value,
            'cash': self.cash,
            'positions_value': positions_value
        })
    
    async def _make_trading_decision(self):
        """Make a simple trading decision."""
        # Simple momentum strategy
        symbol = random.choice(self.symbols)
        current_price = self.prices[symbol]
        
        # Random decision for testing
        action = random.choice(['buy', 'sell', 'hold'])
        
        if action == 'buy' and self.cash > 0:
            # Buy with 10% of cash
            amount = self.cash * 0.1
            quantity = amount / current_price
            
            await self._execute_trade(symbol, 'buy', quantity, current_price)
            
        elif action == 'sell' and symbol in self.positions:
            # Sell 50% of position
            quantity = self.positions[symbol] * 0.5
            
            await self._execute_trade(symbol, 'sell', quantity, current_price)
    
    async def _execute_trade(self, symbol: str, side: str, quantity: float, price: float):
        """Execute a paper trade."""
        cost = quantity * price
        
        if side == 'buy':
            if cost > self.cash:
                return  # Not enough cash
            
            self.cash -= cost
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
            
        elif side == 'sell':
            if symbol not in self.positions or self.positions[symbol] < quantity:
                return  # Not enough to sell
            
            self.cash += cost
            self.positions[symbol] -= quantity
            
            if self.positions[symbol] < 0.001:
                del self.positions[symbol]
        
        # Record trade
        trade = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'cost': cost,
            'portfolio_value': self.portfolio_value
        }
        
        self.performance['trades'].append(trade)
        self.trades_executed += 1
        
        logger.info(f"Trade executed: {side.upper()} {quantity:.4f} {symbol} @ ${price:.2f}")
    
    def _log_status(self):
        """Log current status."""
        pnl = self.portfolio_value - 100000
        pnl_pct = (pnl / 100000) * 100
        
        logger.info("=" * 50)
        logger.info(f"Portfolio Value: ${self.portfolio_value:,.2f}")
        logger.info(f"P&L: ${pnl:,.2f} ({pnl_pct:.2f}%)")
        logger.info(f"Cash: ${self.cash:,.2f}")
        logger.info(f"Trades Executed: {self.trades_executed}")
        
        if self.positions:
            logger.info("Positions:")
            for symbol, qty in self.positions.items():
                value = qty * self.prices[symbol]
                logger.info(f"  {symbol}: {qty:.4f} (${value:,.2f})")
        
        logger.info("=" * 50)
    
    def _save_performance(self):
        """Save performance to file."""
        try:
            # Calculate metrics
            self.performance['total_pnl'] = self.portfolio_value - 100000
            
            if len(self.performance['trades']) > 0:
                # Simple win rate calculation
                winning_trades = sum(
                    1 for i, trade in enumerate(self.performance['trades'])
                    if i > 0 and trade['portfolio_value'] > self.performance['trades'][i-1]['portfolio_value']
                )
                self.performance['win_rate'] = winning_trades / len(self.performance['trades'])
            
            # Save to file
            perf_file = Path("data/performance/paper_trading_performance.json")
            perf_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(perf_file, 'w') as f:
                json.dump(self.performance, f, indent=2)
            
            logger.info(f"Performance saved to {perf_file}")
            
        except Exception as e:
            logger.error(f"Error saving performance: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    async def shutdown(self):
        """Clean shutdown."""
        logger.info("Shutting down paper trading...")
        
        # Final status
        self._log_status()
        
        # Save final performance
        self._save_performance()
        
        # Save final state
        state = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions': self.positions,
            'trades_executed': self.trades_executed,
            'runtime_seconds': time.time() - self.start_time
        }
        
        state_file = Path("data/state/paper_trading_state.json")
        state_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Final state saved to {state_file}")
        logger.info("Paper trading shutdown complete")


class MetricsExporter:
    """Export metrics for Prometheus monitoring."""
    
    def __init__(self, trader: SimplePaperTrader, port: int = 8000):
        self.trader = trader
        self.port = port
        
        # Import prometheus client
        try:
            from prometheus_client import Gauge, Counter, start_http_server
            
            # Define metrics
            self.portfolio_value = Gauge('trading_portfolio_value', 'Total portfolio value')
            self.cash_balance = Gauge('trading_cash_balance', 'Cash balance')
            self.pnl_total = Gauge('trading_pnl_total', 'Total P&L')
            self.trades_total = Counter('trading_trades_total', 'Total trades executed')
            self.position_count = Gauge('trading_position_count', 'Number of open positions')
            
            # Start HTTP server
            start_http_server(self.port)
            logger.info(f"Metrics server started on port {self.port}")
            
            self.enabled = True
            
        except ImportError:
            logger.warning("prometheus_client not installed, metrics disabled")
            self.enabled = False
    
    def update_metrics(self):
        """Update Prometheus metrics."""
        if not self.enabled:
            return
        
        self.portfolio_value.set(self.trader.portfolio_value)
        self.cash_balance.set(self.trader.cash)
        self.pnl_total.set(self.trader.portfolio_value - 100000)
        self.position_count.set(len(self.trader.positions))
        
        # Counter is automatically incremented
        if self.trader.trades_executed > 0:
            self.trades_total._value._value = self.trader.trades_executed


async def main():
    """Main entry point."""
    # Create directories
    for dir_path in ['logs/trading', 'data/performance', 'data/state']:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Initialize trader
    trader = SimplePaperTrader()
    
    # Initialize metrics exporter
    metrics = MetricsExporter(trader)
    
    # Start metrics update loop
    async def update_metrics_loop():
        while trader.running:
            metrics.update_metrics()
            await asyncio.sleep(5)
    
    # Run trader and metrics updater
    if metrics.enabled:
        await asyncio.gather(
            trader.start(),
            update_metrics_loop()
        )
    else:
        await trader.start()


if __name__ == "__main__":
    logger.info("Starting Simple Paper Trading System...")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Paper trading interrupted")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
