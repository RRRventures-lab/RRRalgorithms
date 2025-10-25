from datetime import datetime
from dotenv import load_dotenv
from engine.exchanges import PaperExchange
from engine.executor import StrategyExecutor
from engine.oms import OrderManager
from engine.portfolio import PortfolioManager
from engine.positions import PositionManager
from functools import lru_cache
import argparse
import asyncio
import logging
import os
import sys


"""
Trading Engine Main Loop
Orchestrates all components and runs the trading system
"""


# Import from worktree packages


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_engine.log"),
    ],
)

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Main Trading Engine
    Coordinates all components and runs the trading loop
    """

    def __init__(
        self,
        mode: str = "paper",
        initial_capital: float = 100000.0,
        max_position_size: float = 0.20,
        max_daily_loss: float = 0.05,
    ):
        """
        Initialize Trading Engine

        Args:
            mode: Trading mode ("paper" or "live")
            initial_capital: Initial capital
            max_position_size: Max position size as % of portfolio
            max_daily_loss: Max daily loss as % of portfolio
        """
        self.mode = mode
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss

        # Load environment variables
        load_dotenv(os.path.join(os.path.dirname(__file__), "../../../../config/api-keys/.env"))

        self.supabase_url = os.getenv("DATABASE_PATH")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")

        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase credentials not found in environment")

        # Components
        self.exchange = None
        self.order_manager = None
        self.position_manager = None
        self.portfolio_manager = None
        self.strategy_executor = None

        # State
        self.running = False
        self.loop_interval_seconds = 5  # Check for signals every 5 seconds

        logger.info(
            f"Initialized Trading Engine in {mode.upper()} mode with "
            f"capital={initial_capital}, max_position={max_position_size*100}%, "
            f"max_daily_loss={max_daily_loss*100}%"
        )

    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing trading engine components...")

        try:
            # Initialize exchange
            if self.mode == "paper":
                self.exchange = PaperExchange(
                    exchange_id="paper_exchange",
                    initial_balance=self.initial_capital,
                )
            else:
                # Live trading implementation with comprehensive safety checks
                logger.warning("=" * 70)
                logger.warning("LIVE TRADING MODE REQUESTED")
                logger.warning("=" * 70)

                # Import credentials manager for safety validation
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), "security"))
                from credentials_manager import get_credentials_manager

                # Get credentials manager
                creds_manager = get_credentials_manager()

                # CRITICAL: Perform comprehensive safety checks
                is_safe, warnings = creds_manager.validate_live_trading_safety()

                if not is_safe:
                    logger.critical("LIVE TRADING SAFETY CHECKS FAILED:")
                    for warning in warnings:
                        logger.critical(f"  - {warning}")
                    logger.critical("=" * 70)
                    raise RuntimeError(
                        "Live trading safety validation failed. Please review configuration:\n"
                        "1. Set PAPER_TRADING=false in config/api-keys/.env.coinbase\n"
                        "2. Set LIVE_TRADING_ENABLED=true\n"
                        "3. Set ENVIRONMENT=production\n"
                        "4. Configure all risk limits (MAX_ORDER_SIZE_USD, etc.)\n"
                        "5. Ensure Coinbase API credentials are correct\n\n"
                        "USE PAPER TRADING MODE FOR TESTING: --mode paper"
                    )

                logger.warning("Safety checks passed - proceeding with live trading")
                logger.warning("Real money will be used for orders!")
                logger.warning("=" * 70)

                # Import live exchange connectors
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), "exchanges"))
                from coinbase_exchange import CoinbaseExchange

                # Get Coinbase credentials
                coinbase_creds = creds_manager.get_coinbase_credentials()

                # Initialize Coinbase exchange in LIVE mode
                self.exchange = CoinbaseExchange(paper_trading=False)

                logger.info("Live Coinbase exchange initialized")
                logger.info(f"Organization: {coinbase_creds['organization_id']}")
                logger.info(f"Risk limits: {creds_manager.get_risk_limits()}")

            await self.exchange.connect()
            logger.info(f"Exchange connected: {self.exchange.exchange_id}")

            # Initialize managers
            self.position_manager = PositionManager(
                supabase_url=self.supabase_url,
                supabase_key=self.supabase_key,
            )
            logger.info("Position Manager initialized")

            self.portfolio_manager = PortfolioManager(
                supabase_url=self.supabase_url,
                supabase_key=self.supabase_key,
                position_manager=self.position_manager,
                initial_capital=self.initial_capital,
            )
            logger.info("Portfolio Manager initialized")

            self.order_manager = OrderManager(
                exchange=self.exchange,
                supabase_url=self.supabase_url,
                supabase_key=self.supabase_key,
            )
            logger.info("Order Manager initialized")

            self.strategy_executor = StrategyExecutor(
                order_manager=self.order_manager,
                position_manager=self.position_manager,
                portfolio_manager=self.portfolio_manager,
                supabase_url=self.supabase_url,
                supabase_key=self.supabase_key,
                max_position_size=self.max_position_size,
                max_daily_loss=self.max_daily_loss,
            )
            logger.info("Strategy Executor initialized")

            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}", exc_info=True)
            raise

    async def run(self):
        """Run the main trading loop"""
        logger.info("Starting trading engine...")
        self.running = True

        # Start portfolio snapshot loop in background
        snapshot_task = asyncio.create_task(self.portfolio_manager.start_snapshot_loop())

        try:
            while self.running:
                await self._trading_loop_iteration()
                await asyncio.sleep(self.loop_interval_seconds)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down...")
            self.running = False

        except Exception as e:
            logger.error(f"Fatal error in trading loop: {e}", exc_info=True)
            self.running = False

        finally:
            # Cleanup - properly handle task cancellation
            snapshot_task.cancel()
            try:
                # Wait for the task to be cancelled
                await snapshot_task
            except asyncio.CancelledError:
                # This is expected when cancelling a task
                logger.debug("Snapshot task cancelled successfully")
            except Exception as e:
                logger.warning(f"Error during snapshot task cancellation: {e}")

            await self.shutdown()

    async def _trading_loop_iteration(self):
        """Single iteration of the trading loop"""
        try:
            iteration_start = datetime.utcnow()
            logger.debug("=== Trading Loop Iteration ===")

            # 1. Update market prices (fetch from exchange or data feed)
            market_prices = await self._fetch_market_prices()

            # 2. Update portfolio with current prices
            await self.portfolio_manager.update_portfolio(market_prices)

            # 3. Process new trading signals
            results = await self.strategy_executor.process_signals()

            if results:
                logger.info(f"Processed {len(results)} signals")
                for result in results:
                    if result.get("success"):
                        logger.info(
                            f"Signal {result['signal_id']}: {result['action']} executed successfully"
                        )
                    else:
                        logger.warning(
                            f"Signal {result['signal_id']}: execution failed - {result.get('reason')}"
                        )

            # 4. Log portfolio status
            portfolio = self.portfolio_manager.portfolio
            logger.info(
                f"Portfolio: Value=${portfolio['total_value']:.2f}, "
                f"P&L=${portfolio['total_pnl']:.2f} ({portfolio['total_return_pct']:.2f}%), "
                f"Cash=${portfolio['cash']:.2f}, Equity=${portfolio['equity']:.2f}"
            )

            # 5. Check risk limits
            risk_check = await self.portfolio_manager.check_risk_limits(
                self.max_position_size, self.max_daily_loss
            )

            if not risk_check["within_limits"]:
                logger.warning(f"Risk limits exceeded: {risk_check}")

            iteration_duration = (datetime.utcnow() - iteration_start).total_seconds()
            logger.debug(f"Iteration completed in {iteration_duration:.2f}s")

        except Exception as e:
            logger.error(f"Error in trading loop iteration: {e}", exc_info=True)

    async def _fetch_market_prices(self) -> dict:
        """
        Fetch current market prices for all positions

        Returns:
            Dict of symbol -> price
        """
        try:
            # Get all open positions
            positions = await self.position_manager.get_all_open_positions()

            market_prices = {}

            for position in positions:
                symbol = position["symbol"]
                # Fetch price from exchange
                price = await self.exchange.get_current_price(symbol)
                market_prices[symbol] = price

            return market_prices

        except Exception as e:
            logger.error(f"Failed to fetch market prices: {e}", exc_info=True)
            return {}

    async def shutdown(self):
        """Shutdown the trading engine gracefully"""
        logger.info("Shutting down trading engine...")

        try:
            # Take final portfolio snapshot
            await self.portfolio_manager.take_snapshot()

            # Disconnect from exchange
            if self.exchange:
                await self.exchange.disconnect()

            logger.info("Trading engine shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)

    def get_status(self) -> dict:
        """Get current status of the trading engine"""
        return {
            "running": self.running,
            "mode": self.mode,
            "exchange_connected": self.exchange.connected if self.exchange else False,
            "portfolio": self.portfolio_manager.portfolio if self.portfolio_manager else None,
        }


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Trading Engine")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["paper", "live"],
        default="paper",
        help="Trading mode (paper or live)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000.0,
        help="Initial capital",
    )
    parser.add_argument(
        "--max-position",
        type=float,
        default=0.20,
        help="Maximum position size as fraction of portfolio (0.20 = 20%%)",
    )
    parser.add_argument(
        "--max-daily-loss",
        type=float,
        default=0.05,
        help="Maximum daily loss as fraction of portfolio (0.05 = 5%%)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # Verify paper trading for safety
    if args.mode == "live":
        logger.error("LIVE TRADING IS NOT YET IMPLEMENTED AND DISABLED FOR SAFETY")
        logger.error("Please use --mode paper for paper trading")
        sys.exit(1)

    # Print startup banner
    logger.info("=" * 60)
    logger.info("RRRalgorithms Trading Engine")
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Initial Capital: ${args.capital:,.2f}")
    logger.info(f"Max Position Size: {args.max_position*100:.1f}%")
    logger.info(f"Max Daily Loss: {args.max_daily_loss*100:.1f}%")
    logger.info("=" * 60)

    # Create and run engine
    engine = TradingEngine(
        mode=args.mode,
        initial_capital=args.capital,
        max_position_size=args.max_position,
        max_daily_loss=args.max_daily_loss,
    )

    await engine.initialize()
    await engine.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
