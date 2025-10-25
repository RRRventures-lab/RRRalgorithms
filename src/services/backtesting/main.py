from backtest.data import SupabaseDataLoader
from backtest.engine import BacktestEngine
from backtest.metrics import PerformanceCalculator
from backtest.optimization import WalkForwardOptimizer
from backtest.reports import ReportGenerator
from backtest.simulation import MonteCarloSimulator
from backtest.strategies import SimpleMomentumStrategy, MeanReversionStrategy
from datetime import datetime
from pathlib import Path
import argparse
import logging
import os
import sys

#!/usr/bin/env python3
"""
Main Backtesting CLI

Command-line interface for running backtests.

Usage:
    python main.py --strategy momentum --symbol BTC-USD --start 2024-01-01 --end 2024-12-31
"""


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Strategy registry
STRATEGIES = {
    'momentum': SimpleMomentumStrategy,
    'mean_reversion': MeanReversionStrategy,
}


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run cryptocurrency trading strategy backtests'
    )

    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        choices=list(STRATEGIES.keys()),
        help='Strategy to backtest'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default='BTC-USD',
        help='Trading symbol (default: BTC-USD)'
    )

    parser.add_argument(
        '--start',
        type=str,
        required=True,
        help='Start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end',
        type=str,
        required=True,
        help='End date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--timeframe',
        type=str,
        default='1h',
        choices=['1m', '5m', '15m', '1h', '4h', '1d'],
        help='Data timeframe (default: 1h)'
    )

    parser.add_argument(
        '--initial-capital',
        type=float,
        default=100000.0,
        help='Initial capital in USD (default: 100000)'
    )

    parser.add_argument(
        '--commission',
        type=float,
        default=0.001,
        help='Commission rate (default: 0.001 = 0.1%%)'
    )

    parser.add_argument(
        '--slippage',
        type=float,
        default=5.0,
        help='Slippage in basis points (default: 5.0)'
    )

    parser.add_argument(
        '--monte-carlo',
        action='store_true',
        help='Run Monte Carlo simulation'
    )

    parser.add_argument(
        '--n-simulations',
        type=int,
        default=1000,
        help='Number of Monte Carlo simulations (default: 1000)'
    )

    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Run walk-forward optimization'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports',
        help='Output directory for reports (default: reports)'
    )

    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip report generation'
    )

    # Strategy-specific parameters
    parser.add_argument(
        '--fast-period',
        type=int,
        default=20,
        help='Fast MA period for momentum strategy (default: 20)'
    )

    parser.add_argument(
        '--slow-period',
        type=int,
        default=50,
        help='Slow MA period for momentum strategy (default: 50)'
    )

    parser.add_argument(
        '--bb-period',
        type=int,
        default=20,
        help='Bollinger Band period for mean reversion (default: 20)'
    )

    parser.add_argument(
        '--bb-std',
        type=float,
        default=2.0,
        help='Bollinger Band std dev for mean reversion (default: 2.0)'
    )

    return parser.parse_args()


def create_strategy(args):
    """Create strategy instance from arguments."""
    strategy_class = STRATEGIES[args.strategy]

    if args.strategy == 'momentum':
        return strategy_class(
            fast_period=args.fast_period,
            slow_period=args.slow_period
        )
    elif args.strategy == 'mean_reversion':
        return strategy_class(
            period=args.bb_period,
            std_dev=args.bb_std
        )
    else:
        return strategy_class()


def main():
    """Main execution function."""
    args = parse_arguments()

    logger.info("=" * 70)
    logger.info("BACKTESTING FRAMEWORK")
    logger.info("=" * 70)
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Initial Capital: ${args.initial_capital:,.2f}")
    logger.info("=" * 70)

    try:
        # Load data
        logger.info("\n[1/5] Loading historical data...")
        data_loader = SupabaseDataLoader()

        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date = datetime.strptime(args.end, '%Y-%m-%d')

        data = data_loader.load_crypto_data(
            symbol=args.symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=args.timeframe
        )

        if data.empty:
            logger.error("No data loaded. Please check your date range and symbol.")
            return 1

        logger.info(f"Loaded {len(data)} bars")

        # Run optimization if requested
        if args.optimize:
            logger.info("\n[2/5] Running walk-forward optimization...")

            # Define parameter grid based on strategy
            if args.strategy == 'momentum':
                param_grid = {
                    'fast_period': [10, 20, 30],
                    'slow_period': [40, 50, 60]
                }
            elif args.strategy == 'mean_reversion':
                param_grid = {
                    'period': [15, 20, 25],
                    'std_dev': [1.5, 2.0, 2.5]
                }
            else:
                param_grid = {}

            engine = BacktestEngine(
                initial_capital=args.initial_capital,
                commission_rate=args.commission,
                slippage_bps=args.slippage
            )

            optimizer = WalkForwardOptimizer(n_splits=3)
            opt_result = optimizer.optimize_parameters(
                engine=engine,
                strategy_class=STRATEGIES[args.strategy],
                data=data,
                parameter_grid=param_grid,
                symbol=args.symbol
            )

            logger.info(f"Optimal parameters: {opt_result.best_parameters}")

            # Use optimal parameters
            if args.strategy == 'momentum':
                strategy = SimpleMomentumStrategy(**opt_result.best_parameters)
            elif args.strategy == 'mean_reversion':
                strategy = MeanReversionStrategy(**opt_result.best_parameters)
            else:
                strategy = create_strategy(args)
        else:
            logger.info("\n[2/5] Creating strategy...")
            strategy = create_strategy(args)

        # Run backtest
        logger.info("\n[3/5] Running backtest...")
        engine = BacktestEngine(
            initial_capital=args.initial_capital,
            commission_rate=args.commission,
            slippage_bps=args.slippage
        )

        result = engine.run_backtest(strategy, data, args.symbol)

        # Calculate performance metrics
        logger.info("\n[4/5] Calculating performance metrics...")
        calc = PerformanceCalculator(initial_capital=args.initial_capital)
        metrics = calc.calculate_metrics(
            equity_curve=result.equity_curve,
            trades=result.trades,
            start_date=start_date,
            end_date=end_date
        )

        # Print summary
        calc.print_summary(metrics)

        # Run Monte Carlo simulation if requested
        mc_result = None
        if args.monte_carlo and len(result.trades) >= 10:
            logger.info("\n[5/5] Running Monte Carlo simulation...")
            simulator = MonteCarloSimulator(
                n_simulations=args.n_simulations,
                initial_capital=args.initial_capital
            )
            mc_result = simulator.run_simulation(result.trades)
            simulator.print_summary(mc_result)

        # Generate report
        if not args.no_report:
            logger.info("\nGenerating report...")
            report_gen = ReportGenerator(output_dir=args.output_dir)
            report_path = report_gen.generate_report(
                backtest_result=result,
                performance_metrics=metrics,
                monte_carlo_result=mc_result
            )
            logger.info(f"Report saved to: {report_path}")

        logger.info("\n" + "=" * 70)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"Error during backtest: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
