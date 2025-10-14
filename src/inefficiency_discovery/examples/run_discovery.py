from backtesting import InefficiencyValidator, BacktestEngine
from datetime import datetime
from orchestration.master_orchestrator import MasterOrchestrator
import asyncio
import logging
import os
import sys

"""
Example: Run the full inefficiency discovery system

This script demonstrates how to use the complete system to discover
novel market inefficiencies.
"""


# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inefficiency_discovery.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


async def main():
    """
    Main execution function
    """
    logger.info("="*60)
    logger.info("Market Inefficiency Discovery System")
    logger.info("RRR Ventures © 2025")
    logger.info("="*60)
    
    # Configuration
    symbols = [
        'BTC-USD',
        'ETH-USD',
        'SOL-USD',
        'MATIC-USD',
        'AVAX-USD'
    ]
    
    logger.info(f"\n📊 Monitoring {len(symbols)} symbols:")
    for symbol in symbols:
        logger.info(f"  • {symbol}")
    
    # Initialize orchestrator
    logger.info("\n🚀 Initializing Master Orchestrator...")
    
    orchestrator = MasterOrchestrator(
        enable_sentiment=True  # Enable Perplexity AI sentiment
    )
    
    logger.info("✅ Orchestrator initialized")
    logger.info(f"   Active detectors: {orchestrator.stats.detectors_active}")
    
    # Initialize validator
    validator = InefficiencyValidator()
    
    try:
        # Start discovery (runs until interrupted)
        logger.info("\n🔍 Starting discovery process...")
        logger.info("   Press Ctrl+C to stop\n")
        
        await orchestrator.start(symbols)
        
    except KeyboardInterrupt:
        logger.info("\n\n⏹️  Stopping system...")
        await orchestrator.stop()
        
        # Print final statistics
        logger.info("\n" + "="*60)
        logger.info("📊 Final Statistics")
        logger.info("="*60)
        
        stats = orchestrator.get_statistics()
        
        logger.info(f"\n⏱️  Runtime:")
        logger.info(f"   Uptime: {stats['uptime_seconds']:.0f} seconds")
        logger.info(f"   Detection cycles: {stats['total_cycles']}")
        
        logger.info(f"\n📈 Signal Discovery:")
        logger.info(f"   Total signals: {stats['total_signals']}")
        logger.info(f"   Significant signals: {stats['significant_signals']}")
        logger.info(f"   Signals per hour: {stats['signals_per_hour']:.1f}")
        
        if stats['signals_by_type']:
            logger.info(f"\n🎯 Signals by Type:")
            for signal_type, count in sorted(stats['signals_by_type'].items(), 
                                            key=lambda x: x[1], reverse=True):
                logger.info(f"   {signal_type}: {count}")
        
        logger.info(f"\n💹 Quality Metrics:")
        logger.info(f"   Avg confidence: {stats['avg_confidence']:.1%}")
        logger.info(f"   Avg expected return: {stats['avg_expected_return']:.2%}")
        
        # Get top signals
        recent_signals = orchestrator.get_recent_signals(top_n=5)
        
        if recent_signals:
            logger.info(f"\n🏆 Top 5 Signals:")
            for i, signal in enumerate(recent_signals, 1):
                logger.info(f"\n   {i}. {signal.inefficiency_type.value.upper()}")
                logger.info(f"      Symbols: {', '.join(signal.symbols)}")
                logger.info(f"      Confidence: {signal.confidence:.1%}")
                logger.info(f"      Expected Return: {signal.expected_return:.2%}")
                logger.info(f"      P-value: {signal.p_value:.4f}")
                logger.info(f"      Direction: {signal.direction}")
                
                # Validate signal
                validation = validator.validate_signal(signal)
                
                if validation['is_valid']:
                    logger.info(f"      ✅ VALIDATED (score: {validation['overall_score']:.2f})")
                    for rec in validation['recommendations']:
                        logger.info(f"         → {rec}")
                else:
                    logger.info(f"      ❌ NOT VALIDATED")
                    for warning in validation['warnings']:
                        logger.info(f"         ⚠️  {warning}")
        
        # Detector statistics
        logger.info(f"\n🔬 Detector Performance:")
        detector_stats = orchestrator.get_detector_statistics()
        
        for detector_name, det_stats in detector_stats.items():
            logger.info(f"\n   {detector_name}:")
            logger.info(f"      Signals: {det_stats.get('total_signals', 0)}")
            logger.info(f"      Significant: {det_stats.get('significant_signals', 0)}")
            if det_stats.get('avg_confidence'):
                logger.info(f"      Avg confidence: {det_stats['avg_confidence']:.1%}")
        
        # Validation statistics
        val_stats = validator.get_validation_statistics()
        if val_stats:
            logger.info(f"\n✅ Validation Summary:")
            logger.info(f"   Total validated: {val_stats['total_validations']}")
            logger.info(f"   Passed: {val_stats['valid_signals']}")
            logger.info(f"   Failed: {val_stats['invalid_signals']}")
            logger.info(f"   Pass rate: {val_stats['validation_rate']:.1%}")
        
        logger.info("\n" + "="*60)
        logger.info("✅ System shutdown complete")
        logger.info("="*60 + "\n")
        
        # Save results
        logger.info("💾 Saving results to inefficiency_results.log")
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        await orchestrator.stop()


def run_backtest_example():
    """
    Example: Run a backtest on discovered signals
    """
    import pandas as pd
    import numpy as np
    from base import InefficiencySignal, InefficiencyType
    from backtesting import BacktestEngine
    
    logger.info("\n" + "="*60)
    logger.info("🧪 Backtest Example")
    logger.info("="*60)
    
    # Generate mock price data
    dates = pd.date_range(start='2024-01-01', end='2025-10-12', freq='1H')
    prices = 50000 + np.cumsum(np.random.normal(0, 100, len(dates)))
    
    price_data = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'BTC-USD',
        'price': prices
    })
    
    # Generate mock signals
    signals = []
    for i in range(0, len(dates), 100):
        signal = InefficiencySignal(
            signal_id=f"signal_{i}",
            timestamp=dates[i],
            inefficiency_type=InefficiencyType.LATENCY_ARBITRAGE,
            symbols=['BTC-USD'],
            confidence=np.random.uniform(0.7, 0.95),
            expected_return=np.random.uniform(0.5, 3.0),
            p_value=np.random.uniform(0.001, 0.05),
            direction='long' if np.random.random() > 0.5 else 'short',
            expected_duration=3600
        )
        signals.append(signal)
    
    # Run backtest
    engine = BacktestEngine(initial_capital=100000)
    
    result = engine.backtest_strategy(
        signals=signals,
        price_data=price_data,
        strategy_name="Latency Arbitrage v1"
    )
    
    if result:
        logger.info(f"\n✅ Backtest completed successfully")
        logger.info(f"   Strategy is {'VIABLE ✅' if result.is_viable() else 'NOT VIABLE ❌'}")


if __name__ == "__main__":
    # Check for command line argument
    if len(sys.argv) > 1 and sys.argv[1] == '--backtest':
        run_backtest_example()
    else:
        # Run main discovery system
        asyncio.run(main())

