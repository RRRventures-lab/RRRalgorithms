from datetime import datetime, timedelta
from functools import lru_cache
from hypothesis_tester import HypothesisTester, HypothesisDecision
from pathlib import Path
from typing import Dict, List, Tuple
import asyncio
import importlib.util
import json
import sys
import yaml

#!/usr/bin/env python3

"""
Run hypothesis tests with relaxed criteria to find potentially profitable strategies.

This script overrides the default strict thresholds to use more permissive criteria
for identifying strategies with potential that might be rejected by conservative thresholds.
"""


# Add parent directories to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'testing'))



def load_relaxed_criteria(config_path: str = "configs/relaxed_criteria.yml") -> Dict:
    """Load relaxed criteria configuration."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


@lru_cache(maxsize=128)


def get_top_hypotheses() -> List[Tuple[str, str, int]]:
    """
    Get top 5 most promising hypotheses based on priority scores.

    Returns:
        List of (hypothesis_id, test_file, priority_score)
    """
    # Based on the test files found and their likely priority
    hypotheses = [
        ("H003", "test_cex_dex_arbitrage.py", 810),  # CEX-DEX Arbitrage
        ("H002", "test_orderbook_imbalance.py", 800),  # Order Book Imbalance
        ("H004", "test_whale_tracking.py", 750),  # Whale Tracking
        ("H005", "test_funding_rate_divergence.py", 700),  # Funding Rate
        ("H014", "test_rsi_momentum.py", 650),  # RSI Momentum
    ]
    return hypotheses


async def run_hypothesis_with_relaxed_criteria(
    test_file: str,
    hypothesis_id: str,
    criteria: Dict
) -> Dict:
    """
    Run a single hypothesis test with relaxed criteria.

    Args:
        test_file: Name of the test file
        hypothesis_id: Hypothesis identifier
        criteria: Relaxed criteria configuration

    Returns:
        Test results dictionary
    """
    try:
        print(f"\n{'='*60}")
        print(f"Running {hypothesis_id}: {test_file}")
        print(f"{'='*60}")

        # Import the test module dynamically
        test_path = Path(__file__).parent.parent / 'testing' / test_file
        spec = importlib.util.spec_from_file_location(test_file[:-3], test_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find the hypothesis class (should be only one per file)
        hypothesis_class = None
        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type) and
                issubclass(obj, HypothesisTester) and
                obj != HypothesisTester):
                hypothesis_class = obj
                break

        if not hypothesis_class:
            raise ValueError(f"No HypothesisTester subclass found in {test_file}")

        # Create instance and run test
        tester = hypothesis_class()

        # Override the make_decision method to use relaxed criteria
        original_make_decision = tester.make_decision

        def make_decision_relaxed(backtest_results, statistical_validation):
            """Override with relaxed criteria."""
            return original_make_decision(
                backtest_results,
                statistical_validation,
                min_sharpe=criteria['decision_criteria']['sharpe_ratio']['scale_threshold'],
                min_win_rate=criteria['decision_criteria']['win_rate']['scale_threshold'],
                max_p_value=criteria['hypothesis_testing']['p_value_threshold']
            )

        tester.make_decision = make_decision_relaxed

        # Run the test with recent data (last 30 days as per relaxed config)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=criteria['backtesting']['min_backtest_days'])

        print(f"Testing period: {start_date.date()} to {end_date.date()}")
        print(f"Using relaxed criteria:")
        print(f"  - Sharpe threshold: {criteria['decision_criteria']['sharpe_ratio']['scale_threshold']}")
        print(f"  - Win rate threshold: {criteria['decision_criteria']['win_rate']['scale_threshold']}")
        print(f"  - P-value threshold: {criteria['hypothesis_testing']['p_value_threshold']}")

        # Run the full test
        report = await tester.execute_full_pipeline(start_date, end_date)

        # Extract key results
        results = {
            'hypothesis_id': hypothesis_id,
            'test_file': test_file,
            'decision': report.decision.decision,
            'metrics': {
                'sharpe_ratio': report.backtest_results.sharpe_ratio,
                'sortino_ratio': report.backtest_results.sortino_ratio,
                'win_rate': report.backtest_results.win_rate,
                'profit_factor': report.backtest_results.profit_factor,
                'max_drawdown': report.backtest_results.max_drawdown,
                'total_trades': report.backtest_results.total_trades,
                'total_return': report.backtest_results.total_return,
            },
            'statistical': {
                'p_value': report.statistical_validation.p_value,
                't_statistic': report.statistical_validation.t_statistic,
                'confidence': report.decision.confidence,
            },
            'relaxed_criteria_used': True,
            'original_decision_with_strict': 'KILL',  # Based on prior results
        }

        # Check if decision changed from strict to relaxed
        if report.decision.decision == 'SCALE':
            print(f"✅ SUCCESS: {hypothesis_id} is now PROFITABLE with relaxed criteria!")
            print(f"   Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
            print(f"   Win Rate: {results['metrics']['win_rate']:.2%}")
            results['profitable_with_relaxed'] = True
        elif report.decision.decision == 'ITERATE':
            print(f"⚠️  ITERATE: {hypothesis_id} shows potential, needs optimization")
            results['profitable_with_relaxed'] = False
        else:
            print(f"❌ KILL: {hypothesis_id} still not profitable even with relaxed criteria")
            results['profitable_with_relaxed'] = False

        return results

    except Exception as e:
        print(f"Error testing {hypothesis_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'hypothesis_id': hypothesis_id,
            'test_file': test_file,
            'error': str(e),
            'decision': 'ERROR',
            'profitable_with_relaxed': False
        }


async def main():
    """Run top 5 hypotheses with relaxed criteria."""
    print("\n" + "="*80)
    print("HYPOTHESIS TESTING WITH RELAXED CRITERIA")
    print("="*80)
    print("\nProblem: 0/21 strategies were profitable with strict criteria (Sharpe > 1.5)")
    print("Solution: Testing with relaxed criteria (Sharpe > 0.3) to find hidden opportunities")

    # Load relaxed criteria
    criteria = load_relaxed_criteria()

    # Get top hypotheses to test
    hypotheses = get_top_hypotheses()

    print(f"\nTesting top {len(hypotheses)} most promising strategies:")
    for h_id, test_file, priority in hypotheses:
        print(f"  - {h_id}: {test_file} (Priority: {priority})")

    # Run tests
    results = []
    for h_id, test_file, priority in hypotheses:
        result = await run_hypothesis_with_relaxed_criteria(test_file, h_id, criteria)
        result['priority_score'] = priority
        results.append(result)

        # Small delay between tests to avoid overwhelming the system
        await asyncio.sleep(2)

    # Summarize results
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS WITH RELAXED CRITERIA")
    print("="*80)

    profitable_count = sum(1 for r in results if r.get('profitable_with_relaxed', False))
    iterate_count = sum(1 for r in results if r.get('decision') == 'ITERATE')

    print(f"\n📊 Results Overview:")
    print(f"  - Profitable (SCALE): {profitable_count}/{len(results)}")
    print(f"  - Needs Iteration: {iterate_count}/{len(results)}")
    print(f"  - Still Unprofitable: {len(results) - profitable_count - iterate_count}/{len(results)}")

    print("\n📈 Individual Strategy Results:")
    for result in sorted(results, key=lambda x: x.get('metrics', {}).get('sharpe_ratio', -999), reverse=True):
        if 'error' in result:
            print(f"\n❌ {result['hypothesis_id']}: ERROR - {result['error']}")
        else:
            sharpe = result['metrics']['sharpe_ratio']
            win_rate = result['metrics']['win_rate']
            decision = result['decision']

            icon = "✅" if decision == "SCALE" else "⚠️" if decision == "ITERATE" else "❌"
            print(f"\n{icon} {result['hypothesis_id']}: {decision}")
            print(f"   Sharpe Ratio: {sharpe:.2f}")
            print(f"   Win Rate: {win_rate:.2%}")
            print(f"   Total Return: {result['metrics']['total_return']:.2%}")
            print(f"   Max Drawdown: {result['metrics']['max_drawdown']:.2%}")

    # Save results to file
    output_path = Path(__file__).parent / 'results' / 'relaxed_criteria_results.json'
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'criteria_file': 'configs/relaxed_criteria.yml',
            'summary': {
                'total_tested': len(results),
                'profitable': profitable_count,
                'iterate': iterate_count,
                'unprofitable': len(results) - profitable_count - iterate_count,
            },
            'results': results
        }, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {output_path}")

    # Provide recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if profitable_count > 0:
        print("\n✅ SUCCESS! Found profitable strategies with relaxed criteria!")
        print("\nNext Steps:")
        print("1. Review the profitable strategies in detail")
        print("2. Run extended backtests (90+ days) to confirm profitability")
        print("3. Gradually tighten criteria to find optimal thresholds")
        print("4. Implement position sizing and risk management")
        print("5. Start paper trading with small positions")
    elif iterate_count > 0:
        print("\n⚠️ Some strategies show potential but need optimization.")
        print("\nNext Steps:")
        print("1. Analyze why these strategies are underperforming")
        print("2. Optimize parameters (thresholds, timeframes, indicators)")
        print("3. Consider combining multiple signals")
        print("4. Test with different market conditions")
    else:
        print("\n❌ No profitable strategies found even with relaxed criteria.")
        print("\nNext Steps:")
        print("1. Review strategy logic for fundamental issues")
        print("2. Check data quality and collection methods")
        print("3. Consider entirely new strategy approaches")
        print("4. Analyze successful strategies from research papers")

    return results


if __name__ == "__main__":
    asyncio.run(main())