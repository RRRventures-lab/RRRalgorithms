from datetime import datetime, timedelta
from pathlib import Path
from report_generator import HypothesisReportGenerator
from test_cex_dex_arbitrage import CEXDEXArbitrageHypothesis
from test_orderbook_imbalance import OrderBookImbalanceHypothesis
from test_whale_tracking import WhaleTrackingHypothesis
import asyncio
import sys

"""
Master script to run all hypothesis tests in Phase 2A.

This script executes all three priority hypothesis tests in sequence,
collects results, and generates a comprehensive comparison report to
identify which strategies should be KILLED, ITERATED, or SCALED.
"""


sys.path.append(str(Path(__file__).parent))



async def run_all_hypothesis_tests():
    """
    Run all three hypothesis tests and generate comparison report.

    Returns:
        Dict with all test results
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "PHASE 2A: HYPOTHESIS TESTING")
    print(" " * 15 + "Testing 3 High-Priority Market Hypotheses")
    print("=" * 80)

    # Define test period (last 6 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    print(f"\nTest Period: {start_date.date()} to {end_date.date()}")
    print(f"Duration: 180 days\n")

    # Initialize all testers
    testers = [
        ("H003", "CEX-DEX Arbitrage", CEXDEXArbitrageHypothesis(), 810),
        ("H002", "Order Book Imbalance", OrderBookImbalanceHypothesis(), 720),
        ("H001", "Whale Tracking", WhaleTrackingHypothesis(), 640)
    ]

    results = []
    reports = []

    # Run each test sequentially (could be parallelized in production)
    for i, (hypothesis_id, title, tester, priority) in enumerate(testers, 1):
        print("\n" + "-" * 80)
        print(f"Test {i}/3: {hypothesis_id} - {title} (Priority: {priority})")
        print("-" * 80)

        try:
            # Run the test
            report = await tester.execute_full_pipeline(
                start_date=start_date,
                end_date=end_date
            )

            reports.append(report)

            # Collect summary results
            result = {
                'id': hypothesis_id,
                'title': title,
                'priority': priority,
                'decision': report.decision.decision,
                'confidence': report.decision.confidence,
                'sharpe_ratio': report.backtest_results.sharpe_ratio,
                'sortino_ratio': report.backtest_results.sortino_ratio,
                'win_rate': report.backtest_results.win_rate,
                'total_return': report.backtest_results.total_return,
                'annual_return': report.backtest_results.annual_return,
                'max_drawdown': report.backtest_results.max_drawdown,
                'total_trades': report.backtest_results.total_trades,
                'profit_factor': report.backtest_results.profit_factor,
                'p_value': report.statistical_validation.p_value,
                'significant': report.statistical_validation.significant,
                'execution_time': report.execution_time
            }

            results.append(result)

            # Print quick summary
            print(f"\n{hypothesis_id} Results:")
            print(f"  Decision: {report.decision.decision} (Confidence: {report.decision.confidence:.0%})")
            print(f"  Sharpe: {report.backtest_results.sharpe_ratio:.2f}")
            print(f"  Win Rate: {report.backtest_results.win_rate:.1%}")
            print(f"  Return: {report.backtest_results.total_return:.2%}")
            print(f"  Significant: {report.statistical_validation.significant}")

        except Exception as e:
            print(f"\n❌ ERROR testing {hypothesis_id}: {e}")
            import traceback
            traceback.print_exc()

            # Add failed result
            results.append({
                'id': hypothesis_id,
                'title': title,
                'priority': priority,
                'decision': 'FAILED',
                'confidence': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0,
                'p_value': 1.0,
                'significant': False,
                'execution_time': 0.0
            })

    # Generate comparison report
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON REPORT")
    print("=" * 80)

    report_dir = Path("/Volumes/Lexar/RRRVentures/RRRalgorithms/research/results")
    report_gen = HypothesisReportGenerator(report_dir)

    comparison_report_path = report_gen.generate_comparison_report(
        results,
        output_file="phase_2a_comparison_report.md"
    )

    # Print final summary
    print("\n" + "=" * 80)
    print(" " * 25 + "FINAL SUMMARY")
    print("=" * 80)

    # Count decisions
    decisions = {'SCALE': [], 'ITERATE': [], 'KILL': [], 'FAILED': []}
    for result in results:
        decision = result['decision']
        decisions[decision].append(result)

    # Print decision summary
    print(f"\n📊 Decision Summary:")
    print(f"  ✅ SCALE (Ready for Production): {len(decisions['SCALE'])}")
    print(f"  ⚠️  ITERATE (Needs Refinement): {len(decisions['ITERATE'])}")
    print(f"  ❌ KILL (Not Viable): {len(decisions['KILL'])}")
    if decisions['FAILED']:
        print(f"  ⛔ FAILED (Errors): {len(decisions['FAILED'])}")

    # Print strategies to scale
    if decisions['SCALE']:
        print(f"\n✅ Strategies Ready for Production:")
        for result in sorted(decisions['SCALE'], key=lambda x: x['sharpe_ratio'], reverse=True):
            print(f"  • {result['id']} - {result['title']}")
            print(f"    Sharpe: {result['sharpe_ratio']:.2f} | "
                  f"Win Rate: {result['win_rate']:.1%} | "
                  f"Return: {result['total_return']:.1%}")
    else:
        print(f"\n⚠️  No strategies ready for immediate production")

    # Print strategies to iterate
    if decisions['ITERATE']:
        print(f"\n⚠️  Strategies That Need Refinement:")
        for result in decisions['ITERATE']:
            print(f"  • {result['id']} - {result['title']}")
            print(f"    Sharpe: {result['sharpe_ratio']:.2f} | "
                  f"Shows promise but needs work")

    # Print strategies to kill
    if decisions['KILL']:
        print(f"\n❌ Strategies Not Viable:")
        for result in decisions['KILL']:
            print(f"  • {result['id']} - {result['title']}")
            print(f"    Sharpe: {result['sharpe_ratio']:.2f} | "
                  f"Statistical significance: {result['significant']}")

    # Print recommendations
    print(f"\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if len(decisions['SCALE']) >= 2:
        print(f"\n✅ SUCCESS: {len(decisions['SCALE'])} strategies ready for production!")
        print(f"\nNext Steps:")
        print(f"  1. Proceed to Phase 2B: Strategy Implementation")
        print(f"  2. Implement production versions of SCALE strategies")
        print(f"  3. Set up paper trading for validation")
        print(f"  4. Build monitoring dashboards")
    elif len(decisions['SCALE']) == 1:
        print(f"\n⚠️  PARTIAL SUCCESS: 1 strategy ready, could test more hypotheses")
        print(f"\nNext Steps:")
        print(f"  1. Implement the winning strategy (Phase 2B)")
        print(f"  2. OR test 2-3 additional hypotheses to find more opportunities")
        print(f"  3. Focus on refining ITERATE strategies")
    else:
        print(f"\n❌ NO STRATEGIES READY: Need to refine approach")
        print(f"\nNext Steps:")
        print(f"  1. Analyze why strategies failed")
        print(f"  2. Improve feature engineering for ITERATE strategies")
        print(f"  3. Test 3-5 additional hypotheses")
        print(f"  4. Consider adjusting thresholds and parameters")

    # Print report location
    print(f"\n" + "=" * 80)
    print(f"📄 Detailed comparison report saved to:")
    print(f"   {comparison_report_path}")
    print("=" * 80 + "\n")

    return {
        'results': results,
        'reports': reports,
        'decisions': decisions,
        'comparison_report_path': comparison_report_path
    }


async def main():
    """Main entry point."""
    start_time = datetime.now()

    # Run all tests
    output = await run_all_hypothesis_tests()

    # Calculate total execution time
    execution_time = (datetime.now() - start_time).total_seconds()

    print(f"\n⏱️  Total Execution Time: {execution_time:.1f} seconds ({execution_time/60:.1f} minutes)")

    # Save summary to JSON
    import json
    from pathlib import Path

    summary_file = Path("/Volumes/Lexar/RRRVentures/RRRalgorithms/research/results/phase_2a_summary.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        'execution_date': datetime.now().isoformat(),
        'execution_time_seconds': execution_time,
        'results': output['results'],
        'decision_counts': {
            'SCALE': len(output['decisions']['SCALE']),
            'ITERATE': len(output['decisions']['ITERATE']),
            'KILL': len(output['decisions']['KILL']),
            'FAILED': len(output['decisions']['FAILED'])
        }
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n💾 Summary saved to: {summary_file}")

    return output


if __name__ == "__main__":
    # Run all tests
    output = asyncio.run(main())

    # Print Phase 2A completion status
    print("\n" + "🎉" * 40)
    print("\n" + " " * 30 + "PHASE 2A COMPLETE")
    print("\n" + "🎉" * 40 + "\n")
