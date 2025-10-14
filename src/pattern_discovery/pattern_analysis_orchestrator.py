from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import asyncio
import json
import logging
import numpy as np
import pandas as pd

"""
Pattern Analysis Orchestrator
==============================

Coordinates parallel pattern discovery across all downloaded cryptocurrency data.
Uses multiple subagents to analyze 12+ million bars for statistically significant patterns.

Author: RRR Ventures
Date: 2025-10-11
"""


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatternAnalysisOrchestrator:
    """
    Orchestrates parallel pattern discovery across all assets and timeframes
    """

    def __init__(self, data_dir: str = "/Volumes/Lexar/RRRVentures/RRRalgorithms/worktrees/data-pipeline"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path("/Volumes/Lexar/RRRVentures/RRRalgorithms/results/pattern_discovery")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.cryptocurrencies = [
            "BTC", "ETH", "SOL", "ADA", "DOT",
            "MATIC", "AVAX", "ATOM", "LINK", "UNI"
        ]

        self.timeframes = ["1min", "5min", "15min", "1hr", "4hr", "1day"]

        self.pattern_database = []
        self.statistics_summary = {}

    def load_historical_data(self, crypto: str, timeframe: str) -> pd.DataFrame:
        """
        Load historical data for a cryptocurrency and timeframe

        Searches for data in multiple formats (CSV, JSON, Supabase)
        """
        logger.info(f"Loading {crypto} {timeframe} data...")

        # Try multiple file locations and formats
        possible_paths = [
            self.data_dir / f"data/historical/{crypto.lower()}usd/{crypto.lower()}usd_{timeframe}.csv",
            self.data_dir / f"data/json_backfill/{crypto.lower()}usd_{timeframe}.json",
            self.data_dir / f"{crypto.lower()}_data/{crypto.lower()}_{timeframe}.csv",
        ]

        for path in possible_paths:
            if path.exists():
                logger.info(f"Found data at: {path}")

                if path.suffix == '.csv':
                    df = pd.read_csv(path)
                elif path.suffix == '.json':
                    df = pd.read_json(path)
                else:
                    continue

                # Ensure required columns exist
                required_cols = ['close', 'high', 'low', 'open', 'volume']
                if all(col in df.columns for col in required_cols):
                    logger.info(f"Loaded {len(df)} bars for {crypto} {timeframe}")
                    return df

        logger.warning(f"No data found for {crypto} {timeframe}")
        return pd.DataFrame()

    async def analyze_single_asset_timeframe(self, crypto: str, timeframe: str) -> Dict:
        """
        Analyze patterns for a single cryptocurrency and timeframe

        This is the task that will be delegated to parallel agents
        """
        logger.info(f"Analyzing patterns for {crypto} {timeframe}...")

        # Load data
        df = self.load_historical_data(crypto, timeframe)

        if df.empty:
            return {
                'crypto': crypto,
                'timeframe': timeframe,
                'status': 'no_data',
                'patterns_found': []
            }

        # Import pattern detectors
        import sys
        sys.path.append('/Volumes/Lexar/RRRVentures/RRRalgorithms/src')
        from pattern_discovery.pattern_detector import (
            PricePatternDetector,
            TechnicalIndicatorPatternDetector,
            MarketRegimeClassifier,
            calculate_pattern_statistics
        )

        # Detect price patterns
        price_detector = PricePatternDetector()
        price_patterns = price_detector.detect_all_patterns(df)

        # Detect technical indicator patterns
        tech_detector = TechnicalIndicatorPatternDetector()
        ma_crossovers = tech_detector.detect_ma_crossover(df['close'].values)
        rsi_divergences = tech_detector.detect_rsi_divergence(df['close'].values)

        # Classify market regimes
        regime_classifier = MarketRegimeClassifier()
        regimes = regime_classifier.classify_regime(df)

        # Calculate statistics for each pattern type
        pattern_results = []

        # Statistics for price patterns
        for pattern in price_patterns:
            stats = calculate_pattern_statistics([pattern.parameters], df)
            pattern_results.append({
                'pattern_type': pattern.pattern_type,
                'pattern_name': pattern.name,
                'crypto': crypto,
                'timeframe': timeframe,
                'statistics': stats
            })

        # Statistics for MA crossovers
        if ma_crossovers:
            stats = calculate_pattern_statistics(ma_crossovers, df)
            pattern_results.append({
                'pattern_type': 'technical',
                'pattern_name': 'MA Crossover',
                'crypto': crypto,
                'timeframe': timeframe,
                'statistics': stats
            })

        # Statistics for RSI divergences
        if rsi_divergences:
            stats = calculate_pattern_statistics(rsi_divergences, df)
            pattern_results.append({
                'pattern_type': 'technical',
                'pattern_name': 'RSI Divergence',
                'crypto': crypto,
                'timeframe': timeframe,
                'statistics': stats
            })

        result = {
            'crypto': crypto,
            'timeframe': timeframe,
            'status': 'success',
            'total_bars_analyzed': len(df),
            'patterns_found': pattern_results,
            'price_patterns_count': len(price_patterns),
            'ma_crossovers_count': len(ma_crossovers),
            'rsi_divergences_count': len(rsi_divergences),
            'regime_distribution': regimes.value_counts().to_dict()
        }

        # Save individual result
        result_file = self.results_dir / f"{crypto}_{timeframe}_patterns.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"Completed analysis for {crypto} {timeframe}")
        logger.info(f"Found {len(pattern_results)} pattern types")

        return result

    async def run_parallel_pattern_analysis(self):
        """
        Run pattern analysis across all assets and timeframes in parallel

        Creates analysis tasks for all 60 combinations (10 cryptos × 6 timeframes)
        """
        logger.info("=" * 80)
        logger.info("STARTING PARALLEL PATTERN ANALYSIS")
        logger.info("=" * 80)

        start_time = datetime.now()

        # Create tasks for all combinations
        tasks = []
        for crypto in self.cryptocurrencies:
            for timeframe in self.timeframes:
                task = self.analyze_single_asset_timeframe(crypto, timeframe)
                tasks.append(task)

        logger.info(f"Created {len(tasks)} analysis tasks")
        logger.info("Executing in parallel...")

        # Run all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful = 0
        failed = 0
        total_patterns = 0

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Analysis failed: {result}")
                failed += 1
                continue

            if result.get('status') == 'success':
                successful += 1
                total_patterns += len(result.get('patterns_found', []))
            else:
                failed += 1

        duration = (datetime.now() - start_time).total_seconds()

        # Generate summary
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'duration_seconds': duration,
            'total_combinations': len(tasks),
            'successful_analyses': successful,
            'failed_analyses': failed,
            'total_patterns_discovered': total_patterns,
            'results': [r for r in results if not isinstance(r, Exception)]
        }

        # Save summary
        summary_file = self.results_dir / "pattern_analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("=" * 80)
        logger.info("PATTERN ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        logger.info(f"Successful: {successful}/{len(tasks)}")
        logger.info(f"Total patterns discovered: {total_patterns}")
        logger.info(f"Summary saved to: {summary_file}")

        return summary

    def rank_patterns_by_significance(self, min_sharpe: float = 2.0,
                                      min_win_rate: float = 0.55,
                                      min_occurrences: int = 30,
                                      max_p_value: float = 0.01) -> List[Dict]:
        """
        Rank all discovered patterns by statistical significance

        Returns top patterns that meet all criteria
        """
        logger.info("Ranking patterns by statistical significance...")

        # Load all pattern results
        all_patterns = []

        for result_file in self.results_dir.glob("*_patterns.json"):
            with open(result_file, 'r') as f:
                result = json.load(f)

            if result.get('status') == 'success':
                patterns = result.get('patterns_found', [])
                all_patterns.extend(patterns)

        logger.info(f"Loaded {len(all_patterns)} total pattern occurrences")

        # Filter by criteria
        significant_patterns = []

        for pattern in all_patterns:
            stats = pattern.get('statistics', {})

            if (stats.get('sharpe_ratio', 0) >= min_sharpe and
                stats.get('win_rate', 0) >= min_win_rate and
                stats.get('occurrences', 0) >= min_occurrences and
                stats.get('p_value', 1) <= max_p_value):

                significant_patterns.append({
                    **pattern,
                    'significance_score': (
                        stats['sharpe_ratio'] * stats['win_rate'] *
                        np.log(stats['occurrences']) / (stats['p_value'] + 0.001)
                    )
                })

        # Sort by significance score
        significant_patterns.sort(key=lambda x: x['significance_score'], reverse=True)

        logger.info(f"Found {len(significant_patterns)} statistically significant patterns")

        # Save ranked patterns
        ranked_file = self.results_dir / "ranked_significant_patterns.json"
        with open(ranked_file, 'w') as f:
            json.dump(significant_patterns[:50], f, indent=2, default=str)  # Top 50

        logger.info(f"Top 50 patterns saved to: {ranked_file}")

        return significant_patterns

    def generate_pattern_discovery_report(self):
        """Generate comprehensive pattern discovery report"""
        logger.info("Generating pattern discovery report...")

        report = {
            'title': 'Pattern Discovery Report',
            'generated_at': datetime.now().isoformat(),
            'summary': {},
            'top_patterns': [],
            'by_crypto': {},
            'by_timeframe': {},
            'by_pattern_type': {}
        }

        # Load summary
        summary_file = self.results_dir / "pattern_analysis_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            report['summary'] = summary

        # Load ranked patterns
        ranked_file = self.results_dir / "ranked_significant_patterns.json"
        if ranked_file.exists():
            with open(ranked_file, 'r') as f:
                top_patterns = json.load(f)
            report['top_patterns'] = top_patterns[:10]  # Top 10

        # Save report
        report_file = self.results_dir / "pattern_discovery_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Generate markdown report
        md_report = self._generate_markdown_report(report)
        md_file = self.results_dir / "PATTERN_DISCOVERY_REPORT.md"
        with open(md_file, 'w') as f:
            f.write(md_report)

        logger.info(f"Report saved to: {report_file}")
        logger.info(f"Markdown report saved to: {md_file}")

        return report

    def _generate_markdown_report(self, report: Dict) -> str:
        """Generate markdown-formatted report"""
        md = "# Pattern Discovery Report\n\n"
        md += f"**Generated:** {report['generated_at']}\n\n"

        md += "## Summary\n\n"
        summary = report.get('summary', {})
        md += f"- **Total Analyses:** {summary.get('total_combinations', 0)}\n"
        md += f"- **Successful:** {summary.get('successful_analyses', 0)}\n"
        md += f"- **Total Patterns Found:** {summary.get('total_patterns_discovered', 0)}\n"
        md += f"- **Duration:** {summary.get('duration_seconds', 0):.2f} seconds\n\n"

        md += "## Top 10 Statistically Significant Patterns\n\n"
        md += "| Rank | Pattern | Crypto | Timeframe | Sharpe | Win Rate | Occurrences | p-value |\n"
        md += "|------|---------|--------|-----------|--------|----------|-------------|----------|\n"

        for i, pattern in enumerate(report.get('top_patterns', [])[:10], 1):
            stats = pattern.get('statistics', {})
            md += f"| {i} | {pattern['pattern_name']} | {pattern['crypto']} | {pattern['timeframe']} | "
            md += f"{stats.get('sharpe_ratio', 0):.2f} | {stats.get('win_rate', 0):.1%} | "
            md += f"{stats.get('occurrences', 0)} | {stats.get('p_value', 1):.4f} |\n"

        md += "\n## Next Steps\n\n"
        md += "1. Generate strategy variations from top patterns\n"
        md += "2. Run comprehensive backtesting\n"
        md += "3. Perform Monte Carlo validation\n"
        md += "4. Build ensemble strategy\n"

        return md


async def main():
    """Main entry point for pattern analysis"""
    orchestrator = PatternAnalysisOrchestrator()

    # Run parallel pattern analysis
    summary = await orchestrator.run_parallel_pattern_analysis()

    # Rank patterns
    significant_patterns = orchestrator.rank_patterns_by_significance()

    # Generate report
    report = orchestrator.generate_pattern_discovery_report()

    logger.info("=" * 80)
    logger.info("PATTERN ANALYSIS COMPLETE")
    logger.info(f"Found {len(significant_patterns)} significant patterns")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
