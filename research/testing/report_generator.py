from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""
Report generation utilities for hypothesis testing.

Generates comprehensive markdown reports with charts and visualizations.
"""



class HypothesisReportGenerator:
    """Generate comprehensive hypothesis test reports."""

    def __init__(self, output_dir: Path):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)

    def generate_report(
        self,
        hypothesis_id: str,
        title: str,
        backtest_results: Any,
        statistical_validation: Any,
        decision: Any,
        metadata: Dict[str, Any]
    ) -> Path:
        """
        Generate comprehensive markdown report.

        Args:
            hypothesis_id: Hypothesis ID
            title: Hypothesis title
            backtest_results: BacktestResults object
            statistical_validation: StatisticalValidation object
            decision: HypothesisDecision object
            metadata: Additional metadata

        Returns:
            Path to generated report
        """
        # Create report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"{hypothesis_id}_report_{timestamp}.md"

        # Generate sections
        sections = []
        sections.append(self._generate_header(hypothesis_id, title, metadata))
        sections.append(self._generate_executive_summary(decision, backtest_results))
        sections.append(self._generate_performance_metrics(backtest_results))
        sections.append(self._generate_statistical_analysis(statistical_validation))
        sections.append(self._generate_decision_section(decision))
        sections.append(self._generate_visualizations(hypothesis_id, backtest_results))
        sections.append(self._generate_next_steps(decision))

        # Write report
        with open(report_file, 'w') as f:
            f.write('\n\n'.join(sections))

        print(f"Report generated: {report_file}")
        return report_file

    def _generate_header(
        self,
        hypothesis_id: str,
        title: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Generate report header."""
        return f"""# Hypothesis Test Report: {hypothesis_id}

## {title}

**Test Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Hypothesis ID**: {hypothesis_id}
**Category**: {metadata.get('category', 'Unknown')}
**Priority Score**: {metadata.get('priority_score', 0)}
**Test Duration**: {metadata.get('execution_time', 0):.1f} seconds

---"""

    def _generate_executive_summary(
        self,
        decision: Any,
        backtest_results: Any
    ) -> str:
        """Generate executive summary."""
        decision_emoji = {
            'SCALE': '✅',
            'ITERATE': '⚠️',
            'KILL': '❌'
        }

        emoji = decision_emoji.get(decision.decision, '❓')

        return f"""## Executive Summary

**Decision**: {emoji} **{decision.decision}** (Confidence: {decision.confidence:.0%})

**Key Metrics**:
- Sharpe Ratio: {backtest_results.sharpe_ratio:.2f}
- Win Rate: {backtest_results.win_rate:.1%}
- Total Return: {backtest_results.total_return:.1%}
- Max Drawdown: {backtest_results.max_drawdown:.1%}

**Recommendation**: {decision.reasoning[0] if decision.reasoning else 'No reasoning provided'}

---"""

    def _generate_performance_metrics(self, backtest_results: Any) -> str:
        """Generate performance metrics table."""
        return f"""## Performance Metrics

### Returns
| Metric | Value |
|--------|-------|
| Total Return | {backtest_results.total_return:.2%} |
| Annual Return | {backtest_results.annual_return:.2%} |
| Volatility (Annual) | {backtest_results.volatility:.2%} |
| Sharpe Ratio | {backtest_results.sharpe_ratio:.2f} |
| Sortino Ratio | {backtest_results.sortino_ratio:.2f} |

### Risk Metrics
| Metric | Value |
|--------|-------|
| Max Drawdown | {backtest_results.max_drawdown:.2%} |
| Best Trade | {backtest_results.best_trade:.2%} |
| Worst Trade | {backtest_results.worst_trade:.2%} |

### Trade Statistics
| Metric | Value |
|--------|-------|
| Total Trades | {backtest_results.total_trades} |
| Win Rate | {backtest_results.win_rate:.1%} |
| Profit Factor | {backtest_results.profit_factor:.2f} |
| Average Win | {backtest_results.avg_win:.2%} |
| Average Loss | {backtest_results.avg_loss:.2%} |

---"""

    def _generate_statistical_analysis(self, validation: Any) -> str:
        """Generate statistical analysis section."""
        significance = "✅ Significant" if validation.significant else "❌ Not Significant"

        return f"""## Statistical Analysis

**Significance**: {significance}

| Metric | Value |
|--------|-------|
| P-value | {validation.p_value:.4f} |
| T-statistic | {validation.t_statistic:.2f} |
| Correlation | {validation.correlation:.3f} |
| Information Coefficient | {validation.information_coefficient:.3f} |

**Interpretation**:
- P-value < 0.05 indicates statistical significance
- Correlation measures linear relationship with returns
- IC (Spearman) measures rank correlation

**Notes**: {validation.notes}

---"""

    def _generate_decision_section(self, decision: Any) -> str:
        """Generate decision section."""
        return f"""## Decision: {decision.decision}

**Confidence**: {decision.confidence:.0%}

### Reasoning
{chr(10).join([f'- {reason}' for reason in decision.reasoning])}

### Production Ready
{'✅ Yes' if decision.ready_for_production else '❌ No'}

---"""

    def _generate_visualizations(
        self,
        hypothesis_id: str,
        backtest_results: Any
    ) -> str:
        """Generate visualizations."""
        # Save equity curve chart
        chart_file = self.output_dir / f"{hypothesis_id}_equity_curve.png"

        plt.figure(figsize=(12, 6))
        plt.plot(backtest_results.equity_curve.index, backtest_results.equity_curve.values)
        plt.title('Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()

        # Save drawdown chart
        dd_chart_file = self.output_dir / f"{hypothesis_id}_drawdown.png"

        running_max = backtest_results.equity_curve.expanding().max()
        drawdown = (backtest_results.equity_curve - running_max) / running_max

        plt.figure(figsize=(12, 4))
        plt.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        plt.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        plt.title('Drawdown')
        plt.xlabel('Time')
        plt.ylabel('Drawdown %')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(dd_chart_file, dpi=150, bbox_inches='tight')
        plt.close()

        return f"""## Visualizations

### Equity Curve
![Equity Curve]({chart_file.name})

### Drawdown Analysis
![Drawdown]({dd_chart_file.name})

---"""

    def _generate_next_steps(self, decision: Any) -> str:
        """Generate next steps section."""
        return f"""## Next Steps

{chr(10).join([f'{i+1}. {step}' for i, step in enumerate(decision.next_steps)])}

---

**Report Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    def generate_comparison_report(
        self,
        hypotheses: List[Dict[str, Any]],
        output_file: str = "comparison_report.md"
    ) -> Path:
        """
        Generate comparison report for multiple hypotheses.

        Args:
            hypotheses: List of hypothesis test results
            output_file: Output filename

        Returns:
            Path to comparison report
        """
        report_file = self.output_dir / output_file

        # Sort by Sharpe ratio
        sorted_hypotheses = sorted(
            hypotheses,
            key=lambda x: x.get('sharpe_ratio', 0),
            reverse=True
        )

        # Generate report
        content = [
            "# Hypothesis Testing Comparison Report",
            f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n**Total Hypotheses Tested**: {len(hypotheses)}",
            "\n---\n",
            "## Summary\n"
        ]

        # Count decisions
        decisions = {}
        for h in hypotheses:
            decision = h.get('decision', 'UNKNOWN')
            decisions[decision] = decisions.get(decision, 0) + 1

        content.append("**Decisions**:")
        for decision, count in decisions.items():
            emoji = {'SCALE': '✅', 'ITERATE': '⚠️', 'KILL': '❌'}.get(decision, '❓')
            content.append(f"- {emoji} {decision}: {count}")

        content.append("\n---\n")
        content.append("## Detailed Results\n")

        # Results table
        content.append("| Rank | ID | Title | Decision | Sharpe | Win Rate | Return | Max DD |")
        content.append("|------|----|----|----------|--------|----------|--------|--------|")

        for i, h in enumerate(sorted_hypotheses, 1):
            decision_emoji = {'SCALE': '✅', 'ITERATE': '⚠️', 'KILL': '❌'}.get(h.get('decision', ''), '❓')
            content.append(
                f"| {i} | {h.get('id', 'N/A')} | {h.get('title', 'N/A')[:30]} | "
                f"{decision_emoji} {h.get('decision', 'N/A')} | "
                f"{h.get('sharpe_ratio', 0):.2f} | "
                f"{h.get('win_rate', 0):.1%} | "
                f"{h.get('total_return', 0):.1%} | "
                f"{h.get('max_drawdown', 0):.1%} |"
            )

        content.append("\n---\n")
        content.append("## Recommendations\n")

        scale_hypotheses = [h for h in sorted_hypotheses if h.get('decision') == 'SCALE']

        if scale_hypotheses:
            content.append(f"\n**{len(scale_hypotheses)} hypotheses ready for production:**\n")
            for h in scale_hypotheses:
                content.append(f"- **{h.get('id')}**: {h.get('title')} (Sharpe: {h.get('sharpe_ratio', 0):.2f})")
        else:
            content.append("\n**No hypotheses ready for immediate production.**")

        iterate_hypotheses = [h for h in sorted_hypotheses if h.get('decision') == 'ITERATE']
        if iterate_hypotheses:
            content.append(f"\n**{len(iterate_hypotheses)} hypotheses need refinement:**\n")
            for h in iterate_hypotheses[:3]:  # Top 3
                content.append(f"- **{h.get('id')}**: {h.get('title')}")

        # Write report
        with open(report_file, 'w') as f:
            f.write('\n'.join(content))

        print(f"Comparison report generated: {report_file}")
        return report_file


def create_summary_json(
    hypothesis_id: str,
    results: Dict[str, Any],
    output_dir: Path
) -> Path:
    """
    Create JSON summary of results.

    Args:
        hypothesis_id: Hypothesis ID
        results: Dictionary of results
        output_dir: Output directory

    Returns:
        Path to JSON file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_file = output_dir / f"{hypothesis_id}_summary.json"

    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return json_file
