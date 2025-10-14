from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Any
import json
import logging

#!/usr/bin/env python3
"""
Monte Carlo Simulation Reporter

Generates comprehensive reports from Monte Carlo simulation results

Author: AI Psychology Team
Date: 2025-10-11
"""


logger = logging.getLogger(__name__)


class MonteCarloReporter:
    """
    Generates reports from Monte Carlo simulation results

    Supports multiple output formats: JSON, Markdown, HTML
    """

    def __init__(self):
        logger.info("MonteCarloReporter initialized")

    def generate_summary_report(
        self,
        engine_stats: Dict[str, Any],
        detailed: bool = False
    ) -> str:
        """
        Generate summary report in Markdown format

        Args:
            engine_stats: Statistics from MonteCarloEngine
            detailed: Include detailed breakdown

        Returns:
            Markdown formatted report
        """
        report = []

        # Header
        report.append("# Monte Carlo Simulation Report")
        report.append(f"\n**Generated**: {datetime.utcnow().isoformat()}Z\n")

        # Overall Statistics
        report.append("## Overall Statistics\n")
        report.append(f"- **Total Scenarios**: {engine_stats['total_scenarios']:,}")
        report.append(f"- **Passed**: {engine_stats['passed']:,}")
        report.append(f"- **Failed**: {engine_stats['failed']:,}")
        report.append(f"- **Pass Rate**: {engine_stats['pass_rate']*100:.1f}%\n")

        # By Category
        if 'results_by_category' in engine_stats:
            report.append("## Results by Category\n")
            report.append("| Category | Passed | Failed | Pass Rate |")
            report.append("|----------|--------|--------|-----------|")

            for category, results in engine_stats['results_by_category'].items():
                total = results['passed'] + results['failed']
                pass_rate = results['passed'] / total * 100 if total > 0 else 0
                report.append(
                    f"| {category} | {results['passed']:,} | {results['failed']:,} | {pass_rate:.1f}% |"
                )

            report.append("")

        # Critical Metrics
        report.append("## Critical Metrics\n")
        report.append(f"- **Hallucinations Detected**: {engine_stats.get('total_hallucinations', 0):,}")
        report.append(f"- **Decisions Rejected**: {engine_stats.get('total_decisions_rejected', 0):,}")
        report.append(f"- **Validation Failures**: {engine_stats.get('total_validation_failures', 0):,}")
        report.append(f"- **Unstable Systems**: {engine_stats.get('unstable_systems', 0):,}\n")

        # Detailed breakdown
        if detailed:
            report.append("## Detailed Analysis\n")
            report.append("### Top Failure Scenarios\n")
            report.append("(To be implemented with full result data)\n")

        # Recommendations
        report.append("## Recommendations\n")

        if engine_stats['pass_rate'] < 0.90:
            report.append("- ⚠️ **CRITICAL**: Pass rate below 90% threshold")
            report.append("- Review failed scenarios and improve system robustness\n")
        elif engine_stats['pass_rate'] < 0.95:
            report.append("- ⚠️ **WARNING**: Pass rate below 95% target")
            report.append("- Investigate failure patterns\n")
        else:
            report.append("- ✅ System passing 95%+ scenarios")
            report.append("- Continue monitoring for regressions\n")

        return "\n".join(report)

    def generate_json_report(
        self,
        engine_stats: Dict[str, Any],
        agent_stats: Dict[str, Any],
        validation_stats: Dict[str, Any]
    ) -> str:
        """
        Generate comprehensive JSON report

        Args:
            engine_stats: Monte Carlo engine statistics
            agent_stats: Superthink agent statistics
            validation_stats: Validation system statistics

        Returns:
            JSON formatted report
        """
        report = {
            "report_metadata": {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "report_version": "1.0.0",
                "report_type": "monte_carlo_validation"
            },
            "monte_carlo_simulation": engine_stats,
            "superthink_agents": agent_stats,
            "validation_system": validation_stats,
            "summary": {
                "overall_health": self._calculate_health_score(
                    engine_stats,
                    agent_stats,
                    validation_stats
                ),
                "recommendations": self._generate_recommendations(
                    engine_stats,
                    agent_stats,
                    validation_stats
                )
            }
        }

        return json.dumps(report, indent=2)

    def generate_html_report(
        self,
        engine_stats: Dict[str, Any]
    ) -> str:
        """
        Generate HTML report

        Args:
            engine_stats: Statistics from MonteCarloEngine

        Returns:
            HTML formatted report
        """
        html = []

        html.append("<!DOCTYPE html>")
        html.append("<html><head>")
        html.append("<title>Monte Carlo Simulation Report</title>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 40px; }")
        html.append("h1 { color: #333; }")
        html.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
        html.append("th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }")
        html.append("th { background-color: #4CAF50; color: white; }")
        html.append(".pass { color: green; font-weight: bold; }")
        html.append(".fail { color: red; font-weight: bold; }")
        html.append("</style>")
        html.append("</head><body>")

        # Header
        html.append("<h1>Monte Carlo Simulation Report</h1>")
        html.append(f"<p><strong>Generated:</strong> {datetime.utcnow().isoformat()}Z</p>")

        # Summary Table
        html.append("<h2>Summary</h2>")
        html.append("<table>")
        html.append("<tr><th>Metric</th><th>Value</th></tr>")
        html.append(f"<tr><td>Total Scenarios</td><td>{engine_stats['total_scenarios']:,}</td></tr>")
        html.append(f"<tr><td>Passed</td><td class='pass'>{engine_stats['passed']:,}</td></tr>")
        html.append(f"<tr><td>Failed</td><td class='fail'>{engine_stats['failed']:,}</td></tr>")
        html.append(f"<tr><td>Pass Rate</td><td>{engine_stats['pass_rate']*100:.1f}%</td></tr>")
        html.append("</table>")

        # By Category
        if 'results_by_category' in engine_stats:
            html.append("<h2>Results by Category</h2>")
            html.append("<table>")
            html.append("<tr><th>Category</th><th>Passed</th><th>Failed</th><th>Pass Rate</th></tr>")

            for category, results in engine_stats['results_by_category'].items():
                total = results['passed'] + results['failed']
                pass_rate = results['passed'] / total * 100 if total > 0 else 0
                html.append(f"<tr>")
                html.append(f"<td>{category}</td>")
                html.append(f"<td class='pass'>{results['passed']:,}</td>")
                html.append(f"<td class='fail'>{results['failed']:,}</td>")
                html.append(f"<td>{pass_rate:.1f}%</td>")
                html.append(f"</tr>")

            html.append("</table>")

        html.append("</body></html>")

        return "\n".join(html)

    def _calculate_health_score(
        self,
        engine_stats: Dict[str, Any],
        agent_stats: Dict[str, Any],
        validation_stats: Dict[str, Any]
    ) -> str:
        """Calculate overall system health score"""

        pass_rate = engine_stats.get('pass_rate', 0)

        if pass_rate >= 0.95:
            return "EXCELLENT"
        elif pass_rate >= 0.90:
            return "GOOD"
        elif pass_rate >= 0.80:
            return "FAIR"
        elif pass_rate >= 0.70:
            return "POOR"
        else:
            return "CRITICAL"

    def _generate_recommendations(
        self,
        engine_stats: Dict[str, Any],
        agent_stats: Dict[str, Any],
        validation_stats: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on results"""

        recommendations = []

        pass_rate = engine_stats.get('pass_rate', 0)

        if pass_rate < 0.90:
            recommendations.append("URGENT: Pass rate below 90% - system not ready for production")
            recommendations.append("Review and fix failing scenarios before deployment")

        if engine_stats.get('total_hallucinations', 0) > 0:
            recommendations.append("Hallucinations detected - review hallucination detection thresholds")

        if engine_stats.get('unstable_systems', 0) > 0:
            recommendations.append("System stability issues detected - investigate resource constraints")

        if not recommendations:
            recommendations.append("System performing well - continue monitoring")

        return recommendations


def generate_report_cli():
    """CLI entry point for report generation"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Monte Carlo simulation reports")
    parser.add_argument('--input', required=True, help='Input statistics JSON file')
    parser.add_argument('--format', choices=['markdown', 'json', 'html'], default='markdown')
    parser.add_argument('--output', help='Output file (default: stdout)')
    parser.add_argument('--detailed', action='store_true', help='Generate detailed report')

    args = parser.parse_args()

    # Load stats
    with open(args.input, 'r') as f:
        stats = json.load(f)

    # Generate report
    reporter = MonteCarloReporter()

    if args.format == 'markdown':
        report = reporter.generate_summary_report(stats, detailed=args.detailed)
    elif args.format == 'json':
        report = reporter.generate_json_report(stats, {}, {})
    elif args.format == 'html':
        report = reporter.generate_html_report(stats)

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    generate_report_cli()
