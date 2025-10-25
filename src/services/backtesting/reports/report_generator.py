from datetime import datetime
from pathlib import Path
from typing import Optional, List
import logging
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
Report Generator

Generates comprehensive backtest reports with visualizations.
"""


logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate comprehensive backtest reports with visualizations.

    Creates:
    - Equity curve charts
    - Drawdown charts
    - Trade distribution
    - Monthly returns heatmap
    - Performance summary
    """

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')

        logger.info(f"Initialized ReportGenerator, output dir: {self.output_dir}")

    def generate_report(
        self,
        backtest_result,
        performance_metrics,
        monte_carlo_result=None,
        output_file: Optional[str] = None
    ):
        """
        Generate comprehensive backtest report.

        Args:
            backtest_result: BacktestResult object
            performance_metrics: PerformanceMetrics object
            monte_carlo_result: Optional MonteCarloResult object
            output_file: Optional output filename
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"backtest_report_{timestamp}.png"

        output_path = self.output_dir / output_file

        logger.info(f"Generating backtest report: {output_path}")

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Equity Curve
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_equity_curve(ax1, backtest_result.equity_curve)

        # 2. Drawdown Curve
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_drawdown(ax2, backtest_result.drawdown_curve)

        # 3. Trade Distribution
        ax3 = fig.add_subplot(gs[2, 0])
        self._plot_trade_distribution(ax3, backtest_result.trades)

        # 4. Monthly Returns
        ax4 = fig.add_subplot(gs[2, 1])
        self._plot_monthly_returns(ax4, backtest_result.equity_curve)

        # 5. Performance Metrics Text
        ax5 = fig.add_subplot(gs[3, 0])
        self._plot_metrics_table(ax5, performance_metrics)

        # 6. Monte Carlo (if available) or Trade Timeline
        ax6 = fig.add_subplot(gs[3, 1])
        if monte_carlo_result:
            self._plot_monte_carlo_dist(ax6, monte_carlo_result)
        else:
            self._plot_trade_timeline(ax6, backtest_result.trades)

        # Add title
        strategy_name = backtest_result.metadata.get('strategy', 'Unknown')
        fig.suptitle(
            f'Backtest Report: {strategy_name}\n'
            f'Total Return: {performance_metrics.total_return:.2f}% | '
            f'Sharpe: {performance_metrics.sharpe_ratio:.2f} | '
            f'Max DD: {performance_metrics.max_drawdown:.2f}%',
            fontsize=16,
            fontweight='bold'
        )

        # Save figure
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Report saved to: {output_path}")

        # Also generate HTML report
        html_path = output_path.with_suffix('.html')
        self._generate_html_report(
            html_path,
            backtest_result,
            performance_metrics,
            monte_carlo_result
        )

        return output_path

    def _plot_equity_curve(self, ax, equity_curve: pd.Series):
        """Plot equity curve."""
        ax.plot(equity_curve.index, equity_curve.values, linewidth=2, color='#2E86AB')
        ax.fill_between(
            equity_curve.index,
            equity_curve.values,
            alpha=0.3,
            color='#2E86AB'
        )
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity ($)')
        ax.grid(True, alpha=0.3)

        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    def _plot_drawdown(self, ax, drawdown_curve: pd.Series):
        """Plot drawdown curve."""
        ax.fill_between(
            drawdown_curve.index,
            drawdown_curve.values * 100,
            0,
            color='#A23B72',
            alpha=0.6
        )
        ax.plot(drawdown_curve.index, drawdown_curve.values * 100, linewidth=2, color='#A23B72')
        ax.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    def _plot_trade_distribution(self, ax, trades: List):
        """Plot trade P&L distribution."""
        trade_pnls = [t.pnl for t in trades if t.pnl is not None]

        if not trade_pnls:
            ax.text(0.5, 0.5, 'No trade data available', ha='center', va='center')
            return

        wins = [pnl for pnl in trade_pnls if pnl > 0]
        losses = [pnl for pnl in trade_pnls if pnl <= 0]

        ax.hist(wins, bins=20, alpha=0.6, color='#06A77D', label=f'Wins ({len(wins)})')
        ax.hist(losses, bins=20, alpha=0.6, color='#D62246', label=f'Losses ({len(losses)})')

        ax.set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('P&L ($)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

    def _plot_monthly_returns(self, ax, equity_curve: pd.Series):
        """Plot monthly returns heatmap."""
        if len(equity_curve) == 0:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            return

        # Calculate monthly returns
        monthly_equity = equity_curve.resample('M').last()
        monthly_returns = monthly_equity.pct_change() * 100

        if len(monthly_returns) < 2:
            ax.text(0.5, 0.5, 'Insufficient data for monthly returns', ha='center', va='center')
            return

        # Create pivot table for heatmap
        df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })

        pivot = df.pivot(index='month', columns='year', values='return')

        # Plot heatmap
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)

        # Set ticks
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(12))
        ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

        ax.set_title('Monthly Returns (%)', fontsize=14, fontweight='bold')

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Return (%)')

    def _plot_metrics_table(self, ax, metrics):
        """Plot performance metrics as a table."""
        ax.axis('tight')
        ax.axis('off')

        metrics_data = [
            ['Metric', 'Value'],
            ['Total Return', f"{metrics.total_return:.2f}%"],
            ['CAGR', f"{metrics.cagr:.2f}%"],
            ['Sharpe Ratio', f"{metrics.sharpe_ratio:.2f}"],
            ['Sortino Ratio', f"{metrics.sortino_ratio:.2f}"],
            ['Max Drawdown', f"{metrics.max_drawdown:.2f}%"],
            ['Win Rate', f"{metrics.win_rate * 100:.2f}%"],
            ['Profit Factor', f"{metrics.profit_factor:.2f}"],
            ['Total Trades', f"{metrics.total_trades}"],
            ['Expectancy', f"${metrics.expectancy:.2f}"],
            ['SQN', f"{metrics.sqn:.2f}"],
        ]

        table = ax.table(
            cellText=metrics_data,
            cellLoc='left',
            loc='center',
            colWidths=[0.6, 0.4]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(metrics_data)):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F0F0F0')

        ax.set_title('Performance Metrics', fontsize=14, fontweight='bold', pad=20)

    def _plot_monte_carlo_dist(self, ax, mc_result):
        """Plot Monte Carlo return distribution."""
        ax.hist(
            mc_result.total_return_dist,
            bins=50,
            alpha=0.7,
            color='#2E86AB',
            edgecolor='black'
        )

        # Add percentile lines
        ax.axvline(
            mc_result.percentile_5,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'5th %ile: {mc_result.percentile_5:.1f}%'
        )
        ax.axvline(
            mc_result.median_return,
            color='green',
            linestyle='--',
            linewidth=2,
            label=f'Median: {mc_result.median_return:.1f}%'
        )
        ax.axvline(
            mc_result.percentile_95,
            color='blue',
            linestyle='--',
            linewidth=2,
            label=f'95th %ile: {mc_result.percentile_95:.1f}%'
        )

        ax.set_title('Monte Carlo Return Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Total Return (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_trade_timeline(self, ax, trades: List):
        """Plot trade timeline."""
        trade_times = [t.timestamp for t in trades if t.pnl is not None]
        trade_pnls = [t.pnl for t in trades if t.pnl is not None]

        if not trade_times:
            ax.text(0.5, 0.5, 'No trade data available', ha='center', va='center')
            return

        colors = ['#06A77D' if pnl > 0 else '#D62246' for pnl in trade_pnls]

        ax.scatter(trade_times, trade_pnls, c=colors, alpha=0.6, s=50)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        ax.set_title('Trade Timeline', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Trade P&L ($)')
        ax.grid(True, alpha=0.3)

    def _generate_html_report(
        self,
        output_path: Path,
        backtest_result,
        performance_metrics,
        monte_carlo_result
    ):
        """Generate HTML report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2E86AB;
            border-bottom: 3px solid #2E86AB;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #333;
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #2E86AB;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .metric-label {{
            font-weight: bold;
            color: #666;
        }}
        .metric-value {{
            font-size: 1.2em;
            color: #2E86AB;
        }}
        .positive {{
            color: #06A77D;
        }}
        .negative {{
            color: #D62246;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Backtest Report</h1>

        <p><strong>Strategy:</strong> {backtest_result.metadata.get('strategy', 'Unknown')}</p>
        <p><strong>Period:</strong> {performance_metrics.start_date.strftime('%Y-%m-%d')} to {performance_metrics.end_date.strftime('%Y-%m-%d')}</p>
        <p><strong>Initial Capital:</strong> ${backtest_result.metadata.get('initial_capital', 0):,.2f}</p>

        <h2>Performance Summary</h2>
        <div>
            <div class="metric">
                <span class="metric-label">Total Return:</span>
                <span class="metric-value {'positive' if performance_metrics.total_return > 0 else 'negative'}">{performance_metrics.total_return:.2f}%</span>
            </div>
            <div class="metric">
                <span class="metric-label">CAGR:</span>
                <span class="metric-value">{performance_metrics.cagr:.2f}%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Sharpe Ratio:</span>
                <span class="metric-value">{performance_metrics.sharpe_ratio:.2f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Max Drawdown:</span>
                <span class="metric-value negative">{performance_metrics.max_drawdown:.2f}%</span>
            </div>
        </div>

        <h2>Trade Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Trades</td>
                <td>{performance_metrics.total_trades}</td>
            </tr>
            <tr>
                <td>Winning Trades</td>
                <td class="positive">{performance_metrics.winning_trades}</td>
            </tr>
            <tr>
                <td>Losing Trades</td>
                <td class="negative">{performance_metrics.losing_trades}</td>
            </tr>
            <tr>
                <td>Win Rate</td>
                <td>{performance_metrics.win_rate * 100:.2f}%</td>
            </tr>
            <tr>
                <td>Profit Factor</td>
                <td>{performance_metrics.profit_factor:.2f}</td>
            </tr>
            <tr>
                <td>Average Win</td>
                <td class="positive">${performance_metrics.avg_win:.2f}</td>
            </tr>
            <tr>
                <td>Average Loss</td>
                <td class="negative">${performance_metrics.avg_loss:.2f}</td>
            </tr>
        </table>

        <h2>Risk Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Sortino Ratio</td>
                <td>{performance_metrics.sortino_ratio:.2f}</td>
            </tr>
            <tr>
                <td>Calmar Ratio</td>
                <td>{performance_metrics.calmar_ratio:.2f}</td>
            </tr>
            <tr>
                <td>Max Drawdown Duration</td>
                <td>{performance_metrics.max_drawdown_duration} days</td>
            </tr>
            <tr>
                <td>Recovery Factor</td>
                <td>{performance_metrics.recovery_factor:.2f}</td>
            </tr>
        </table>

        <p style="margin-top: 40px; color: #666; font-size: 0.9em;">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </div>
</body>
</html>
"""

        with open(output_path, 'w') as f:
            f.write(html_content)

        logger.info(f"HTML report saved to: {output_path}")
