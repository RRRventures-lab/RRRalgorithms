from collections import deque
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional
import sys
import time


"""
Console-based monitoring for local development.
Lightweight alternative to Prometheus + Grafana.
"""



try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Stub types for type annotations when Rich is not available
    Layout = None  # type: ignore
    Table = None  # type: ignore
    Console = None  # type: ignore


class LocalMonitor:
    """
    Console-based monitoring for local development.
    Shows key metrics and system status in real-time.
    """
    
    def __init__(self, use_rich: bool = True):
        """
        Initialize local monitor.
        
        Args:
            use_rich: Use rich library for better formatting (if available)
        """
        self.use_rich = use_rich and RICH_AVAILABLE
        self.console = Console() if self.use_rich else None
        
        # Metrics storage
        self.portfolio_value = 10000.0
        self.cash = 10000.0
        self.pnl = 0.0
        self.daily_pnl = 0.0
        
        # Trading metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Performance metrics
        self.metrics_history = deque(maxlen=100)
        
        # System status
        self.services_status = {}
        self.last_update = time.time()
        
    def update_portfolio(
        self,
        value: float,
        cash: float,
        pnl: float,
        daily_pnl: float
    ):
        """Update portfolio metrics."""
        self.portfolio_value = value
        self.cash = cash
        self.pnl = pnl
        self.daily_pnl = daily_pnl
        self.last_update = time.time()
        
        # Store in history
        self.metrics_history.append({
            'timestamp': time.time(),
            'value': value,
            'pnl': pnl
        })
    
    def record_trade(self, side: str, profit: Optional[float] = None):
        """Record a trade execution."""
        self.total_trades += 1
        if profit is not None:
            if profit > 0:
                self.winning_trades += 1
            elif profit < 0:
                self.losing_trades += 1
    
    def update_service_status(self, service: str, status: str):
        """Update service status."""
        self.services_status[service] = {
            'status': status,
            'timestamp': time.time()
        }
    
    @lru_cache(maxsize=128)
    
    def get_win_rate(self) -> float:
        """Calculate win rate."""
        total = self.winning_trades + self.losing_trades
        return self.winning_trades / total if total > 0 else 0.0
    
    def display(self):
        """Display current metrics."""
        if self.use_rich:
            self._display_rich()
        else:
            self._display_simple()
    
    def _display_rich(self):
        """Display metrics using rich library."""
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # Header
        header = Panel(
            f"[bold cyan]RRRalgorithms Trading System[/bold cyan] - Local Development",
            style="bold white on blue"
        )
        layout["header"].update(header)
        
        # Body with metrics
        body = Layout()
        body.split_row(
            Layout(name="portfolio"),
            Layout(name="trading")
        )
        
        # Portfolio metrics table
        portfolio_table = Table(title="Portfolio", show_header=False)
        portfolio_table.add_column("Metric", style="cyan")
        portfolio_table.add_column("Value", style="green")
        
        portfolio_table.add_row("Total Value", f"${self.portfolio_value:,.2f}")
        portfolio_table.add_row("Cash", f"${self.cash:,.2f}")
        portfolio_table.add_row("P&L", f"${self.pnl:+,.2f}")
        
        pnl_color = "green" if self.daily_pnl >= 0 else "red"
        portfolio_table.add_row(
            "Daily P&L",
            f"[{pnl_color}]${self.daily_pnl:+,.2f}[/{pnl_color}]"
        )
        
        body["portfolio"].update(Panel(portfolio_table))
        
        # Trading metrics table
        trading_table = Table(title="Trading", show_header=False)
        trading_table.add_column("Metric", style="cyan")
        trading_table.add_column("Value", style="yellow")
        
        trading_table.add_row("Total Trades", str(self.total_trades))
        trading_table.add_row("Winning", str(self.winning_trades))
        trading_table.add_row("Losing", str(self.losing_trades))
        trading_table.add_row("Win Rate", f"{self.get_win_rate():.1%}")
        
        body["trading"].update(Panel(trading_table))
        
        layout["body"].update(body)
        
        # Footer with services status
        services_status = " | ".join([
            f"{service}: {'✓' if info['status'] == 'running' else '✗'}"
            for service, info in self.services_status.items()
        ])
        footer = Panel(
            f"Services: {services_status} | "
            f"Updated: {datetime.now().strftime('%H:%M:%S')}",
            style="dim"
        )
        layout["footer"].update(footer)
        
        self.console.print(layout)
    
    def _display_simple(self):
        """Display metrics using simple text."""
        print("\n" + "="*60)
        print("RRRalgorithms Trading System - Local Development")
        print("="*60)
        
        print(f"\n📊 PORTFOLIO:")
        print(f"  Total Value:  ${self.portfolio_value:,.2f}")
        print(f"  Cash:         ${self.cash:,.2f}")
        print(f"  P&L:          ${self.pnl:+,.2f}")
        print(f"  Daily P&L:    ${self.daily_pnl:+,.2f}")
        
        print(f"\n💰 TRADING:")
        print(f"  Total Trades: {self.total_trades}")
        print(f"  Winning:      {self.winning_trades}")
        print(f"  Losing:       {self.losing_trades}")
        print(f"  Win Rate:     {self.get_win_rate():.1%}")
        
        print(f"\n⚙️  SERVICES:")
        for service, info in self.services_status.items():
            status = "✓ Running" if info['status'] == 'running' else "✗ Stopped"
            print(f"  {service:.<20} {status}")
        
        print(f"\n🕐 Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60 + "\n")
    
    def display_live(self, refresh_rate: float = 1.0):
        """
        Display metrics with live updates.
        
        Args:
            refresh_rate: Seconds between updates
        """
        if not self.use_rich:
            print("Live display requires 'rich' library. Install with: pip install rich")
            print("Falling back to periodic updates...")
            try:
                while True:
                    self.display()
                    time.sleep(refresh_rate)
            except KeyboardInterrupt:
                pass
        else:
            try:
                with Live(self._create_dashboard(), refresh_per_second=1) as live:
                    while True:
                        live.update(self._create_dashboard())
                        time.sleep(refresh_rate)
            except KeyboardInterrupt:
                pass
    
    def _create_dashboard(self) -> Layout:
        """Create dashboard layout for live display."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # Header
        layout["header"].update(Panel(
            f"[bold cyan]RRRalgorithms Trading System[/bold cyan] - "
            f"{datetime.now().strftime('%H:%M:%S')}",
            style="bold white on blue"
        ))
        
        # Body
        body = Layout()
        body.split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # Portfolio panel
        portfolio_content = (
            f"[cyan]Total Value:[/cyan]  ${self.portfolio_value:,.2f}\n"
            f"[cyan]Cash:[/cyan]         ${self.cash:,.2f}\n"
            f"[cyan]P&L:[/cyan]          ${self.pnl:+,.2f}\n"
        )
        pnl_color = "green" if self.daily_pnl >= 0 else "red"
        portfolio_content += f"[cyan]Daily P&L:[/cyan]    [{pnl_color}]${self.daily_pnl:+,.2f}[/{pnl_color}]"
        
        body["left"].update(Panel(portfolio_content, title="📊 Portfolio"))
        
        # Trading panel
        trading_content = (
            f"[yellow]Total Trades:[/yellow]  {self.total_trades}\n"
            f"[green]Winning:[/green]        {self.winning_trades}\n"
            f"[red]Losing:[/red]         {self.losing_trades}\n"
            f"[yellow]Win Rate:[/yellow]      {self.get_win_rate():.1%}"
        )
        
        body["right"].update(Panel(trading_content, title="💰 Trading"))
        
        layout["body"].update(body)
        
        # Footer
        services = " | ".join([
            f"{s}: {'[green]✓[/green]' if i['status'] == 'running' else '[red]✗[/red]'}"
            for s, i in self.services_status.items()
        ])
        layout["footer"].update(Panel(f"Services: {services}", style="dim"))
        
        return layout
    
    def log(self, level: str, message: str):
        """Log a message."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if self.use_rich:
            level_colors = {
                'INFO': 'blue',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'SUCCESS': 'green'
            }
            color = level_colors.get(level, 'white')
            self.console.print(f"[{color}][{timestamp}] {level}:[/{color}] {message}")
        else:
            print(f"[{timestamp}] {level}: {message}")


# Global monitor instance
_global_monitor: Optional[LocalMonitor] = None


@lru_cache(maxsize=128)


def get_monitor() -> LocalMonitor:
    """Get global monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = LocalMonitor()
    return _global_monitor


if __name__ == "__main__":
    # Test the monitor
    print("🧪 Testing Local Monitor\n")
    
    monitor = LocalMonitor()
    
    # Update some metrics
    monitor.update_service_status("data_pipeline", "running")
    monitor.update_service_status("trading_engine", "running")
    monitor.update_service_status("risk_management", "running")
    
    monitor.update_portfolio(
        value=10500,
        cash=5000,
        pnl=500,
        daily_pnl=150
    )
    
    monitor.record_trade("buy", profit=50)
    monitor.record_trade("sell", profit=30)
    monitor.record_trade("buy", profit=-20)
    
    # Display
    monitor.display()
    
    print("\n💡 To see live updates, run:")
    print("   monitor.display_live()")

