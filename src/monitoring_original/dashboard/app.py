from alerts.alert_manager import AlertManager
from datetime import datetime, timedelta
from dotenv import load_dotenv
from functools import lru_cache
from monitoring.database_monitor import DatabaseMonitor
from monitoring.performance_monitor import get_performance_monitor
from pathlib import Path
from typing import Dict, List, Any, Optional
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import sys


"""
Real-Time Trading Dashboard

Web-based dashboard using Streamlit for monitoring trading system.
Auto-refreshes every 5 seconds to display real-time metrics.
"""


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# Load environment variables
env_path = Path(__file__).resolve().parents[4] / "config" / "api-keys" / ".env"
load_dotenv(env_path)

# Page configuration
st.set_page_config(
    page_title="RRR Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive {
        color: #00ff00;
    }
    .negative {
        color: #ff0000;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# Initialize monitors
@st.cache_resource
@lru_cache(maxsize=128)
def get_monitors():
    """Get monitor instances (cached)"""
    return {
        'db_monitor': DatabaseMonitor(),
        'perf_monitor': get_performance_monitor(),
        'alert_manager': AlertManager()
    }


def format_currency(value: float) -> str:
    """Format value as currency"""
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value:.2f}%"


@lru_cache(maxsize=128)


def get_portfolio_metrics(db_monitor: DatabaseMonitor) -> Dict[str, Any]:
    """Get current portfolio metrics"""
    snapshot = db_monitor.get_latest_portfolio_snapshot()

    if not snapshot:
        return {
            'total_value': 0.0,
            'cash': 0.0,
            'positions_value': 0.0,
            'daily_pnl': 0.0,
            'daily_return': 0.0
        }

    # Calculate daily P&L
    try:
        today = datetime.utcnow().date().isoformat()
        response = db_monitor.supabase.table("portfolio_snapshots") \
            .select("*") \
            .gte("timestamp", today) \
            .order("timestamp", desc=False) \
            .limit(2) \
            .execute()

        if response.data and len(response.data) >= 2:
            start_value = response.data[0].get("total_value", 0)
            current_value = response.data[-1].get("total_value", 0)
            daily_pnl = current_value - start_value
            daily_return = (daily_pnl / start_value * 100) if start_value > 0 else 0.0
        else:
            daily_pnl = 0.0
            daily_return = 0.0
    except:
        daily_pnl = 0.0
        daily_return = 0.0

    return {
        'total_value': snapshot.get('total_value', 0.0),
        'cash': snapshot.get('cash', 0.0),
        'positions_value': snapshot.get('positions_value', 0.0),
        'daily_pnl': daily_pnl,
        'daily_return': daily_return
    }


def render_overview_page(monitors: Dict[str, Any]):
    """Render overview dashboard page"""
    st.title("📈 Trading System Overview")

    db_monitor = monitors['db_monitor']
    perf_monitor = monitors['perf_monitor']

    # Get metrics
    portfolio_metrics = get_portfolio_metrics(db_monitor)
    open_positions = db_monitor.get_open_positions()
    perf_health = perf_monitor.check_performance_health()

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Portfolio Value",
            format_currency(portfolio_metrics['total_value']),
            format_currency(portfolio_metrics['daily_pnl'])
        )

    with col2:
        st.metric(
            "Daily Return",
            format_percentage(portfolio_metrics['daily_return']),
            delta_color="normal" if portfolio_metrics['daily_return'] >= 0 else "inverse"
        )

    with col3:
        st.metric(
            "Open Positions",
            len(open_positions),
            ""
        )

    with col4:
        st.metric(
            "Avg Latency",
            f"{perf_health['metrics'].get('avg_latency_ms', 0):.2f}ms",
            ""
        )

    # Portfolio composition
    st.subheader("Portfolio Composition")
    col1, col2 = st.columns(2)

    with col1:
        # Cash vs Positions pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Cash', 'Positions'],
            values=[portfolio_metrics['cash'], portfolio_metrics['positions_value']],
            hole=0.4,
            marker_colors=['#636EFA', '#EF553B']
        )])
        fig.update_layout(
            title="Cash vs Positions",
            height=300,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Positions breakdown
        if open_positions:
            position_data = pd.DataFrame([
                {
                    'Symbol': pos.get('symbol', 'N/A'),
                    'Value': pos.get('current_value', 0),
                    'Quantity': pos.get('quantity', 0)
                }
                for pos in open_positions
            ])

            fig = px.bar(
                position_data,
                x='Symbol',
                y='Value',
                title="Positions by Value",
                color='Value',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No open positions")

    # Recent activity
    st.subheader("Recent Activity")
    recent_trades = db_monitor.get_recent_trades(limit=10)

    if recent_trades:
        trades_df = pd.DataFrame(recent_trades)
        # Format timestamp
        if 'created_at' in trades_df.columns:
            trades_df['created_at'] = pd.to_datetime(trades_df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')

        st.dataframe(
            trades_df[[
                'created_at', 'symbol', 'side', 'quantity',
                'price', 'status'
            ]].head(10),
            use_container_width=True
        )
    else:
        st.info("No recent trades")


def render_trades_page(monitors: Dict[str, Any]):
    """Render trades page"""
    st.title("📊 Trading Activity")

    db_monitor = monitors['db_monitor']

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        limit = st.selectbox("Number of trades", [10, 25, 50, 100], index=1)

    with col2:
        status_filter = st.multiselect(
            "Status",
            ["pending", "filled", "cancelled", "rejected"],
            default=["filled"]
        )

    with col3:
        side_filter = st.multiselect(
            "Side",
            ["buy", "sell"],
            default=["buy", "sell"]
        )

    # Get trades
    trades = db_monitor.get_recent_trades(limit=limit)

    if trades:
        trades_df = pd.DataFrame(trades)

        # Apply filters
        if status_filter:
            trades_df = trades_df[trades_df['status'].isin(status_filter)]
        if side_filter:
            trades_df = trades_df[trades_df['side'].isin(side_filter)]

        # Format timestamp
        if 'created_at' in trades_df.columns:
            trades_df['created_at'] = pd.to_datetime(trades_df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Trades", len(trades_df))

        with col2:
            buy_trades = len(trades_df[trades_df['side'] == 'buy'])
            st.metric("Buy Orders", buy_trades)

        with col3:
            sell_trades = len(trades_df[trades_df['side'] == 'sell'])
            st.metric("Sell Orders", sell_trades)

        with col4:
            if 'price' in trades_df.columns and 'quantity' in trades_df.columns:
                total_volume = (trades_df['price'] * trades_df['quantity']).sum()
                st.metric("Total Volume", format_currency(total_volume))

        # Trades table
        st.subheader("Trade History")
        st.dataframe(trades_df, use_container_width=True)

        # Trade distribution
        col1, col2 = st.columns(2)

        with col1:
            if 'symbol' in trades_df.columns:
                symbol_counts = trades_df['symbol'].value_counts()
                fig = px.pie(
                    values=symbol_counts.values,
                    names=symbol_counts.index,
                    title="Trades by Symbol"
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if 'side' in trades_df.columns:
                side_counts = trades_df['side'].value_counts()
                fig = px.bar(
                    x=side_counts.index,
                    y=side_counts.values,
                    title="Trades by Side",
                    labels={'x': 'Side', 'y': 'Count'},
                    color=side_counts.index,
                    color_discrete_map={'buy': 'green', 'sell': 'red'}
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trades found")


def render_performance_page(monitors: Dict[str, Any]):
    """Render performance page"""
    st.title("🎯 Performance Metrics")

    db_monitor = monitors['db_monitor']
    perf_monitor = monitors['perf_monitor']

    # Performance health
    perf_health = perf_monitor.check_performance_health()

    # Status indicator
    status = perf_health['status']
    status_color = {
        'healthy': '🟢',
        'degraded': '🟡',
        'unhealthy': '🔴'
    }.get(status, '⚪')

    st.markdown(f"## System Status: {status_color} {status.upper()}")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    metrics = perf_health['metrics']

    with col1:
        st.metric(
            "Avg Latency",
            f"{metrics.get('avg_latency_ms', 0):.2f}ms"
        )

    with col2:
        st.metric(
            "P95 Latency",
            f"{metrics.get('p95_latency_ms', 0):.2f}ms"
        )

    with col3:
        st.metric(
            "Throughput",
            f"{metrics.get('throughput_rps', 0):.2f} req/s"
        )

    with col4:
        st.metric(
            "Error Rate",
            f"{metrics.get('error_rate_pct', 0):.2f}%"
        )

    # Latency distribution
    st.subheader("Latency Distribution")
    percentiles = {
        'P50': metrics.get('p50_latency_ms', 0),
        'P95': metrics.get('p95_latency_ms', 0),
        'P99': metrics.get('p99_latency_ms', 0)
    }

    fig = go.Figure(data=[
        go.Bar(
            x=list(percentiles.keys()),
            y=list(percentiles.values()),
            marker_color=['green', 'yellow', 'red']
        )
    ])
    fig.update_layout(
        title="Latency Percentiles",
        yaxis_title="Latency (ms)",
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

    # Equity curve
    st.subheader("Portfolio Equity Curve")

    try:
        response = db_monitor.supabase.table("portfolio_snapshots") \
            .select("timestamp, total_value") \
            .order("timestamp", desc=False) \
            .limit(1000) \
            .execute()

        if response.data:
            equity_df = pd.DataFrame(response.data)
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_df['timestamp'],
                y=equity_df['total_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ))

            fig.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Time",
                yaxis_title="Value ($)",
                height=400,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No portfolio history available")
    except Exception as e:
        st.error(f"Error loading equity curve: {e}")

    # Issues
    if perf_health.get('issues'):
        st.subheader("⚠️ Performance Issues")
        for issue in perf_health['issues']:
            st.warning(issue)


def render_risk_page(monitors: Dict[str, Any]):
    """Render risk management page"""
    st.title("⚠️ Risk Management")

    db_monitor = monitors['db_monitor']
    alert_manager = monitors['alert_manager']

    # Risk metrics
    portfolio_metrics = get_portfolio_metrics(db_monitor)
    open_positions = db_monitor.get_open_positions()

    # Top metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        leverage = (portfolio_metrics['positions_value'] / portfolio_metrics['total_value']) if portfolio_metrics['total_value'] > 0 else 0
        st.metric("Portfolio Leverage", f"{leverage:.2f}x")

    with col2:
        st.metric("Daily P&L", format_currency(portfolio_metrics['daily_pnl']))

    with col3:
        max_position = max([p.get('current_value', 0) for p in open_positions]) if open_positions else 0
        max_position_pct = (max_position / portfolio_metrics['total_value'] * 100) if portfolio_metrics['total_value'] > 0 else 0
        st.metric("Largest Position", format_percentage(max_position_pct))

    # Position sizes
    st.subheader("Position Sizes")

    if open_positions and portfolio_metrics['total_value'] > 0:
        position_data = []
        for pos in open_positions:
            position_value = pos.get('current_value', 0)
            position_pct = (position_value / portfolio_metrics['total_value']) * 100

            position_data.append({
                'Symbol': pos.get('symbol', 'N/A'),
                'Value': position_value,
                'Percentage': position_pct,
                'Quantity': pos.get('quantity', 0)
            })

        pos_df = pd.DataFrame(position_data)

        fig = px.bar(
            pos_df,
            x='Symbol',
            y='Percentage',
            title="Position Sizes (% of Portfolio)",
            color='Percentage',
            color_continuous_scale='RdYlGn_r'
        )

        # Add max position size line
        max_position_threshold = float(os.getenv("MAX_POSITION_SIZE", "0.20")) * 100
        fig.add_hline(
            y=max_position_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Max Position Size ({max_position_threshold}%)"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(pos_df, use_container_width=True)
    else:
        st.info("No open positions")

    # Risk alerts
    st.subheader("Risk Alerts")

    loss_alert = alert_manager.check_portfolio_losses()
    risk_alerts = alert_manager.check_risk_limits()

    if loss_alert:
        st.error(f"⚠️ {loss_alert['message']}")

    if risk_alerts:
        for alert in risk_alerts:
            st.warning(f"⚠️ {alert['message']}")

    if not loss_alert and not risk_alerts:
        st.success("✅ No risk alerts")


def render_system_health_page(monitors: Dict[str, Any]):
    """Render system health page"""
    st.title("🏥 System Health")

    db_monitor = monitors['db_monitor']

    # Database health
    db_health = db_monitor.check_health()

    col1, col2 = st.columns(2)

    with col1:
        status = db_health['status']
        status_emoji = {
            'healthy': '🟢',
            'degraded': '🟡',
            'unhealthy': '🔴'
        }.get(status, '⚪')

        st.markdown(f"### Database Status: {status_emoji} {status.upper()}")

    with col2:
        errors = db_health['metrics'].get('errors_last_hour', 0)
        st.metric("Errors (Last Hour)", errors)

    # Table row counts
    st.subheader("Database Tables")

    row_counts = db_health['metrics'].get('row_counts', {})
    table_df = pd.DataFrame([
        {'Table': table, 'Rows': count}
        for table, count in row_counts.items()
    ])

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(table_df, use_container_width=True)

    with col2:
        fig = px.bar(
            table_df,
            x='Table',
            y='Rows',
            title="Table Row Counts",
            color='Rows',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Query performance
    st.subheader("Query Performance")

    query_times = db_health['metrics'].get('query_times_ms', {})
    query_df = pd.DataFrame([
        {'Table': table, 'Latency (ms)': time_ms}
        for table, time_ms in query_times.items()
    ])

    fig = go.Figure(data=[
        go.Bar(
            x=query_df['Table'],
            y=query_df['Latency (ms)'],
            marker_color=['red' if t > 1000 else 'yellow' if t > 500 else 'green'
                          for t in query_df['Latency (ms)']]
        )
    ])

    fig.update_layout(
        title="Database Query Latency",
        yaxis_title="Latency (ms)",
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)

    # API usage
    st.subheader("API Usage (Last 24h)")

    api_usage = db_monitor.get_api_usage_stats(hours=24)

    if api_usage:
        usage_df = pd.DataFrame([
            {'Endpoint': endpoint, 'Requests': count}
            for endpoint, count in api_usage.items()
        ]).sort_values('Requests', ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(usage_df, use_container_width=True)

        with col2:
            fig = px.pie(
                usage_df,
                values='Requests',
                names='Endpoint',
                title="API Requests by Endpoint"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No API usage data")

    # Issues
    if db_health.get('issues'):
        st.subheader("⚠️ Issues Detected")
        for issue in db_health['issues']:
            st.warning(issue)


def main():
    """Main dashboard application"""

    # Sidebar
    st.sidebar.title("RRR Trading Dashboard")
    st.sidebar.markdown("---")

    # Page selection
    page = st.sidebar.radio(
        "Navigation",
        ["📈 Overview", "📊 Trades", "🎯 Performance", "⚠️ Risk", "🏥 System Health"]
    )

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=True)
    if auto_refresh:
        st.sidebar.info("Dashboard refreshes every 5 seconds")

    # Timestamp
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Last Updated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # Get monitors
    monitors = get_monitors()

    # Render selected page
    if page == "📈 Overview":
        render_overview_page(monitors)
    elif page == "📊 Trades":
        render_trades_page(monitors)
    elif page == "🎯 Performance":
        render_performance_page(monitors)
    elif page == "⚠️ Risk":
        render_risk_page(monitors)
    elif page == "🏥 System Health":
        render_system_health_page(monitors)

    # Auto-refresh
    if auto_refresh:
        import time
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
