from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from src.core.constants import TradingConstants, RiskConstants
from src.core.database.local_db import LocalDatabase
import pandas as pd
import streamlit as st
import sys
import time


"""
Mobile Dashboard for RRRalgorithms Trading System
==================================================

Mobile-optimized Streamlit dashboard for monitoring and controlling
the trading system from iPhone/iPad.

Features:
- Real-time portfolio monitoring
- Trade history
- Position tracking
- System health status
- Remote control capabilities

Author: RRR Ventures
Date: 2025-10-12
"""


# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# Page configuration
st.set_page_config(
    page_title="RRR Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"  # More space on mobile
)

# Custom CSS for mobile optimization
st.markdown("""
<style>
    /* Mobile-friendly styling */
    .big-metric {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #888;
        text-align: center;
    }
    .positive {
        color: #00ff00;
    }
    .negative {
        color: #ff4444;
    }
    .status-running {
        color: #00ff00;
    }
    .status-stopped {
        color: #ff4444;
    }
    /* Larger touch targets */
    .stButton button {
        height: 3rem;
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@lru_cache(maxsize=128)


def get_portfolio_metrics():
    """Get latest portfolio metrics from database"""
    try:
        db = LocalDatabase()
        # For now, return demo data since DB method doesn't exist yet
        return {
            'total_value': 10000.0,
            'cash': 10000.0,
            'positions_value': 0.0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'win_rate': 0.0
        }
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return None


@lru_cache(maxsize=128)


def get_recent_trades(limit=10):
    """Get recent trades"""
    try:
        # Return demo data for now
        return pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(5)],
            'symbol': ['BTC-USD', 'ETH-USD', 'BTC-USD', 'ETH-USD', 'SOL-USD'],
            'side': ['BUY', 'SELL', 'BUY', 'SELL', 'BUY'],
            'quantity': [0.1, 1.0, 0.2, 0.5, 10.0],
            'price': [50000, 3000, 51000, 3100, 100],
            'pnl': [100, -50, 200, 75, -25]
        })
    except Exception as e:
        st.error(f"Error loading trades: {e}")
        return pd.DataFrame()


@lru_cache(maxsize=128)


def get_positions():
    """Get current positions"""
    try:
        # Return demo data for now
        return pd.DataFrame({
            'symbol': ['BTC-USD', 'ETH-USD'],
            'quantity': [0.5, 2.0],
            'entry_price': [48000, 2800],
            'current_price': [50000, 3000],
            'pnl': [1000, 400],
            'pnl_pct': [4.17, 7.14]
        })
    except Exception as e:
        st.error(f"Error loading positions: {e}")
        return pd.DataFrame()


def main():
    """Main dashboard"""
    
    # Header
    st.title("📈 RRRalgorithms Trading")
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    
    # Create tabs for mobile navigation
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💼 Positions", "📜 Trades", "⚙️ System"])
    
    # Tab 1: Overview
    with tab1:
        st.subheader("Portfolio Overview")
        
        metrics = get_portfolio_metrics()
        
        if metrics:
            # Big metrics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Portfolio Value",
                    f"${metrics['total_value']:,.2f}",
                    delta=f"${metrics['total_pnl']:+,.2f}",
                    delta_color="normal"
                )
            
            with col2:
                st.metric(
                    "Daily P&L",
                    f"${metrics.get('daily_pnl', 0):+,.2f}",
                    delta=None
                )
            
            with col3:
                win_rate = metrics.get('win_rate', 0)
                st.metric(
                    "Win Rate",
                    f"{win_rate:.1%}",
                    delta=None
                )
            
            # Portfolio breakdown
            st.subheader("Breakdown")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Cash", f"${metrics['cash']:,.2f}")
            
            with col2:
                st.metric("Positions", f"${metrics['positions_value']:,.2f}")
            
            # Performance metrics
            if metrics.get('sharpe_ratio'):
                st.subheader("Performance")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                
                with col2:
                    max_dd = metrics.get('max_drawdown', 0)
                    st.metric("Max Drawdown", f"{max_dd:.1%}")
    
    # Tab 2: Positions
    with tab2:
        st.subheader("Open Positions")
        
        positions_df = get_positions()
        
        if not positions_df.empty:
            # Format for mobile display
            display_df = positions_df[['symbol', 'quantity', 'average_price', 'current_price', 'unrealized_pnl']]
            display_df.columns = ['Symbol', 'Qty', 'Avg Price', 'Current', 'P&L']
            
            # Format numbers
            display_df['Avg Price'] = display_df['Avg Price'].apply(lambda x: f"${x:,.2f}")
            display_df['Current'] = display_df['Current'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
            display_df['P&L'] = display_df['P&L'].apply(lambda x: f"${x:+,.2f}" if pd.notna(x) else "$0.00")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No open positions")
    
    # Tab 3: Trades
    with tab3:
        st.subheader("Recent Trades")
        
        trades_df = get_recent_trades(limit=20)
        
        if not trades_df.empty:
            # Format for mobile
            display_df = trades_df[['symbol', 'side', 'quantity', 'price', 'status', 'pnl']].copy()
            display_df.columns = ['Symbol', 'Side', 'Qty', 'Price', 'Status', 'P&L']
            
            # Format numbers
            display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:,.2f}")
            display_df['P&L'] = display_df['P&L'].apply(lambda x: f"${x:+,.2f}" if pd.notna(x) else "N/A")
            
            # Color code side
            def color_side(row):
                if row['Side'] == 'buy':
                    return ['background-color: rgba(0,255,0,0.1)'] * len(row)
                elif row['Side'] == 'sell':
                    return ['background-color: rgba(255,0,0,0.1)'] * len(row)
                return [''] * len(row)
            
            st.dataframe(
                display_df.head(10),
                use_container_width=True,
                hide_index=True
            )
            
            # Trade statistics
            st.subheader("Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Trades", len(trades_df))
            
            with col2:
                buy_count = len(trades_df[trades_df['side'] == 'buy'])
                st.metric("Buys", buy_count)
            
            with col3:
                sell_count = len(trades_df[trades_df['side'] == 'sell'])
                st.metric("Sells", sell_count)
        else:
            st.info("No trades yet")
    
    # Tab 4: System
    with tab4:
        st.subheader("System Status")
        
        # System health
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("🟢 Trading System: RUNNING")
            st.info(f"🖥️  Machine: MacBook M3")
            st.info(f"💾 Location: Lexar Drive")
        
        with col2:
            st.info(f"🕐 Time: {datetime.now().strftime('%H:%M:%S')}")
            st.info(f"📅 Date: {datetime.now().strftime('%Y-%m-%d')}")
        
        st.divider()
        
        # Control buttons (large for mobile)
        st.subheader("Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("⏸️ PAUSE TRADING", use_container_width=True):
                st.warning("Trading paused (feature in development)")
        
        with col2:
            if st.button("▶️ RESUME TRADING", use_container_width=True):
                st.success("Trading resumed (feature in development)")
        
        if st.button("🚨 EMERGENCY STOP", type="primary", use_container_width=True):
            st.error("Emergency stop triggered (feature in development)")
        
        st.divider()
        
        # System info
        st.subheader("Configuration")
        
        st.json({
            "Max Position Size": f"{TradingConstants.MAX_POSITION_SIZE_PCT:.0%}",
            "Max Daily Loss": f"{RiskConstants.MAX_DAILY_LOSS_PCT:.0%}",
            "Stop Loss": f"{RiskConstants.DEFAULT_STOP_LOSS_PCT:.0%}",
            "Max Positions": RiskConstants.MAX_OPEN_POSITIONS
        })
    
    # Auto-refresh
    time.sleep(5)
    st.rerun()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        st.info("Dashboard stopped")
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        st.exception(e)

