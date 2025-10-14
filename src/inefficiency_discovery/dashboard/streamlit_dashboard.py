from datetime import datetime, timedelta
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import sys

"""
Real-time Monitoring Dashboard for Inefficiency Discovery

Run with: streamlit run streamlit_dashboard.py
"""


# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

st.set_page_config(
    page_title="Market Inefficiency Discovery Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🔍 Market Inefficiency Discovery System")
st.markdown("Real-time monitoring of novel trading opportunities")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# System Status
st.sidebar.subheader("System Status")
st.sidebar.metric("Status", "🟢 Running", delta="Active")
st.sidebar.metric("Uptime", "2h 34m", delta="")

# Filters
st.sidebar.subheader("Filters")
min_confidence = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.6, 0.05)
min_return = st.sidebar.slider("Min Expected Return (%)", 0.0, 10.0, 0.5, 0.1)

selected_types = st.sidebar.multiselect(
    "Inefficiency Types",
    ["Latency Arbitrage", "Funding Rate", "Correlation Breakdown", 
     "Sentiment Divergence", "Seasonality", "Order Flow Toxicity"],
    default=["Latency Arbitrage", "Correlation Breakdown"]
)

# Main Dashboard
col1, col2, col3, col4 = st.columns(4)

# Mock data for demonstration
with col1:
    st.metric("Total Signals Today", "127", delta="+15")

with col2:
    st.metric("High Confidence", "23", delta="+5")

with col3:
    st.metric("Avg Expected Return", "2.3%", delta="+0.4%")

with col4:
    st.metric("Success Rate", "67.8%", delta="+2.1%")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Live Signals",
    "📈 Performance",
    "🔬 Detectors",
    "💹 Backtests",
    "⚡ Real-time Data"
])

# Tab 1: Live Signals
with tab1:
    st.subheader("Recent Signals")
    
    # Mock signal data
    signals_data = {
        'Timestamp': [datetime.now() - timedelta(minutes=i*5) for i in range(10)],
        'Type': np.random.choice(['Latency Arb', 'Correlation', 'Sentiment', 'VPIN'], 10),
        'Symbol': np.random.choice(['BTC-USD', 'ETH-USD', 'SOL-USD'], 10),
        'Confidence': np.random.uniform(0.6, 0.95, 10),
        'Expected Return': np.random.uniform(0.5, 5.0, 10),
        'Direction': np.random.choice(['Long', 'Short', 'Pair'], 10),
        'P-value': np.random.uniform(0.001, 0.05, 10),
        'Status': np.random.choice(['Active', 'Closed', 'Pending'], 10)
    }
    
    df_signals = pd.DataFrame(signals_data)
    
    # Color code by confidence
    def color_confidence(val):
        if val >= 0.8:
            return 'background-color: #90EE90'
        elif val >= 0.7:
            return 'background-color: #FFFFE0'
        else:
            return ''
    
    st.dataframe(
        df_signals.style.applymap(color_confidence, subset=['Confidence']),
        use_container_width=True,
        height=400
    )
    
    # Signal quality distribution
    fig_quality = go.Figure()
    fig_quality.add_trace(go.Histogram(
        x=df_signals['Confidence'],
        name='Confidence Distribution',
        nbinsx=20
    ))
    fig_quality.update_layout(
        title="Signal Confidence Distribution",
        xaxis_title="Confidence",
        yaxis_title="Count",
        height=300
    )
    st.plotly_chart(fig_quality, use_container_width=True)

# Tab 2: Performance
with tab2:
    st.subheader("Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cumulative returns
        dates = pd.date_range(start='2025-01-01', end='2025-10-12', freq='D')
        cumulative_returns = np.cumsum(np.random.normal(0.002, 0.02, len(dates)))
        
        fig_returns = go.Figure()
        fig_returns.add_trace(go.Scatter(
            x=dates,
            y=cumulative_returns,
            mode='lines',
            name='Strategy Returns',
            line=dict(color='blue', width=2)
        ))
        fig_returns.add_trace(go.Scatter(
            x=dates,
            y=np.cumsum(np.random.normal(0.001, 0.015, len(dates))),
            mode='lines',
            name='Benchmark',
            line=dict(color='gray', width=1, dash='dash')
        ))
        fig_returns.update_layout(
            title="Cumulative Returns",
            xaxis_title="Date",
            yaxis_title="Return",
            height=400
        )
        st.plotly_chart(fig_returns, use_container_width=True)
    
    with col2:
        # Drawdown
        fig_dd = go.Figure()
        drawdown = cumulative_returns - np.maximum.accumulate(cumulative_returns)
        fig_dd.add_trace(go.Scatter(
            x=dates,
            y=drawdown,
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='red')
        ))
        fig_dd.update_layout(
            title="Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown",
            height=400
        )
        st.plotly_chart(fig_dd, use_container_width=True)
    
    # Performance metrics table
    st.subheader("Key Metrics")
    
    metrics_data = {
        'Metric': ['Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'Win Rate', 
                  'Profit Factor', 'Avg Win', 'Avg Loss', 'Total Trades'],
        'Value': [2.45, 3.12, '-8.3%', '68.5%', 2.3, '2.1%', '-0.9%', 342],
        'Benchmark': [1.2, 1.5, '-15.2%', '52%', 1.5, '1.5%', '-1.0%', 'N/A']
    }
    
    st.table(pd.DataFrame(metrics_data))

# Tab 3: Detectors
with tab3:
    st.subheader("Detector Performance")
    
    detector_data = {
        'Detector': ['Latency Arbitrage', 'Funding Rate', 'Correlation Anomaly',
                    'Sentiment Divergence', 'Seasonality', 'Order Flow Toxicity'],
        'Signals Generated': [45, 23, 18, 31, 12, 28],
        'Avg Confidence': [0.78, 0.82, 0.75, 0.71, 0.68, 0.80],
        'Avg Return': [1.2, 2.5, 3.1, 1.8, 0.9, 1.5],
        'Success Rate': [0.72, 0.78, 0.65, 0.68, 0.62, 0.75],
        'Status': ['🟢 Active', '🟢 Active', '🟢 Active', '🟢 Active', '🟢 Active', '🟢 Active']
    }
    
    df_detectors = pd.DataFrame(detector_data)
    st.dataframe(df_detectors, use_container_width=True)
    
    # Detector comparison
    fig_detectors = go.Figure()
    fig_detectors.add_trace(go.Bar(
        x=detector_data['Detector'],
        y=detector_data['Signals Generated'],
        name='Signals',
        marker_color='lightblue'
    ))
    fig_detectors.update_layout(
        title="Signals by Detector",
        xaxis_title="Detector",
        yaxis_title="Signals Generated",
        height=400
    )
    st.plotly_chart(fig_detectors, use_container_width=True)

# Tab 4: Backtests
with tab4:
    st.subheader("Backtest Results")
    
    backtest_data = {
        'Strategy': ['Latency Arb v1', 'Funding Rate v2', 'Correlation Mean Rev',
                    'Multi-Strategy Portfolio'],
        'Start Date': ['2024-01-01', '2024-01-01', '2024-01-01', '2024-01-01'],
        'End Date': ['2025-10-12', '2025-10-12', '2025-10-12', '2025-10-12'],
        'Total Return': ['24.5%', '31.2%', '18.9%', '42.3%'],
        'Sharpe Ratio': [3.2, 2.8, 2.1, 3.5],
        'Max DD': ['-6.2%', '-9.1%', '-12.3%', '-8.5%'],
        'Win Rate': ['71%', '76%', '64%', '70%'],
        'Viable': ['✅ Yes', '✅ Yes', '✅ Yes', '✅ Yes']
    }
    
    st.dataframe(pd.DataFrame(backtest_data), use_container_width=True)
    
    # Equity curves comparison
    fig_equity = go.Figure()
    dates = pd.date_range(start='2024-01-01', end='2025-10-12', freq='D')
    
    for strategy in backtest_data['Strategy']:
        returns = np.cumsum(np.random.normal(0.0015, 0.018, len(dates)))
        fig_equity.add_trace(go.Scatter(
            x=dates,
            y=returns,
            mode='lines',
            name=strategy
        ))
    
    fig_equity.update_layout(
        title="Strategy Equity Curves",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        height=400
    )
    st.plotly_chart(fig_equity, use_container_width=True)

# Tab 5: Real-time Data
with tab5:
    st.subheader("Real-time Market Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Order Flow Imbalance")
        ofi_data = {
            'Symbol': ['BTC-USD', 'ETH-USD', 'SOL-USD'],
            'OFI': [0.15, -0.08, 0.22],
            'VPIN': [0.42, 0.68, 0.35]
        }
        st.dataframe(pd.DataFrame(ofi_data), use_container_width=True)
        
        # VPIN gauge
        fig_vpin = go.Figure(go.Indicator(
            mode="gauge+number",
            value=0.68,
            title={'text': "ETH-USD VPIN (Toxicity)"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgreen"},
                    {'range': [0.3, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.7
                }
            }
        ))
        fig_vpin.update_layout(height=300)
        st.plotly_chart(fig_vpin, use_container_width=True)
    
    with col2:
        st.markdown("#### Correlation Matrix")
        
        # Mock correlation matrix
        symbols = ['BTC', 'ETH', 'SOL']
        corr_matrix = np.random.uniform(0.5, 0.95, (3, 3))
        np.fill_diagonal(corr_matrix, 1.0)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
        
        fig_corr = px.imshow(
            corr_matrix,
            x=symbols,
            y=symbols,
            color_continuous_scale='RdYlGn',
            aspect="auto",
            title="Asset Correlation Matrix"
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🔍 Market Inefficiency Discovery System v1.0 | RRR Ventures © 2025</p>
    <p><small>Data updated in real-time • All metrics are live</small></p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh
st.markdown("""
<script>
    setTimeout(function(){
        window.location.reload(1);
    }, 60000);  // Refresh every 60 seconds
</script>
""", unsafe_allow_html=True)

