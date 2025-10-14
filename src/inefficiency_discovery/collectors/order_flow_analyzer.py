from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd

"""
Order flow analysis and anomaly detection
"""


logger = logging.getLogger(__name__)


@dataclass
class OrderFlowAnomaly:
    """Detected order flow anomaly"""
    timestamp: datetime
    symbol: str
    anomaly_type: str
    severity: float  # 0-1
    description: str
    metrics: Dict[str, float]


class OrderFlowAnalyzer:
    """
    Analyzes order flow for anomalies and patterns
    """
    
    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        self.historical_metrics: Dict[str, List[float]] = {}
        
    def detect_spoofing(self, orderbook_snapshots: List) -> Optional[OrderFlowAnomaly]:
        """
        Detect spoofing patterns (large orders that disappear)
        
        Spoofing characteristics:
        - Large order appears in book
        - Order disappears before execution
        - Pattern repeats
        """
        if len(orderbook_snapshots) < 10:
            return None
        
        # Track large orders at best bid/ask
        large_orders = []
        
        for i in range(len(orderbook_snapshots) - 1):
            curr_snapshot = orderbook_snapshots[i]
            next_snapshot = orderbook_snapshots[i + 1]
            
            # Check for large bid disappearing
            if curr_snapshot.bids and next_snapshot.bids:
                curr_best_bid_size = curr_snapshot.bids[0][1]
                next_best_bid_size = next_snapshot.bids[0][1]
                
                # Large order disappeared
                if curr_best_bid_size > next_best_bid_size * 3:
                    large_orders.append({
                        'timestamp': curr_snapshot.timestamp,
                        'side': 'bid',
                        'size': curr_best_bid_size,
                        'price': curr_snapshot.bids[0][0]
                    })
        
        # If multiple large orders disappeared, likely spoofing
        if len(large_orders) >= 3:
            return OrderFlowAnomaly(
                timestamp=datetime.now(),
                symbol=orderbook_snapshots[0].symbol,
                anomaly_type='spoofing',
                severity=min(len(large_orders) / 10, 1.0),
                description=f"Detected {len(large_orders)} potential spoofing events",
                metrics={'spoofing_events': len(large_orders)}
            )
        
        return None
    
    def detect_iceberg_orders(self, ticks: List, orderbook_snapshots: List) -> Optional[OrderFlowAnomaly]:
        """
        Detect iceberg orders (hidden liquidity)
        
        Indicators:
        - Large trade executed
        - Order book depth didn't change proportionally
        - Repeated at same price level
        """
        if len(ticks) < 5 or len(orderbook_snapshots) < 5:
            return None
        
        iceberg_events = []
        
        # Find large trades
        avg_trade_size = np.mean([t.size for t in ticks])
        large_trades = [t for t in ticks if t.size > avg_trade_size * 3]
        
        if len(large_trades) >= 2:
            # Check if trades happened at similar price levels
            prices = [t.price for t in large_trades]
            price_std = np.std(prices)
            
            if price_std / np.mean(prices) < 0.001:  # Less than 0.1% variation
                return OrderFlowAnomaly(
                    timestamp=datetime.now(),
                    symbol=ticks[0].symbol,
                    anomaly_type='iceberg_order',
                    severity=0.7,
                    description=f"Detected potential iceberg order: {len(large_trades)} large trades at similar prices",
                    metrics={
                        'large_trades': len(large_trades),
                        'avg_size': np.mean([t.size for t in large_trades])
                    }
                )
        
        return None
    
    def calculate_depth_imbalance_zscore(self, symbol: str, current_imbalance: float) -> float:
        """
        Calculate z-score for depth imbalance
        
        Returns number of standard deviations from historical mean
        """
        if symbol not in self.historical_metrics:
            self.historical_metrics[symbol] = []
        
        self.historical_metrics[symbol].append(current_imbalance)
        
        # Keep only recent history
        if len(self.historical_metrics[symbol]) > self.lookback_periods:
            self.historical_metrics[symbol] = self.historical_metrics[symbol][-self.lookback_periods:]
        
        if len(self.historical_metrics[symbol]) < 10:
            return 0.0
        
        mean = np.mean(self.historical_metrics[symbol])
        std = np.std(self.historical_metrics[symbol])
        
        if std == 0:
            return 0.0
        
        z_score = (current_imbalance - mean) / std
        return z_score
    
    def detect_flash_crash(self, ticks: List, threshold_pct: float = 0.05) -> Optional[OrderFlowAnomaly]:
        """
        Detect flash crash events
        
        Characteristics:
        - Rapid price drop (>5% in short time)
        - Followed by partial recovery
        - Low volume during crash
        """
        if len(ticks) < 20:
            return None
        
        prices = np.array([t.price for t in ticks])
        
        # Find maximum drawdown in recent window
        running_max = np.maximum.accumulate(prices)
        drawdown = (prices - running_max) / running_max
        
        max_dd = np.min(drawdown)
        
        if max_dd < -threshold_pct:
            # Check if there was recovery
            dd_idx = np.argmin(drawdown)
            if dd_idx < len(prices) - 5:
                recovery = (prices[-1] - prices[dd_idx]) / prices[dd_idx]
                
                if recovery > threshold_pct / 2:  # Recovered at least half
                    return OrderFlowAnomaly(
                        timestamp=datetime.now(),
                        symbol=ticks[0].symbol,
                        anomaly_type='flash_crash',
                        severity=abs(max_dd),
                        description=f"Flash crash detected: {max_dd*100:.2f}% drop with {recovery*100:.2f}% recovery",
                        metrics={
                            'max_drawdown': max_dd,
                            'recovery': recovery
                        }
                    )
        
        return None
    
    def detect_order_book_squeeze(self, orderbook_snapshots: List) -> Optional[OrderFlowAnomaly]:
        """
        Detect order book squeeze (spread widening)
        
        Characteristics:
        - Spread suddenly widens
        - Low liquidity on both sides
        """
        if len(orderbook_snapshots) < 10:
            return None
        
        spreads = [s.spread_bps for s in orderbook_snapshots if s.spread_bps > 0]
        
        if len(spreads) < 10:
            return None
        
        avg_spread = np.mean(spreads[:-3])
        current_spread = spreads[-1]
        
        # Spread widened significantly
        if current_spread > avg_spread * 2:
            return OrderFlowAnomaly(
                timestamp=datetime.now(),
                symbol=orderbook_snapshots[0].symbol,
                anomaly_type='spread_squeeze',
                severity=min(current_spread / avg_spread / 2, 1.0),
                description=f"Spread widened from {avg_spread:.1f} to {current_spread:.1f} bps",
                metrics={
                    'avg_spread_bps': avg_spread,
                    'current_spread_bps': current_spread,
                    'ratio': current_spread / avg_spread
                }
            )
        
        return None
    
    def analyze_market_impact(self, ticks: List, time_window_seconds: int = 60) -> Dict[str, float]:
        """
        Analyze market impact of trades
        
        Returns:
            Dictionary with impact metrics
        """
        if len(ticks) < 10:
            return {}
        
        # Group trades into time buckets
        df = pd.DataFrame([{
            'timestamp': t.timestamp,
            'price': t.price,
            'size': t.size,
            'trade_sign': t.trade_sign,
            'dollar_volume': t.dollar_volume
        } for t in ticks])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        # Calculate returns
        df['returns'] = df['price'].pct_change()
        df['signed_volume'] = df['size'] * df['trade_sign']
        
        # Calculate correlation between signed volume and returns
        if len(df) > 10:
            correlation = df['signed_volume'].corr(df['returns'])
            
            # Calculate average impact per unit volume
            impact_per_volume = np.abs(df['returns'].values[1:]) / (np.abs(df['signed_volume'].values[:-1]) + 1e-10)
            avg_impact = np.mean(impact_per_volume[np.isfinite(impact_per_volume)])
            
            return {
                'volume_price_correlation': correlation,
                'avg_impact_per_volume': avg_impact,
                'total_volume': df['dollar_volume'].sum(),
                'price_volatility': df['returns'].std()
            }
        
        return {}

