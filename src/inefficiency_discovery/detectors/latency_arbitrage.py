from ..base import BaseInefficiencyDetector, InefficiencySignal, InefficiencyType
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from scipy import stats
from typing import Dict, List, Optional, Tuple
import asyncio
import logging
import numpy as np
import pandas as pd
import uuid


"""
Latency Arbitrage Detector

Detects price update delays across exchanges to profit from information asymmetry.
When Exchange A updates before Exchange B, we can trade on B based on A's signal.
"""



logger = logging.getLogger(__name__)


@dataclass
class ExchangePriceUpdate:
    """Price update from an exchange"""
    timestamp: datetime
    exchange: str
    symbol: str
    price: float
    volume: float


class LatencyArbitrageDetector(BaseInefficiencyDetector):
    """
    Detects latency arbitrage opportunities across exchanges
    
    Methodology:
    1. Track same symbol on multiple exchanges
    2. Measure time delta between price updates
    3. Calculate lead-lag correlation (Granger causality)
    4. Generate signals when leading exchange moves
    
    Expected Sharpe: 3-5 (if execution < 50ms possible)
    """
    
    def __init__(self, exchanges: List[str] = None, latency_threshold_ms: float = 100):
        super().__init__("LatencyArbitrageDetector")
        
        self.exchanges = exchanges or ['coinbase', 'binance', 'kraken']
        self.latency_threshold_ms = latency_threshold_ms
        
        # Price update buffers per exchange
        self.price_updates: Dict[str, List[ExchangePriceUpdate]] = {
            exchange: [] for exchange in self.exchanges
        }
        
        # Lead-lag relationships
        self.lead_lag_matrix: Dict[Tuple[str, str], float] = {}  # (leader, follower) -> correlation
        
        # Historical latency tracking
        self.latency_history: Dict[Tuple[str, str], List[float]] = {}  # (exchange_a, exchange_b) -> latencies
        
        # Buffer size
        self.buffer_size = 1000
        
    async def detect(self, data: pd.DataFrame) -> List[InefficiencySignal]:
        """
        Detect latency arbitrage opportunities
        
        Args:
            data: DataFrame with columns: timestamp, exchange, symbol, price, volume
            
        Returns:
            List of inefficiency signals
        """
        if data.empty:
            return []
        
        signals = []
        
        # Group by symbol
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol]
            
            # Update price buffers
            for _, row in symbol_data.iterrows():
                update = ExchangePriceUpdate(
                    timestamp=row['timestamp'],
                    exchange=row['exchange'],
                    symbol=row['symbol'],
                    price=row['price'],
                    volume=row.get('volume', 0)
                )
                
                if update.exchange in self.price_updates:
                    self.price_updates[update.exchange].append(update)
                    
                    # Limit buffer size
                    if len(self.price_updates[update.exchange]) > self.buffer_size:
                        self.price_updates[update.exchange] = self.price_updates[update.exchange][-self.buffer_size:]
            
            # Calculate lead-lag relationships
            self._calculate_lead_lag_relationships(symbol)
            
            # Generate signals
            symbol_signals = self._generate_signals(symbol)
            signals.extend(symbol_signals)
        
        # Filter and validate
        return self.filter_signals(signals)
    
    def _calculate_lead_lag_relationships(self, symbol: str):
        """
        Calculate lead-lag relationships using Granger causality
        
        Updates self.lead_lag_matrix with correlation coefficients
        """
        # Need at least 50 data points per exchange
        min_points = 50
        
        for i, exchange_a in enumerate(self.exchanges):
            for exchange_b in self.exchanges[i+1:]:
                # Get recent price updates for both exchanges
                updates_a = [u for u in self.price_updates.get(exchange_a, []) 
                           if u.symbol == symbol][-min_points:]
                updates_b = [u for u in self.price_updates.get(exchange_b, []) 
                           if u.symbol == symbol][-min_points:]
                
                if len(updates_a) < min_points or len(updates_b) < min_points:
                    continue
                
                # Align timestamps (find common time window)
                timestamps_a = [u.timestamp for u in updates_a]
                timestamps_b = [u.timestamp for u in updates_b]
                
                start_time = max(min(timestamps_a), min(timestamps_b))
                end_time = min(max(timestamps_a), max(timestamps_b))
                
                # Filter to common window
                aligned_a = [u for u in updates_a if start_time <= u.timestamp <= end_time]
                aligned_b = [u for u in updates_b if start_time <= u.timestamp <= end_time]
                
                if len(aligned_a) < 30 or len(aligned_b) < 30:
                    continue
                
                # Calculate cross-correlation at different lags
                correlation_ab = self._calculate_cross_correlation(aligned_a, aligned_b)
                correlation_ba = self._calculate_cross_correlation(aligned_b, aligned_a)
                
                # Store relationships
                self.lead_lag_matrix[(exchange_a, exchange_b)] = correlation_ab
                self.lead_lag_matrix[(exchange_b, exchange_a)] = correlation_ba
                
                # Calculate average latency
                avg_latency = self._calculate_average_latency(aligned_a, aligned_b)
                
                key = (exchange_a, exchange_b)
                if key not in self.latency_history:
                    self.latency_history[key] = []
                
                self.latency_history[key].append(avg_latency)
                
                if len(self.latency_history[key]) > 100:
                    self.latency_history[key] = self.latency_history[key][-100:]
    
    def _calculate_cross_correlation(self, updates_a: List[ExchangePriceUpdate], 
                                     updates_b: List[ExchangePriceUpdate]) -> float:
        """
        Calculate cross-correlation with A leading B
        
        Returns correlation coefficient (0-1)
        """
        if len(updates_a) < 10 or len(updates_b) < 10:
            return 0.0
        
        # Convert to price returns
        prices_a = np.array([u.price for u in updates_a])
        prices_b = np.array([u.price for u in updates_b])
        
        if len(prices_a) < 2 or len(prices_b) < 2:
            return 0.0
        
        returns_a = np.diff(np.log(prices_a))
        returns_b = np.diff(np.log(prices_b))
        
        # Align lengths
        min_len = min(len(returns_a), len(returns_b))
        returns_a = returns_a[:min_len]
        returns_b = returns_b[:min_len]
        
        if min_len < 10:
            return 0.0
        
        # Calculate correlation with A leading B
        # Correlate A[:-1] with B[1:] (A leads by 1 tick)
        if len(returns_a) > 1:
            correlation, p_value = stats.pearsonr(returns_a[:-1], returns_b[1:])
            
            # Only return significant correlations
            if p_value < 0.05:
                return abs(correlation)
        
        return 0.0
    
    def _calculate_average_latency(self, updates_a: List[ExchangePriceUpdate],
                                   updates_b: List[ExchangePriceUpdate]) -> float:
        """
        Calculate average time delay between exchanges
        
        Returns latency in milliseconds
        """
        latencies = []
        
        # For each update in A, find nearest update in B
        for update_a in updates_a:
            # Find nearest update in B that came after A
            future_updates_b = [u for u in updates_b if u.timestamp > update_a.timestamp]
            
            if future_updates_b:
                nearest_b = min(future_updates_b, key=lambda u: abs((u.timestamp - update_a.timestamp).total_seconds()))
                latency_ms = (nearest_b.timestamp - update_a.timestamp).total_seconds() * 1000
                
                # Only consider reasonable latencies (< 5 seconds)
                if 0 < latency_ms < 5000:
                    latencies.append(latency_ms)
        
        return np.mean(latencies) if latencies else 0.0
    
    def _generate_signals(self, symbol: str) -> List[InefficiencySignal]:
        """
        Generate latency arbitrage signals
        
        Logic:
        - If Exchange A consistently leads Exchange B
        - And recent price move on A is significant
        - Generate signal to trade on B
        """
        signals = []
        
        # Find strongest lead-lag relationships
        for (leader, follower), correlation in self.lead_lag_matrix.items():
            if correlation < 0.5:  # Need strong correlation
                continue
            
            # Get recent updates
            leader_updates = [u for u in self.price_updates.get(leader, []) 
                            if u.symbol == symbol][-10:]
            follower_updates = [u for u in self.price_updates.get(follower, []) 
                              if u.symbol == symbol][-10:]
            
            if len(leader_updates) < 5 or len(follower_updates) < 5:
                continue
            
            # Check for recent price movement on leader
            leader_prices = [u.price for u in leader_updates]
            price_change = (leader_prices[-1] - leader_prices[0]) / leader_prices[0]
            
            # Need significant move (> 0.1%)
            if abs(price_change) < 0.001:
                continue
            
            # Check latency
            key = (leader, follower)
            if key in self.latency_history:
                avg_latency = np.mean(self.latency_history[key][-10:])
                
                # Only generate signal if latency is within threshold
                if avg_latency > self.latency_threshold_ms:
                    # Calculate expected profit
                    expected_return = price_change * correlation
                    
                    # Account for transaction costs (0.1% per trade)
                    transaction_cost = 0.001
                    net_return = expected_return - (2 * transaction_cost)
                    
                    if net_return > 0:
                        # Create signal
                        signal = InefficiencySignal(
                            signal_id=str(uuid.uuid4()),
                            timestamp=datetime.now(),
                            inefficiency_type=InefficiencyType.LATENCY_ARBITRAGE,
                            symbols=[symbol],
                            exchange=follower,
                            confidence=correlation,
                            expected_return=net_return * 100,  # As percentage
                            expected_duration=int(avg_latency / 1000),  # Convert to seconds
                            direction='long' if price_change > 0 else 'short',
                            description=f"{leader} leads {follower} by {avg_latency:.0f}ms with {correlation:.2%} correlation. Trade on {follower} based on {leader} signal.",
                            metadata={
                                'leader_exchange': leader,
                                'follower_exchange': follower,
                                'correlation': correlation,
                                'latency_ms': avg_latency,
                                'price_change_leader': price_change,
                                'leader_price': leader_prices[-1],
                                'follower_price': [u.price for u in follower_updates][-1]
                            }
                        )
                        
                        # Calculate statistics
                        signal = self.calculate_statistics(signal)
                        signals.append(signal)
        
        return signals
    
    def calculate_statistics(self, signal: InefficiencySignal) -> InefficiencySignal:
        """
        Calculate statistical metrics for signal
        """
        # Get metadata
        correlation = signal.metadata.get('correlation', 0)
        
        # Calculate z-score (how unusual is this correlation)
        # Compare to historical correlations
        all_correlations = [c for c in self.lead_lag_matrix.values()]
        
        if len(all_correlations) > 10:
            mean_corr = np.mean(all_correlations)
            std_corr = np.std(all_correlations)
            
            if std_corr > 0:
                signal.z_score = (correlation - mean_corr) / std_corr
            else:
                signal.z_score = 0.0
        else:
            signal.z_score = 0.0
        
        # P-value: probability this correlation is random
        # Use Fisher transformation for correlation significance
        n = 50  # Assume 50 observations
        if abs(correlation) > 0:
            z = 0.5 * np.log((1 + correlation) / (1 - correlation))
            se = 1 / np.sqrt(n - 3)
            signal.p_value = 2 * (1 - stats.norm.cdf(abs(z / se)))
        else:
            signal.p_value = 1.0
        
        # Estimate Sharpe ratio
        # Assume we can capture 50% of the price move
        capture_ratio = 0.5
        expected_return_per_trade = signal.expected_return / 100 * capture_ratio
        
        # Assume volatility of 2% per trade
        volatility_per_trade = 0.02
        
        # Annualize (assume 100 trades per day)
        trades_per_year = 100 * 252
        annual_return = expected_return_per_trade * trades_per_year
        annual_volatility = volatility_per_trade * np.sqrt(trades_per_year)
        
        if annual_volatility > 0:
            signal.sharpe_ratio = annual_return / annual_volatility
        else:
            signal.sharpe_ratio = 0.0
        
        return signal
    
    @lru_cache(maxsize=128)
    
    def get_lead_lag_matrix(self) -> pd.DataFrame:
        """
        Get lead-lag correlation matrix as DataFrame
        
        Returns:
            DataFrame with exchanges as rows/columns, correlations as values
        """
        matrix = pd.DataFrame(0.0, index=self.exchanges, columns=self.exchanges)
        
        for (leader, follower), correlation in self.lead_lag_matrix.items():
            matrix.loc[leader, follower] = correlation
        
        return matrix
    
    @lru_cache(maxsize=128)
    
    def get_latency_statistics(self) -> Dict[str, Any]:
        """Get statistics about exchange latencies"""
        stats_dict = {}
        
        for (exchange_a, exchange_b), latencies in self.latency_history.items():
            if latencies:
                stats_dict[f"{exchange_a}_to_{exchange_b}"] = {
                    'avg_latency_ms': np.mean(latencies),
                    'median_latency_ms': np.median(latencies),
                    'std_latency_ms': np.std(latencies),
                    'min_latency_ms': np.min(latencies),
                    'max_latency_ms': np.max(latencies),
                    'samples': len(latencies)
                }
        
        return stats_dict

