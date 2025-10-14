from ..base import BaseInefficiencyDetector, InefficiencySignal, InefficiencyType
from datetime import datetime, timedelta
from functools import lru_cache
from itertools import combinations
from scipy import stats
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd
import uuid


"""
Correlation Anomaly Detector

Detects when asset correlations break from historical norms.
Mean-reversion strategy: bet correlation returns to normal range.
"""



logger = logging.getLogger(__name__)


class CorrelationAnomalyDetector(BaseInefficiencyDetector):
    """
    Detects correlation breakdown opportunities
    
    Strategy:
    - Track correlation matrix for multiple assets
    - Detect when correlation deviates significantly from historical range
    - Generate pairs trade: long strong, short weak
    - Exit when correlation normalizes
    
    Expected Performance: 10-20% per trade, medium risk
    """
    
    def __init__(self, lookback_window: int = 90, correlation_threshold: float = 2.0):
        """
        Args:
            lookback_window: Days of history for correlation calculation
            correlation_threshold: Z-score threshold for anomaly (default: 2 std devs)
        """
        super().__init__("CorrelationAnomalyDetector")
        
        self.lookback_window = lookback_window
        self.correlation_threshold = correlation_threshold
        
        # Price history for correlation calculation
        self.price_history: Dict[str, List[Dict]] = {}  # symbol -> list of {timestamp, price}
        
        # Historical correlation matrix
        self.correlation_history: Dict[Tuple[str, str], List[float]] = {}  # (symbol1, symbol2) -> correlations
        
        # Current correlation matrix
        self.current_correlations: Dict[Tuple[str, str], float] = {}
    
    async def detect(self, data: pd.DataFrame) -> List[InefficiencySignal]:
        """
        Detect correlation anomalies
        
        Args:
            data: DataFrame with columns: timestamp, symbol, price, returns (optional)
            
        Returns:
            List of inefficiency signals
        """
        if data.empty:
            return []
        
        # Update price history
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol]
            
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            for _, row in symbol_data.iterrows():
                self.price_history[symbol].append({
                    'timestamp': row['timestamp'],
                    'price': row['price']
                })
            
            # Limit history
            if len(self.price_history[symbol]) > 10000:
                self.price_history[symbol] = self.price_history[symbol][-10000:]
        
        # Need at least 2 symbols
        if len(self.price_history) < 2:
            return []
        
        # Calculate correlation matrix
        self._update_correlation_matrix()
        
        # Detect anomalies
        signals = self._detect_anomalies()
        
        return self.filter_signals(signals)
    
    def _update_correlation_matrix(self):
        """Update correlation matrix for all asset pairs"""
        
        symbols = list(self.price_history.keys())
        
        # Calculate correlations for all pairs
        for symbol1, symbol2 in combinations(symbols, 2):
            correlation = self._calculate_correlation(symbol1, symbol2)
            
            if correlation is not None:
                pair = tuple(sorted([symbol1, symbol2]))
                self.current_correlations[pair] = correlation
                
                # Store in history
                if pair not in self.correlation_history:
                    self.correlation_history[pair] = []
                
                self.correlation_history[pair].append(correlation)
                
                # Limit history
                if len(self.correlation_history[pair]) > 1000:
                    self.correlation_history[pair] = self.correlation_history[pair][-1000:]
    
    def _calculate_correlation(self, symbol1: str, symbol2: str, 
                               method: str = 'pearson') -> Optional[float]:
        """
        Calculate correlation between two symbols
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            method: 'pearson' or 'spearman'
            
        Returns:
            Correlation coefficient (-1 to 1) or None
        """
        if symbol1 not in self.price_history or symbol2 not in self.price_history:
            return None
        
        # Get recent price history
        days = self.lookback_window
        cutoff_time = datetime.now() - timedelta(days=days)
        
        history1 = [h for h in self.price_history[symbol1] 
                   if h['timestamp'] >= cutoff_time]
        history2 = [h for h in self.price_history[symbol2] 
                   if h['timestamp'] >= cutoff_time]
        
        if len(history1) < 30 or len(history2) < 30:
            return None
        
        # Convert to DataFrame for easier alignment
        df1 = pd.DataFrame(history1).set_index('timestamp')
        df2 = pd.DataFrame(history2).set_index('timestamp')
        
        # Merge on timestamp (inner join for common timestamps)
        merged = df1.join(df2, how='inner', lsuffix='_1', rsuffix='_2')
        
        if len(merged) < 30:
            return None
        
        # Calculate returns
        returns1 = merged['price_1'].pct_change().dropna()
        returns2 = merged['price_2'].pct_change().dropna()
        
        if len(returns1) < 20 or len(returns2) < 20:
            return None
        
        # Calculate correlation
        if method == 'pearson':
            correlation, _ = stats.pearsonr(returns1, returns2)
        else:  # spearman
            correlation, _ = stats.spearmanr(returns1, returns2)
        
        return correlation
    
    def _detect_anomalies(self) -> List[InefficiencySignal]:
        """Detect correlation anomalies and generate signals"""
        
        signals = []
        
        for pair, current_corr in self.current_correlations.items():
            if pair not in self.correlation_history or len(self.correlation_history[pair]) < 30:
                continue
            
            symbol1, symbol2 = pair
            
            # Calculate historical statistics
            historical_corrs = self.correlation_history[pair]
            mean_corr = np.mean(historical_corrs)
            std_corr = np.std(historical_corrs)
            
            if std_corr == 0:
                continue
            
            # Calculate z-score
            z_score = (current_corr - mean_corr) / std_corr
            
            # Check if correlation is significantly different
            if abs(z_score) > self.correlation_threshold:
                # Get recent prices
                price1 = self.price_history[symbol1][-1]['price']
                price2 = self.price_history[symbol2][-1]['price']
                
                # Determine which asset is stronger
                returns1 = self._calculate_recent_return(symbol1, days=5)
                returns2 = self._calculate_recent_return(symbol2, days=5)
                
                # Generate pairs trade
                if returns1 > returns2:
                    # Symbol1 outperforming, symbol2 underperforming
                    # If correlation broken down: long symbol2 (underperformer), short symbol1
                    long_symbol = symbol2
                    short_symbol = symbol1
                    long_price = price2
                    short_price = price1
                else:
                    long_symbol = symbol1
                    short_symbol = symbol2
                    long_price = price1
                    short_price = price2
                
                # Calculate expected return
                # Assume 50% reversion to mean correlation
                price_divergence = abs(returns1 - returns2)
                expected_reversion = price_divergence * 0.5
                
                # Account for transaction costs
                transaction_costs = 0.002  # 0.2% total
                expected_return = expected_reversion - transaction_costs
                
                if expected_return > 0.005:  # At least 0.5% profit potential
                    signal = InefficiencySignal(
                        signal_id=str(uuid.uuid4()),
                        timestamp=datetime.now(),
                        inefficiency_type=InefficiencyType.CORRELATION_BREAKDOWN,
                        symbols=[long_symbol, short_symbol],
                        confidence=min(abs(z_score) / 5, 1.0),  # Cap at z=5
                        expected_return=expected_return * 100,
                        expected_duration=7 * 24 * 3600,  # 7 days
                        direction='pair',
                        description=f"Correlation breakdown: {symbol1}-{symbol2} correlation is {current_corr:.2f} (historical: {mean_corr:.2f} ± {std_corr:.2f}). Z-score: {z_score:.2f}. Long {long_symbol}, short {short_symbol}.",
                        metadata={
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'current_correlation': current_corr,
                            'historical_mean': mean_corr,
                            'historical_std': std_corr,
                            'z_score': z_score,
                            'long_symbol': long_symbol,
                            'short_symbol': short_symbol,
                            'long_price': long_price,
                            'short_price': short_price,
                            'price_divergence': price_divergence,
                            'returns1': returns1,
                            'returns2': returns2
                        }
                    )
                    
                    signal = self.calculate_statistics(signal)
                    signals.append(signal)
        
        return signals
    
    def _calculate_recent_return(self, symbol: str, days: int = 5) -> float:
        """Calculate recent return for a symbol"""
        
        if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
            return 0.0
        
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_prices = [h['price'] for h in self.price_history[symbol] 
                        if h['timestamp'] >= cutoff_time]
        
        if len(recent_prices) < 2:
            return 0.0
        
        return (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
    
    def calculate_statistics(self, signal: InefficiencySignal) -> InefficiencySignal:
        """Calculate statistical metrics"""
        
        z_score = signal.metadata.get('z_score', 0)
        signal.z_score = abs(z_score)
        
        # P-value from z-score
        signal.p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Estimate Sharpe ratio
        # Correlation breakdown trades typically have Sharpe 1.5-2.5
        estimated_sharpe = 2.0 * (abs(z_score) / 3)  # Scale by z-score
        signal.sharpe_ratio = min(estimated_sharpe, 5.0)  # Cap at 5
        
        return signal
    
    @lru_cache(maxsize=128)
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get current correlation matrix as DataFrame"""
        
        symbols = sorted(set(s for pair in self.current_correlations.keys() for s in pair))
        
        if not symbols:
            return pd.DataFrame()
        
        matrix = pd.DataFrame(1.0, index=symbols, columns=symbols)
        
        for (s1, s2), corr in self.current_correlations.items():
            matrix.loc[s1, s2] = corr
            matrix.loc[s2, s1] = corr
        
        return matrix
    
    @lru_cache(maxsize=128)
    
    def get_correlation_statistics(self, symbol1: str, symbol2: str) -> Dict:
        """Get correlation statistics for a pair"""
        
        pair = tuple(sorted([symbol1, symbol2]))
        
        if pair not in self.correlation_history or len(self.correlation_history[pair]) < 10:
            return {}
        
        correlations = self.correlation_history[pair]
        
        return {
            'pair': f"{symbol1}-{symbol2}",
            'current': self.current_correlations.get(pair, 0),
            'mean': np.mean(correlations),
            'median': np.median(correlations),
            'std': np.std(correlations),
            'min': np.min(correlations),
            'max': np.max(correlations),
            'samples': len(correlations),
            'current_z_score': (self.current_correlations.get(pair, 0) - np.mean(correlations)) / np.std(correlations) if np.std(correlations) > 0 else 0
        }

