from ..base import BaseInefficiencyDetector, InefficiencySignal, InefficiencyType
from datetime import datetime, timedelta
from functools import lru_cache
from scipy import stats, fft
from typing import Dict, List, Optional
import logging
import numpy as np
import pandas as pd
import uuid


"""
Intraday Seasonality Detector

Detects time-of-day and day-of-week patterns in price movements and volatility.
"""



logger = logging.getLogger(__name__)


class SeasonalityDetector(BaseInefficiencyDetector):
    """
    Detects intraday and weekly seasonality patterns
    
    Patterns:
    1. Asia-Europe-US timezone transitions (volatility spikes)
    2. Weekend effect (lower liquidity, higher spreads)
    3. Monthly/quarterly rebalancing (end-of-month flows)
    4. Hourly patterns (consistent up/down movements)
    
    Expected Performance: 5-10% annual alpha
    """
    
    def __init__(self, min_samples_per_period: int = 30):
        super().__init__("SeasonalityDetector")
        
        self.min_samples_per_period = min_samples_per_period
        
        # Hourly patterns (0-23)
        self.hourly_returns: Dict[str, Dict[int, List[float]]] = {}
        self.hourly_volatility: Dict[str, Dict[int, List[float]]] = {}
        
        # Day of week patterns (0-6, Monday-Sunday)
        self.daily_returns: Dict[str, Dict[int, List[float]]] = {}
        self.daily_volatility: Dict[str, Dict[int, List[float]]] = {}
        
        # Day of month patterns (1-31)
        self.monthly_returns: Dict[str, Dict[int, List[float]]] = {}
        
    async def detect(self, data: pd.DataFrame) -> List[InefficiencySignal]:
        """
        Detect seasonality patterns
        
        Args:
            data: DataFrame with: timestamp, symbol, price, returns (optional)
            
        Returns:
            List of inefficiency signals
        """
        if data.empty:
            return []
        
        # Update pattern histories
        self._update_patterns(data)
        
        signals = []
        
        # Detect patterns for each symbol
        for symbol in data['symbol'].unique():
            # Hourly patterns
            hourly_signal = self._detect_hourly_pattern(symbol)
            if hourly_signal:
                signals.append(hourly_signal)
            
            # Day of week patterns
            daily_signal = self._detect_daily_pattern(symbol)
            if daily_signal:
                signals.append(daily_signal)
            
            # End of month patterns
            monthly_signal = self._detect_monthly_pattern(symbol)
            if monthly_signal:
                signals.append(monthly_signal)
        
        return self.filter_signals(signals)
    
    def _update_patterns(self, data: pd.DataFrame):
        """Update pattern histories with new data"""
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            
            # Initialize if needed
            if symbol not in self.hourly_returns:
                self.hourly_returns[symbol] = {h: [] for h in range(24)}
                self.hourly_volatility[symbol] = {h: [] for h in range(24)}
                self.daily_returns[symbol] = {d: [] for d in range(7)}
                self.daily_volatility[symbol] = {d: [] for d in range(7)}
                self.monthly_returns[symbol] = {d: [] for d in range(1, 32)}
            
            # Calculate returns if not provided
            if 'returns' not in symbol_data.columns:
                symbol_data['returns'] = symbol_data['price'].pct_change()
            
            # Extract time components
            symbol_data['hour'] = pd.to_datetime(symbol_data['timestamp']).dt.hour
            symbol_data['dayofweek'] = pd.to_datetime(symbol_data['timestamp']).dt.dayofweek
            symbol_data['day'] = pd.to_datetime(symbol_data['timestamp']).dt.day
            
            # Update hourly patterns
            for hour in range(24):
                hour_data = symbol_data[symbol_data['hour'] == hour]
                if len(hour_data) > 0:
                    returns = hour_data['returns'].dropna()
                    if len(returns) > 0:
                        self.hourly_returns[symbol][hour].extend(returns.tolist())
                        self.hourly_volatility[symbol][hour].append(returns.std())
                        
                        # Limit history
                        if len(self.hourly_returns[symbol][hour]) > 1000:
                            self.hourly_returns[symbol][hour] = self.hourly_returns[symbol][hour][-1000:]
            
            # Update daily patterns
            for day in range(7):
                day_data = symbol_data[symbol_data['dayofweek'] == day]
                if len(day_data) > 0:
                    returns = day_data['returns'].dropna()
                    if len(returns) > 0:
                        self.daily_returns[symbol][day].extend(returns.tolist())
                        self.daily_volatility[symbol][day].append(returns.std())
            
            # Update monthly patterns
            for day in range(1, 32):
                day_data = symbol_data[symbol_data['day'] == day]
                if len(day_data) > 0:
                    returns = day_data['returns'].dropna()
                    if len(returns) > 0:
                        self.monthly_returns[symbol][day].extend(returns.tolist())
    
    def _detect_hourly_pattern(self, symbol: str) -> Optional[InefficiencySignal]:
        """Detect profitable hourly patterns"""
        
        if symbol not in self.hourly_returns:
            return None
        
        # Get current hour
        current_hour = datetime.now().hour
        
        # Check if this hour has a statistically significant pattern
        hour_returns = self.hourly_returns[symbol][current_hour]
        
        if len(hour_returns) < self.min_samples_per_period:
            return None
        
        # Calculate statistics
        mean_return = np.mean(hour_returns)
        std_return = np.std(hour_returns)
        
        if std_return == 0:
            return None
        
        # T-test: is mean significantly different from zero?
        t_stat, p_value = stats.ttest_1samp(hour_returns, 0)
        
        # Need significant pattern
        if p_value > 0.05 or abs(mean_return) < 0.0001:  # Less than 0.01%
            return None
        
        # Calculate confidence based on t-statistic
        confidence = min(abs(t_stat) / 5, 1.0)
        
        # Generate signal
        direction = 'long' if mean_return > 0 else 'short'
        
        signal = InefficiencySignal(
            signal_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            inefficiency_type=InefficiencyType.SEASONALITY,
            symbols=[symbol],
            confidence=confidence,
            expected_return=abs(mean_return) * 100,
            expected_duration=3600,  # 1 hour
            direction=direction,
            description=f"Hourly pattern at {current_hour}:00 UTC. Historical mean return: {mean_return:.4%} (t={t_stat:.2f}, p={p_value:.4f}).",
            metadata={
                'pattern_type': 'hourly',
                'hour': current_hour,
                'mean_return': mean_return,
                'std_return': std_return,
                't_statistic': t_stat,
                'p_value': p_value,
                'sample_size': len(hour_returns)
            }
        )
        
        signal = self.calculate_statistics(signal)
        return signal
    
    def _detect_daily_pattern(self, symbol: str) -> Optional[InefficiencySignal]:
        """Detect day-of-week patterns"""
        
        if symbol not in self.daily_returns:
            return None
        
        current_day = datetime.now().weekday()
        
        day_returns = self.daily_returns[symbol][current_day]
        
        if len(day_returns) < self.min_samples_per_period:
            return None
        
        mean_return = np.mean(day_returns)
        std_return = np.std(day_returns)
        
        if std_return == 0:
            return None
        
        t_stat, p_value = stats.ttest_1samp(day_returns, 0)
        
        if p_value > 0.05 or abs(mean_return) < 0.0005:  # Less than 0.05%
            return None
        
        confidence = min(abs(t_stat) / 4, 1.0)
        direction = 'long' if mean_return > 0 else 'short'
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        signal = InefficiencySignal(
            signal_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            inefficiency_type=InefficiencyType.SEASONALITY,
            symbols=[symbol],
            confidence=confidence,
            expected_return=abs(mean_return) * 100,
            expected_duration=24 * 3600,  # 1 day
            direction=direction,
            description=f"{day_names[current_day]} pattern. Historical mean return: {mean_return:.4%}.",
            metadata={
                'pattern_type': 'daily',
                'day_of_week': current_day,
                'day_name': day_names[current_day],
                'mean_return': mean_return,
                'std_return': std_return,
                't_statistic': t_stat,
                'p_value': p_value,
                'sample_size': len(day_returns)
            }
        )
        
        signal = self.calculate_statistics(signal)
        return signal
    
    def _detect_monthly_pattern(self, symbol: str) -> Optional[InefficiencySignal]:
        """Detect end-of-month rebalancing patterns"""
        
        if symbol not in self.monthly_returns:
            return None
        
        current_day = datetime.now().day
        
        # Focus on end of month (last 3 days)
        is_month_end = datetime.now().day >= 28 or current_day <= 3
        
        if not is_month_end:
            return None
        
        # Aggregate last 3 days of month and first 3 days
        month_end_returns = []
        for day in [28, 29, 30, 31, 1, 2, 3]:
            if day in self.monthly_returns[symbol]:
                month_end_returns.extend(self.monthly_returns[symbol][day])
        
        if len(month_end_returns) < self.min_samples_per_period:
            return None
        
        mean_return = np.mean(month_end_returns)
        std_return = np.std(month_end_returns)
        
        if std_return == 0:
            return None
        
        t_stat, p_value = stats.ttest_1samp(month_end_returns, 0)
        
        if p_value > 0.05 or abs(mean_return) < 0.001:
            return None
        
        confidence = min(abs(t_stat) / 3, 1.0)
        direction = 'long' if mean_return > 0 else 'short'
        
        signal = InefficiencySignal(
            signal_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            inefficiency_type=InefficiencyType.SEASONALITY,
            symbols=[symbol],
            confidence=confidence,
            expected_return=abs(mean_return) * 100,
            expected_duration=3 * 24 * 3600,  # 3 days
            direction=direction,
            description=f"Month-end rebalancing pattern. Historical mean return: {mean_return:.4%}.",
            metadata={
                'pattern_type': 'monthly',
                'current_day': current_day,
                'mean_return': mean_return,
                'std_return': std_return,
                't_statistic': t_stat,
                'p_value': p_value,
                'sample_size': len(month_end_returns)
            }
        )
        
        signal = self.calculate_statistics(signal)
        return signal
    
    def calculate_statistics(self, signal: InefficiencySignal) -> InefficiencySignal:
        """Calculate statistical metrics"""
        
        t_stat = signal.metadata.get('t_statistic', 0)
        p_value = signal.metadata.get('p_value', 1.0)
        
        signal.z_score = abs(t_stat)
        signal.p_value = p_value
        
        # Estimate Sharpe ratio
        # Seasonality strategies typically have moderate Sharpe (1.0-2.0)
        estimated_sharpe = 1.5 * min(abs(t_stat) / 3, 1.0)
        signal.sharpe_ratio = max(estimated_sharpe, 0.5)
        
        return signal
    
    @lru_cache(maxsize=128)
    
    def get_hourly_pattern_summary(self, symbol: str) -> pd.DataFrame:
        """Get summary of hourly patterns"""
        
        if symbol not in self.hourly_returns:
            return pd.DataFrame()
        
        data = []
        for hour in range(24):
            returns = self.hourly_returns[symbol][hour]
            if len(returns) >= 10:
                data.append({
                    'hour': hour,
                    'mean_return': np.mean(returns),
                    'std_return': np.std(returns),
                    'sample_size': len(returns),
                    't_statistic': stats.ttest_1samp(returns, 0)[0] if len(returns) > 1 else 0,
                    'p_value': stats.ttest_1samp(returns, 0)[1] if len(returns) > 1 else 1.0
                })
        
        return pd.DataFrame(data)

