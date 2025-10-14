from ..base import BaseInefficiencyDetector, InefficiencySignal, InefficiencyType
from datetime import datetime
from functools import lru_cache
from scipy import stats
from typing import Dict, List, Optional
import logging
import numpy as np
import pandas as pd
import uuid


"""
Funding Rate Arbitrage Detector

Detects opportunities to profit from perpetual futures funding rates.
Strategy: Delta-neutral position collecting funding payments.
"""



logger = logging.getLogger(__name__)


class FundingRateArbitrageDetector(BaseInefficiencyDetector):
    """
    Detects funding rate arbitrage opportunities
    
    Strategy:
    - When funding rate > threshold: Short perpetual, long spot
    - When funding rate < -threshold: Long perpetual, short spot
    - Collect funding payments while delta-neutral
    
    Historical Performance: 15-30% APY with low volatility
    """
    
    def __init__(self, funding_rate_threshold: float = 0.001):
        """
        Args:
            funding_rate_threshold: Minimum funding rate to trade (0.001 = 0.1%)
        """
        super().__init__("FundingRateArbitrageDetector")
        
        self.funding_rate_threshold = funding_rate_threshold
        
        # Historical funding rates
        self.funding_history: Dict[str, List[Dict]] = {}  # symbol -> list of {timestamp, rate}
        
        # Track funding rate trends
        self.funding_trends: Dict[str, float] = {}  # symbol -> trend (positive/negative)
    
    async def detect(self, data: pd.DataFrame) -> List[InefficiencySignal]:
        """
        Detect funding rate arbitrage opportunities
        
        Args:
            data: DataFrame with columns: timestamp, symbol, funding_rate, spot_price, perp_price
            
        Returns:
            List of inefficiency signals
        """
        if data.empty:
            return []
        
        signals = []
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].iloc[-1]  # Most recent
            
            funding_rate = symbol_data.get('funding_rate', 0)
            spot_price = symbol_data.get('spot_price', 0)
            perp_price = symbol_data.get('perp_price', 0)
            timestamp = symbol_data.get('timestamp', datetime.now())
            
            if spot_price == 0 or perp_price == 0:
                continue
            
            # Store history
            if symbol not in self.funding_history:
                self.funding_history[symbol] = []
            
            self.funding_history[symbol].append({
                'timestamp': timestamp,
                'rate': funding_rate,
                'spot_price': spot_price,
                'perp_price': perp_price
            })
            
            # Limit history
            if len(self.funding_history[symbol]) > 1000:
                self.funding_history[symbol] = self.funding_history[symbol][-1000:]
            
            # Check if funding rate exceeds threshold
            if abs(funding_rate) > self.funding_rate_threshold:
                signal = self._generate_signal(
                    symbol=symbol,
                    funding_rate=funding_rate,
                    spot_price=spot_price,
                    perp_price=perp_price,
                    timestamp=timestamp
                )
                
                if signal:
                    signals.append(signal)
        
        return self.filter_signals(signals)
    
    def _generate_signal(self, symbol: str, funding_rate: float, 
                        spot_price: float, perp_price: float,
                        timestamp: datetime) -> Optional[InefficiencySignal]:
        """Generate funding rate arbitrage signal"""
        
        # Calculate expected profit
        # Funding payments typically occur every 8 hours (3 times per day)
        payments_per_day = 3
        daily_rate = funding_rate * payments_per_day
        
        # Annualize
        annual_rate = daily_rate * 365
        
        # Account for transaction costs (0.1% to enter + 0.1% to exit)
        transaction_costs = 0.002
        
        # Calculate basis (perp - spot)
        basis = (perp_price - spot_price) / spot_price
        
        # Net expected return
        holding_period_days = 7  # Typical holding period
        expected_profit = (funding_rate * payments_per_day * holding_period_days) - transaction_costs
        
        if expected_profit <= 0:
            return None
        
        # Determine direction
        if funding_rate > self.funding_rate_threshold:
            direction = "pair"  # Short perp, long spot
            description = f"Positive funding rate {funding_rate:.4%}. Short perpetual, long spot to collect payments."
        else:
            direction = "pair"  # Long perp, short spot
            description = f"Negative funding rate {funding_rate:.4%}. Long perpetual, short spot to pay less."
        
        # Calculate trend
        trend = self._calculate_funding_trend(symbol)
        
        # Create signal
        signal = InefficiencySignal(
            signal_id=str(uuid.uuid4()),
            timestamp=timestamp,
            inefficiency_type=InefficiencyType.FUNDING_RATE,
            symbols=[symbol],
            confidence=min(abs(funding_rate) / 0.01, 1.0),  # Higher rate = higher confidence
            expected_return=expected_profit * 100,  # As percentage
            expected_duration=holding_period_days * 24 * 3600,  # In seconds
            direction=direction,
            entry_price=spot_price,
            description=description,
            metadata={
                'funding_rate': funding_rate,
                'daily_rate': daily_rate,
                'annual_rate': annual_rate,
                'spot_price': spot_price,
                'perp_price': perp_price,
                'basis': basis,
                'trend': trend,
                'payments_per_day': payments_per_day
            }
        )
        
        # Calculate statistics
        signal = self.calculate_statistics(signal)
        
        return signal
    
    def _calculate_funding_trend(self, symbol: str, lookback: int = 24) -> float:
        """
        Calculate funding rate trend
        
        Returns:
            Positive = increasing, negative = decreasing
        """
        if symbol not in self.funding_history or len(self.funding_history[symbol]) < 10:
            return 0.0
        
        recent_rates = [h['rate'] for h in self.funding_history[symbol][-lookback:]]
        
        if len(recent_rates) < 5:
            return 0.0
        
        # Linear regression slope
        x = np.arange(len(recent_rates))
        slope, _ = np.polyfit(x, recent_rates, 1)
        
        return slope
    
    def calculate_statistics(self, signal: InefficiencySignal) -> InefficiencySignal:
        """Calculate statistical metrics"""
        
        funding_rate = signal.metadata.get('funding_rate', 0)
        
        # Get historical funding rates for this symbol
        symbol = signal.symbols[0]
        
        if symbol in self.funding_history and len(self.funding_history[symbol]) > 30:
            historical_rates = [h['rate'] for h in self.funding_history[symbol]]
            
            mean_rate = np.mean(historical_rates)
            std_rate = np.std(historical_rates)
            
            # Z-score: how unusual is this funding rate
            if std_rate > 0:
                signal.z_score = (abs(funding_rate) - mean_rate) / std_rate
            else:
                signal.z_score = 0.0
            
            # T-test: is this rate significantly different from zero?
            t_stat, p_value = stats.ttest_1samp([funding_rate], 0)
            signal.p_value = p_value
        else:
            signal.z_score = 0.0
            signal.p_value = 1.0
        
        # Estimate Sharpe ratio
        annual_return = signal.metadata.get('annual_rate', 0)
        
        # Funding rate strategies typically have low volatility (2-5% annual)
        estimated_volatility = 0.03
        
        if estimated_volatility > 0:
            signal.sharpe_ratio = annual_return / estimated_volatility
        else:
            signal.sharpe_ratio = 0.0
        
        return signal
    
    @lru_cache(maxsize=128)
    
    def get_funding_rate_statistics(self, symbol: str) -> Dict:
        """Get statistics for a symbol's funding rates"""
        
        if symbol not in self.funding_history or len(self.funding_history[symbol]) < 10:
            return {}
        
        rates = [h['rate'] for h in self.funding_history[symbol]]
        
        return {
            'symbol': symbol,
            'current_rate': rates[-1],
            'mean_rate': np.mean(rates),
            'median_rate': np.median(rates),
            'std_rate': np.std(rates),
            'min_rate': np.min(rates),
            'max_rate': np.max(rates),
            'positive_rate_pct': np.sum(np.array(rates) > 0) / len(rates),
            'annualized_mean': np.mean(rates) * 3 * 365,  # 3 payments per day
            'samples': len(rates)
        }

