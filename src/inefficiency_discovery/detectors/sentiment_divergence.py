from ..base import BaseInefficiencyDetector, InefficiencySignal, InefficiencyType
from datetime import datetime, timedelta
from functools import lru_cache
from scipy import stats
from typing import Dict, List, Optional
import logging
import numpy as np
import pandas as pd
import uuid


"""
Sentiment-Price Divergence Detector

Detects when sentiment and price move in opposite directions.
Hypothesis: Sentiment leads or lags price in predictable patterns.
"""



logger = logging.getLogger(__name__)


class SentimentDivergenceDetector(BaseInefficiencyDetector):
    """
    Detects sentiment-price divergences
    
    Types of divergences:
    1. Bullish divergence: Price down, sentiment up (accumulation)
    2. Bearish divergence: Price up, sentiment down (distribution)
    3. Sentiment lead: Sentiment changes before price
    
    Expected Performance: Sharpe 1.5-2.5
    """
    
    def __init__(self, divergence_threshold: float = 1.5, lookback_days: int = 7):
        """
        Args:
            divergence_threshold: Z-score threshold for divergence
            lookback_days: Days to look back for price/sentiment comparison
        """
        super().__init__("SentimentDivergenceDetector")
        
        self.divergence_threshold = divergence_threshold
        self.lookback_days = lookback_days
        
        # Historical data
        self.price_history: Dict[str, List[Dict]] = {}
        self.sentiment_history: Dict[str, List[Dict]] = {}
        
    async def detect(self, data: pd.DataFrame) -> List[InefficiencySignal]:
        """
        Detect sentiment-price divergences
        
        Args:
            data: DataFrame with: timestamp, symbol, price, sentiment_score
            
        Returns:
            List of inefficiency signals
        """
        if data.empty:
            return []
        
        signals = []
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol]
            
            # Update histories
            for _, row in symbol_data.iterrows():
                # Price history
                if symbol not in self.price_history:
                    self.price_history[symbol] = []
                
                self.price_history[symbol].append({
                    'timestamp': row['timestamp'],
                    'price': row['price']
                })
                
                # Sentiment history
                if 'sentiment_score' in row and not pd.isna(row['sentiment_score']):
                    if symbol not in self.sentiment_history:
                        self.sentiment_history[symbol] = []
                    
                    self.sentiment_history[symbol].append({
                        'timestamp': row['timestamp'],
                        'sentiment': row['sentiment_score']
                    })
            
            # Limit history
            for hist in [self.price_history, self.sentiment_history]:
                if symbol in hist and len(hist[symbol]) > 10000:
                    hist[symbol] = hist[symbol][-10000:]
            
            # Detect divergences
            divergence = self._detect_divergence(symbol)
            
            if divergence:
                signals.append(divergence)
        
        return self.filter_signals(signals)
    
    def _detect_divergence(self, symbol: str) -> Optional[InefficiencySignal]:
        """Detect sentiment-price divergence for a symbol"""
        
        if symbol not in self.price_history or symbol not in self.sentiment_history:
            return None
        
        if len(self.price_history[symbol]) < 10 or len(self.sentiment_history[symbol]) < 5:
            return None
        
        # Get recent data
        cutoff_time = datetime.now() - timedelta(days=self.lookback_days)
        
        recent_prices = [h for h in self.price_history[symbol] 
                        if h['timestamp'] >= cutoff_time]
        recent_sentiments = [h for h in self.sentiment_history[symbol] 
                            if h['timestamp'] >= cutoff_time]
        
        if len(recent_prices) < 5 or len(recent_sentiments) < 3:
            return None
        
        # Calculate price trend
        prices = [h['price'] for h in recent_prices]
        price_change = (prices[-1] - prices[0]) / prices[0]
        
        # Calculate sentiment trend
        sentiments = [h['sentiment'] for h in recent_sentiments]
        sentiment_change = sentiments[-1] - sentiments[0]
        
        # Normalize to same scale
        normalized_price = np.sign(price_change) * min(abs(price_change) / 0.1, 1.0)
        normalized_sentiment = np.sign(sentiment_change) * min(abs(sentiment_change), 1.0)
        
        # Calculate divergence
        divergence_score = normalized_sentiment - normalized_price
        
        # Calculate z-score
        historical_divergences = self._get_historical_divergences(symbol)
        
        if len(historical_divergences) > 10:
            mean_div = np.mean(historical_divergences)
            std_div = np.std(historical_divergences)
            
            if std_div > 0:
                z_score = (divergence_score - mean_div) / std_div
            else:
                z_score = 0.0
        else:
            z_score = divergence_score / 0.5  # Assume std of 0.5
        
        # Check if significant divergence
        if abs(z_score) > self.divergence_threshold:
            # Determine type
            if divergence_score > 0:
                # Sentiment more positive than price action
                if price_change < 0:
                    divergence_type = "bullish_divergence"
                    direction = "long"
                    description = f"Bullish divergence: Price down {price_change:.2%} but sentiment up. Potential buying opportunity."
                else:
                    divergence_type = "sentiment_lead"
                    direction = "long"
                    description = f"Sentiment leading price higher. Sentiment change: {sentiment_change:.2f}, Price change: {price_change:.2%}."
            else:
                # Sentiment more negative than price action
                if price_change > 0:
                    divergence_type = "bearish_divergence"
                    direction = "short"
                    description = f"Bearish divergence: Price up {price_change:.2%} but sentiment down. Potential selling opportunity."
                else:
                    divergence_type = "sentiment_lead"
                    direction = "short"
                    description = f"Sentiment leading price lower. Sentiment change: {sentiment_change:.2f}, Price change: {price_change:.2%}."
            
            # Calculate expected return
            # Assume 30% of divergence will correct
            expected_correction = abs(divergence_score) * 0.3
            expected_return = expected_correction * 0.1  # Scale to reasonable return
            
            # Account for transaction costs
            transaction_costs = 0.002
            net_return = expected_return - transaction_costs
            
            if net_return > 0:
                signal = InefficiencySignal(
                    signal_id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    inefficiency_type=InefficiencyType.SENTIMENT_DIVERGENCE,
                    symbols=[symbol],
                    confidence=min(abs(z_score) / 3, 1.0),
                    expected_return=net_return * 100,
                    expected_duration=3 * 24 * 3600,  # 3 days
                    direction=direction,
                    entry_price=prices[-1],
                    description=description,
                    metadata={
                        'divergence_type': divergence_type,
                        'price_change': price_change,
                        'sentiment_change': sentiment_change,
                        'divergence_score': divergence_score,
                        'z_score': z_score,
                        'current_price': prices[-1],
                        'current_sentiment': sentiments[-1],
                        'lookback_days': self.lookback_days
                    }
                )
                
                signal = self.calculate_statistics(signal)
                return signal
        
        return None
    
    def _get_historical_divergences(self, symbol: str, max_samples: int = 100) -> List[float]:
        """Calculate historical divergence scores"""
        
        if symbol not in self.price_history or symbol not in self.sentiment_history:
            return []
        
        divergences = []
        
        # Sample at different time points
        price_hist = self.price_history[symbol]
        sent_hist = self.sentiment_history[symbol]
        
        if len(price_hist) < 20 or len(sent_hist) < 10:
            return []
        
        # Calculate divergences for multiple windows
        for i in range(min(max_samples, len(price_hist) - 10)):
            idx = -(i + 10)
            
            window_prices = [h['price'] for h in price_hist[idx-5:idx]]
            window_sents = [h['sentiment'] for h in sent_hist if price_hist[idx-5]['timestamp'] <= h['timestamp'] <= price_hist[idx]['timestamp']]
            
            if len(window_prices) >= 3 and len(window_sents) >= 2:
                price_change = (window_prices[-1] - window_prices[0]) / window_prices[0]
                sent_change = window_sents[-1] - window_sents[0]
                
                norm_price = np.sign(price_change) * min(abs(price_change) / 0.1, 1.0)
                norm_sent = np.sign(sent_change) * min(abs(sent_change), 1.0)
                
                divergence = norm_sent - norm_price
                divergences.append(divergence)
        
        return divergences
    
    def calculate_statistics(self, signal: InefficiencySignal) -> InefficiencySignal:
        """Calculate statistical metrics"""
        
        z_score = signal.metadata.get('z_score', 0)
        signal.z_score = abs(z_score)
        
        # P-value from z-score
        signal.p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Estimate Sharpe ratio
        # Sentiment divergence strategies typically have Sharpe 1.5-2.5
        estimated_sharpe = 2.0 * min(abs(z_score) / 3, 1.0)
        signal.sharpe_ratio = max(estimated_sharpe, 1.0)
        
        return signal
    
    @lru_cache(maxsize=128)
    
    def get_current_divergence(self, symbol: str) -> Optional[Dict]:
        """Get current divergence metrics for a symbol"""
        
        if symbol not in self.price_history or symbol not in self.sentiment_history:
            return None
        
        if len(self.price_history[symbol]) < 2 or len(self.sentiment_history[symbol]) < 2:
            return None
        
        # Recent price change
        prices = [h['price'] for h in self.price_history[symbol][-10:]]
        price_change = (prices[-1] - prices[0]) / prices[0]
        
        # Recent sentiment change
        sentiments = [h['sentiment'] for h in self.sentiment_history[symbol][-5:]]
        sentiment_change = sentiments[-1] - sentiments[0]
        
        # Calculate divergence
        norm_price = np.sign(price_change) * min(abs(price_change) / 0.1, 1.0)
        norm_sent = np.sign(sentiment_change) * min(abs(sentiment_change), 1.0)
        divergence = norm_sent - norm_price
        
        return {
            'symbol': symbol,
            'price_change': price_change,
            'sentiment_change': sentiment_change,
            'divergence_score': divergence,
            'current_price': prices[-1],
            'current_sentiment': sentiments[-1]
        }

