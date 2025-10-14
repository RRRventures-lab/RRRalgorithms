from dataclasses import dataclass, field
from scipy import stats
from scipy.signal import find_peaks, argrelextrema
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
import pandas as pd

"""
Pattern Discovery System for Cryptocurrency Trading
====================================================

Analyzes millions of historical data points to discover statistically
significant trading patterns using:
- Statistical pattern recognition
- Machine learning clustering
- Technical indicator analysis
- Market regime classification

Author: RRR Ventures
Date: 2025-10-11
"""


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """Represents a discovered trading pattern"""
    pattern_id: str
    pattern_type: str  # 'price', 'volume', 'technical', 'regime'
    name: str
    description: str

    # Pattern characteristics
    min_bars: int  # Minimum bars needed to identify pattern
    max_bars: int  # Maximum bars in pattern

    # Statistical metrics
    occurrences: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    p_value: float = 1.0

    # Market conditions
    best_market_regime: Optional[str] = None
    best_timeframe: Optional[str] = None
    best_volatility_range: Optional[Tuple[float, float]] = None

    # Pattern parameters
    parameters: Dict = field(default_factory=dict)

    def is_significant(self, alpha: float = 0.01) -> bool:
        """Check if pattern is statistically significant"""
        return (
            self.p_value < alpha and
            self.win_rate > 0.55 and
            self.sharpe_ratio > 2.0 and
            self.occurrences >= 30
        )


class PricePatternDetector:
    """
    Detects classical price patterns:
    - Head and shoulders / Inverse head and shoulders
    - Double top / Double bottom
    - Triple top / Triple bottom
    - Triangles (ascending, descending, symmetrical)
    - Channels and ranges
    - Cup and handle
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.patterns_found: List[Pattern] = []

    def detect_head_and_shoulders(self, prices: np.ndarray) -> List[Dict]:
        """
        Detect head and shoulders pattern

        Pattern: Left shoulder, Head (higher), Right shoulder
        Returns list of pattern occurrences with indices
        """
        patterns = []

        # Find local maxima (peaks)
        peaks, _ = find_peaks(prices, distance=10)

        if len(peaks) < 3:
            return patterns

        # Check each consecutive triple of peaks
        for i in range(len(peaks) - 2):
            left_shoulder_idx = peaks[i]
            head_idx = peaks[i + 1]
            right_shoulder_idx = peaks[i + 2]

            left_shoulder = prices[left_shoulder_idx]
            head = prices[head_idx]
            right_shoulder = prices[right_shoulder_idx]

            # Criteria for head and shoulders:
            # 1. Head is higher than both shoulders
            # 2. Shoulders are roughly equal (within 2%)
            shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder

            if head > left_shoulder and head > right_shoulder and shoulder_diff < 0.02:
                # Find neckline (low points between peaks)
                neckline_left = np.min(prices[left_shoulder_idx:head_idx])
                neckline_right = np.min(prices[head_idx:right_shoulder_idx])

                patterns.append({
                    'start_idx': left_shoulder_idx,
                    'end_idx': right_shoulder_idx,
                    'head_idx': head_idx,
                    'neckline': (neckline_left + neckline_right) / 2,
                    'confidence': 1.0 - shoulder_diff,
                    'expected_move': head - (neckline_left + neckline_right) / 2
                })

        return patterns

    def detect_double_top(self, prices: np.ndarray) -> List[Dict]:
        """Detect double top pattern (bearish reversal)"""
        patterns = []
        peaks, _ = find_peaks(prices, distance=15)

        for i in range(len(peaks) - 1):
            peak1_idx = peaks[i]
            peak2_idx = peaks[i + 1]

            peak1 = prices[peak1_idx]
            peak2 = prices[peak2_idx]

            # Peaks should be roughly equal (within 1.5%)
            peak_diff = abs(peak1 - peak2) / peak1

            if peak_diff < 0.015:
                # Find trough between peaks
                trough_idx = peak1_idx + np.argmin(prices[peak1_idx:peak2_idx])
                trough = prices[trough_idx]

                patterns.append({
                    'start_idx': peak1_idx,
                    'end_idx': peak2_idx,
                    'trough_idx': trough_idx,
                    'resistance': (peak1 + peak2) / 2,
                    'support': trough,
                    'confidence': 1.0 - peak_diff,
                    'expected_move': (peak1 + peak2) / 2 - trough
                })

        return patterns

    def detect_double_bottom(self, prices: np.ndarray) -> List[Dict]:
        """Detect double bottom pattern (bullish reversal)"""
        patterns = []
        troughs = argrelextrema(prices, np.less, order=15)[0]

        for i in range(len(troughs) - 1):
            trough1_idx = troughs[i]
            trough2_idx = troughs[i + 1]

            trough1 = prices[trough1_idx]
            trough2 = prices[trough2_idx]

            # Troughs should be roughly equal (within 1.5%)
            trough_diff = abs(trough1 - trough2) / trough1

            if trough_diff < 0.015:
                # Find peak between troughs
                peak_idx = trough1_idx + np.argmax(prices[trough1_idx:trough2_idx])
                peak = prices[peak_idx]

                patterns.append({
                    'start_idx': trough1_idx,
                    'end_idx': trough2_idx,
                    'peak_idx': peak_idx,
                    'support': (trough1 + trough2) / 2,
                    'resistance': peak,
                    'confidence': 1.0 - trough_diff,
                    'expected_move': peak - (trough1 + trough2) / 2
                })

        return patterns

    def detect_all_patterns(self, df: pd.DataFrame) -> List[Pattern]:
        """Detect all price patterns in a dataset"""
        prices = df['close'].values

        logger.info(f"Analyzing {len(prices)} price bars for patterns...")

        # Detect each pattern type
        hs_patterns = self.detect_head_and_shoulders(prices)
        dt_patterns = self.detect_double_top(prices)
        db_patterns = self.detect_double_bottom(prices)

        logger.info(f"Found {len(hs_patterns)} head-and-shoulders patterns")
        logger.info(f"Found {len(dt_patterns)} double-top patterns")
        logger.info(f"Found {len(db_patterns)} double-bottom patterns")

        # Convert to Pattern objects
        patterns = []

        for hs in hs_patterns:
            pattern = Pattern(
                pattern_id=f"HS_{hs['start_idx']}",
                pattern_type="price",
                name="Head and Shoulders",
                description="Bearish reversal pattern",
                min_bars=30,
                max_bars=100,
                occurrences=1,
                parameters=hs
            )
            patterns.append(pattern)

        for dt in dt_patterns:
            pattern = Pattern(
                pattern_id=f"DT_{dt['start_idx']}",
                pattern_type="price",
                name="Double Top",
                description="Bearish reversal pattern",
                min_bars=20,
                max_bars=80,
                occurrences=1,
                parameters=dt
            )
            patterns.append(pattern)

        for db in db_patterns:
            pattern = Pattern(
                pattern_id=f"DB_{db['start_idx']}",
                pattern_type="price",
                name="Double Bottom",
                description="Bullish reversal pattern",
                min_bars=20,
                max_bars=80,
                occurrences=1,
                parameters=db
            )
            patterns.append(pattern)

        self.patterns_found.extend(patterns)
        return patterns


class TechnicalIndicatorPatternDetector:
    """
    Detects patterns based on technical indicators:
    - MA crossovers (golden cross, death cross)
    - RSI divergences
    - MACD divergences
    - Bollinger Band squeezes
    - Volume breakouts
    """

    def __init__(self):
        self.patterns_found: List[Pattern] = []

    def calculate_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        return pd.Series(prices).rolling(window=period).mean().values

    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = pd.Series(gains).rolling(window=period).mean().values
        avg_loss = pd.Series(losses).rolling(window=period).mean().values

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def detect_ma_crossover(self, prices: np.ndarray, fast_period: int = 50,
                           slow_period: int = 200) -> List[Dict]:
        """Detect moving average crossovers (Golden Cross / Death Cross)"""
        fast_ma = self.calculate_sma(prices, fast_period)
        slow_ma = self.calculate_sma(prices, slow_period)

        crossovers = []

        for i in range(slow_period, len(prices) - 1):
            # Golden cross: fast MA crosses above slow MA
            if fast_ma[i-1] <= slow_ma[i-1] and fast_ma[i] > slow_ma[i]:
                crossovers.append({
                    'idx': i,
                    'type': 'golden_cross',
                    'signal': 'bullish',
                    'fast_ma': fast_ma[i],
                    'slow_ma': slow_ma[i],
                    'price': prices[i]
                })

            # Death cross: fast MA crosses below slow MA
            elif fast_ma[i-1] >= slow_ma[i-1] and fast_ma[i] < slow_ma[i]:
                crossovers.append({
                    'idx': i,
                    'type': 'death_cross',
                    'signal': 'bearish',
                    'fast_ma': fast_ma[i],
                    'slow_ma': slow_ma[i],
                    'price': prices[i]
                })

        return crossovers

    def detect_rsi_divergence(self, prices: np.ndarray, rsi: Optional[np.ndarray] = None) -> List[Dict]:
        """Detect bullish/bearish RSI divergences"""
        if rsi is None:
            rsi = self.calculate_rsi(prices)

        divergences = []

        # Find price peaks and troughs
        price_peaks = find_peaks(prices, distance=10)[0]
        price_troughs = argrelextrema(prices, np.less, order=10)[0]

        # Find RSI peaks and troughs
        rsi_peaks = find_peaks(rsi, distance=10)[0]
        rsi_troughs = argrelextrema(rsi, np.less, order=10)[0]

        # Bullish divergence: price makes lower low, RSI makes higher low
        for i in range(1, len(price_troughs)):
            if price_troughs[i] in rsi_troughs:
                prev_price_trough_idx = price_troughs[i-1]
                curr_price_trough_idx = price_troughs[i]

                if prices[curr_price_trough_idx] < prices[prev_price_trough_idx]:
                    # Price lower low
                    prev_rsi_idx = np.where(rsi_troughs == prev_price_trough_idx)[0]
                    if len(prev_rsi_idx) > 0 and rsi[curr_price_trough_idx] > rsi[prev_price_trough_idx]:
                        # RSI higher low - bullish divergence
                        divergences.append({
                            'idx': curr_price_trough_idx,
                            'type': 'bullish_divergence',
                            'signal': 'bullish',
                            'price_trend': 'lower_low',
                            'rsi_trend': 'higher_low'
                        })

        # Bearish divergence: price makes higher high, RSI makes lower high
        for i in range(1, len(price_peaks)):
            if price_peaks[i] in rsi_peaks:
                prev_price_peak_idx = price_peaks[i-1]
                curr_price_peak_idx = price_peaks[i]

                if prices[curr_price_peak_idx] > prices[prev_price_peak_idx]:
                    # Price higher high
                    if rsi[curr_price_peak_idx] < rsi[prev_price_peak_idx]:
                        # RSI lower high - bearish divergence
                        divergences.append({
                            'idx': curr_price_peak_idx,
                            'type': 'bearish_divergence',
                            'signal': 'bearish',
                            'price_trend': 'higher_high',
                            'rsi_trend': 'lower_high'
                        })

        return divergences


class MarketRegimeClassifier:
    """
    Classifies market regimes:
    - Trending (uptrend/downtrend)
    - Ranging (sideways)
    - High volatility
    - Low volatility
    """

    def classify_regime(self, df: pd.DataFrame, window: int = 50) -> pd.Series:
        """
        Classify market regime for each bar

        Returns: Series with regime labels
        """
        prices = df['close'].values

        regimes = []

        for i in range(window, len(prices)):
            window_prices = prices[i-window:i]

            # Calculate trend using linear regression slope
            x = np.arange(window)
            slope, _ = np.polyfit(x, window_prices, 1)

            # Calculate volatility (standard deviation)
            volatility = np.std(window_prices) / np.mean(window_prices)

            # Classify regime
            if abs(slope) / np.mean(window_prices) > 0.001:  # Trending
                if slope > 0:
                    regime = 'uptrend'
                else:
                    regime = 'downtrend'
            else:
                regime = 'ranging'

            # Add volatility classification
            if volatility > 0.03:
                regime += '_high_vol'
            elif volatility < 0.01:
                regime += '_low_vol'
            else:
                regime += '_normal_vol'

            regimes.append(regime)

        # Pad beginning with 'unknown'
        regimes = ['unknown'] * window + regimes

        return pd.Series(regimes, index=df.index)


def calculate_pattern_statistics(pattern_occurrences: List[Dict],
                                 df: pd.DataFrame,
                                 forward_window: int = 20) -> Dict:
    """
    Calculate statistical metrics for discovered patterns

    Args:
        pattern_occurrences: List of pattern occurrence dicts with 'idx' key
        df: DataFrame with price data
        forward_window: Bars to look forward for return calculation

    Returns:
        Dictionary with statistical metrics
    """
    returns = []
    wins = 0
    losses = 0

    for occurrence in pattern_occurrences:
        idx = occurrence.get('idx') or occurrence.get('end_idx')

        if idx is None or idx + forward_window >= len(df):
            continue

        entry_price = df.iloc[idx]['close']
        exit_price = df.iloc[idx + forward_window]['close']

        ret = (exit_price - entry_price) / entry_price
        returns.append(ret)

        if ret > 0:
            wins += 1
        else:
            losses += 1

    if len(returns) == 0:
        return {
            'occurrences': 0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'sharpe_ratio': 0.0,
            'profit_factor': 0.0,
            'p_value': 1.0
        }

    returns_array = np.array(returns)

    # Calculate metrics
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    avg_return = np.mean(returns_array)
    sharpe_ratio = (np.mean(returns_array) / np.std(returns_array)) * np.sqrt(252) if np.std(returns_array) > 0 else 0

    gross_profit = np.sum(returns_array[returns_array > 0])
    gross_loss = abs(np.sum(returns_array[returns_array < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # T-test: Are returns significantly different from zero?
    t_stat, p_value = stats.ttest_1samp(returns_array, 0)

    return {
        'occurrences': len(returns),
        'win_rate': win_rate,
        'avg_return': avg_return * 100,  # as percentage
        'sharpe_ratio': sharpe_ratio,
        'profit_factor': profit_factor,
        'p_value': p_value,
        'wins': wins,
        'losses': losses
    }


if __name__ == "__main__":
    # Test with sample data
    logger.info("Testing pattern detection system...")

    # Generate sample price data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)

    df = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'high': prices + np.random.rand(1000),
        'low': prices - np.random.rand(1000),
        'open': prices + np.random.randn(1000) * 0.5,
        'volume': np.random.rand(1000) * 1000000
    })

    # Test price pattern detector
    price_detector = PricePatternDetector()
    patterns = price_detector.detect_all_patterns(df)

    logger.info(f"Detected {len(patterns)} total patterns")

    # Test technical indicator detector
    tech_detector = TechnicalIndicatorPatternDetector()
    ma_crossovers = tech_detector.detect_ma_crossover(prices, 20, 50)

    logger.info(f"Detected {len(ma_crossovers)} MA crossovers")

    # Test regime classifier
    regime_classifier = MarketRegimeClassifier()
    regimes = regime_classifier.classify_regime(df)

    logger.info(f"Regime distribution: {regimes.value_counts().to_dict()}")

    logger.info("Pattern detection system test complete!")
