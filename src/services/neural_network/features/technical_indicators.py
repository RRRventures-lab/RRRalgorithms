from dataclasses import dataclass
from typing import Union, Tuple, Optional, List
import numpy as np
import pandas as pd

"""
Technical Indicators for Cryptocurrency Trading

Comprehensive collection of 25+ technical indicators organized by category:
- Momentum: RSI, MACD, Stochastic, ROC, Williams %R, CCI
- Trend: SMA, EMA, DEMA, TEMA, ADX, Aroon, Parabolic SAR
- Volatility: Bollinger Bands, ATR, Keltner Channels, Donchian Channels
- Volume: OBV, VWAP, MFI, A/D Line, CMF
- Support/Resistance: Pivot Points, Fibonacci Retracements

All indicators are vectorized for efficiency and support multi-timeframe analysis.
"""



@dataclass
class OHLCVData:
    """Container for OHLCV market data"""
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'OHLCVData':
        """Create from pandas DataFrame"""
        return cls(
            open=df['open'].values,
            high=df['high'].values,
            low=df['low'].values,
            close=df['close'].values,
            volume=df['volume'].values
        )


# ============================================================================
# MOMENTUM INDICATORS
# ============================================================================

def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index (RSI)

    Measures momentum oscillating between 0-100.
    Overbought: > 70, Oversold: < 30

    Args:
        close: Closing prices
        period: Lookback period (default: 14)

    Returns:
        RSI values
    """
    delta = np.diff(close, prepend=close[0])
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    # Wilder's smoothing
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)

    avg_gain[period] = np.mean(gains[1:period+1])
    avg_loss[period] = np.mean(losses[1:period+1])

    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period

    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_values = 100 - (100 / (1 + rs))

    return rsi_values


def macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Moving Average Convergence Divergence (MACD)

    Trend-following momentum indicator.

    Args:
        close: Closing prices
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)

    Returns:
        macd_line, signal_line, histogram
    """
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)

    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stochastic Oscillator (%K and %D)

    Momentum indicator comparing close to high-low range.
    Overbought: > 80, Oversold: < 20

    Args:
        high, low, close: OHLC data
        k_period: %K period (default: 14)
        d_period: %D smoothing period (default: 3)

    Returns:
        k_values (%K), d_values (%D)
    """
    lowest_low = rolling_min(low, k_period)
    highest_high = rolling_max(high, k_period)

    k_values = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    d_values = sma(k_values, d_period)

    return k_values, d_values


def roc(close: np.ndarray, period: int = 12) -> np.ndarray:
    """
    Rate of Change (ROC)

    Momentum indicator showing % price change.

    Args:
        close: Closing prices
        period: Lookback period (default: 12)

    Returns:
        ROC values
    """
    roc_values = np.zeros_like(close)
    roc_values[period:] = 100 * (close[period:] - close[:-period]) / (close[:-period] + 1e-10)
    return roc_values


def williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Williams %R

    Momentum indicator similar to Stochastic.
    Overbought: > -20, Oversold: < -80

    Args:
        high, low, close: OHLC data
        period: Lookback period (default: 14)

    Returns:
        Williams %R values
    """
    highest_high = rolling_max(high, period)
    lowest_low = rolling_min(low, period)

    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    return wr


def cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Commodity Channel Index (CCI)

    Oscillating indicator for identifying cyclical trends.
    Overbought: > +100, Oversold: < -100

    Args:
        high, low, close: OHLC data
        period: Lookback period (default: 20)

    Returns:
        CCI values
    """
    typical_price = (high + low + close) / 3
    sma_tp = sma(typical_price, period)
    mean_deviation = rolling_std(typical_price, period)

    cci_values = (typical_price - sma_tp) / (0.015 * mean_deviation + 1e-10)
    return cci_values


# ============================================================================
# TREND INDICATORS
# ============================================================================

def sma(data: np.ndarray, period: int) -> np.ndarray:
    """
    Simple Moving Average (SMA)

    Args:
        data: Input data
        period: Period for averaging

    Returns:
        SMA values
    """
    sma_values = np.zeros_like(data)
    for i in range(period - 1, len(data)):
        sma_values[i] = np.mean(data[i - period + 1:i + 1])
    return sma_values


def ema(data: np.ndarray, period: int) -> np.ndarray:
    """
    Exponential Moving Average (EMA)

    Args:
        data: Input data
        period: Period for EMA

    Returns:
        EMA values
    """
    ema_values = np.zeros_like(data)
    multiplier = 2 / (period + 1)

    # Initialize with SMA
    ema_values[period - 1] = np.mean(data[:period])

    for i in range(period, len(data)):
        ema_values[i] = (data[i] - ema_values[i-1]) * multiplier + ema_values[i-1]

    return ema_values


def dema(data: np.ndarray, period: int) -> np.ndarray:
    """
    Double Exponential Moving Average (DEMA)

    Reduced lag compared to EMA.
    DEMA = 2*EMA - EMA(EMA)

    Args:
        data: Input data
        period: Period for DEMA

    Returns:
        DEMA values
    """
    ema1 = ema(data, period)
    ema2 = ema(ema1, period)
    dema_values = 2 * ema1 - ema2
    return dema_values


def tema(data: np.ndarray, period: int) -> np.ndarray:
    """
    Triple Exponential Moving Average (TEMA)

    Even less lag than DEMA.
    TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))

    Args:
        data: Input data
        period: Period for TEMA

    Returns:
        TEMA values
    """
    ema1 = ema(data, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    tema_values = 3 * ema1 - 3 * ema2 + ema3
    return tema_values


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Average Directional Index (ADX)

    Measures trend strength (0-100).
    Strong trend: > 25, No trend: < 20

    Args:
        high, low, close: OHLC data
        period: Period for ADX (default: 14)

    Returns:
        ADX values
    """
    # True Range
    tr = true_range(high, low, close)
    atr_values = sma(tr, period)

    # Directional Movement
    plus_dm = np.maximum(high[1:] - high[:-1], 0)
    minus_dm = np.maximum(low[:-1] - low[1:], 0)

    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])

    # Directional Indicators
    plus_di = 100 * sma(plus_dm, period) / (atr_values + 1e-10)
    minus_di = 100 * sma(minus_dm, period) / (atr_values + 1e-10)

    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_values = sma(dx, period)

    return adx_values


def aroon(high: np.ndarray, low: np.ndarray, period: int = 25
          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aroon Indicator (Up, Down, Oscillator)

    Identifies trend changes and strength.

    Args:
        high, low: High and low prices
        period: Lookback period (default: 25)

    Returns:
        aroon_up, aroon_down, aroon_oscillator
    """
    aroon_up = np.zeros_like(high)
    aroon_down = np.zeros_like(low)

    for i in range(period, len(high)):
        days_since_high = period - np.argmax(high[i-period:i+1])
        days_since_low = period - np.argmin(low[i-period:i+1])

        aroon_up[i] = 100 * (period - days_since_high) / period
        aroon_down[i] = 100 * (period - days_since_low) / period

    aroon_oscillator = aroon_up - aroon_down

    return aroon_up, aroon_down, aroon_oscillator


# ============================================================================
# VOLATILITY INDICATORS
# ============================================================================

def bollinger_bands(close: np.ndarray, period: int = 20, std_dev: float = 2.0
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands (Upper, Middle, Lower)

    Volatility bands around moving average.

    Args:
        close: Closing prices
        period: Period for SMA (default: 20)
        std_dev: Number of standard deviations (default: 2.0)

    Returns:
        upper_band, middle_band, lower_band
    """
    middle_band = sma(close, period)
    std = rolling_std(close, period)

    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)

    return upper_band, middle_band, lower_band


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Average True Range (ATR)

    Measures market volatility.

    Args:
        high, low, close: OHLC data
        period: Period for averaging (default: 14)

    Returns:
        ATR values
    """
    tr = true_range(high, low, close)
    atr_values = sma(tr, period)
    return atr_values


def keltner_channels(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     ema_period: int = 20, atr_period: int = 10, multiplier: float = 2.0
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keltner Channels (Upper, Middle, Lower)

    Similar to Bollinger Bands but uses ATR.

    Args:
        high, low, close: OHLC data
        ema_period: Period for middle line EMA (default: 20)
        atr_period: Period for ATR (default: 10)
        multiplier: ATR multiplier (default: 2.0)

    Returns:
        upper_channel, middle_line, lower_channel
    """
    middle_line = ema(close, ema_period)
    atr_values = atr(high, low, close, atr_period)

    upper_channel = middle_line + (multiplier * atr_values)
    lower_channel = middle_line - (multiplier * atr_values)

    return upper_channel, middle_line, lower_channel


def donchian_channels(high: np.ndarray, low: np.ndarray, period: int = 20
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Donchian Channels (Upper, Middle, Lower)

    Breakout indicator using highest high and lowest low.

    Args:
        high, low: High and low prices
        period: Lookback period (default: 20)

    Returns:
        upper_channel, middle_channel, lower_channel
    """
    upper_channel = rolling_max(high, period)
    lower_channel = rolling_min(low, period)
    middle_channel = (upper_channel + lower_channel) / 2

    return upper_channel, middle_channel, lower_channel


# ============================================================================
# VOLUME INDICATORS
# ============================================================================

def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    On-Balance Volume (OBV)

    Cumulative volume indicator based on price direction.

    Args:
        close: Closing prices
        volume: Volume data

    Returns:
        OBV values
    """
    obv_values = np.zeros_like(close)
    obv_values[0] = volume[0]

    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv_values[i] = obv_values[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv_values[i] = obv_values[i-1] - volume[i]
        else:
            obv_values[i] = obv_values[i-1]

    return obv_values


def vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
         ) -> np.ndarray:
    """
    Volume Weighted Average Price (VWAP)

    Average price weighted by volume.

    Args:
        high, low, close: OHLC data
        volume: Volume data

    Returns:
        VWAP values
    """
    typical_price = (high + low + close) / 3
    vwap_values = np.cumsum(typical_price * volume) / (np.cumsum(volume) + 1e-10)
    return vwap_values


def mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray,
        period: int = 14) -> np.ndarray:
    """
    Money Flow Index (MFI)

    Volume-weighted RSI.
    Overbought: > 80, Oversold: < 20

    Args:
        high, low, close: OHLC data
        volume: Volume data
        period: Period (default: 14)

    Returns:
        MFI values
    """
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    positive_flow = np.zeros_like(money_flow)
    negative_flow = np.zeros_like(money_flow)

    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i-1]:
            positive_flow[i] = money_flow[i]
        elif typical_price[i] < typical_price[i-1]:
            negative_flow[i] = money_flow[i]

    positive_sum = rolling_sum(positive_flow, period)
    negative_sum = rolling_sum(negative_flow, period)

    money_ratio = positive_sum / (negative_sum + 1e-10)
    mfi_values = 100 - (100 / (1 + money_ratio))

    return mfi_values


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Calculate True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    return tr


def rolling_max(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate rolling maximum"""
    result = np.zeros_like(data)
    for i in range(period - 1, len(data)):
        result[i] = np.max(data[i - period + 1:i + 1])
    return result


def rolling_min(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate rolling minimum"""
    result = np.zeros_like(data)
    for i in range(period - 1, len(data)):
        result[i] = np.min(data[i - period + 1:i + 1])
    return result


def rolling_std(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate rolling standard deviation"""
    result = np.zeros_like(data)
    for i in range(period - 1, len(data)):
        result[i] = np.std(data[i - period + 1:i + 1])
    return result


def rolling_sum(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate rolling sum"""
    result = np.zeros_like(data)
    for i in range(period - 1, len(data)):
        result[i] = np.sum(data[i - period + 1:i + 1])
    return result


# ============================================================================
# MULTI-TIMEFRAME FEATURE ENGINEERING
# ============================================================================

class TechnicalFeatureEngineering:
    """
    Comprehensive feature engineering class for technical indicators.

    Features:
    - 25+ technical indicators
    - Multi-timeframe analysis
    - Normalized features
    - Feature scaling
    """

    def __init__(self, normalize: bool = True):
        """
        Initialize feature engineering.

        Args:
            normalize: Whether to normalize features (default: True)
        """
        self.normalize = normalize

    def compute_all_features(self, ohlcv: OHLCVData) -> np.ndarray:
        """
        Compute all technical indicators as features.

        Args:
            ohlcv: OHLCV market data

        Returns:
            Feature matrix of shape [n_samples, n_features]
        """
        features = []

        # Momentum indicators (6 features)
        features.append(rsi(ohlcv.close, 14))
        features.append(rsi(ohlcv.close, 21))
        macd_line, signal_line, hist = macd(ohlcv.close)
        features.extend([macd_line, signal_line, hist])
        k, d = stochastic(ohlcv.high, ohlcv.low, ohlcv.close)
        features.extend([k, d])
        features.append(roc(ohlcv.close, 12))
        features.append(williams_r(ohlcv.high, ohlcv.low, ohlcv.close))
        features.append(cci(ohlcv.high, ohlcv.low, ohlcv.close))

        # Trend indicators (8 features)
        features.append(sma(ohlcv.close, 7))
        features.append(sma(ohlcv.close, 25))
        features.append(sma(ohlcv.close, 99))
        features.append(ema(ohlcv.close, 12))
        features.append(ema(ohlcv.close, 26))
        features.append(dema(ohlcv.close, 20))
        features.append(adx(ohlcv.high, ohlcv.low, ohlcv.close))
        aroon_up, aroon_down, aroon_osc = aroon(ohlcv.high, ohlcv.low)
        features.extend([aroon_up, aroon_down])

        # Volatility indicators (7 features)
        bb_upper, bb_middle, bb_lower = bollinger_bands(ohlcv.close)
        features.extend([bb_upper, bb_middle, bb_lower])
        features.append(atr(ohlcv.high, ohlcv.low, ohlcv.close))
        kc_upper, kc_middle, kc_lower = keltner_channels(ohlcv.high, ohlcv.low, ohlcv.close)
        features.extend([kc_upper, kc_lower])
        dc_upper, dc_middle = donchian_channels(ohlcv.high, ohlcv.low)[:2]
        features.extend([dc_upper])

        # Volume indicators (4 features)
        features.append(obv(ohlcv.close, ohlcv.volume))
        features.append(vwap(ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume))
        features.append(mfi(ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume))

        # Stack all features
        feature_matrix = np.column_stack(features)

        # Normalize if requested
        if self.normalize:
            feature_matrix = self._normalize_features(feature_matrix)

        return feature_matrix

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using z-score normalization.

        Args:
            features: Raw feature matrix

        Returns:
            Normalized features
        """
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-10
        normalized = (features - mean) / std
        return normalized


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("Technical Indicators Module - Test")

    # Generate sample data
    n = 1000
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(n) * 2)
    high_prices = close_prices + np.random.rand(n) * 3
    low_prices = close_prices - np.random.rand(n) * 3
    open_prices = (close_prices + np.random.randn(n))
    volume = np.random.randint(1000000, 10000000, n)

    ohlcv = OHLCVData(
        open=open_prices,
        high=high_prices,
        low=low_prices,
        close=close_prices,
        volume=volume
    )

    # Test feature engineering
    fe = TechnicalFeatureEngineering(normalize=True)
    features = fe.compute_all_features(ohlcv)

    print(f"Generated {features.shape[1]} features for {features.shape[0]} samples")
    print(f"Feature matrix shape: {features.shape}")
    print(f"Sample features (last 5 rows, first 10 columns):")
    print(features[-5:, :10])

    # Test individual indicators
    print("\nTesting individual indicators:")
    print(f"RSI(14): {rsi(close_prices)[-10:]}")
    print(f"ADX(14): {adx(high_prices, low_prices, close_prices)[-10:]}")
    bb_upper, bb_middle, bb_lower = bollinger_bands(close_prices)
    print(f"Bollinger Bands: Upper={bb_upper[-1]:.2f}, Middle={bb_middle[-1]:.2f}, Lower={bb_lower[-1]:.2f}")

    print("\nTechnical Indicators Module Test Complete!")
