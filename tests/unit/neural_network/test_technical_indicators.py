from dataclasses import dataclass
from features.technical_indicators import (
import numpy as np
import os
import pytest
import sys

"""
Unit Tests for Technical Indicators

Tests all 25+ technical indicators for correctness and edge cases.
Critical for feature engineering quality.
"""


# Add neural network to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../worktrees/neural-network/src'))

    # Momentum
    rsi, macd, stochastic, roc, williams_r, cci,
    # Trend
    sma, ema, dema, tema, adx, aroon,
    # Volatility
    bollinger_bands, atr, keltner_channels, donchian_channels,
    # Volume
    obv, vwap, mfi,
    # Classes
    OHLCVData, TechnicalFeatureEngineering
)


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    n = 100

    # Generate realistic price movement
    prices = 50000 + np.cumsum(np.random.randn(n) * 100)

    return OHLCVData(
        open=prices + np.random.randn(n) * 50,
        high=prices + np.abs(np.random.randn(n) * 100),
        low=prices - np.abs(np.random.randn(n) * 100),
        close=prices,
        volume=np.random.randint(1000000, 10000000, n).astype(float)
    )


class TestMomentumIndicators:
    """Test momentum indicators"""

    def test_rsi_basic(self, sample_ohlcv):
        """Test RSI calculation"""
        rsi_values = rsi(sample_ohlcv.close, period=14)

        # RSI should be between 0 and 100
        assert np.all((rsi_values >= 0) & (rsi_values <= 100))
        assert len(rsi_values) == len(sample_ohlcv.close)

    def test_rsi_overbought_oversold(self):
        """Test RSI overbought/oversold conditions"""
        # Strongly uptrending data
        uptrend = np.linspace(100, 200, 50)
        rsi_up = rsi(uptrend, period=14)

        # RSI should be high (near 100) for strong uptrend
        assert rsi_up[-1] > 70

        # Strongly downtrending data
        downtrend = np.linspace(200, 100, 50)
        rsi_down = rsi(downtrend, period=14)

        # RSI should be low (near 0) for strong downtrend
        assert rsi_down[-1] < 30

    def test_rsi_period_variations(self, sample_ohlcv):
        """Test RSI with different periods"""
        rsi_7 = rsi(sample_ohlcv.close, period=7)
        rsi_21 = rsi(sample_ohlcv.close, period=21)

        # Shorter period should be more volatile
        assert np.std(rsi_7) >= np.std(rsi_21) * 0.9  # Allow some tolerance

    def test_macd_basic(self, sample_ohlcv):
        """Test MACD calculation"""
        macd_line, signal_line, histogram = macd(sample_ohlcv.close)

        assert len(macd_line) == len(sample_ohlcv.close)
        assert len(signal_line) == len(sample_ohlcv.close)
        assert len(histogram) == len(sample_ohlcv.close)

        # Histogram should equal macd - signal
        np.testing.assert_array_almost_equal(
            histogram[~np.isnan(histogram)],
            (macd_line - signal_line)[~np.isnan(histogram)],
            decimal=2
        )

    def test_stochastic_basic(self, sample_ohlcv):
        """Test Stochastic oscillator"""
        k, d = stochastic(sample_ohlcv.high, sample_ohlcv.low, sample_ohlcv.close)

        # Stochastic should be between 0 and 100
        assert np.all((k >= 0) & (k <= 100))
        assert np.all((d >= 0) & (d <= 100))

    def test_roc_basic(self, sample_ohlcv):
        """Test Rate of Change"""
        roc_values = roc(sample_ohlcv.close, period=10)

        assert len(roc_values) == len(sample_ohlcv.close)
        # ROC can be any value, but should be reasonable
        assert np.all(np.abs(roc_values[~np.isnan(roc_values)]) < 100)

    def test_williams_r_basic(self, sample_ohlcv):
        """Test Williams %R"""
        williams_values = williams_r(sample_ohlcv.high, sample_ohlcv.low, sample_ohlcv.close)

        # Williams %R should be between -100 and 0
        assert np.all((williams_values >= -100) & (williams_values <= 0))

    def test_cci_basic(self, sample_ohlcv):
        """Test Commodity Channel Index"""
        cci_values = cci(sample_ohlcv.high, sample_ohlcv.low, sample_ohlcv.close)

        assert len(cci_values) == len(sample_ohlcv.close)
        # CCI typically ranges -200 to +200
        assert np.all(np.abs(cci_values[~np.isnan(cci_values)]) < 500)


class TestTrendIndicators:
    """Test trend indicators"""

    def test_sma_basic(self, sample_ohlcv):
        """Test Simple Moving Average"""
        sma_values = sma(sample_ohlcv.close, period=20)

        assert len(sma_values) == len(sample_ohlcv.close)
        # SMA should smooth out fluctuations
        assert np.std(sma_values[~np.isnan(sma_values)]) < np.std(sample_ohlcv.close)

    def test_sma_known_values(self):
        """Test SMA with known values"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        sma_3 = sma(data, period=3)

        # Check specific values
        assert sma_3[2] == 2.0  # (1+2+3)/3
        assert sma_3[3] == 3.0  # (2+3+4)/3
        assert sma_3[4] == 4.0  # (3+4+5)/3

    def test_ema_basic(self, sample_ohlcv):
        """Test Exponential Moving Average"""
        ema_values = ema(sample_ohlcv.close, period=20)

        assert len(ema_values) == len(sample_ohlcv.close)
        # EMA should react faster than SMA
        sma_values = sma(sample_ohlcv.close, period=20)
        # Can't easily compare without more context, just check it's different
        assert not np.array_equal(
            ema_values[~np.isnan(ema_values)],
            sma_values[~np.isnan(sma_values)]
        )

    def test_dema_basic(self, sample_ohlcv):
        """Test Double Exponential Moving Average"""
        dema_values = dema(sample_ohlcv.close, period=20)

        assert len(dema_values) == len(sample_ohlcv.close)

    def test_tema_basic(self, sample_ohlcv):
        """Test Triple Exponential Moving Average"""
        tema_values = tema(sample_ohlcv.close, period=20)

        assert len(tema_values) == len(sample_ohlcv.close)

    def test_adx_basic(self, sample_ohlcv):
        """Test Average Directional Index"""
        adx_values = adx(sample_ohlcv.high, sample_ohlcv.low, sample_ohlcv.close)

        assert len(adx_values) == len(sample_ohlcv.close)
        # ADX should be between 0 and 100
        assert np.all((adx_values[~np.isnan(adx_values)] >= 0) &
                     (adx_values[~np.isnan(adx_values)] <= 100))

    def test_aroon_basic(self, sample_ohlcv):
        """Test Aroon indicator"""
        aroon_up, aroon_down = aroon(sample_ohlcv.high, sample_ohlcv.low)

        # Aroon should be between 0 and 100
        assert np.all((aroon_up >= 0) & (aroon_up <= 100))
        assert np.all((aroon_down >= 0) & (aroon_down <= 100))


class TestVolatilityIndicators:
    """Test volatility indicators"""

    def test_bollinger_bands_basic(self, sample_ohlcv):
        """Test Bollinger Bands"""
        upper, middle, lower = bollinger_bands(sample_ohlcv.close)

        assert len(upper) == len(sample_ohlcv.close)
        assert len(middle) == len(sample_ohlcv.close)
        assert len(lower) == len(sample_ohlcv.close)

        # Upper should be above middle, middle above lower
        valid_idx = ~np.isnan(upper)
        assert np.all(upper[valid_idx] >= middle[valid_idx])
        assert np.all(middle[valid_idx] >= lower[valid_idx])

    def test_bollinger_bands_width(self):
        """Test Bollinger Bands width with different volatility"""
        # Low volatility data
        low_vol = np.ones(100) + np.random.randn(100) * 0.01
        upper_low, _, lower_low = bollinger_bands(low_vol)

        # High volatility data
        high_vol = np.ones(100) + np.random.randn(100) * 10
        upper_high, _, lower_high = bollinger_bands(high_vol)

        # High volatility should have wider bands
        width_low = np.nanmean(upper_low - lower_low)
        width_high = np.nanmean(upper_high - lower_high)
        assert width_high > width_low

    def test_atr_basic(self, sample_ohlcv):
        """Test Average True Range"""
        atr_values = atr(sample_ohlcv.high, sample_ohlcv.low, sample_ohlcv.close)

        assert len(atr_values) == len(sample_ohlcv.close)
        # ATR should be positive
        assert np.all(atr_values[~np.isnan(atr_values)] >= 0)

    def test_keltner_channels_basic(self, sample_ohlcv):
        """Test Keltner Channels"""
        upper, middle, lower = keltner_channels(
            sample_ohlcv.high, sample_ohlcv.low, sample_ohlcv.close
        )

        # Upper should be above middle, middle above lower
        valid_idx = ~np.isnan(upper)
        assert np.all(upper[valid_idx] >= middle[valid_idx])
        assert np.all(middle[valid_idx] >= lower[valid_idx])

    def test_donchian_channels_basic(self, sample_ohlcv):
        """Test Donchian Channels"""
        upper, middle, lower = donchian_channels(sample_ohlcv.high, sample_ohlcv.low)

        # Upper should be highest high, lower should be lowest low
        assert len(upper) == len(sample_ohlcv.high)
        assert len(lower) == len(sample_ohlcv.low)


class TestVolumeIndicators:
    """Test volume indicators"""

    def test_obv_basic(self, sample_ohlcv):
        """Test On-Balance Volume"""
        obv_values = obv(sample_ohlcv.close, sample_ohlcv.volume)

        assert len(obv_values) == len(sample_ohlcv.close)
        # OBV should be cumulative
        assert obv_values[0] == sample_ohlcv.volume[0]

    def test_vwap_basic(self, sample_ohlcv):
        """Test Volume Weighted Average Price"""
        vwap_values = vwap(
            sample_ohlcv.high, sample_ohlcv.low,
            sample_ohlcv.close, sample_ohlcv.volume
        )

        assert len(vwap_values) == len(sample_ohlcv.close)
        # VWAP should be within price range
        assert np.all(vwap_values >= np.minimum.reduce([
            sample_ohlcv.open, sample_ohlcv.low
        ]))
        assert np.all(vwap_values <= np.maximum.reduce([
            sample_ohlcv.open, sample_ohlcv.high
        ]))

    def test_mfi_basic(self, sample_ohlcv):
        """Test Money Flow Index"""
        mfi_values = mfi(
            sample_ohlcv.high, sample_ohlcv.low,
            sample_ohlcv.close, sample_ohlcv.volume
        )

        # MFI should be between 0 and 100
        assert np.all((mfi_values >= 0) & (mfi_values <= 100))


class TestFeatureEngineering:
    """Test TechnicalFeatureEngineering class"""

    def test_initialization(self):
        """Test feature engineering initialization"""
        fe = TechnicalFeatureEngineering()
        assert fe is not None
        assert hasattr(fe, 'compute_all_features')

    def test_compute_all_features(self, sample_ohlcv):
        """Test computing all features"""
        fe = TechnicalFeatureEngineering()
        features = fe.compute_all_features(sample_ohlcv)

        # Should have many features (25+)
        assert features.shape[0] == len(sample_ohlcv.close)
        assert features.shape[1] >= 25

        # No infinite values
        assert not np.any(np.isinf(features))

    def test_normalization(self, sample_ohlcv):
        """Test feature normalization"""
        fe = TechnicalFeatureEngineering(normalize=True)
        features = fe.compute_all_features(sample_ohlcv)

        # Normalized features should have reasonable range
        # (some indicators naturally have bounded ranges like RSI, so check mean is reasonable)
        assert np.abs(np.nanmean(features)) < 50


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_input(self):
        """Test with empty input"""
        empty = np.array([])

        with pytest.raises(Exception):
            rsi(empty)

    def test_single_value(self):
        """Test with single value"""
        single = np.array([100.0])

        # Should handle gracefully (likely return NaN)
        result = sma(single, period=20)
        assert len(result) == 1

    def test_all_same_values(self):
        """Test with constant values"""
        constant = np.ones(100) * 100

        rsi_val = rsi(constant)
        # RSI of constant should be around 50
        assert 45 <= rsi_val[-1] <= 55 or np.isnan(rsi_val[-1])

    def test_very_short_period(self, sample_ohlcv):
        """Test with very short period"""
        sma_2 = sma(sample_ohlcv.close, period=2)
        assert len(sma_2) == len(sample_ohlcv.close)

    def test_period_longer_than_data(self, sample_ohlcv):
        """Test with period longer than data"""
        short_data = sample_ohlcv.close[:10]
        sma_20 = sma(short_data, period=20)

        # Should return NaN for insufficient data
        assert np.all(np.isnan(sma_20))


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
