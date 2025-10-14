from src.data_pipeline.mock_data_source import MockDataSource, MockOrderBook, MockSentimentData
from src.neural_network.mock_predictor import MockPredictor, EnsemblePredictor
import pytest
import time

"""
Unit tests for mock data source and predictor.
"""



@pytest.mark.unit
def test_mock_data_source_initialization():
    """Test mock data source initializes correctly."""
    source = MockDataSource(symbols=['BTC-USD'], volatility=0.02)
    
    assert 'BTC-USD' in source.symbols
    assert source.volatility == 0.02
    assert 'BTC-USD' in source.current_prices


@pytest.mark.unit
def test_mock_data_generation(mock_data_source):
    """Test mock data generation produces valid OHLCV."""
    data = mock_data_source.get_latest_data('BTC-USD')
    
    assert 'BTC-USD' in data
    ohlcv = data['BTC-USD']
    
    # Check all required fields
    assert 'open' in ohlcv
    assert 'high' in ohlcv
    assert 'low' in ohlcv
    assert 'close' in ohlcv
    assert 'volume' in ohlcv
    assert 'timestamp' in ohlcv
    
    # Check OHLC consistency
    assert ohlcv['high'] >= ohlcv['open']
    assert ohlcv['high'] >= ohlcv['close']
    assert ohlcv['low'] <= ohlcv['open']
    assert ohlcv['low'] <= ohlcv['close']


@pytest.mark.unit
def test_historical_data_generation(mock_data_source):
    """Test historical data generation."""
    df = mock_data_source.get_historical_data('BTC-USD', periods=50)
    
    assert len(df) == 50
    assert 'open' in df.columns
    assert 'close' in df.columns
    assert 'volume' in df.columns
    
    # Timestamps should be sequential
    timestamps = df['timestamp'].values
    assert all(timestamps[i] < timestamps[i+1] for i in range(len(timestamps)-1))


@pytest.mark.unit
def test_mock_orderbook():
    """Test mock order book generation."""
    orderbook = MockOrderBook('BTC-USD', mid_price=50000)
    book = orderbook.get_orderbook(levels=10)
    
    assert 'bids' in book
    assert 'asks' in book
    assert len(book['bids']) == 10
    assert len(book['asks']) == 10
    
    # Bids should be decreasing
    bid_prices = [bid[0] for bid in book['bids']]
    assert all(bid_prices[i] >= bid_prices[i+1] for i in range(len(bid_prices)-1))
    
    # Asks should be increasing
    ask_prices = [ask[0] for ask in book['asks']]
    assert all(ask_prices[i] <= ask_prices[i+1] for i in range(len(ask_prices)-1))


@pytest.mark.unit
def test_mock_sentiment():
    """Test mock sentiment generation."""
    sentiment = MockSentimentData()
    data = sentiment.get_sentiment('BTC-USD')
    
    assert 'sentiment' in data
    assert 'confidence' in data
    assert 0 <= data['sentiment'] <= 1
    assert 0 <= data['confidence'] <= 1


@pytest.mark.unit
def test_mock_predictor_trend_following():
    """Test trend following predictor."""
    predictor = MockPredictor(model_type='trend_following', random_seed=42)
    
    # Build price history (uptrend)
    prices = [50000, 50500, 51000, 51500]
    for price in prices:
        predictor.predict('BTC-USD', price)
    
    # Should predict up
    pred = predictor.predict('BTC-USD', 52000)
    assert pred['direction'] in ['up', 'neutral']
    assert pred['predicted_price'] > 51000  # Should predict higher


@pytest.mark.unit
def test_mock_predictor_mean_reversion():
    """Test mean reversion predictor."""
    predictor = MockPredictor(model_type='mean_reversion', random_seed=42)
    
    # Build price history
    prices = [50000, 50000, 50000, 50000, 55000]  # Sudden jump
    for price in prices:
        predictor.predict('BTC-USD', price)
    
    # Should predict reversion (down)
    pred = predictor.predict('BTC-USD', 55000)
    # Mean reversion should predict return to mean
    assert 'mean_price' in pred


@pytest.mark.unit
def test_predictor_output_format(mock_predictor):
    """Test predictor output has required fields."""
    pred = mock_predictor.predict('BTC-USD', 50000)
    
    required_fields = [
        'symbol', 'timestamp', 'predicted_price',
        'predicted_change', 'direction', 'confidence',
        'model_type'
    ]
    
    for field in required_fields:
        assert field in pred, f"Missing field: {field}"
    
    assert pred['direction'] in ['up', 'down', 'neutral']
    assert 0 <= pred['confidence'] <= 1


@pytest.mark.unit
def test_multi_horizon_prediction(mock_predictor):
    """Test multi-horizon predictions."""
    predictions = mock_predictor.predict_multi_horizon(
        'BTC-USD', 50000, horizons=[1, 5, 15, 60]
    )
    
    assert len(predictions) == 4
    assert 1 in predictions
    assert 60 in predictions
    
    # Longer horizons should have larger changes
    change_1 = abs(predictions[1]['predicted_change'])
    change_60 = abs(predictions[60]['predicted_change'])
    # Due to horizon multiplier
    assert change_60 >= change_1


@pytest.mark.unit
def test_ensemble_predictor():
    """Test ensemble predictor combines multiple strategies."""
    ensemble = EnsemblePredictor(strategies=['trend_following', 'mean_reversion'])
    
    pred = ensemble.predict('BTC-USD', 50000)
    
    assert pred['model_type'] == 'ensemble'
    assert 'component_predictions' in pred
    assert len(pred['component_predictions']) == 2


@pytest.mark.unit
def test_feature_importance(mock_predictor):
    """Test feature importance extraction."""
    importance = mock_predictor.get_feature_importance()
    
    assert isinstance(importance, dict)
    assert len(importance) > 0
    
    # Should sum to approximately 1
    total = sum(importance.values())
    assert 0.99 <= total <= 1.01


@pytest.mark.unit
def test_model_stats(mock_predictor):
    """Test model statistics."""
    # Make some predictions
    for i in range(5):
        mock_predictor.predict('BTC-USD', 50000 + i * 100)
    
    stats = mock_predictor.get_model_stats()
    
    assert 'predictions_made' in stats
    assert stats['predictions_made'] == 5
    assert 'model_type' in stats
    assert 'win_rate' in stats

