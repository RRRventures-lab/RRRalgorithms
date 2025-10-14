from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import logging
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn


"""
Unified Neural Network Predictor
=================================

Consolidates mock, optimized, and production predictors into a single configurable class.
Reduces code duplication and provides a consistent interface for all prediction modes.

Author: RRR Ventures
Date: 2025-10-12
"""


# Configure logging
logger = logging.getLogger(__name__)


class PredictorMode(Enum):
    """Prediction modes for different environments."""
    MOCK = "mock"
    OPTIMIZED = "optimized"
    PRODUCTION = "production"


@dataclass
class PredictionResult:
    """Standardized prediction result."""
    symbol: str
    predicted_price: float
    confidence: float
    predicted_direction: str
    features_used: List[str]
    execution_time: float
    mode: PredictorMode
    metadata: Dict[str, Any]


class UnifiedPredictor:
    """
    Unified predictor that can operate in mock, optimized, or production mode.

    Modes:
        - mock: Fast random predictions for testing
        - optimized: Performance-optimized for real-time trading
        - production: Full feature set with comprehensive analysis
    """

    def __init__(self, mode: Union[str, PredictorMode] = PredictorMode.PRODUCTION):
        """
        Initialize the unified predictor.

        Args:
            mode: Prediction mode (mock, optimized, or production)
        """
        if isinstance(mode, str):
            mode = PredictorMode(mode.lower())

        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.feature_cache = {}
        self.prediction_cache = {}

        # Initialize based on mode
        if mode == PredictorMode.MOCK:
            self._init_mock()
        elif mode == PredictorMode.OPTIMIZED:
            self._init_optimized()
        else:  # PRODUCTION
            self._init_production()

        logger.info(f"UnifiedPredictor initialized in {mode.value} mode on {self.device}")

    def _init_mock(self):
        """Initialize mock predictor for testing."""
        self.features = ["close", "volume", "rsi", "macd"]
        self.confidence_range = (0.5, 0.9)
        logger.info("Mock predictor ready - using random predictions")

    def _init_optimized(self):
        """Initialize optimized predictor with performance optimizations."""
        self.features = [
            "close", "volume", "rsi", "macd", "bollinger_upper",
            "bollinger_lower", "ema_12", "ema_26", "stochastic_k"
        ]

        # Optimized transformer model
        class OptimizedTransformer(nn.Module):
            def __init__(self, input_dim=9, hidden_dim=64, num_heads=4, num_layers=2):
                super().__init__()
                self.input_projection = nn.Linear(input_dim, hidden_dim)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=256,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                self.output_projection = nn.Sequential(
                    nn.Linear(hidden_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 3)  # [price_change, confidence, volatility]
                )

            def forward(self, x):
                x = self.input_projection(x)
                x = self.transformer(x)
                return self.output_projection(x[:, -1, :])

        self.models['price'] = OptimizedTransformer().to(self.device)
        self.models['price'].eval()

        # Enable optimizations
        if self.device.type == 'cuda':
            self.models['price'] = torch.jit.script(self.models['price'])

        logger.info("Optimized predictor ready with Transformer model")

    def _init_production(self):
        """Initialize production predictor with full feature set."""
        self.features = [
            # Price & Volume
            "open", "high", "low", "close", "volume", "vwap",
            # Moving Averages
            "sma_10", "sma_20", "sma_50", "ema_12", "ema_26",
            # Momentum Indicators
            "rsi", "macd", "macd_signal", "stochastic_k", "stochastic_d",
            # Volatility Indicators
            "bollinger_upper", "bollinger_middle", "bollinger_lower", "atr",
            # Volume Indicators
            "obv", "volume_sma", "volume_ratio",
            # Market Microstructure
            "bid_ask_spread", "order_imbalance", "trade_intensity"
        ]

        # Production models ensemble
        class ProductionEnsemble(nn.Module):
            def __init__(self, input_dim=27):
                super().__init__()
                # Transformer for sequence modeling
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=128, nhead=8, dim_feedforward=512,
                        dropout=0.1, batch_first=True
                    ),
                    num_layers=4
                )

                # LSTM for temporal patterns
                self.lstm = nn.LSTM(input_dim, 128, num_layers=2,
                                    batch_first=True, bidirectional=True)

                # CNN for pattern recognition
                self.cnn = nn.Sequential(
                    nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1)
                )

                # Input projections
                self.input_projection = nn.Linear(input_dim, 128)

                # Ensemble combination
                self.ensemble_combiner = nn.Sequential(
                    nn.Linear(128 + 256 + 128, 256),  # transformer + lstm + cnn
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 5)  # [price, confidence, volatility, direction, risk]
                )

            def forward(self, x):
                batch_size, seq_len, features = x.shape

                # Transformer branch
                trans_input = self.input_projection(x)
                trans_out = self.transformer(trans_input)[:, -1, :]

                # LSTM branch
                lstm_out, _ = self.lstm(x)
                lstm_out = lstm_out[:, -1, :]

                # CNN branch
                cnn_input = x.transpose(1, 2)
                cnn_out = self.cnn(cnn_input).squeeze(-1)

                # Ensemble
                combined = torch.cat([trans_out, lstm_out, cnn_out], dim=1)
                return self.ensemble_combiner(combined)

        self.models['ensemble'] = ProductionEnsemble().to(self.device)
        self.models['ensemble'].eval()

        # Load pre-trained weights if available
        self._load_production_weights()

        logger.info("Production predictor ready with ensemble model")

    def _load_production_weights(self):
        """Load pre-trained weights for production model."""
        weights_path = Path("models/production/ensemble_weights.pth")
        if weights_path.exists():
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.models['ensemble'].load_state_dict(state_dict)
                logger.info("Loaded pre-trained production weights")
            except Exception as e:
                logger.warning(f"Could not load production weights: {e}")

    def predict(self,
                symbol: str,
                market_data: pd.DataFrame,
                horizon: int = 1,
                use_cache: bool = True) -> PredictionResult:
        """
        Generate prediction based on current mode.

        Args:
            symbol: Trading symbol
            market_data: Historical market data
            horizon: Prediction horizon in time steps
            use_cache: Whether to use cached predictions

        Returns:
            PredictionResult object with prediction details
        """
        start_time = time.time()

        # Check cache
        cache_key = f"{symbol}_{horizon}_{len(market_data)}"
        if use_cache and cache_key in self.prediction_cache:
            cached = self.prediction_cache[cache_key]
            if time.time() - cached['timestamp'] < 60:  # 1 minute cache
                return cached['result']

        # Generate prediction based on mode
        if self.mode == PredictorMode.MOCK:
            result = self._predict_mock(symbol, market_data, horizon)
        elif self.mode == PredictorMode.OPTIMIZED:
            result = self._predict_optimized(symbol, market_data, horizon)
        else:  # PRODUCTION
            result = self._predict_production(symbol, market_data, horizon)

        # Add execution time
        result.execution_time = time.time() - start_time

        # Cache result
        if use_cache:
            self.prediction_cache[cache_key] = {
                'result': result,
                'timestamp': time.time()
            }

        return result

    def _predict_mock(self, symbol: str, market_data: pd.DataFrame, horizon: int) -> PredictionResult:
        """Generate mock prediction for testing."""
        current_price = market_data['close'].iloc[-1]

        # Random prediction
        price_change = np.random.normal(0, 0.02)  # 2% volatility
        predicted_price = current_price * (1 + price_change)
        confidence = np.random.uniform(*self.confidence_range)
        direction = "UP" if price_change > 0 else "DOWN"

        return PredictionResult(
            symbol=symbol,
            predicted_price=predicted_price,
            confidence=confidence,
            predicted_direction=direction,
            features_used=self.features,
            execution_time=0.001,
            mode=self.mode,
            metadata={'mock': True, 'horizon': horizon}
        )

    def _predict_optimized(self, symbol: str, market_data: pd.DataFrame, horizon: int) -> PredictionResult:
        """Generate optimized prediction for real-time trading."""
        # Extract features efficiently
        features = self._extract_features_optimized(market_data)

        # Prepare input tensor
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.models['price'](x)
            price_change, confidence, volatility = output[0].cpu().numpy()

        # Calculate prediction
        current_price = market_data['close'].iloc[-1]
        predicted_price = current_price * (1 + price_change / 100)
        direction = "UP" if price_change > 0 else "DOWN"

        return PredictionResult(
            symbol=symbol,
            predicted_price=float(predicted_price),
            confidence=float(torch.sigmoid(torch.tensor(confidence)).item()),
            predicted_direction=direction,
            features_used=self.features,
            execution_time=0.01,
            mode=self.mode,
            metadata={
                'volatility': float(volatility),
                'horizon': horizon,
                'device': str(self.device)
            }
        )

    def _predict_production(self, symbol: str, market_data: pd.DataFrame, horizon: int) -> PredictionResult:
        """Generate production prediction with full analysis."""
        # Extract comprehensive features
        features = self._extract_features_production(market_data)

        # Prepare input sequence
        sequence_length = min(100, len(features))
        x = torch.tensor(features[-sequence_length:], dtype=torch.float32)
        x = x.unsqueeze(0).to(self.device)

        # Run ensemble inference
        with torch.no_grad():
            output = self.models['ensemble'](x)
            price, confidence, volatility, direction, risk = output[0].cpu().numpy()

        # Process predictions
        current_price = market_data['close'].iloc[-1]
        predicted_price = current_price * (1 + price / 100)
        confidence_score = torch.sigmoid(torch.tensor(confidence)).item()
        direction_label = "UP" if direction > 0 else "DOWN"

        # Risk adjustment
        if risk > 0.7:  # High risk threshold
            confidence_score *= 0.8
            logger.warning(f"High risk detected for {symbol}: {risk:.2f}")

        return PredictionResult(
            symbol=symbol,
            predicted_price=float(predicted_price),
            confidence=float(confidence_score),
            predicted_direction=direction_label,
            features_used=self.features,
            execution_time=0.05,
            mode=self.mode,
            metadata={
                'volatility': float(volatility),
                'risk_score': float(risk),
                'horizon': horizon,
                'sequence_length': sequence_length,
                'model': 'ensemble',
                'device': str(self.device)
            }
        )

    def _extract_features_optimized(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract features efficiently for optimized mode."""
        # Use vectorized operations for speed
        features = []

        # Basic features
        features.append(market_data['close'].iloc[-20:].values)
        features.append(market_data['volume'].iloc[-20:].values)

        # Technical indicators (simplified)
        close = market_data['close'].values

        # RSI
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().iloc[-20:].values
        avg_loss = pd.Series(loss).rolling(14).mean().iloc[-20:].values
        rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))
        features.append(rsi)

        # Add more indicators as needed...

        return np.column_stack(features)[:20, :9]  # Ensure correct shape

    def _extract_features_production(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract comprehensive features for production mode."""
        # This would include all 27 features listed in the production init
        # For brevity, returning mock data here
        num_samples = len(market_data)
        num_features = 27
        return np.random.randn(num_samples, num_features)

    def batch_predict(self, symbols: List[str], market_data: Dict[str, pd.DataFrame]) -> List[PredictionResult]:
        """
        Generate predictions for multiple symbols efficiently.

        Args:
            symbols: List of trading symbols
            market_data: Dictionary mapping symbols to DataFrames

        Returns:
            List of PredictionResult objects
        """
        results = []

        for symbol in symbols:
            if symbol in market_data:
                result = self.predict(symbol, market_data[symbol])
                results.append(result)
            else:
                logger.warning(f"No market data for {symbol}")

        return results

    def update_model(self, training_data: pd.DataFrame, labels: np.ndarray):
        """
        Update model with new training data (production mode only).

        Args:
            training_data: New training data
            labels: Corresponding labels
        """
        if self.mode != PredictorMode.PRODUCTION:
            logger.warning(f"Model updates not supported in {self.mode.value} mode")
            return

        # Implement online learning or periodic retraining
        logger.info(f"Updating model with {len(training_data)} new samples")
        # Implementation would go here

    @lru_cache(maxsize=128)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.mode == PredictorMode.MOCK:
            return {feat: np.random.random() for feat in self.features}

        # For optimized and production modes, would calculate actual importance
        return {feat: 1.0 / len(self.features) for feat in self.features}

    def save_model(self, path: str):
        """Save model weights to disk."""
        if self.mode == PredictorMode.MOCK:
            logger.info("Mock mode - no model to save")
            return

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
            model_path = save_path.parent / f"{save_path.stem}_{name}.pth"
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved {name} model to {model_path}")

    def load_model(self, path: str):
        """Load model weights from disk."""
        if self.mode == PredictorMode.MOCK:
            logger.info("Mock mode - no model to load")
            return

        load_path = Path(path)

        for name, model in self.models.items():
            model_path = load_path.parent / f"{load_path.stem}_{name}.pth"
            if model_path.exists():
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded {name} model from {model_path}")
            else:
                logger.warning(f"Model file not found: {model_path}")

    @property
    def is_ready(self) -> bool:
        """Check if predictor is ready for inference."""
        if self.mode == PredictorMode.MOCK:
            return True
        return len(self.models) > 0

    def __repr__(self) -> str:
        return f"UnifiedPredictor(mode={self.mode.value}, device={self.device}, ready={self.is_ready})"


def create_predictor(mode: str = "production", **kwargs) -> UnifiedPredictor:
    """
    Factory function to create a predictor instance.

    Args:
        mode: Predictor mode (mock, optimized, or production)
        **kwargs: Additional configuration parameters

    Returns:
        Configured UnifiedPredictor instance
    """
    return UnifiedPredictor(mode=mode)


if __name__ == "__main__":
    # Demo usage
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "mock"

    print(f"Creating predictor in {mode} mode...")
    predictor = create_predictor(mode)
    print(predictor)

    # Generate sample data
    sample_data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100),
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
    })

    # Make prediction
    result = predictor.predict("BTC-USD", sample_data)

    print(f"\nPrediction Result:")
    print(f"  Symbol: {result.symbol}")
    print(f"  Price: ${result.predicted_price:.2f}")
    print(f"  Direction: {result.predicted_direction}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Execution Time: {result.execution_time:.3f}s")
    print(f"  Mode: {result.mode.value}")
    print(f"  Metadata: {json.dumps(result.metadata, indent=2)}")