"""
Mock neural-network style predictors used across tests and local demos.

Historically these lived under ``src/neural-network/`` (note the hyphen) which
made ``import src.neural_network.mock_predictor`` impossible. Restoring the
module fixes imports for the CLI entry points and the unit test fixtures.
"""

from __future__ import annotations

from collections import deque
import math
import random
import statistics
import time
from typing import Deque, Dict, Iterable, List, Optional, Sequence


def _build_rng(seed: Optional[int]) -> random.Random:
    return random.Random(seed if seed is not None else int(time.time() * 1000))


class MockPredictor:
    """Heuristic predictor that mimics a lightweight ML model."""

    def __init__(
        self,
        model_type: str = "trend_following",
        random_seed: Optional[int] = None,
        history_window: int = 200,
    ) -> None:
        self.model_type = model_type
        self.history_window = history_window
        self._rng = _build_rng(random_seed)
        self._history: Dict[str, Deque[float]] = {}
        self._prediction_count: int = 0
        self._last_prediction_ts: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def predict(self, symbol: str, current_price: float) -> Dict[str, float]:
        features = self._update_history_and_features(symbol, current_price)
        prediction = self._build_prediction(symbol, current_price, features, horizon=1)
        self._prediction_count += 1
        self._last_prediction_ts = prediction["timestamp"]
        return prediction

    def predict_multi_horizon(
        self,
        symbol: str,
        current_price: float,
        horizons: Sequence[int],
    ) -> Dict[int, Dict[str, float]]:
        if not horizons:
            raise ValueError("horizons must contain at least one value")

        features = self._update_history_and_features(symbol, current_price)
        results: Dict[int, Dict[str, float]] = {}

        for horizon in horizons:
            results[int(horizon)] = self._build_prediction(symbol, current_price, features, horizon=horizon)

        self._prediction_count += len(horizons)
        self._last_prediction_ts = results[int(horizons[-1])]["timestamp"]
        return results

    def get_feature_importance(self) -> Dict[str, float]:
        if self.model_type == "mean_reversion":
            importance = {
                "price_deviation": 0.5,
                "volatility": 0.2,
                "momentum": 0.2,
                "sentiment": 0.1,
            }
        else:
            importance = {
                "momentum": 0.4,
                "trend": 0.3,
                "volatility": 0.2,
                "volume_profile": 0.1,
            }

        normaliser = sum(importance.values())
        return {key: value / normaliser for key, value in importance.items()}

    def get_model_stats(self) -> Dict[str, float]:
        return {
            "model_type": self.model_type,
            "predictions_made": self._prediction_count,
            "last_prediction_at": self._last_prediction_ts,
            "history_symbols": len(self._history),
            "win_rate": 0.55 if self.model_type == "trend_following" else 0.52,
            "avg_confidence": 0.68,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_history(self, symbol: str) -> Deque[float]:
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=self.history_window)
        return self._history[symbol]

    def _update_history_and_features(self, symbol: str, price: float) -> Dict[str, float]:
        history = self._get_history(symbol)
        history.append(price)

        if len(history) >= 2:
            momentum = history[-1] - history[-2]
            trend_pct = (history[-1] - history[0]) / max(history[0], 1e-6)
            volatility = statistics.pstdev(history) if len(history) > 2 else abs(momentum)
        else:
            momentum = 0.0
            trend_pct = 0.0
            volatility = 0.0

        mean_price = statistics.mean(history) if history else price

        return {
            "momentum": momentum,
            "trend_pct": trend_pct,
            "volatility": volatility,
            "mean_price": mean_price,
            "history_length": len(history),
        }

    def _base_signal(self, current_price: float, features: Dict[str, float]) -> float:
        momentum = features["momentum"]
        trend_pct = features["trend_pct"]
        volatility = features["volatility"]
        mean_price = features["mean_price"]

        if self.model_type == "trend_following":
            return momentum * 0.8 + (trend_pct * current_price) * 0.6
        if self.model_type == "mean_reversion":
            return (mean_price - current_price) * 0.75
        if self.model_type == "momentum":
            return momentum * 1.1 + volatility * 0.2
        if self.model_type == "volatility_breakout":
            return (volatility + abs(momentum)) * self._rng.choice([-1, 1])
        return self._rng.gauss(0.0, current_price * 0.002)

    def _build_prediction(
        self,
        symbol: str,
        current_price: float,
        features: Dict[str, float],
        horizon: int,
    ) -> Dict[str, float]:
        horizon = max(1, int(horizon))

        base_change = self._base_signal(current_price, features)
        horizon_scale = math.sqrt(horizon)

        noise_scale = max(features["volatility"], current_price * 0.001)
        noise = self._rng.gauss(0.0, noise_scale * 0.25 * horizon_scale)

        predicted_change = base_change * horizon_scale + noise
        predicted_price = max(current_price + predicted_change, 0.01)

        magnitude_pct = abs(predicted_change) / max(current_price, 1.0)
        if magnitude_pct < 0.0005:
            direction = "neutral"
        else:
            direction = "up" if predicted_change > 0 else "down"

        confidence = min(0.95, max(0.4, 0.55 + magnitude_pct * 4))
        if direction == "neutral":
            confidence = min(confidence, 0.6)

        prediction = {
            "symbol": symbol,
            "timestamp": time.time(),
            "predicted_price": predicted_price,
            "predicted_change": predicted_change,
            "direction": direction,
            "confidence": confidence,
            "model_type": self.model_type,
            "horizon_minutes": horizon,
            "feature_snapshot": features.copy(),
        }

        if self.model_type == "mean_reversion":
            prediction["mean_price"] = features["mean_price"]
        if self.model_type == "trend_following":
            prediction["momentum"] = features["momentum"]

        return prediction


class EnsemblePredictor:
    """Combine multiple mock predictors into a simple ensemble."""

    def __init__(
        self,
        strategies: Optional[Iterable[str]] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        self.strategies = list(strategies) if strategies else ["trend_following", "mean_reversion", "momentum"]
        base_seed = random_seed if random_seed is not None else int(time.time() * 1000)
        self._predictors: List[MockPredictor] = [
            MockPredictor(model_type=strategy, random_seed=base_seed + idx)
            for idx, strategy in enumerate(self.strategies)
        ]

    def predict(self, symbol: str, current_price: float) -> Dict[str, float]:
        component_predictions = [predictor.predict(symbol, current_price) for predictor in self._predictors]
        predicted_price = sum(item["predicted_price"] for item in component_predictions) / len(component_predictions)
        predicted_change = sum(item["predicted_change"] for item in component_predictions) / len(component_predictions)
        confidence = sum(item["confidence"] for item in component_predictions) / len(component_predictions)

        if abs(predicted_change) < current_price * 0.0005:
            direction = "neutral"
        else:
            direction = "up" if predicted_change > 0 else "down"

        return {
            "symbol": symbol,
            "timestamp": time.time(),
            "predicted_price": predicted_price,
            "predicted_change": predicted_change,
            "direction": direction,
            "confidence": min(0.97, max(0.45, confidence)),
            "model_type": "ensemble",
            "component_predictions": component_predictions,
        }

    def predict_multi_horizon(
        self,
        symbol: str,
        current_price: float,
        horizons: Sequence[int],
    ) -> Dict[int, Dict[str, float]]:
        return {
            horizon: self.predict(symbol, current_price)
            for horizon in horizons
        }

    def get_feature_importance(self) -> Dict[str, float]:
        # Aggregate and normalise importances from the individual predictors.
        aggregate: Dict[str, float] = {}
        for predictor in self._predictors:
            for feature, value in predictor.get_feature_importance().items():
                aggregate[feature] = aggregate.get(feature, 0.0) + value

        total = sum(aggregate.values()) or 1.0
        return {feature: value / total for feature, value in aggregate.items()}


__all__ = ["MockPredictor", "EnsemblePredictor"]
