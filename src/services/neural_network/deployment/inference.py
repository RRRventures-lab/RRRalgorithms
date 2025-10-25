"""
Model Inference Engine

Provides real-time and batch inference capabilities for trained neural network models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import time
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for prediction results."""
    predictions: Dict[str, Dict[str, np.ndarray]]  # {horizon: {metric: values}}
    timestamp: datetime
    latency_ms: float
    model_version: str
    confidence_scores: Dict[str, float]


class ModelInference:
    """
    Base inference engine for neural network models.

    Handles model loading, preprocessing, and prediction.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        use_amp: bool = True,
        batch_size: int = 32,
        model_version: str = 'v1.0'
    ):
        """
        Initialize inference engine.

        Args:
            model: Trained neural network model
            device: Device to run inference on
            use_amp: Use automatic mixed precision
            batch_size: Batch size for inference
            model_version: Model version identifier
        """
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.batch_size = batch_size
        self.model_version = model_version

        # Load model
        self.model = model.to(device)
        self.model.eval()

        # Inference statistics
        self.inference_count = 0
        self.total_latency = 0.0
        self.latency_history = deque(maxlen=1000)

        logger.info(f"Initialized inference engine on {device} (AMP: {self.use_amp})")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_class: type,
        model_config: Dict,
        device: str = 'cuda',
        **kwargs
    ):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            model_class: Model class
            model_config: Model configuration
            device: Device to load on
            **kwargs: Additional arguments

        Returns:
            ModelInference instance
        """
        # Create model
        model = model_class(**model_config)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"Loaded model from {checkpoint_path}")

        return cls(model=model, device=device, **kwargs)

    def preprocess(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess input data.

        Args:
            data: Input data [batch_size, seq_len, features] or [seq_len, features]

        Returns:
            Preprocessed tensor
        """
        # Convert to tensor if numpy
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        # Add batch dimension if needed
        if data.dim() == 2:
            data = data.unsqueeze(0)

        return data.to(self.device)

    def postprocess(
        self,
        outputs: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Postprocess model outputs.

        Args:
            outputs: Raw model outputs

        Returns:
            Processed predictions
        """
        predictions = {}

        for horizon, horizon_outputs in outputs.items():
            predictions[horizon] = {}

            for key, value in horizon_outputs.items():
                if isinstance(value, torch.Tensor):
                    predictions[horizon][key] = value.detach().cpu().numpy()

            # Add predicted class
            if 'probs' in predictions[horizon]:
                predictions[horizon]['predicted_class'] = \
                    np.argmax(predictions[horizon]['probs'], axis=-1)

                # Map to labels
                class_names = ['down', 'flat', 'up']
                predictions[horizon]['predicted_label'] = \
                    np.array([class_names[c] for c in predictions[horizon]['predicted_class']])

        return predictions

    @torch.no_grad()
    def predict(
        self,
        data: Union[np.ndarray, torch.Tensor],
        return_all: bool = True
    ) -> PredictionResult:
        """
        Run inference on input data.

        Args:
            data: Input data
            return_all: Return all outputs or just predictions

        Returns:
            Prediction results
        """
        start_time = time.time()

        # Preprocess
        inputs = self.preprocess(data)

        # Forward pass
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
        else:
            outputs = self.model(inputs)

        # Postprocess
        predictions = self.postprocess(outputs)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Update statistics
        self.inference_count += 1
        self.total_latency += latency_ms
        self.latency_history.append(latency_ms)

        # Extract confidence scores
        confidence_scores = {}
        for horizon in predictions:
            if 'confidence' in predictions[horizon]:
                confidence_scores[horizon] = float(predictions[horizon]['confidence'].mean())
            elif 'probs' in predictions[horizon]:
                # Use max probability as confidence
                confidence_scores[horizon] = float(predictions[horizon]['probs'].max(axis=-1).mean())

        # Create result
        result = PredictionResult(
            predictions=predictions if return_all else {
                h: {'predicted_class': v['predicted_class'], 'predicted_label': v['predicted_label']}
                for h, v in predictions.items()
            },
            timestamp=datetime.now(),
            latency_ms=latency_ms,
            model_version=self.model_version,
            confidence_scores=confidence_scores
        )

        return result

    def get_statistics(self) -> Dict[str, float]:
        """
        Get inference statistics.

        Returns:
            Dictionary of statistics
        """
        if not self.latency_history:
            return {}

        latencies = list(self.latency_history)

        return {
            'total_inferences': self.inference_count,
            'avg_latency_ms': self.total_latency / self.inference_count,
            'recent_avg_latency_ms': np.mean(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies)
        }


class BatchInference(ModelInference):
    """
    Batch inference engine for processing multiple samples efficiently.
    """

    def predict_batch(
        self,
        data_list: List[Union[np.ndarray, torch.Tensor]],
        batch_size: Optional[int] = None
    ) -> List[PredictionResult]:
        """
        Run inference on batch of samples.

        Args:
            data_list: List of input samples
            batch_size: Batch size (uses default if None)

        Returns:
            List of prediction results
        """
        batch_size = batch_size or self.batch_size
        results = []

        # Process in batches
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]

            # Stack into single tensor
            if isinstance(batch[0], np.ndarray):
                batch_tensor = np.stack(batch)
            else:
                batch_tensor = torch.stack(batch)

            # Run inference
            result = self.predict(batch_tensor)
            results.append(result)

        return results


class StreamingInference:
    """
    Streaming inference engine for real-time predictions.

    Maintains a sliding window buffer for sequential data.
    """

    def __init__(
        self,
        model: nn.Module,
        seq_len: int = 100,
        device: str = 'cuda',
        **kwargs
    ):
        """
        Initialize streaming inference.

        Args:
            model: Trained model
            seq_len: Sequence length
            device: Device
            **kwargs: Additional arguments for ModelInference
        """
        self.seq_len = seq_len
        self.buffer = deque(maxlen=seq_len)
        self.inference_engine = ModelInference(model=model, device=device, **kwargs)

    def update(self, new_data: np.ndarray):
        """
        Update buffer with new data point.

        Args:
            new_data: New data point [features]
        """
        self.buffer.append(new_data)

    def predict(self, return_all: bool = True) -> Optional[PredictionResult]:
        """
        Run inference on current buffer.

        Args:
            return_all: Return all outputs

        Returns:
            Prediction result or None if buffer not full
        """
        if len(self.buffer) < self.seq_len:
            logger.warning(f"Buffer not full ({len(self.buffer)}/{self.seq_len})")
            return None

        # Convert buffer to array
        sequence = np.array(list(self.buffer))

        # Run inference
        return self.inference_engine.predict(sequence, return_all=return_all)

    def predict_next(
        self,
        new_data: np.ndarray,
        return_all: bool = True
    ) -> PredictionResult:
        """
        Update buffer and predict in one call.

        Args:
            new_data: New data point
            return_all: Return all outputs

        Returns:
            Prediction result
        """
        self.update(new_data)
        return self.predict(return_all=return_all)

    def reset(self):
        """Reset buffer."""
        self.buffer.clear()

    def get_statistics(self) -> Dict[str, float]:
        """Get inference statistics."""
        stats = self.inference_engine.get_statistics()
        stats['buffer_size'] = len(self.buffer)
        stats['buffer_full'] = len(self.buffer) == self.seq_len
        return stats


class EnsembleInference:
    """
    Ensemble inference combining multiple models.
    """

    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        device: str = 'cuda',
        **kwargs
    ):
        """
        Initialize ensemble inference.

        Args:
            models: List of trained models
            weights: Optional weights for each model
            device: Device
            **kwargs: Additional arguments
        """
        self.models = [ModelInference(m, device=device, **kwargs) for m in models]
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.weights = np.array(self.weights)
        self.weights = self.weights / self.weights.sum()  # Normalize

        logger.info(f"Initialized ensemble with {len(models)} models")

    def predict(
        self,
        data: Union[np.ndarray, torch.Tensor],
        aggregation: str = 'weighted_vote'
    ) -> PredictionResult:
        """
        Run ensemble inference.

        Args:
            data: Input data
            aggregation: Aggregation method ('weighted_vote', 'average_probs')

        Returns:
            Ensemble prediction result
        """
        start_time = time.time()

        # Get predictions from all models
        all_results = [model.predict(data) for model in self.models]

        # Aggregate predictions
        ensemble_predictions = {}
        horizons = all_results[0].predictions.keys()

        for horizon in horizons:
            if aggregation == 'weighted_vote':
                # Weighted voting on predicted classes
                all_preds = np.array([
                    r.predictions[horizon]['predicted_class']
                    for r in all_results
                ])

                # Weighted vote
                votes = np.zeros((all_preds.shape[1], 3))
                for i, pred in enumerate(all_preds):
                    for j, p in enumerate(pred):
                        votes[j, p] += self.weights[i]

                ensemble_class = np.argmax(votes, axis=1)

            elif aggregation == 'average_probs':
                # Average probabilities
                all_probs = np.array([
                    r.predictions[horizon]['probs']
                    for r in all_results
                ])

                weighted_probs = np.average(all_probs, axis=0, weights=self.weights)
                ensemble_class = np.argmax(weighted_probs, axis=-1)

            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")

            # Create ensemble predictions
            class_names = ['down', 'flat', 'up']
            ensemble_predictions[horizon] = {
                'predicted_class': ensemble_class,
                'predicted_label': np.array([class_names[c] for c in ensemble_class])
            }

            # Add average confidence
            avg_confidence = np.mean([
                r.confidence_scores[horizon]
                for r in all_results
            ])
            ensemble_predictions[horizon]['confidence'] = avg_confidence

        latency_ms = (time.time() - start_time) * 1000

        result = PredictionResult(
            predictions=ensemble_predictions,
            timestamp=datetime.now(),
            latency_ms=latency_ms,
            model_version=f'ensemble_{len(self.models)}',
            confidence_scores={h: v['confidence'] for h, v in ensemble_predictions.items()}
        )

        return result


if __name__ == "__main__":
    # Test inference engines
    print("Testing Inference Engines...")

    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(6, 64)
            self.heads = nn.ModuleDict({
                '5min': nn.Linear(64, 3),
                '15min': nn.Linear(64, 3),
                '1hr': nn.Linear(64, 3)
            })

        def forward(self, x):
            x = self.fc(x).mean(dim=1)
            return {
                h: {
                    'logits': self.heads[h](x),
                    'probs': torch.softmax(self.heads[h](x), dim=-1),
                    'confidence': torch.rand(x.size(0), 1)
                }
                for h in ['5min', '15min', '1hr']
            }

    model = DummyModel()

    # Test basic inference
    print("\n1. Testing ModelInference...")
    inference = ModelInference(model, device='cpu', use_amp=False)

    data = np.random.randn(1, 100, 6).astype(np.float32)
    result = inference.predict(data)

    print(f"   Latency: {result.latency_ms:.2f}ms")
    print(f"   Predictions: {list(result.predictions.keys())}")
    print(f"   Confidence: {result.confidence_scores}")

    # Test batch inference
    print("\n2. Testing BatchInference...")
    batch_inference = BatchInference(model, device='cpu', batch_size=4)

    data_list = [np.random.randn(100, 6).astype(np.float32) for _ in range(10)]
    results = batch_inference.predict_batch(data_list, batch_size=4)

    print(f"   Processed {len(data_list)} samples in {len(results)} batches")
    avg_latency = np.mean([r.latency_ms for r in results])
    print(f"   Average latency: {avg_latency:.2f}ms")

    # Test streaming inference
    print("\n3. Testing StreamingInference...")
    streaming = StreamingInference(model, seq_len=100, device='cpu')

    # Simulate streaming data
    for i in range(150):
        new_point = np.random.randn(6).astype(np.float32)
        streaming.update(new_point)

        if i >= 99 and i % 10 == 0:
            result = streaming.predict()
            if result:
                print(f"   Step {i}: Prediction = {result.predictions['5min']['predicted_label'][0]}")

    stats = streaming.get_statistics()
    print(f"   Statistics: {stats}")

    print("\nInference engines tested successfully!")
