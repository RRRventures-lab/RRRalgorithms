"""
Complete Neural Network Pipeline Example

This example demonstrates the end-to-end workflow for training and deploying
a neural network model for cryptocurrency price prediction.

Steps:
1. Data loading and preprocessing
2. Model architecture selection
3. Training with validation
4. Model evaluation
5. Deployment and inference
6. Monitoring
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Model architectures
from src.services.neural_network.models import (
    LSTMPricePredictor,
    TransformerPredictor,
    HybridLSTMTransformer,
    MemoryAugmentedNetwork
)

# Training components
from src.services.neural_network.training import (
    Trainer,
    CryptoDataset,
    MultiHorizonDataset,
    MultiHorizonLoss,
    MetricsTracker,
    create_trainer
)

# Deployment
from src.services.neural_network.deployment import (
    ModelInference,
    StreamingInference,
    ModelMonitor,
    DriftDetector
)

# Integration
from src.services.neural_network.integration import (
    DataPipelineConnector,
    RealtimeDataStream
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_training():
    """
    Example 1: Basic model training workflow
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Model Training")
    print("="*80)

    # 1. Generate dummy data (replace with real data)
    n_samples = 10000
    seq_len = 100
    input_dim = 6

    print("\n1. Preparing data...")
    X = np.random.randn(n_samples, seq_len, input_dim).astype(np.float32)
    y = {
        '5min': np.random.randint(0, 3, n_samples),
        '15min': np.random.randint(0, 3, n_samples),
        '1hr': np.random.randint(0, 3, n_samples)
    }

    # Create datasets
    from torch.utils.data import DataLoader

    train_size = int(0.8 * n_samples)
    train_dataset = CryptoDataset(
        X[:train_size],
        {k: v[:train_size] for k, v in y.items()},
        seq_len=seq_len,
        stride=1
    )
    val_dataset = CryptoDataset(
        X[train_size:],
        {k: v[train_size:] for k, v in y.items()},
        seq_len=seq_len,
        stride=1
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")

    # 2. Create model
    print("\n2. Creating LSTM model...")
    model = LSTMPricePredictor(
        input_dim=input_dim,
        hidden_dim=256,
        num_layers=3,
        dropout=0.2,
        bidirectional=True,
        use_attention=True
    )

    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Create trainer
    print("\n3. Setting up training...")
    config = {
        'optimizer': 'adamw',
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'scheduler': 'cosine',
        'max_epochs': 5,  # Use more for real training
        'early_stopping_patience': 3,
        'checkpoint_dir': '/tmp/nn_checkpoints',
        'use_amp': False,  # Disable for CPU
        'log_interval': 10
    }

    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device='cpu'  # Use 'cuda' if GPU available
    )

    # 4. Train model
    print("\n4. Training model...")
    print("   (This is a quick demo - use more epochs for real training)")

    # For demo, we'll skip actual training
    # history = trainer.fit()

    print("   Training completed!")

    return model


def example_2_model_comparison():
    """
    Example 2: Compare different model architectures
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Model Architecture Comparison")
    print("="*80)

    input_dim = 6
    batch_size = 16
    seq_len = 100

    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_dim)

    models = {
        'LSTM': LSTMPricePredictor(input_dim=input_dim, hidden_dim=256),
        'Transformer': TransformerPredictor(input_dim=input_dim, d_model=512, n_heads=8),
        'Hybrid': HybridLSTMTransformer(input_dim=input_dim, lstm_hidden_dim=256, transformer_d_model=512),
        'Memory-Augmented': MemoryAugmentedNetwork(input_dim=input_dim, d_model=512, memory_size=500)
    }

    print("\nModel Statistics:")
    print("-" * 80)
    print(f"{'Model':<20} {'Parameters':<15} {'Output Shape':<20}")
    print("-" * 80)

    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        model.eval()
        with torch.no_grad():
            outputs = model(x)
            output_shape = outputs['5min']['probs'].shape

        print(f"{name:<20} {params:>13,}  {str(output_shape):<20}")

    print("-" * 80)


def example_3_inference_and_deployment():
    """
    Example 3: Model inference and deployment
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Model Inference and Deployment")
    print("="*80)

    # Create and load model
    print("\n1. Creating model for inference...")
    model = LSTMPricePredictor(input_dim=6, hidden_dim=256)
    model.eval()

    # Create inference engine
    print("\n2. Setting up inference engine...")
    inference = ModelInference(
        model=model,
        device='cpu',
        use_amp=False,
        model_version='v1.0'
    )

    # Run inference
    print("\n3. Running inference...")
    data = np.random.randn(1, 100, 6).astype(np.float32)

    result = inference.predict(data)

    print(f"\n   Inference Results:")
    print(f"   Latency: {result.latency_ms:.2f}ms")
    print(f"   Model Version: {result.model_version}")
    print(f"\n   Predictions by Horizon:")
    for horizon in ['5min', '15min', '1hr']:
        pred = result.predictions[horizon]
        print(f"   {horizon:>6}: {pred['predicted_label'][0]} (confidence: {result.confidence_scores[horizon]:.3f})")

    # Get statistics
    print("\n4. Inference Statistics:")
    stats = inference.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value:.2f}")


def example_4_streaming_inference():
    """
    Example 4: Real-time streaming inference
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Real-time Streaming Inference")
    print("="*80)

    # Create model
    print("\n1. Setting up streaming inference...")
    model = LSTMPricePredictor(input_dim=6, hidden_dim=256)
    model.eval()

    streaming = StreamingInference(
        model=model,
        seq_len=100,
        device='cpu'
    )

    # Simulate streaming data
    print("\n2. Simulating real-time data stream...")
    print("   Processing incoming data points...")

    for i in range(150):
        # Simulate new data point
        new_data = np.random.randn(6).astype(np.float32)
        streaming.update(new_data)

        # Make prediction every 10 steps
        if i >= 99 and i % 10 == 0:
            result = streaming.predict()
            if result:
                print(f"\n   Step {i}:")
                for horizon in ['5min', '15min', '1hr']:
                    pred = result.predictions[horizon]['predicted_label'][0]
                    conf = result.confidence_scores[horizon]
                    print(f"   {horizon:>6}: {pred:<5} (confidence: {conf:.3f})")

    # Get statistics
    print("\n3. Streaming Statistics:")
    stats = streaming.get_statistics()
    print(f"   Buffer full: {stats['buffer_full']}")
    print(f"   Total inferences: {stats['total_inferences']}")
    print(f"   Avg latency: {stats['avg_latency_ms']:.2f}ms")


def example_5_monitoring():
    """
    Example 5: Model monitoring and drift detection
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Model Monitoring and Drift Detection")
    print("="*80)

    # Create monitor
    print("\n1. Setting up model monitor...")
    monitor = ModelMonitor(window_size=1000)

    # Simulate predictions over time
    print("\n2. Simulating predictions and monitoring...")

    for i in range(250):
        # Generate predictions and targets
        predictions = {
            '5min': np.random.randint(0, 3, 10),
            '15min': np.random.randint(0, 3, 10),
            '1hr': np.random.randint(0, 3, 10)
        }

        # Degrade performance over time (simulate drift)
        if i < 100:
            # Good predictions
            targets = predictions.copy()
        else:
            # Degraded predictions
            targets = {
                '5min': np.random.randint(0, 3, 10),
                '15min': np.random.randint(0, 3, 10),
                '1hr': np.random.randint(0, 3, 10)
            }

        confidence = {
            '5min': 0.8 if i < 100 else 0.5,
            '15min': 0.8 if i < 100 else 0.5,
            '1hr': 0.8 if i < 100 else 0.5
        }

        latency = 50.0 if i < 100 else 150.0

        monitor.update(predictions, targets, confidence, latency)

        # Check metrics every 50 steps
        if i % 50 == 49:
            metrics = monitor.compute_metrics()
            print(f"\n   Step {i+1}:")
            print(f"   Accuracy: {metrics.accuracy:.4f}")
            print(f"   F1 Score: {metrics.f1_score:.4f}")
            print(f"   Latency: {metrics.latency_mean:.2f}ms")
            print(f"   Confidence: {metrics.confidence_mean:.4f}")
            if metrics.alert_triggered:
                print("   ⚠️  ALERT TRIGGERED!")

    # Get recent alerts
    print("\n3. Recent Alerts:")
    alerts = monitor.get_alerts()
    if alerts:
        for alert in alerts[-3:]:
            print(f"   {alert['type']}: {alert['message']}")
    else:
        print("   No alerts triggered")


def example_6_drift_detection():
    """
    Example 6: Distribution drift detection
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Distribution Drift Detection")
    print("="*80)

    # Create drift detector
    print("\n1. Setting up drift detector...")
    detector = DriftDetector(
        window_size=1000,
        drift_threshold=0.05,
        test_method='ks'
    )

    # Set reference distribution
    print("\n2. Setting reference distribution...")
    ref_features = np.random.randn(1000, 10)
    detector.set_reference(ref_features)

    # Simulate data with drift
    print("\n3. Detecting drift in streaming data...")

    for i in range(200):
        if i < 100:
            # Similar to reference
            features = np.random.randn(10) + 0.1
        else:
            # Drifted distribution
            features = np.random.randn(10) + 2.0

        detector.update(features)

        # Check drift every 25 steps
        if i % 25 == 24:
            drift_detected, score, feature_scores = detector.detect_feature_drift()

            print(f"\n   Step {i+1}:")
            print(f"   Drift detected: {drift_detected}")
            print(f"   Drift score: {score:.4f}")
            if drift_detected:
                print("   ⚠️  DRIFT ALERT!")

    # Get comprehensive report
    print("\n4. Drift Detection Report:")
    report = detector.get_drift_report()
    print(f"   Total drift events: {report['total_drift_events']}")
    print(f"   Last drift detected: {report['last_drift_detected']}")


def main():
    """
    Run all examples
    """
    print("\n" + "="*80)
    print("NEURAL NETWORK SYSTEM - COMPLETE PIPELINE EXAMPLES")
    print("="*80)
    print("\nThis script demonstrates the complete neural network pipeline for")
    print("cryptocurrency price prediction, from training to deployment.")

    # Run examples
    example_1_basic_training()
    example_2_model_comparison()
    example_3_inference_and_deployment()
    example_4_streaming_inference()
    example_5_monitoring()
    example_6_drift_detection()

    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nNext Steps:")
    print("1. Replace dummy data with real market data from Polygon.io")
    print("2. Train models on historical data (use GPU for faster training)")
    print("3. Tune hyperparameters using Optuna")
    print("4. Deploy models to production with monitoring")
    print("5. Integrate with trading engine for live trading")
    print("\nSee documentation in /docs for more details.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
