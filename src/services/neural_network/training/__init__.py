"""
Neural Network Training Pipeline

This module provides comprehensive training infrastructure for neural network models
including data loading, training loops, validation, checkpointing, and monitoring.
"""

from .trainer import Trainer, DistributedTrainer
from .dataset import CryptoDataset, MultiHorizonDataset
from .losses import FocalLoss, LabelSmoothingCrossEntropy, MultiHorizonLoss
from .metrics import MetricsTracker, TradingMetrics

__all__ = [
    'Trainer',
    'DistributedTrainer',
    'CryptoDataset',
    'MultiHorizonDataset',
    'FocalLoss',
    'LabelSmoothingCrossEntropy',
    'MultiHorizonLoss',
    'MetricsTracker',
    'TradingMetrics',
]
