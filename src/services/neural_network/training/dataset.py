"""
PyTorch Datasets for Cryptocurrency Trading

This module provides dataset classes for loading and preprocessing cryptocurrency
data for neural network training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CryptoDataset(Dataset):
    """
    PyTorch dataset for cryptocurrency OHLCV data.

    Loads sequences of market data for time-series prediction.
    """

    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        labels: Dict[str, np.ndarray],
        seq_len: int = 100,
        stride: int = 1,
        normalize: bool = True
    ):
        """
        Initialize cryptocurrency dataset.

        Args:
            data: OHLCV data [n_samples, n_features]
            labels: Dictionary of labels for each horizon
            seq_len: Sequence length for sliding window
            stride: Stride for sliding window
            normalize: Whether to normalize features
        """
        self.seq_len = seq_len
        self.stride = stride
        self.normalize = normalize

        # Convert DataFrame to numpy if needed
        if isinstance(data, pd.DataFrame):
            self.data = data.values
        else:
            self.data = data

        self.labels = labels
        self.horizons = list(labels.keys())

        # Normalize if requested
        if normalize:
            self.data = self._normalize(self.data)

        # Create sequences
        self.sequences = []
        self.sequence_labels = {horizon: [] for horizon in self.horizons}

        for i in range(0, len(self.data) - seq_len, stride):
            self.sequences.append(self.data[i:i + seq_len])
            for horizon in self.horizons:
                self.sequence_labels[horizon].append(self.labels[horizon][i + seq_len - 1])

        self.sequences = np.array(self.sequences, dtype=np.float32)
        for horizon in self.horizons:
            self.sequence_labels[horizon] = np.array(self.sequence_labels[horizon], dtype=np.int64)

        logger.info(f"Created dataset with {len(self.sequences)} sequences")

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize features using z-score normalization.

        Args:
            data: Raw features

        Returns:
            Normalized features
        """
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        return (data - mean) / std

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get item by index.

        Args:
            idx: Index

        Returns:
            Sequence and labels for all horizons
        """
        sequence = torch.from_numpy(self.sequences[idx])
        labels = {
            horizon: torch.tensor(self.sequence_labels[horizon][idx], dtype=torch.long)
            for horizon in self.horizons
        }

        return sequence, labels


class MultiHorizonDataset(Dataset):
    """
    Dataset for multi-horizon price prediction with technical indicators.

    Includes computed technical indicators as additional features.
    """

    def __init__(
        self,
        ohlcv_data: pd.DataFrame,
        technical_features: Optional[np.ndarray] = None,
        horizons: Dict[str, int] = None,
        seq_len: int = 100,
        thresholds: Dict[str, float] = None,
        train: bool = True,
        val_split: float = 0.2
    ):
        """
        Initialize multi-horizon dataset.

        Args:
            ohlcv_data: DataFrame with OHLCV data
            technical_features: Pre-computed technical features
            horizons: Prediction horizons in steps
            seq_len: Sequence length
            thresholds: Price change thresholds for classification
            train: Whether this is training data
            val_split: Validation split ratio
        """
        self.seq_len = seq_len
        self.train = train

        if horizons is None:
            horizons = {'5min': 5, '15min': 15, '1hr': 60}

        if thresholds is None:
            thresholds = {'5min': 0.002, '15min': 0.005, '1hr': 0.01}

        self.horizons = horizons
        self.thresholds = thresholds

        # Extract OHLCV features
        ohlcv_features = ohlcv_data[['open', 'high', 'low', 'close', 'volume']].values

        # Combine with technical indicators if provided
        if technical_features is not None:
            # Ensure same length
            min_len = min(len(ohlcv_features), len(technical_features))
            features = np.concatenate([
                ohlcv_features[-min_len:],
                technical_features[-min_len:]
            ], axis=1)
        else:
            features = ohlcv_features

        # Create labels
        close_prices = ohlcv_data['close'].values
        labels = self._create_labels(close_prices, horizons, thresholds)

        # Split train/val
        split_idx = int(len(features) * (1 - val_split))

        if train:
            self.features = features[:split_idx]
            self.labels = {k: v[:split_idx] for k, v in labels.items()}
        else:
            self.features = features[split_idx:]
            self.labels = {k: v[split_idx:] for k, v in labels.items()}

        # Normalize features
        self.features = self._normalize_features(self.features)

        # Create sequences
        self.sequences, self.sequence_labels = self._create_sequences()

        logger.info(
            f"Created {'training' if train else 'validation'} dataset "
            f"with {len(self.sequences)} sequences"
        )

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using robust scaling."""
        # Use robust scaling (less sensitive to outliers)
        q25 = np.percentile(features, 25, axis=0)
        q75 = np.percentile(features, 75, axis=0)
        iqr = q75 - q25 + 1e-8
        median = np.median(features, axis=0)

        return (features - median) / iqr

    def _create_labels(
        self,
        close_prices: np.ndarray,
        horizons: Dict[str, int],
        thresholds: Dict[str, float]
    ) -> Dict[str, np.ndarray]:
        """
        Create classification labels for price movements.

        Labels:
            0: Down (price decreases > threshold)
            1: Flat (price change within threshold)
            2: Up (price increases > threshold)
        """
        labels = {}

        for horizon_name, horizon_steps in horizons.items():
            threshold = thresholds.get(horizon_name, 0.005)
            horizon_labels = []

            for i in range(len(close_prices)):
                if i + horizon_steps >= len(close_prices):
                    # Pad with neutral label
                    horizon_labels.append(1)
                else:
                    current_price = close_prices[i]
                    future_price = close_prices[i + horizon_steps]
                    price_change = (future_price - current_price) / (current_price + 1e-8)

                    if price_change < -threshold:
                        label = 0  # Down
                    elif price_change > threshold:
                        label = 2  # Up
                    else:
                        label = 1  # Flat

                    horizon_labels.append(label)

            labels[horizon_name] = np.array(horizon_labels, dtype=np.int64)

        return labels

    def _create_sequences(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Create sliding window sequences."""
        n_samples = len(self.features) - self.seq_len
        n_features = self.features.shape[1]

        sequences = np.zeros((n_samples, self.seq_len, n_features), dtype=np.float32)
        sequence_labels = {
            horizon: np.zeros(n_samples, dtype=np.int64)
            for horizon in self.horizons.keys()
        }

        for i in range(n_samples):
            sequences[i] = self.features[i:i + self.seq_len]
            for horizon in self.horizons.keys():
                sequence_labels[horizon][i] = self.labels[horizon][i + self.seq_len - 1]

        return sequences, sequence_labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get item by index."""
        sequence = torch.from_numpy(self.sequences[idx])
        labels = {
            horizon: torch.tensor(self.sequence_labels[horizon][idx], dtype=torch.long)
            for horizon in self.horizons.keys()
        }

        return sequence, labels

    def get_class_weights(self) -> Dict[str, torch.Tensor]:
        """
        Compute class weights for balanced training.

        Returns:
            Dictionary of class weights for each horizon
        """
        class_weights = {}

        for horizon in self.horizons.keys():
            labels = self.sequence_labels[horizon]
            unique, counts = np.unique(labels, return_counts=True)

            # Inverse frequency weighting
            weights = 1.0 / (counts + 1e-8)
            weights = weights / weights.sum() * len(unique)

            # Create full weight tensor
            weight_tensor = torch.zeros(3)
            for label, weight in zip(unique, weights):
                weight_tensor[label] = weight

            class_weights[horizon] = weight_tensor

        return class_weights


class TimeSeriesDataModule:
    """
    Data module for managing train/val/test splits and dataloaders.
    """

    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        """
        Initialize data module.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Optional test dataset
            batch_size: Batch size
            num_workers: Number of data loading workers
            pin_memory: Pin memory for GPU
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Get test dataloader."""
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
        return None


if __name__ == "__main__":
    # Test datasets
    print("Testing Cryptocurrency Datasets...")

    # Create dummy data
    n_samples = 10000
    n_features = 6

    data = np.random.randn(n_samples, n_features).astype(np.float32)
    labels = {
        '5min': np.random.randint(0, 3, n_samples),
        '15min': np.random.randint(0, 3, n_samples),
        '1hr': np.random.randint(0, 3, n_samples)
    }

    # Test CryptoDataset
    print("\n1. Testing CryptoDataset...")
    dataset = CryptoDataset(data, labels, seq_len=100, stride=1)
    print(f"   Dataset size: {len(dataset)}")

    seq, labs = dataset[0]
    print(f"   Sequence shape: {seq.shape}")
    print(f"   Labels: {labs}")

    # Test DataLoader
    print("\n2. Testing DataLoader...")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch_seq, batch_labs = next(iter(loader))
    print(f"   Batch sequence shape: {batch_seq.shape}")
    print(f"   Batch labels shapes: {[f'{k}: {v.shape}' for k, v in batch_labs.items()]}")

    # Test class weights
    print("\n3. Testing MultiHorizonDataset...")
    df = pd.DataFrame({
        'open': np.random.randn(n_samples) + 100,
        'high': np.random.randn(n_samples) + 101,
        'low': np.random.randn(n_samples) + 99,
        'close': np.random.randn(n_samples) + 100,
        'volume': np.random.randint(1000000, 10000000, n_samples)
    })

    train_dataset = MultiHorizonDataset(df, train=True, val_split=0.2)
    val_dataset = MultiHorizonDataset(df, train=False, val_split=0.2)

    print(f"   Train dataset size: {len(train_dataset)}")
    print(f"   Val dataset size: {len(val_dataset)}")

    weights = train_dataset.get_class_weights()
    print(f"   Class weights: {weights}")

    print("\nDataset tests completed successfully!")
