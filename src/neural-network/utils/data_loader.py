from pathlib import Path
from typing import Tuple, Dict, Optional
import logging
import numpy as np
import pandas as pd

"""
Data Loading Utilities for Neural Network Training

This module provides utilities to load cryptocurrency OHLCV data from CSV files
and prepare it for neural network training.
"""


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoDataLoader:
    """
    Load and preprocess cryptocurrency data from CSV files.
    """

    def __init__(self, data_root: str = "/Volumes/Lexar/RRRVentures/RRRalgorithms/worktrees/data-pipeline/data"):
        self.data_root = Path(data_root)

    def load_link_data(self, timeframe: str = "1min") -> pd.DataFrame:
        """
        Load LINK/USD data from CSV.

        Args:
            timeframe: Time frame ('1min', '5min', '15min', '1hr', '4hr', '1day')

        Returns:
            DataFrame with OHLCV data
        """
        file_path = self.data_root / f"linkusd/{timeframe}/X_LINKUSD_{timeframe}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        logger.info(f"Loading LINK data from {file_path}")
        df = pd.read_csv(file_path)

        # Standardize column names
        df = self._standardize_columns(df)

        logger.info(f"Loaded {len(df)} rows of LINK/{timeframe} data")
        return df

    def load_matic_data(self, timeframe: str = "1min") -> pd.DataFrame:
        """
        Load MATIC/USD data from CSV.

        Args:
            timeframe: Time frame

        Returns:
            DataFrame with OHLCV data
        """
        file_path = self.data_root / f"historical/maticusd/maticusd_{timeframe}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        logger.info(f"Loading MATIC data from {file_path}")
        df = pd.read_csv(file_path)

        # Standardize column names
        df = self._standardize_columns(df)

        logger.info(f"Loaded {len(df)} rows of MATIC/{timeframe} data")
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to lowercase.
        """
        # Map common column names to standard format
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Timestamp': 'timestamp',
            'timestamp': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vw': 'vwap',
            't': 'timestamp'
        }

        # Rename columns
        df = df.rename(columns={col: column_mapping.get(col, col.lower()) for col in df.columns})

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Add VWAP if not present (approximate as (high + low + close) / 3)
        if 'vwap' not in df.columns:
            df['vwap'] = (df['high'] + df['low'] + df['close']) / 3

        # Convert timestamp if present
        if 'timestamp' in df.columns:
            # Try to parse timestamp (could be Unix timestamp or datetime string)
            try:
                # If it's numeric, assume Unix timestamp in milliseconds
                if pd.api.types.is_numeric_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                else:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            except Exception as e:
                logger.warning(f"Could not parse timestamp: {e}")

        return df

    def prepare_price_prediction_data(
        self,
        df: pd.DataFrame,
        seq_len: int = 100,
        horizons: Dict[str, int] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Prepare data for price prediction model.

        Args:
            df: DataFrame with OHLCV data
            seq_len: Sequence length for input
            horizons: Prediction horizons in steps (e.g., {'5min': 5, '15min': 15, '1hr': 60})

        Returns:
            features: Normalized OHLCV features
            labels: Dictionary of labels for each horizon
        """
        if horizons is None:
            horizons = {'5min': 5, '15min': 15, '1hr': 60}

        # Extract OHLCV features
        features = df[['open', 'high', 'low', 'close', 'volume', 'vwap']].values

        # Normalize features
        features = self._normalize_features(features)

        # Create labels for each horizon
        close_prices = df['close'].values
        labels = self._create_price_labels(close_prices, horizons)

        return features, labels

    def _normalize_features(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize OHLCV features using log returns and z-score.
        """
        # Calculate log returns for OHLC
        prices = data[:, :4]
        log_returns = np.diff(np.log(prices + 1e-8), axis=0, prepend=prices[:1])

        # Normalize volume (log scale + z-score)
        volume = data[:, 4:5]
        log_volume = np.log(volume + 1)
        volume_norm = (log_volume - np.mean(log_volume)) / (np.std(log_volume) + 1e-8)

        # Combine features
        features = np.concatenate([log_returns, volume_norm, data[:, 5:]], axis=1)

        # Z-score normalization
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)

        return features

    def _create_price_labels(
        self,
        close_prices: np.ndarray,
        horizons: Dict[str, int],
        thresholds: Dict[str, float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Create classification labels for price movements.

        Labels:
            0: Down (price decreases > threshold)
            1: Flat (price change within threshold)
            2: Up (price increases > threshold)
        """
        if thresholds is None:
            thresholds = {
                '5min': 0.002,   # 0.2% threshold
                '15min': 0.005,  # 0.5% threshold
                '1hr': 0.01      # 1.0% threshold
            }

        labels = {}

        for horizon_name, horizon_steps in horizons.items():
            threshold = thresholds.get(horizon_name, 0.005)
            horizon_labels = []

            for i in range(len(close_prices) - horizon_steps):
                current_price = close_prices[i]
                future_price = close_prices[i + horizon_steps]
                price_change = (future_price - current_price) / current_price

                if price_change < -threshold:
                    label = 0  # Down
                elif price_change > threshold:
                    label = 2  # Up
                else:
                    label = 1  # Flat

                horizon_labels.append(label)

            # Pad the end with neutral labels
            horizon_labels.extend([1] * horizon_steps)
            labels[horizon_name] = np.array(horizon_labels)

        return labels

    def create_sequences(
        self,
        features: np.ndarray,
        labels: Dict[str, np.ndarray],
        seq_len: int = 100
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Create sliding window sequences from features and labels.
        """
        n_samples = len(features) - seq_len
        n_features = features.shape[1]

        sequences = np.zeros((n_samples, seq_len, n_features))
        sequence_labels = {
            horizon: np.zeros(n_samples, dtype=np.int64)
            for horizon in labels.keys()
        }

        for i in range(n_samples):
            sequences[i] = features[i:i + seq_len]
            for horizon in labels.keys():
                sequence_labels[horizon][i] = labels[horizon][i + seq_len - 1]

        return sequences, sequence_labels


def load_and_prepare_crypto_data(
    symbol: str = 'LINK',
    timeframe: str = '1min',
    seq_len: int = 100,
    train_split: float = 0.8
) -> Dict[str, any]:
    """
    Convenience function to load and prepare cryptocurrency data.

    Args:
        symbol: Symbol to load ('LINK' or 'MATIC')
        timeframe: Time frame to use
        seq_len: Sequence length
        train_split: Fraction of data to use for training

    Returns:
        Dictionary with train/val data and metadata
    """
    loader = CryptoDataLoader()

    # Load data
    if symbol.upper() == 'LINK':
        df = loader.load_link_data(timeframe)
    elif symbol.upper() == 'MATIC':
        df = loader.load_matic_data(timeframe)
    else:
        raise ValueError(f"Unknown symbol: {symbol}")

    # Prepare features and labels
    features, labels = loader.prepare_price_prediction_data(df, seq_len=seq_len)

    # Create sequences
    sequences, sequence_labels = loader.create_sequences(features, labels, seq_len=seq_len)

    # Train/validation split
    split_idx = int(train_split * len(sequences))

    result = {
        'train_sequences': sequences[:split_idx],
        'val_sequences': sequences[split_idx:],
        'train_labels': {k: v[:split_idx] for k, v in sequence_labels.items()},
        'val_labels': {k: v[split_idx:] for k, v in sequence_labels.items()},
        'n_features': features.shape[1],
        'seq_len': seq_len,
        'symbol': symbol,
        'timeframe': timeframe,
        'total_samples': len(sequences),
        'train_samples': split_idx,
        'val_samples': len(sequences) - split_idx
    }

    logger.info(f"Prepared {result['total_samples']} sequences ({result['train_samples']} train, {result['val_samples']} val)")

    return result


if __name__ == "__main__":
    # Test the data loader
    print("Testing CryptoDataLoader...")

    # Load LINK data
    data = load_and_prepare_crypto_data(symbol='LINK', timeframe='1min', seq_len=100)

    print(f"\nLoaded {data['symbol']} data:")
    print(f"  Total samples: {data['total_samples']}")
    print(f"  Train samples: {data['train_samples']}")
    print(f"  Val samples: {data['val_samples']}")
    print(f"  Features: {data['n_features']}")
    print(f"  Sequence length: {data['seq_len']}")

    print("\nLabel distribution (5min horizon):")
    train_labels = data['train_labels']['5min']
    unique, counts = np.unique(train_labels, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = ['Down', 'Flat', 'Up'][label]
        print(f"  {label_name}: {count} ({count/len(train_labels)*100:.1f}%)")

    print("\nData loader test passed!")
