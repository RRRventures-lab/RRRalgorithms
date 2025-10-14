from datetime import datetime, timedelta
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, List, Tuple, Optional
import json
import logging
import numpy as np
import pandas as pd
import sqlite3
import warnings

#!/usr/bin/env python3
"""
ML Data Optimization Pipeline
Transforms raw trading data into ML-optimized formats
"""


warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLDataOptimizer:
    """Optimize trading data for neural network consumption"""

    def __init__(self, db_path: str, output_dir: str):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)

    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicators for feature engineering"""
        logger.info("Generating technical indicators...")

        # Ensure data is sorted
        df = df.sort_values(['symbol', 'timestamp'])

        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_df = df[mask].copy()

            # Price-based indicators
            df.loc[mask, 'sma_5'] = symbol_df['close'].rolling(5, min_periods=1).mean()
            df.loc[mask, 'sma_20'] = symbol_df['close'].rolling(20, min_periods=1).mean()
            df.loc[mask, 'ema_12'] = symbol_df['close'].ewm(span=12, adjust=False).mean()
            df.loc[mask, 'ema_26'] = symbol_df['close'].ewm(span=26, adjust=False).mean()

            # MACD
            df.loc[mask, 'macd'] = df.loc[mask, 'ema_12'] - df.loc[mask, 'ema_26']
            df.loc[mask, 'macd_signal'] = df.loc[mask, 'macd'].ewm(span=9, adjust=False).mean()
            df.loc[mask, 'macd_histogram'] = df.loc[mask, 'macd'] - df.loc[mask, 'macd_signal']

            # RSI
            delta = symbol_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, 1e-10)
            df.loc[mask, 'rsi'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            bb_sma = symbol_df['close'].rolling(20, min_periods=1).mean()
            bb_std = symbol_df['close'].rolling(20, min_periods=1).std()
            df.loc[mask, 'bb_upper'] = bb_sma + (bb_std * 2)
            df.loc[mask, 'bb_middle'] = bb_sma
            df.loc[mask, 'bb_lower'] = bb_sma - (bb_std * 2)
            df.loc[mask, 'bb_width'] = df.loc[mask, 'bb_upper'] - df.loc[mask, 'bb_lower']
            df.loc[mask, 'bb_position'] = (symbol_df['close'] - df.loc[mask, 'bb_lower']) / (
                df.loc[mask, 'bb_width'].replace(0, 1e-10)
            )

            # Volume indicators
            df.loc[mask, 'volume_sma'] = symbol_df['volume'].rolling(10, min_periods=1).mean()
            df.loc[mask, 'volume_ratio'] = symbol_df['volume'] / df.loc[mask, 'volume_sma'].replace(0, 1e-10)

            # Price ratios and changes
            df.loc[mask, 'price_change_1'] = symbol_df['close'].pct_change(1)
            df.loc[mask, 'price_change_5'] = symbol_df['close'].pct_change(5)
            df.loc[mask, 'high_low_ratio'] = symbol_df['high'] / symbol_df['low'].replace(0, 1e-10)
            df.loc[mask, 'close_open_ratio'] = symbol_df['close'] / symbol_df['open'].replace(0, 1e-10)

            # Volatility
            df.loc[mask, 'volatility_5'] = symbol_df['close'].pct_change().rolling(5, min_periods=1).std()
            df.loc[mask, 'volatility_20'] = symbol_df['close'].pct_change().rolling(20, min_periods=1).std()

            # Support/Resistance levels
            df.loc[mask, 'resistance_1'] = symbol_df['high'].rolling(20, min_periods=1).max()
            df.loc[mask, 'support_1'] = symbol_df['low'].rolling(20, min_periods=1).min()

        return df

    def create_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Create lag features for time series prediction"""
        logger.info(f"Creating lag features for lags: {lags}")

        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol

            for lag in lags:
                df.loc[mask, f'close_lag_{lag}'] = df.loc[mask, 'close'].shift(lag)
                df.loc[mask, f'volume_lag_{lag}'] = df.loc[mask, 'volume'].shift(lag)
                df.loc[mask, f'rsi_lag_{lag}'] = df.loc[mask, 'rsi'].shift(lag)

        return df

    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window statistics"""
        logger.info("Creating rolling window features...")

        windows = [5, 10, 20, 50]

        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol

            for window in windows:
                # Price statistics
                df.loc[mask, f'close_mean_{window}'] = df.loc[mask, 'close'].rolling(window, min_periods=1).mean()
                df.loc[mask, f'close_std_{window}'] = df.loc[mask, 'close'].rolling(window, min_periods=1).std()
                df.loc[mask, f'close_min_{window}'] = df.loc[mask, 'close'].rolling(window, min_periods=1).min()
                df.loc[mask, f'close_max_{window}'] = df.loc[mask, 'close'].rolling(window, min_periods=1).max()

                # Volume statistics
                df.loc[mask, f'volume_mean_{window}'] = df.loc[mask, 'volume'].rolling(window, min_periods=1).mean()

        return df

    def normalize_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Normalize features for neural network input"""
        logger.info("Normalizing features...")

        scalers = {}

        # Keep metadata columns unchanged
        metadata_cols = ['symbol', 'timestamp', 'id', 'created_at', 'datetime']

        # Separate features by type for different scaling strategies
        price_features = [col for col in df.columns if col not in metadata_cols and
                         ('price' in col or 'close' in col or 'open' in col
                         or 'high' in col or 'low' in col or 'sma' in col or 'ema' in col or 'bb_' in col)]
        volume_features = [col for col in df.columns if col not in metadata_cols and 'volume' in col]
        indicator_features = ['rsi', 'macd', 'macd_signal', 'macd_histogram']
        ratio_features = [col for col in df.columns if col not in metadata_cols and 'ratio' in col]

        # Create a copy to preserve original metadata columns
        df_copy = df.copy()

        # Use RobustScaler for price data (handles outliers better)
        if price_features:
            price_scaler = RobustScaler()
            valid_price_features = [col for col in price_features if col in df.columns]
            df_copy[valid_price_features] = price_scaler.fit_transform(df[valid_price_features].fillna(0))
            scalers['price'] = price_scaler

        # Use StandardScaler for volume data
        if volume_features:
            volume_scaler = StandardScaler()
            valid_volume_features = [col for col in volume_features if col in df.columns]
            df_copy[valid_volume_features] = volume_scaler.fit_transform(df[valid_volume_features].fillna(0))
            scalers['volume'] = volume_scaler

        # Use MinMaxScaler for bounded indicators (RSI is 0-100)
        if indicator_features:
            indicator_scaler = MinMaxScaler()
            valid_indicator_features = [col for col in indicator_features if col in df.columns]
            if valid_indicator_features:
                df_copy[valid_indicator_features] = indicator_scaler.fit_transform(df[valid_indicator_features].fillna(0))
                scalers['indicators'] = indicator_scaler

        return df_copy, scalers

    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 50,
                        prediction_horizon: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM/Transformer models"""
        logger.info(f"Creating sequences with length {sequence_length} and horizon {prediction_horizon}")

        sequences = []
        targets = []

        # Check if symbol column exists, if not, assume all data is for same symbol
        if 'symbol' in df.columns:
            symbols = df['symbol'].unique()
        else:
            # Add a dummy symbol column if missing
            df['symbol'] = 'CRYPTO'
            symbols = ['CRYPTO']

        for symbol in symbols:
            symbol_df = df[df['symbol'] == symbol].sort_values('timestamp')

            # Get feature columns (exclude metadata)
            feature_cols = [col for col in symbol_df.columns
                          if col not in ['symbol', 'timestamp', 'id', 'created_at']]

            values = symbol_df[feature_cols].values

            for i in range(sequence_length, len(values) - prediction_horizon):
                sequences.append(values[i - sequence_length:i])
                # Target is the close price change after prediction_horizon steps
                current_close = symbol_df['close'].iloc[i]
                future_close = symbol_df['close'].iloc[i + prediction_horizon]
                target = (future_close - current_close) / current_close if current_close != 0 else 0
                targets.append(target)

        return np.array(sequences), np.array(targets)

    def create_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2,
                               val_size: float = 0.1) -> Dict[str, pd.DataFrame]:
        """Create time-aware train/validation/test splits"""
        logger.info(f"Creating train/test/val splits (test: {test_size}, val: {val_size})")

        # Sort by timestamp
        df = df.sort_values('timestamp')

        n = len(df)
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))

        splits = {
            'train': df.iloc[:train_end],
            'validation': df.iloc[train_end:val_end],
            'test': df.iloc[val_end:]
        }

        logger.info(f"Split sizes - Train: {len(splits['train'])}, "
                   f"Val: {len(splits['validation'])}, Test: {len(splits['test'])}")

        return splits

    def optimize_storage_format(self, df: pd.DataFrame, name: str):
        """Save data in optimized format for fast loading"""
        logger.info(f"Saving {name} in optimized Parquet format...")

        # Save as Parquet with compression
        parquet_path = self.output_dir / f"{name}.parquet"
        df.to_parquet(parquet_path, compression='snappy', index=False)

        # Also save a sample as CSV for human inspection
        csv_path = self.output_dir / f"{name}_sample.csv"
        df.head(1000).to_csv(csv_path, index=False)

        logger.info(f"Saved to {parquet_path}")

        return parquet_path

    def create_feature_metadata(self, df: pd.DataFrame) -> Dict:
        """Create metadata about features for model documentation"""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'features': {
                'total_count': len(df.columns),
                'feature_names': df.columns.tolist(),
                'feature_types': {
                    'price_based': [col for col in df.columns if 'price' in col or 'close' in col],
                    'volume_based': [col for col in df.columns if 'volume' in col],
                    'technical_indicators': [col for col in df.columns if col in ['rsi', 'macd', 'bb_width']],
                    'lag_features': [col for col in df.columns if 'lag_' in col],
                    'rolling_features': [col for col in df.columns if 'mean_' in col or 'std_' in col]
                }
            },
            'statistics': {
                'null_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
                'correlation_matrix': df.select_dtypes(include=[np.number]).corr().to_dict()
            }
        }

        # Save metadata
        metadata_path = self.output_dir / 'feature_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        return metadata

    def run_optimization_pipeline(self):
        """Run the complete data optimization pipeline"""
        logger.info("Starting ML data optimization pipeline...")

        # Load market data
        logger.info("Loading market data from database...")
        df = pd.read_sql_query("SELECT * FROM market_data ORDER BY timestamp", self.conn)

        if df.empty:
            logger.error("No market data found in database!")
            return

        # Add datetime column for easier manipulation
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

        # 1. Create technical indicators
        df = self.create_technical_indicators(df)

        # 2. Create lag features
        df = self.create_lag_features(df)

        # 3. Create rolling features
        df = self.create_rolling_features(df)

        # 4. Handle missing values (forward fill for time series continuity)
        # Preserve symbol column while filling
        symbol_col = df['symbol'].copy()
        df = df.groupby('symbol').ffill().bfill()
        df['symbol'] = symbol_col

        # 5. Normalize features
        df_normalized, scalers = self.normalize_features(df)

        # 6. Create train/test splits
        splits = self.create_train_test_split(df_normalized)

        # 7. Save optimized datasets
        for split_name, split_df in splits.items():
            self.optimize_storage_format(split_df, f"market_data_{split_name}")

        # 8. Create sequences for LSTM
        logger.info("Creating LSTM sequences...")
        X_sequences, y_sequences = self.create_sequences(splits['train'])

        # Save sequences
        np.save(self.output_dir / 'X_train_sequences.npy', X_sequences)
        np.save(self.output_dir / 'y_train_sequences.npy', y_sequences)

        # 9. Create and save feature metadata
        metadata = self.create_feature_metadata(df_normalized)

        # 10. Save scalers for inference
        import pickle
        with open(self.output_dir / 'scalers.pkl', 'wb') as f:
            pickle.dump(scalers, f)

        logger.info(f"✅ Data optimization complete! Files saved to {self.output_dir}")

        # Print summary
        print("\n" + "="*80)
        print("ML DATA OPTIMIZATION SUMMARY")
        print("="*80)
        print(f"\n📊 DATASET STATISTICS:")
        print(f"  • Total records: {len(df):,}")
        print(f"  • Features created: {len(df.columns)}")
        print(f"  • Train samples: {len(splits['train']):,}")
        print(f"  • Validation samples: {len(splits['validation']):,}")
        print(f"  • Test samples: {len(splits['test']):,}")
        print(f"\n🧠 SEQUENCE DATA:")
        print(f"  • Sequence shape: {X_sequences.shape}")
        print(f"  • Target shape: {y_sequences.shape}")
        print(f"\n💾 OUTPUT FILES:")
        print(f"  • Parquet files: {len(list(self.output_dir.glob('*.parquet')))}")
        print(f"  • Numpy arrays: {len(list(self.output_dir.glob('*.npy')))}")
        print(f"  • Location: {self.output_dir}")
        print("="*80)

        return {
            'splits': splits,
            'sequences': (X_sequences, y_sequences),
            'scalers': scalers,
            'metadata': metadata
        }


def main():
    """Main execution function"""
    db_path = "/Volumes/Lexar/RRRVentures/RRRalgorithms/data/local.db"
    output_dir = "/Volumes/Lexar/RRRVentures/RRRalgorithms/data/ml_optimized"

    optimizer = MLDataOptimizer(db_path, output_dir)
    results = optimizer.run_optimization_pipeline()

    return results


if __name__ == "__main__":
    main()