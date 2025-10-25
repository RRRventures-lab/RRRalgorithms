#!/usr/bin/env python3
"""
Data Optimization Pipeline for Neural Network Training
Implements all recommended optimizations from the audit report
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"  # > 95%
    GOOD = "good"           # 85-95%
    FAIR = "fair"           # 70-85%
    POOR = "poor"           # < 70%


@dataclass
class DataValidationResult:
    """Results from data validation"""
    source: str
    total_rows: int
    missing_values: int
    outliers: int
    duplicates: int
    invalid_ohlc: int
    quality_score: float
    quality_level: DataQuality
    issues: List[str]
    recommendations: List[str]


class DataOptimizationPipeline:
    """
    Complete data optimization pipeline for ML model training
    Implements all audit recommendations
    """

    def __init__(self, db_path: str = "data/transparency.db"):
        self.db_path = Path(db_path)
        self.validation_results = {}
        self.optimized_data = {}

    # ========================================================================
    # 1. DATA VALIDATION
    # ========================================================================

    def validate_data_source(self, df: pd.DataFrame, source_name: str) -> DataValidationResult:
        """
        Comprehensive data validation for a single source
        """
        logger.info(f"Validating data source: {source_name}")

        total_rows = len(df)
        issues = []
        recommendations = []

        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            missing_pct = (missing_values / (total_rows * len(df.columns))) * 100
            issues.append(f"Missing values: {missing_values} ({missing_pct:.1f}%)")
            recommendations.append("Apply forward-fill imputation for time series")

        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Duplicate rows: {duplicates}")
            recommendations.append("Remove duplicate entries")

        # Check for outliers (using IQR method)
        outliers = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_condition = ((df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR)))
            outliers += outlier_condition.sum()

        if outliers > 0:
            outlier_pct = (outliers / (total_rows * len(numeric_cols))) * 100
            if outlier_pct > 5:
                issues.append(f"Excessive outliers: {outliers} ({outlier_pct:.1f}%)")
                recommendations.append("Apply winsorization or robust scaling")

        # Check OHLC validity if applicable
        invalid_ohlc = 0
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_ohlc = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            ).sum()

            if invalid_ohlc > 0:
                issues.append(f"Invalid OHLC relationships: {invalid_ohlc}")
                recommendations.append("Fix or remove invalid OHLC bars")

        # Calculate quality score
        quality_score = 100.0
        quality_score -= (missing_values / (total_rows * len(df.columns))) * 20
        quality_score -= (duplicates / total_rows) * 15
        quality_score -= (outliers / (total_rows * len(numeric_cols))) * 10
        quality_score -= (invalid_ohlc / total_rows) * 25
        quality_score = max(0, min(100, quality_score))

        # Determine quality level
        if quality_score >= 95:
            quality_level = DataQuality.EXCELLENT
        elif quality_score >= 85:
            quality_level = DataQuality.GOOD
        elif quality_score >= 70:
            quality_level = DataQuality.FAIR
        else:
            quality_level = DataQuality.POOR

        result = DataValidationResult(
            source=source_name,
            total_rows=total_rows,
            missing_values=missing_values,
            outliers=outliers,
            duplicates=duplicates,
            invalid_ohlc=invalid_ohlc,
            quality_score=quality_score,
            quality_level=quality_level,
            issues=issues,
            recommendations=recommendations
        )

        self.validation_results[source_name] = result
        return result

    # ========================================================================
    # 2. DATA CLEANING
    # ========================================================================

    def clean_data(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """
        Apply data cleaning based on validation results
        """
        logger.info(f"Cleaning data for: {source_name}")
        df_clean = df.copy()

        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        if len(df_clean) < initial_rows:
            logger.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")

        # Handle missing values
        if df_clean.isnull().sum().sum() > 0:
            # Forward fill for time series
            df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
            logger.info("Applied forward-fill imputation for missing values")

        # Fix OHLC relationships
        if all(col in df_clean.columns for col in ['open', 'high', 'low', 'close']):
            # Ensure high is the maximum
            df_clean['high'] = df_clean[['open', 'high', 'low', 'close']].max(axis=1)
            # Ensure low is the minimum
            df_clean['low'] = df_clean[['open', 'high', 'low', 'close']].min(axis=1)
            logger.info("Fixed OHLC relationships")

        # Handle outliers using winsorization
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['timestamp', 'volume']:  # Don't clip these
                lower = df_clean[col].quantile(0.01)
                upper = df_clean[col].quantile(0.99)
                df_clean[col] = df_clean[col].clip(lower, upper)

        logger.info(f"Data cleaning complete. Rows: {len(df_clean)}")
        return df_clean

    # ========================================================================
    # 3. FEATURE ENGINEERING
    # ========================================================================

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features optimized for neural networks
        """
        logger.info("Engineering features for ML")
        df_features = df.copy()

        # Add datetime features if timestamp exists
        if 'timestamp' in df_features.columns:
            df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
            df_features['hour'] = df_features['timestamp'].dt.hour
            df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
            df_features['day_of_month'] = df_features['timestamp'].dt.day
            df_features['month'] = df_features['timestamp'].dt.month

        # Price-based features
        if 'close' in df_features.columns:
            # Returns
            df_features['returns'] = df_features['close'].pct_change()
            df_features['log_returns'] = np.log(df_features['close'] / df_features['close'].shift(1))

            # Moving averages
            for period in [5, 10, 20, 50]:
                df_features[f'ma_{period}'] = df_features['close'].rolling(period).mean()
                df_features[f'ma_{period}_ratio'] = df_features['close'] / df_features[f'ma_{period}']

            # Volatility
            df_features['volatility_20'] = df_features['returns'].rolling(20).std()

            # RSI
            df_features['rsi_14'] = self.calculate_rsi(df_features['close'], 14)

            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df_features['bb_middle'] = df_features['close'].rolling(bb_period).mean()
            bb_std_dev = df_features['close'].rolling(bb_period).std()
            df_features['bb_upper'] = df_features['bb_middle'] + (bb_std_dev * bb_std)
            df_features['bb_lower'] = df_features['bb_middle'] - (bb_std_dev * bb_std)
            df_features['bb_width'] = df_features['bb_upper'] - df_features['bb_lower']
            df_features['bb_position'] = (
                (df_features['close'] - df_features['bb_lower']) /
                (df_features['bb_upper'] - df_features['bb_lower'])
            )

            # Lag features
            for lag in [1, 3, 7, 14, 30]:
                df_features[f'close_lag_{lag}'] = df_features['close'].shift(lag)
                df_features[f'returns_lag_{lag}'] = df_features['returns'].shift(lag)

        # Volume-based features
        if 'volume' in df_features.columns:
            df_features['volume_ma_20'] = df_features['volume'].rolling(20).mean()
            df_features['volume_ratio'] = df_features['volume'] / df_features['volume_ma_20']
            df_features['volume_change'] = df_features['volume'].pct_change()

        # OHLC features
        if all(col in df_features.columns for col in ['open', 'high', 'low', 'close']):
            # Price range
            df_features['hl_range'] = df_features['high'] - df_features['low']
            df_features['oc_range'] = abs(df_features['close'] - df_features['open'])
            df_features['hl_ratio'] = df_features['hl_range'] / df_features['close']

            # Candlestick patterns
            df_features['body_size'] = abs(df_features['close'] - df_features['open'])
            df_features['upper_shadow'] = df_features['high'] - df_features[['close', 'open']].max(axis=1)
            df_features['lower_shadow'] = df_features[['close', 'open']].min(axis=1) - df_features['low']

        logger.info(f"Created {len(df_features.columns) - len(df.columns)} new features")
        return df_features

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # ========================================================================
    # 4. DATA NORMALIZATION
    # ========================================================================

    def normalize_for_nn(self, df: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """
        Normalize data for neural network training
        """
        logger.info(f"Normalizing data using {method} method")
        df_norm = df.copy()

        numeric_cols = df_norm.select_dtypes(include=[np.number]).columns
        # Don't normalize certain columns
        exclude_cols = ['timestamp', 'hour', 'day_of_week', 'day_of_month', 'month']
        cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]

        if method == 'zscore':
            # Z-score normalization
            for col in cols_to_normalize:
                mean = df_norm[col].mean()
                std = df_norm[col].std()
                if std > 0:
                    df_norm[f'{col}_norm'] = (df_norm[col] - mean) / std
                else:
                    df_norm[f'{col}_norm'] = 0

        elif method == 'minmax':
            # Min-Max normalization
            for col in cols_to_normalize:
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                if max_val - min_val > 0:
                    df_norm[f'{col}_norm'] = (df_norm[col] - min_val) / (max_val - min_val)
                else:
                    df_norm[f'{col}_norm'] = 0

        elif method == 'robust':
            # Robust scaling (using median and IQR)
            for col in cols_to_normalize:
                median = df_norm[col].median()
                q1 = df_norm[col].quantile(0.25)
                q3 = df_norm[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    df_norm[f'{col}_norm'] = (df_norm[col] - median) / iqr
                else:
                    df_norm[f'{col}_norm'] = 0

        logger.info(f"Normalized {len(cols_to_normalize)} columns")
        return df_norm

    # ========================================================================
    # 5. DATA SEQUENCING FOR LSTM/TRANSFORMER
    # ========================================================================

    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 30,
        prediction_horizon: int = 1,
        target_col: str = 'close'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series models (LSTM, Transformer)
        """
        logger.info(f"Creating sequences: length={sequence_length}, horizon={prediction_horizon}")

        # Select feature columns (normalized ones)
        feature_cols = [col for col in df.columns if col.endswith('_norm')]
        if not feature_cols:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target from features if present
        if f'{target_col}_norm' in feature_cols:
            feature_cols.remove(f'{target_col}_norm')
        elif target_col in feature_cols:
            feature_cols.remove(target_col)

        # Prepare data
        feature_data = df[feature_cols].values
        target_data = df[target_col].values if target_col in df.columns else df[f'{target_col}_norm'].values

        # Create sequences
        X, y = [], []
        for i in range(len(feature_data) - sequence_length - prediction_horizon + 1):
            X.append(feature_data[i:i + sequence_length])
            y.append(target_data[i + sequence_length + prediction_horizon - 1])

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        return X, y

    # ========================================================================
    # 6. COMPLETE PIPELINE
    # ========================================================================

    def run_complete_pipeline(
        self,
        source_name: str = "transparency_db",
        table_name: str = "performance_snapshots",
        sequence_length: int = 30
    ) -> Dict[str, Any]:
        """
        Run the complete optimization pipeline
        """
        logger.info(f"Running complete pipeline for {source_name}/{table_name}")

        # 1. Load data
        conn = sqlite3.connect(str(self.db_path))
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        logger.info(f"Loaded {len(df)} rows from {table_name}")

        # 2. Validate
        validation_result = self.validate_data_source(df, source_name)
        logger.info(f"Quality Score: {validation_result.quality_score:.1f}% ({validation_result.quality_level.value})")

        # 3. Clean
        df_clean = self.clean_data(df, source_name)

        # 4. Engineer features
        df_features = self.engineer_features(df_clean)

        # 5. Normalize
        df_normalized = self.normalize_for_nn(df_features, method='zscore')

        # 6. Create sequences
        X, y = self.create_sequences(df_normalized, sequence_length=sequence_length)

        # Store optimized data
        self.optimized_data[source_name] = {
            'original_shape': df.shape,
            'cleaned_shape': df_clean.shape,
            'features_shape': df_features.shape,
            'normalized_shape': df_normalized.shape,
            'sequences_shape': X.shape,
            'target_shape': y.shape,
            'validation_result': validation_result,
            'feature_names': df_normalized.columns.tolist(),
            'X': X,
            'y': y
        }

        # Save processed data
        output_dir = Path('data/processed')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as numpy arrays for fast loading
        np.save(output_dir / f'{source_name}_X.npy', X)
        np.save(output_dir / f'{source_name}_y.npy', y)

        # Save metadata
        metadata = {
            'source_name': source_name,
            'table_name': table_name,
            'sequence_length': sequence_length,
            'feature_names': df_normalized.columns.tolist(),
            'original_rows': len(df),
            'sequences_created': len(X),
            'quality_score': validation_result.quality_score,
            'timestamp': datetime.now().isoformat()
        }

        with open(output_dir / f'{source_name}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Pipeline complete. Saved to {output_dir}")

        return {
            'success': True,
            'quality_score': validation_result.quality_score,
            'sequences_created': len(X),
            'features_created': len(df_normalized.columns) - len(df.columns),
            'output_path': str(output_dir)
        }

    # ========================================================================
    # 7. MONITORING & ALERTS
    # ========================================================================

    def check_data_drift(self, current_df: pd.DataFrame, reference_df: pd.DataFrame) -> Dict[str, float]:
        """
        Check for data drift between current and reference datasets
        """
        drift_scores = {}

        numeric_cols = current_df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in reference_df.columns:
                # Calculate KS statistic
                from scipy import stats
                ks_stat, p_value = stats.ks_2samp(
                    reference_df[col].dropna(),
                    current_df[col].dropna()
                )
                drift_scores[col] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'drift_detected': p_value < 0.05
                }

        return drift_scores


# ========================================================================
# MAIN EXECUTION
# ========================================================================

def main():
    """
    Execute the complete data optimization pipeline
    """
    print("\n" + "="*80)
    print("DATA OPTIMIZATION PIPELINE FOR NEURAL NETWORKS")
    print("="*80)

    # Initialize pipeline
    pipeline = DataOptimizationPipeline()

    # Run for performance snapshots
    result = pipeline.run_complete_pipeline(
        source_name="performance_snapshots",
        table_name="performance_snapshots",
        sequence_length=30
    )

    # Print results
    print("\n" + "="*80)
    print("PIPELINE RESULTS")
    print("="*80)
    print(f"✅ Success: {result['success']}")
    print(f"📊 Quality Score: {result['quality_score']:.1f}%")
    print(f"🔢 Sequences Created: {result['sequences_created']:,}")
    print(f"🎯 Features Created: {result['features_created']}")
    print(f"💾 Output Path: {result['output_path']}")

    # Get validation details
    for source, data in pipeline.optimized_data.items():
        validation = data['validation_result']
        print(f"\n📋 Validation for {source}:")
        print(f"   - Total Rows: {validation.total_rows:,}")
        print(f"   - Missing Values: {validation.missing_values}")
        print(f"   - Outliers: {validation.outliers}")
        print(f"   - Quality: {validation.quality_level.value.upper()}")

        if validation.issues:
            print(f"   - Issues: {', '.join(validation.issues)}")
        if validation.recommendations:
            print(f"   - Recommendations: {', '.join(validation.recommendations)}")

    print("\n" + "="*80)
    print("✅ Data optimization complete and ready for neural network training!")
    print("="*80)


if __name__ == "__main__":
    main()