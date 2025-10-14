from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Any
import json
import logging
import numpy as np
import pandas as pd
import sqlite3
import warnings

#!/usr/bin/env python3
"""
Comprehensive Data Validation and Optimization Audit
For RRR Ventures Cryptocurrency Trading Algorithm System
"""


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAuditor:
    """Comprehensive data auditor for trading algorithm training data"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.audit_results = {
            'timestamp': datetime.now().isoformat(),
            'data_inventory': {},
            'quality_metrics': {},
            'integrity_issues': [],
            'optimization_recommendations': [],
            'ml_readiness_score': 0
        }

    def run_full_audit(self) -> Dict:
        """Execute complete data audit pipeline"""
        logger.info("Starting comprehensive data audit...")

        # 1. Data Inventory
        self.inventory_data()

        # 2. Quality Assessment
        self.assess_data_quality()

        # 3. Integrity Validation
        self.validate_integrity()

        # 4. Statistical Analysis
        self.perform_statistical_analysis()

        # 5. ML Readiness Check
        self.assess_ml_readiness()

        # 6. Generate Recommendations
        self.generate_recommendations()

        return self.audit_results

    def inventory_data(self):
        """Inventory all available data sources"""
        logger.info("Inventorying data sources...")

        # Get all tables
        cursor = self.conn.cursor()
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()

        inventory = {}
        for table in tables:
            table_name = table[0]

            # Get row count
            count = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

            # Get schema
            schema = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()

            # Get date range if timestamp exists
            date_range = None
            if any('timestamp' in str(col).lower() for col in schema):
                try:
                    date_query = f"SELECT MIN(timestamp), MAX(timestamp) FROM {table_name} WHERE timestamp IS NOT NULL"
                    dates = cursor.execute(date_query).fetchone()
                    if dates and dates[0]:
                        date_range = {
                            'start': datetime.fromtimestamp(dates[0]) if dates[0] else None,
                            'end': datetime.fromtimestamp(dates[1]) if dates[1] else None
                        }
                except:
                    pass

            inventory[table_name] = {
                'record_count': count,
                'columns': [col[1] for col in schema],
                'date_range': date_range
            }

        self.audit_results['data_inventory'] = inventory
        logger.info(f"Found {len(inventory)} tables with total {sum(t['record_count'] for t in inventory.values())} records")

    def assess_data_quality(self):
        """Assess data quality metrics"""
        logger.info("Assessing data quality...")

        quality_metrics = {}

        # Focus on market_data table for now
        df = pd.read_sql_query("SELECT * FROM market_data", self.conn)

        if not df.empty:
            quality_metrics['market_data'] = {
                'completeness': {
                    'null_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
                    'total_completeness': 100 - (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
                },
                'uniqueness': {
                    'duplicate_rows': len(df[df.duplicated()]),
                    'duplicate_timestamps': len(df[df.duplicated(subset=['symbol', 'timestamp'])])
                },
                'consistency': {
                    'symbols': df['symbol'].unique().tolist(),
                    'timestamp_format_consistent': self._check_timestamp_consistency(df)
                },
                'validity': self._validate_ohlcv_data(df)
            }

        # Check predictions table
        try:
            pred_df = pd.read_sql_query("SELECT * FROM predictions", self.conn)
            if not pred_df.empty:
                quality_metrics['predictions'] = {
                    'completeness': 100 - (pred_df.isnull().sum().sum() / (len(pred_df) * len(pred_df.columns)) * 100),
                    'unique_models': pred_df['model_version'].unique().tolist() if 'model_version' in pred_df else [],
                    'confidence_stats': {
                        'mean': pred_df['confidence'].mean() if 'confidence' in pred_df else None,
                        'std': pred_df['confidence'].std() if 'confidence' in pred_df else None
                    }
                }
        except:
            pass

        self.audit_results['quality_metrics'] = quality_metrics

    def _check_timestamp_consistency(self, df: pd.DataFrame) -> bool:
        """Check if timestamps are consistently formatted"""
        try:
            # Check if all timestamps are numeric (Unix timestamps)
            return df['timestamp'].apply(lambda x: isinstance(x, (int, float))).all()
        except:
            return False

    def _validate_ohlcv_data(self, df: pd.DataFrame) -> Dict:
        """Validate OHLCV data integrity"""
        validity = {
            'ohlc_violations': 0,
            'negative_values': 0,
            'extreme_outliers': 0,
            'suspicious_patterns': []
        }

        # Check OHLC relationships
        ohlc_violations = (
            (df['open'] > df['high']) |
            (df['low'] > df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['high']) |
            (df['close'] > df['high']) |
            (df['open'] < df['low']) |
            (df['close'] < df['low'])
        ).sum()
        validity['ohlc_violations'] = int(ohlc_violations)

        # Check for negative values
        validity['negative_values'] = int((df[['open', 'high', 'low', 'close', 'volume']] < 0).sum().sum())

        # Check for extreme outliers using IQR method
        for col in ['open', 'high', 'low', 'close', 'volume']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR))).sum()
            validity['extreme_outliers'] += int(outliers)

        # Check for suspicious patterns
        if len(df) > 0:
            # Check for frozen prices (no movement)
            frozen = df.groupby('symbol').apply(
                lambda x: (x['close'].diff().abs() < 0.01).sum() / len(x) * 100
            )
            if any(frozen > 50):
                validity['suspicious_patterns'].append('Frozen prices detected')

            # Check for extreme volatility
            returns = df.groupby('symbol')['close'].pct_change()
            if returns.abs().max() > 0.5:  # 50% move in one period
                validity['suspicious_patterns'].append('Extreme volatility detected')

        return validity

    def validate_integrity(self):
        """Validate data integrity across sources"""
        logger.info("Validating data integrity...")

        issues = []

        # Check market data integrity
        cursor = self.conn.cursor()

        # Check for time gaps
        gap_query = """
        WITH time_gaps AS (
            SELECT
                symbol,
                timestamp,
                LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_timestamp,
                timestamp - LAG(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) as gap
            FROM market_data
        )
        SELECT symbol, COUNT(*) as large_gaps
        FROM time_gaps
        WHERE gap > 10  -- More than 10 seconds
        GROUP BY symbol
        """

        gaps = cursor.execute(gap_query).fetchall()
        for symbol, gap_count in gaps:
            if gap_count > 0:
                issues.append({
                    'type': 'TIME_GAPS',
                    'severity': 'MEDIUM',
                    'description': f'{symbol} has {gap_count} time gaps > 10 seconds',
                    'impact': 'May affect time-series model training'
                })

        # Check for data staleness
        latest_query = "SELECT MAX(timestamp) FROM market_data"
        latest_timestamp = cursor.execute(latest_query).fetchone()[0]
        if latest_timestamp:
            age_hours = (datetime.now().timestamp() - latest_timestamp) / 3600
            if age_hours > 24:
                issues.append({
                    'type': 'STALE_DATA',
                    'severity': 'HIGH',
                    'description': f'Latest data is {age_hours:.1f} hours old',
                    'impact': 'Models trained on outdated data'
                })

        self.audit_results['integrity_issues'] = issues

    def perform_statistical_analysis(self):
        """Perform statistical analysis on the data"""
        logger.info("Performing statistical analysis...")

        df = pd.read_sql_query("SELECT * FROM market_data", self.conn)

        if not df.empty:
            stats = {}

            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol].copy()
                symbol_df['returns'] = symbol_df['close'].pct_change()

                stats[symbol] = {
                    'price_stats': {
                        'mean': float(symbol_df['close'].mean()),
                        'std': float(symbol_df['close'].std()),
                        'min': float(symbol_df['close'].min()),
                        'max': float(symbol_df['close'].max())
                    },
                    'volume_stats': {
                        'mean': float(symbol_df['volume'].mean()),
                        'std': float(symbol_df['volume'].std())
                    },
                    'returns_stats': {
                        'mean': float(symbol_df['returns'].mean()),
                        'std': float(symbol_df['returns'].std()),
                        'skewness': float(symbol_df['returns'].skew()),
                        'kurtosis': float(symbol_df['returns'].kurtosis())
                    },
                    'stationarity': self._test_stationarity(symbol_df['close'])
                }

            self.audit_results['statistical_analysis'] = stats

    def _test_stationarity(self, series: pd.Series) -> Dict:
        """Test for stationarity using Augmented Dickey-Fuller test"""
        from statsmodels.tsa.stattools import adfuller

        try:
            result = adfuller(series.dropna())
            return {
                'is_stationary': result[1] < 0.05,
                'p_value': float(result[1]),
                'test_statistic': float(result[0])
            }
        except:
            return {'is_stationary': None, 'p_value': None, 'test_statistic': None}

    def assess_ml_readiness(self):
        """Assess readiness for ML model training"""
        logger.info("Assessing ML readiness...")

        readiness_score = 100
        readiness_factors = []

        # Check data volume
        market_data_count = self.audit_results['data_inventory'].get('market_data', {}).get('record_count', 0)
        if market_data_count < 1000:
            readiness_score -= 30
            readiness_factors.append({
                'factor': 'Insufficient data volume',
                'penalty': -30,
                'recommendation': 'Need at least 1000 data points for training'
            })
        elif market_data_count < 10000:
            readiness_score -= 10
            readiness_factors.append({
                'factor': 'Limited data volume',
                'penalty': -10,
                'recommendation': 'More data would improve model performance'
            })

        # Check data quality
        quality = self.audit_results.get('quality_metrics', {}).get('market_data', {})
        if quality:
            completeness = quality.get('completeness', {}).get('total_completeness', 0)
            if completeness < 95:
                readiness_score -= 20
                readiness_factors.append({
                    'factor': 'Data completeness issues',
                    'penalty': -20,
                    'recommendation': 'Address missing values before training'
                })

            validity = quality.get('validity', {})
            if validity.get('ohlc_violations', 0) > 0:
                readiness_score -= 15
                readiness_factors.append({
                    'factor': 'OHLC data integrity violations',
                    'penalty': -15,
                    'recommendation': 'Fix OHLC relationship violations'
                })

        # Check for critical issues
        critical_issues = [i for i in self.audit_results.get('integrity_issues', [])
                          if i['severity'] == 'HIGH']
        if critical_issues:
            readiness_score -= 25
            readiness_factors.append({
                'factor': 'Critical integrity issues found',
                'penalty': -25,
                'recommendation': 'Resolve critical issues before training'
            })

        self.audit_results['ml_readiness_score'] = max(0, readiness_score)
        self.audit_results['ml_readiness_factors'] = readiness_factors

    def generate_recommendations(self):
        """Generate optimization recommendations"""
        logger.info("Generating optimization recommendations...")

        recommendations = []

        # Data structure recommendations
        recommendations.append({
            'category': 'DATA_FORMAT',
            'priority': 'HIGH',
            'recommendation': 'Convert SQLite data to Parquet format for faster ML training',
            'implementation': 'Use pandas.to_parquet() with snappy compression',
            'expected_improvement': '3-5x faster data loading'
        })

        # Feature engineering recommendations
        recommendations.append({
            'category': 'FEATURE_ENGINEERING',
            'priority': 'HIGH',
            'recommendation': 'Create technical indicators as features',
            'implementation': 'Add RSI, MACD, Bollinger Bands, volume indicators',
            'expected_improvement': 'Improved model predictive power'
        })

        # Data pipeline recommendations
        recommendations.append({
            'category': 'PIPELINE',
            'priority': 'MEDIUM',
            'recommendation': 'Implement sliding window data preparation',
            'implementation': 'Create sequences of length 50-100 for LSTM models',
            'expected_improvement': 'Enable time-series model training'
        })

        # Normalization recommendations
        recommendations.append({
            'category': 'PREPROCESSING',
            'priority': 'HIGH',
            'recommendation': 'Implement feature scaling',
            'implementation': 'Use StandardScaler for price data, MinMaxScaler for indicators',
            'expected_improvement': 'Faster model convergence'
        })

        # Data quality recommendations
        if self.audit_results.get('integrity_issues'):
            recommendations.append({
                'category': 'DATA_QUALITY',
                'priority': 'CRITICAL',
                'recommendation': 'Address data integrity issues',
                'implementation': 'Run data cleaning pipeline to fix violations',
                'expected_improvement': 'Prevent model training on corrupted data'
            })

        # Real-time data recommendations
        latest_data = self.audit_results.get('data_inventory', {}).get('market_data', {}).get('date_range', {})
        if latest_data and latest_data.get('end'):
            age_hours = (datetime.now() - latest_data['end']).total_seconds() / 3600
            if age_hours > 1:
                recommendations.append({
                    'category': 'DATA_FRESHNESS',
                    'priority': 'HIGH',
                    'recommendation': 'Implement real-time data ingestion',
                    'implementation': 'Set up WebSocket connections to Polygon.io',
                    'expected_improvement': 'Enable live trading capabilities'
                })

        self.audit_results['optimization_recommendations'] = recommendations

    def export_report(self, output_path: str):
        """Export audit report to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.audit_results, f, indent=2, default=str)
        logger.info(f"Audit report exported to {output_path}")

    def print_summary(self):
        """Print executive summary of audit results"""
        print("\n" + "="*80)
        print("DATA AUDIT EXECUTIVE SUMMARY")
        print("="*80)

        # Data Inventory
        print("\n📊 DATA INVENTORY:")
        for table, info in self.audit_results['data_inventory'].items():
            print(f"  • {table}: {info['record_count']:,} records")
            if info.get('date_range') and info['date_range'].get('start'):
                print(f"    Date range: {info['date_range']['start']} to {info['date_range']['end']}")

        # Quality Metrics
        print("\n✅ DATA QUALITY:")
        quality = self.audit_results.get('quality_metrics', {}).get('market_data', {})
        if quality:
            print(f"  • Completeness: {quality.get('completeness', {}).get('total_completeness', 0):.1f}%")
            print(f"  • Duplicate rows: {quality.get('uniqueness', {}).get('duplicate_rows', 0)}")
            validity = quality.get('validity', {})
            if validity:
                print(f"  • OHLC violations: {validity.get('ohlc_violations', 0)}")
                print(f"  • Negative values: {validity.get('negative_values', 0)}")

        # Integrity Issues
        print("\n⚠️  INTEGRITY ISSUES:")
        issues = self.audit_results.get('integrity_issues', [])
        if issues:
            for issue in issues[:5]:  # Show top 5 issues
                print(f"  • [{issue['severity']}] {issue['description']}")
        else:
            print("  • No critical issues found")

        # ML Readiness
        print("\n🤖 ML READINESS:")
        score = self.audit_results.get('ml_readiness_score', 0)
        print(f"  • Overall Score: {score}/100")
        factors = self.audit_results.get('ml_readiness_factors', [])
        if factors:
            print("  • Key factors:")
            for factor in factors[:3]:
                print(f"    - {factor['factor']} ({factor['penalty']} points)")

        # Top Recommendations
        print("\n🚀 TOP RECOMMENDATIONS:")
        recs = self.audit_results.get('optimization_recommendations', [])
        for rec in sorted(recs, key=lambda x: x['priority'] == 'CRITICAL', reverse=True)[:5]:
            print(f"  • [{rec['priority']}] {rec['recommendation']}")

        print("\n" + "="*80)


def main():
    """Main execution function"""
    db_path = "/Volumes/Lexar/RRRVentures/RRRalgorithms/data/local.db"

    # Create auditor and run audit
    auditor = DataAuditor(db_path)
    results = auditor.run_full_audit()

    # Print summary
    auditor.print_summary()

    # Export full report
    report_path = "/Volumes/Lexar/RRRVentures/RRRalgorithms/data/audit_report.json"
    auditor.export_report(report_path)

    return results


if __name__ == "__main__":
    main()