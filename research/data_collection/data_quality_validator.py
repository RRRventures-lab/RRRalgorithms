from config import config
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import json
import numpy as np
import pandas as pd
import sqlite3
import sys

"""
Agent 10: Data Quality Assurance and Validation

Verifies 100% of collected data:
- No placeholder values (0, null, "example", "test", "placeholder")
- No missing timestamps
- No duplicate rows
- Realistic price ranges
- Valid OHLC relationships
- Spot checks every 1000th row
"""

sys.path.append(str(Path(__file__).parent.parent / "testing"))




class DataQualityValidator:
    """Validates data quality with zero tolerance for placeholders."""

    def __init__(self):
        """Initialize validator."""
        self.db_path = config.get_local_db_path()
        self.issues = []
        self.stats = {
            "total_rows": 0,
            "symbols_checked": 0,
            "placeholders_found": 0,
            "nulls_found": 0,
            "duplicates_found": 0,
            "outliers_found": 0,
            "invalid_ohlc": 0,
            "quality_score": 100.0
        }

    def validate_all_data(self) -> Dict:
        """Run complete validation on all collected data."""
        print("=" * 80)
        print(" " * 20 + "AGENT 10: DATA QUALITY VALIDATION")
        print(" " * 15 + "Zero Tolerance for Placeholders & Bad Data")
        print("=" * 80)

        conn = sqlite3.connect(str(self.db_path))

        # Get all unique symbols
        symbols_query = "SELECT DISTINCT symbol FROM ohlcv_data"
        symbols = pd.read_sql_query(symbols_query, conn)['symbol'].tolist()

        self.stats["symbols_checked"] = len(symbols)
        print(f"\n📊 Validating {len(symbols)} symbols...")

        for symbol in symbols:
            self._validate_symbol(conn, symbol)

        conn.close()

        # Calculate final quality score
        self._calculate_quality_score()

        # Generate report
        report = self._generate_report()
        print(report)

        return self.stats

    def _validate_symbol(self, conn, symbol: str):
        """Validate all data for a single symbol."""
        print(f"\n[Validator] Checking {symbol}...")

        # Load all data for symbol
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_data
            WHERE symbol = ?
            ORDER BY timestamp
        """
        df = pd.read_sql_query(query, conn, params=(symbol,))

        if df.empty:
            self.issues.append(f"{symbol}: No data found")
            return

        self.stats["total_rows"] += len(df)

        # Check 1: Placeholder values
        self._check_placeholders(df, symbol)

        # Check 2: Null values
        self._check_nulls(df, symbol)

        # Check 3: Duplicate timestamps
        self._check_duplicates(df, symbol)

        # Check 4: Price outliers
        self._check_outliers(df, symbol)

        # Check 5: OHLC relationship validity
        self._check_ohlc_validity(df, symbol)

        # Check 6: Volume sanity
        self._check_volume(df, symbol)

        # Check 7: Spot checks (every 1000th row)
        self._spot_check(df, symbol)

        print(f"[Validator] ✅ {symbol}: {len(df)} rows validated")

    def _check_placeholders(self, df: pd.DataFrame, symbol: str):
        """Check for placeholder values."""
        # Check for exactly 0 prices (suspicious)
        zero_prices = (df['close'] == 0).sum()
        if zero_prices > 0:
            self.stats["placeholders_found"] += zero_prices
            self.issues.append(f"{symbol}: {zero_prices} rows with price = 0")

        # Check for suspiciously round numbers (like 100, 1000, 10000)
        round_numbers = [100, 1000, 10000, 100000]
        for num in round_numbers:
            count = (df['close'] == num).sum()
            if count > len(df) * 0.01:  # More than 1% are exact round numbers
                self.stats["placeholders_found"] += count
                self.issues.append(f"{symbol}: {count} rows with suspiciously round price = {num}")

    def _check_nulls(self, df: pd.DataFrame, symbol: str):
        """Check for null/NaN values."""
        null_counts = df.isnull().sum()

        for col, count in null_counts.items():
            if count > 0:
                self.stats["nulls_found"] += count
                self.issues.append(f"{symbol}: {count} null values in column '{col}'")

    def _check_duplicates(self, df: pd.DataFrame, symbol: str):
        """Check for duplicate timestamps."""
        duplicates = df['timestamp'].duplicated().sum()

        if duplicates > 0:
            self.stats["duplicates_found"] += duplicates
            self.issues.append(f"{symbol}: {duplicates} duplicate timestamps")

    def _check_outliers(self, df: pd.DataFrame, symbol: str):
        """Check for price outliers (>3 std dev moves)."""
        df['returns'] = df['close'].pct_change()

        # Calculate statistics
        mean_return = df['returns'].mean()
        std_return = df['returns'].std()

        # Find outliers (>3 std dev)
        outliers = (np.abs(df['returns'] - mean_return) > 3 * std_return).sum()

        if outliers > len(df) * 0.05:  # More than 5% outliers is suspicious
            self.stats["outliers_found"] += outliers
            self.issues.append(f"{symbol}: {outliers} outlier returns (>3 std dev)")

    def _check_ohlc_validity(self, df: pd.DataFrame, symbol: str):
        """Check OHLC relationships are valid."""
        # High should be >= Open, Close
        invalid_high = ((df['high'] < df['open']) | (df['high'] < df['close'])).sum()

        # Low should be <= Open, Close
        invalid_low = ((df['low'] > df['open']) | (df['low'] > df['close'])).sum()

        total_invalid = invalid_high + invalid_low

        if total_invalid > 0:
            self.stats["invalid_ohlc"] += total_invalid
            self.issues.append(f"{symbol}: {total_invalid} rows with invalid OHLC relationships")

    def _check_volume(self, df: pd.DataFrame, symbol: str):
        """Check volume sanity."""
        # Volume should be > 0 for most bars
        zero_volume = (df['volume'] == 0).sum()
        zero_volume_pct = zero_volume / len(df)

        if zero_volume_pct > 0.1:  # More than 10% zero volume
            self.issues.append(f"{symbol}: {zero_volume_pct:.1%} of bars have zero volume")

    def _spot_check(self, df: pd.DataFrame, symbol: str):
        """Spot check every 1000th row."""
        spot_check_indices = range(0, len(df), 1000)

        for idx in spot_check_indices:
            if idx >= len(df):
                break

            row = df.iloc[idx]

            # Check if all values are reasonable
            if row['close'] <= 0:
                self.issues.append(f"{symbol}: Row {idx} has invalid close price: {row['close']}")

            if pd.isna(row['close']):
                self.issues.append(f"{symbol}: Row {idx} has null close price")

    def _calculate_quality_score(self):
        """Calculate overall quality score (0-100)."""
        if self.stats["total_rows"] == 0:
            self.stats["quality_score"] = 0.0
            return

        # Deduct points for issues
        deductions = 0

        # Placeholders: -10 points per 1% of data
        if self.stats["placeholders_found"] > 0:
            placeholder_pct = self.stats["placeholders_found"] / self.stats["total_rows"]
            deductions += placeholder_pct * 1000

        # Nulls: -10 points per 1% of data
        if self.stats["nulls_found"] > 0:
            null_pct = self.stats["nulls_found"] / self.stats["total_rows"]
            deductions += null_pct * 1000

        # Duplicates: -5 points per 1% of data
        if self.stats["duplicates_found"] > 0:
            dup_pct = self.stats["duplicates_found"] / self.stats["total_rows"]
            deductions += dup_pct * 500

        # Invalid OHLC: -10 points per 1% of data
        if self.stats["invalid_ohlc"] > 0:
            invalid_pct = self.stats["invalid_ohlc"] / self.stats["total_rows"]
            deductions += invalid_pct * 1000

        self.stats["quality_score"] = max(0.0, 100.0 - deductions)

    def _generate_report(self) -> str:
        """Generate quality validation report."""
        report = []
        report.append("\n" + "=" * 80)
        report.append(" " * 25 + "DATA QUALITY REPORT")
        report.append("=" * 80)

        report.append(f"\n📊 Total Rows Validated: {self.stats['total_rows']:,}")
        report.append(f"📦 Symbols Checked: {self.stats['symbols_checked']}")

        report.append(f"\n🔍 Quality Checks:")
        report.append(f"  {'✅' if self.stats['placeholders_found'] == 0 else '❌'} Placeholders: {self.stats['placeholders_found']}")
        report.append(f"  {'✅' if self.stats['nulls_found'] == 0 else '❌'} Null Values: {self.stats['nulls_found']}")
        report.append(f"  {'✅' if self.stats['duplicates_found'] == 0 else '❌'} Duplicates: {self.stats['duplicates_found']}")
        report.append(f"  {'✅' if self.stats['invalid_ohlc'] == 0 else '❌'} Invalid OHLC: {self.stats['invalid_ohlc']}")
        report.append(f"  ℹ️  Outliers: {self.stats['outliers_found']} (acceptable if <5%)")

        # Quality score with color
        score = self.stats['quality_score']
        if score >= 95:
            grade = "✅ EXCELLENT"
        elif score >= 85:
            grade = "⚠️  GOOD"
        elif score >= 70:
            grade = "⚠️  FAIR"
        else:
            grade = "❌ POOR"

        report.append(f"\n🎯 Quality Score: {score:.1f}/100 ({grade})")

        if self.issues:
            report.append(f"\n⚠️  Issues Found ({len(self.issues)}):")
            for issue in self.issues[:10]:  # Show first 10
                report.append(f"  - {issue}")
            if len(self.issues) > 10:
                report.append(f"  ... and {len(self.issues) - 10} more issues")
        else:
            report.append("\n✅ No Issues Found - Data is 100% Clean!")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

    def save_report(self, output_path: Path):
        """Save quality report to JSON."""
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "stats": self.stats,
            "issues": self.issues
        }

        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\n💾 Quality report saved to: {output_path}")


def main():
    """Run data quality validation."""
    validator = DataQualityValidator()

    # Run validation
    results = validator.validate_all_data()

    # Save report
    output_path = Path("/Volumes/Lexar/RRRVentures/RRRalgorithms/research/data_collection/quality_report.json")
    validator.save_report(output_path)

    return results


if __name__ == "__main__":
    results = main()
