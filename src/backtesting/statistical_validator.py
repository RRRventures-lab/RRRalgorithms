from dataclasses import dataclass
from scipy import stats
from typing import List, Dict, Any, Optional
import logging
import numpy as np
import pandas as pd

"""
Statistical Validation Framework
=================================

Validates trading strategies using rigorous statistical tests:
- Hypothesis testing
- Multiple testing corrections
- Robustness checks
- Out-of-sample validation
"""

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results from statistical validation"""
    strategy_id: str
    is_valid: bool
    confidence_level: float

    # Statistical tests
    t_test_passed: bool
    t_statistic: float
    p_value: float
    p_value_adjusted: float  # Bonferroni corrected

    # Robustness tests
    sharpe_ratio: float
    sharpe_std: float  # Std across walk-forward splits
    win_rate: float
    win_rate_std: float
    profit_factor: float
    max_drawdown: float

    # Out-of-sample performance
    in_sample_sharpe: float
    out_sample_sharpe: float
    sharpe_degradation: float  # % degradation from in-sample to out-sample

    # Risk metrics
    var_95: float
    cvar_95: float
    sortino_ratio: float

    # Additional tests
    autocorrelation_test_passed: bool
    normality_test_passed: bool
    stationarity_test_passed: bool

    validation_notes: List[str]

    def get_grade(self) -> str:
        """Get overall validation grade"""
        if not self.is_valid:
            return 'F'

        score = 0

        # Sharpe ratio (0-30 points)
        if self.sharpe_ratio >= 3.0:
            score += 30
        elif self.sharpe_ratio >= 2.0:
            score += 20
        elif self.sharpe_ratio >= 1.5:
            score += 10

        # Win rate (0-20 points)
        if self.win_rate >= 0.65:
            score += 20
        elif self.win_rate >= 0.60:
            score += 15
        elif self.win_rate >= 0.55:
            score += 10

        # Profit factor (0-20 points)
        if self.profit_factor >= 2.5:
            score += 20
        elif self.profit_factor >= 2.0:
            score += 15
        elif self.profit_factor >= 1.5:
            score += 10

        # Statistical significance (0-15 points)
        if self.p_value_adjusted < 0.001:
            score += 15
        elif self.p_value_adjusted < 0.01:
            score += 10
        elif self.p_value_adjusted < 0.05:
            score += 5

        # Out-of-sample performance (0-15 points)
        if self.sharpe_degradation < 0.1:  # Less than 10% degradation
            score += 15
        elif self.sharpe_degradation < 0.25:
            score += 10
        elif self.sharpe_degradation < 0.40:
            score += 5

        # Grading scale
        if score >= 85:
            return 'A+'
        elif score >= 80:
            return 'A'
        elif score >= 75:
            return 'A-'
        elif score >= 70:
            return 'B+'
        elif score >= 65:
            return 'B'
        elif score >= 60:
            return 'B-'
        elif score >= 55:
            return 'C+'
        elif score >= 50:
            return 'C'
        else:
            return 'D'


class StatisticalValidator:
    """
    Validates trading strategies using rigorous statistical methods

    Tests performed:
    1. T-test for mean returns
    2. Sharpe ratio significance
    3. Walk-forward validation
    4. Multiple testing correction (Bonferroni)
    5. Robustness checks
    6. Autocorrelation test
    7. Normality test
    8. Out-of-sample validation
    """

    def __init__(self,
                 significance_level: float = 0.01,
                 min_sharpe: float = 2.0,
                 min_win_rate: float = 0.55,
                 min_profit_factor: float = 1.5,
                 max_drawdown: float = 0.20):
        """
        Initialize validator

        Args:
            significance_level: P-value threshold for statistical significance
            min_sharpe: Minimum Sharpe ratio
            min_win_rate: Minimum win rate
            min_profit_factor: Minimum profit factor
            max_drawdown: Maximum allowed drawdown
        """
        self.significance_level = significance_level
        self.min_sharpe = min_sharpe
        self.min_win_rate = min_win_rate
        self.min_profit_factor = min_profit_factor
        self.max_drawdown = max_drawdown

        self.n_strategies_tested = 0  # For Bonferroni correction

    def validate_strategy(self,
                         strategy_id: str,
                         returns: pd.Series,
                         metrics: Any,  # BacktestMetrics
                         walk_forward_metrics: Optional[List[Any]] = None) -> ValidationResult:
        """
        Validate a trading strategy

        Args:
            strategy_id: Strategy identifier
            returns: Series of strategy returns
            metrics: BacktestMetrics from main backtest
            walk_forward_metrics: List of metrics from walk-forward splits

        Returns:
            ValidationResult with all validation tests
        """
        logger.info(f"Validating strategy: {strategy_id}")

        self.n_strategies_tested += 1
        validation_notes = []

        # 1. T-test for mean returns
        t_statistic, p_value = stats.ttest_1samp(returns.dropna(), 0)
        p_value_adjusted = min(p_value * self.n_strategies_tested, 1.0)  # Bonferroni correction

        t_test_passed = p_value_adjusted < self.significance_level
        if t_test_passed:
            validation_notes.append("✓ Returns significantly different from zero")
        else:
            validation_notes.append("✗ Returns not statistically significant")

        # 2. Performance metrics
        sharpe_ratio = metrics.sharpe_ratio
        win_rate = metrics.win_rate
        profit_factor = metrics.profit_factor
        max_drawdown = abs(metrics.max_drawdown)

        # Performance checks
        sharpe_check = sharpe_ratio >= self.min_sharpe
        win_rate_check = win_rate >= self.min_win_rate
        profit_factor_check = profit_factor >= self.min_profit_factor
        drawdown_check = max_drawdown <= self.max_drawdown

        if sharpe_check:
            validation_notes.append(f"✓ Sharpe ratio {sharpe_ratio:.2f} >= {self.min_sharpe}")
        else:
            validation_notes.append(f"✗ Sharpe ratio {sharpe_ratio:.2f} < {self.min_sharpe}")

        if win_rate_check:
            validation_notes.append(f"✓ Win rate {win_rate:.1%} >= {self.min_win_rate:.1%}")
        else:
            validation_notes.append(f"✗ Win rate {win_rate:.1%} < {self.min_win_rate:.1%}")

        # 3. Walk-forward robustness
        if walk_forward_metrics:
            wf_sharpes = [m.sharpe_ratio for m in walk_forward_metrics]
            wf_win_rates = [m.win_rate for m in walk_forward_metrics]

            sharpe_std = np.std(wf_sharpes)
            win_rate_std = np.std(wf_win_rates)

            # Check consistency
            if sharpe_std < 0.5:  # Low variance is good
                validation_notes.append(f"✓ Consistent Sharpe across splits (std={sharpe_std:.2f})")
            else:
                validation_notes.append(f"⚠ High Sharpe variance across splits (std={sharpe_std:.2f})")

            # In-sample vs out-of-sample
            in_sample_sharpe = sharpe_ratio
            out_sample_sharpe = np.mean(wf_sharpes)
            sharpe_degradation = (in_sample_sharpe - out_sample_sharpe) / in_sample_sharpe if in_sample_sharpe > 0 else 1.0

            if sharpe_degradation < 0.25:
                validation_notes.append(f"✓ Low out-of-sample degradation ({sharpe_degradation:.1%})")
            else:
                validation_notes.append(f"⚠ Significant out-of-sample degradation ({sharpe_degradation:.1%})")
        else:
            sharpe_std = 0
            win_rate_std = 0
            in_sample_sharpe = sharpe_ratio
            out_sample_sharpe = sharpe_ratio
            sharpe_degradation = 0

        # 4. Autocorrelation test (returns should not be autocorrelated)
        autocorrelation_test_passed = self._test_autocorrelation(returns, validation_notes)

        # 5. Normality test (for risk metrics)
        normality_test_passed = self._test_normality(returns, validation_notes)

        # 6. Stationarity test
        stationarity_test_passed = self._test_stationarity(returns, validation_notes)

        # Overall validation
        is_valid = all([
            t_test_passed,
            sharpe_check,
            win_rate_check,
            profit_factor_check,
            drawdown_check
        ])

        if is_valid:
            validation_notes.append("✅ STRATEGY VALIDATED")
        else:
            validation_notes.append("❌ STRATEGY FAILED VALIDATION")

        # Calculate confidence level (0-1)
        confidence_scores = []
        if t_test_passed:
            confidence_scores.append(1.0 - p_value_adjusted)
        if sharpe_check:
            confidence_scores.append(min(sharpe_ratio / (self.min_sharpe * 2), 1.0))
        if win_rate_check:
            confidence_scores.append((win_rate - self.min_win_rate) / (1 - self.min_win_rate))

        confidence_level = np.mean(confidence_scores) if confidence_scores else 0.0

        result = ValidationResult(
            strategy_id=strategy_id,
            is_valid=is_valid,
            confidence_level=confidence_level,
            t_test_passed=t_test_passed,
            t_statistic=t_statistic,
            p_value=p_value,
            p_value_adjusted=p_value_adjusted,
            sharpe_ratio=sharpe_ratio,
            sharpe_std=sharpe_std,
            win_rate=win_rate,
            win_rate_std=win_rate_std,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            in_sample_sharpe=in_sample_sharpe,
            out_sample_sharpe=out_sample_sharpe,
            sharpe_degradation=sharpe_degradation,
            var_95=metrics.var_95,
            cvar_95=metrics.cvar_95,
            sortino_ratio=metrics.sortino_ratio,
            autocorrelation_test_passed=autocorrelation_test_passed,
            normality_test_passed=normality_test_passed,
            stationarity_test_passed=stationarity_test_passed,
            validation_notes=validation_notes
        )

        logger.info(f"Validation complete: {'✅ PASSED' if is_valid else '❌ FAILED'} "
                   f"(Grade: {result.get_grade()}, Confidence: {confidence_level:.1%})")

        return result

    def _test_autocorrelation(self, returns: pd.Series, notes: List[str]) -> bool:
        """Test for autocorrelation in returns (should be minimal)"""
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox

            # Ljung-Box test for autocorrelation
            lb_test = acorr_ljungbox(returns.dropna(), lags=[10], return_df=True)
            p_value = lb_test['lb_pvalue'].iloc[0]

            if p_value > 0.05:  # No significant autocorrelation
                notes.append("✓ No significant autocorrelation detected")
                return True
            else:
                notes.append("⚠ Significant autocorrelation detected (potential overfitting)")
                return False

        except Exception as e:
            notes.append(f"⚠ Autocorrelation test failed: {e}")
            return False

    def _test_normality(self, returns: pd.Series, notes: List[str]) -> bool:
        """Test if returns are normally distributed"""
        try:
            # Shapiro-Wilk test for normality
            stat, p_value = stats.shapiro(returns.dropna()[:5000])  # Limit to 5000 samples

            if p_value > 0.05:
                notes.append("✓ Returns approximately normally distributed")
                return True
            else:
                notes.append("⚠ Returns not normally distributed (use robust risk metrics)")
                return False

        except Exception as e:
            notes.append(f"⚠ Normality test failed: {e}")
            return False

    def _test_stationarity(self, returns: pd.Series, notes: List[str]) -> bool:
        """Test if returns are stationary (ADF test)"""
        try:
            from statsmodels.tsa.stattools import adfuller

            # Augmented Dickey-Fuller test
            result = adfuller(returns.dropna(), autolag='AIC')
            p_value = result[1]

            if p_value < 0.05:
                notes.append("✓ Returns are stationary")
                return True
            else:
                notes.append("⚠ Returns may not be stationary")
                return False

        except Exception as e:
            notes.append(f"⚠ Stationarity test failed: {e}")
            return False

    def validate_multiple_strategies(self,
                                   strategies_data: Dict[str, Dict]) -> List[ValidationResult]:
        """
        Validate multiple strategies with multiple testing correction

        Args:
            strategies_data: Dict mapping strategy_id to dict with 'returns' and 'metrics'

        Returns:
            List of ValidationResult, sorted by confidence
        """
        logger.info(f"Validating {len(strategies_data)} strategies with Bonferroni correction")

        # Reset counter
        self.n_strategies_tested = 0

        results = []
        for strategy_id, data in strategies_data.items():
            result = self.validate_strategy(
                strategy_id=strategy_id,
                returns=data['returns'],
                metrics=data['metrics'],
                walk_forward_metrics=data.get('walk_forward_metrics')
            )
            results.append(result)

        # Sort by confidence
        results.sort(key=lambda r: r.confidence_level, reverse=True)

        # Log summary
        valid_count = sum(1 for r in results if r.is_valid)
        logger.info(f"Validation summary: {valid_count}/{len(results)} strategies passed")

        return results

    def get_top_strategies(self,
                          results: List[ValidationResult],
                          n: int = 10) -> List[ValidationResult]:
        """Get top N validated strategies"""
        valid_results = [r for r in results if r.is_valid]
        return sorted(valid_results, key=lambda r: r.sharpe_ratio, reverse=True)[:n]
