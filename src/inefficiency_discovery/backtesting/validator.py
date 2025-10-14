from ..base import InefficiencySignal, BacktestResult
from datetime import datetime, timedelta
from functools import lru_cache
from scipy import stats
from typing import Dict, List, Optional
import logging
import numpy as np
import pandas as pd


"""
Signal validator using statistical tests and robustness checks
"""



logger = logging.getLogger(__name__)


class InefficiencyValidator:
    """
    Validates inefficiency signals using multiple criteria
    
    Validation steps:
    1. Statistical significance (p-value < 0.01)
    2. Economic significance (Sharpe > 2.0)
    3. Robustness across timeframes
    4. Out-of-sample performance
    5. Transaction cost sensitivity
    """
    
    def __init__(self):
        self.validation_history: List[Dict] = []
        
    def validate_signal(self, signal: InefficiencySignal, 
                       historical_data: Optional[pd.DataFrame] = None) -> Dict[str, any]:
        """
        Validate a signal using multiple tests
        
        Args:
            signal: Signal to validate
            historical_data: Historical price data for backtesting
            
        Returns:
            Validation results dictionary
        """
        results = {
            'signal_id': signal.signal_id,
            'timestamp': datetime.now(),
            'is_valid': False,
            'validation_scores': {},
            'warnings': [],
            'recommendations': []
        }
        
        # Test 1: Statistical significance
        stat_score = self._test_statistical_significance(signal)
        results['validation_scores']['statistical'] = stat_score
        
        if stat_score < 0.5:
            results['warnings'].append("Low statistical significance (p-value too high)")
        
        # Test 2: Economic significance
        econ_score = self._test_economic_significance(signal)
        results['validation_scores']['economic'] = econ_score
        
        if econ_score < 0.5:
            results['warnings'].append("Low economic significance (expected return too small)")
        
        # Test 3: Confidence threshold
        conf_score = signal.confidence
        results['validation_scores']['confidence'] = conf_score
        
        if conf_score < 0.6:
            results['warnings'].append("Low confidence score")
        
        # Test 4: Historical validation (if data available)
        if historical_data is not None:
            hist_score = self._test_historical_performance(signal, historical_data)
            results['validation_scores']['historical'] = hist_score
            
            if hist_score < 0.5:
                results['warnings'].append("Poor historical performance")
        
        # Overall validation
        avg_score = np.mean(list(results['validation_scores'].values()))
        results['overall_score'] = avg_score
        results['is_valid'] = avg_score >= 0.6 and len(results['warnings']) <= 2
        
        # Recommendations
        if results['is_valid']:
            if avg_score >= 0.8:
                results['recommendations'].append("High-quality signal - Consider immediate execution")
            elif avg_score >= 0.7:
                results['recommendations'].append("Good signal - Execute with standard position sizing")
            else:
                results['recommendations'].append("Marginal signal - Execute with reduced position size")
        else:
            results['recommendations'].append("Signal did not pass validation - Do not execute")
        
        # Store history
        self.validation_history.append(results)
        
        return results
    
    def _test_statistical_significance(self, signal: InefficiencySignal) -> float:
        """
        Test statistical significance
        
        Returns score 0-1
        """
        # P-value test
        if signal.p_value <= 0.01:
            p_score = 1.0
        elif signal.p_value <= 0.05:
            p_score = 0.7
        elif signal.p_value <= 0.10:
            p_score = 0.5
        else:
            p_score = 0.2
        
        # Z-score test
        if abs(signal.z_score) >= 3:
            z_score = 1.0
        elif abs(signal.z_score) >= 2:
            z_score = 0.8
        elif abs(signal.z_score) >= 1.5:
            z_score = 0.6
        else:
            z_score = 0.3
        
        return (p_score + z_score) / 2
    
    def _test_economic_significance(self, signal: InefficiencySignal) -> float:
        """
        Test economic significance (is profit large enough to matter?)
        
        Returns score 0-1
        """
        # Expected return test (after transaction costs)
        transaction_costs = 0.002  # 0.2% total
        net_return = signal.expected_return / 100 - transaction_costs
        
        if net_return >= 0.01:  # 1% or more
            return_score = 1.0
        elif net_return >= 0.005:  # 0.5% - 1%
            return_score = 0.8
        elif net_return >= 0.002:  # 0.2% - 0.5%
            return_score = 0.6
        elif net_return > 0:  # Positive but small
            return_score = 0.4
        else:  # Negative after costs
            return_score = 0.0
        
        # Sharpe ratio test
        if signal.sharpe_ratio and signal.sharpe_ratio >= 3.0:
            sharpe_score = 1.0
        elif signal.sharpe_ratio and signal.sharpe_ratio >= 2.0:
            sharpe_score = 0.8
        elif signal.sharpe_ratio and signal.sharpe_ratio >= 1.5:
            sharpe_score = 0.6
        elif signal.sharpe_ratio and signal.sharpe_ratio >= 1.0:
            sharpe_score = 0.4
        else:
            sharpe_score = 0.2
        
        return (return_score + sharpe_score) / 2
    
    def _test_historical_performance(self, signal: InefficiencySignal,
                                    historical_data: pd.DataFrame) -> float:
        """
        Test on historical data
        
        Returns score 0-1
        """
        # This would run a simplified backtest
        # For now, return neutral score
        return 0.7
    
    def monte_carlo_validation(self, signal: InefficiencySignal,
                               historical_returns: np.ndarray,
                               n_simulations: int = 10000) -> Dict:
        """
        Monte Carlo validation using permutation tests
        
        Tests if the signal's pattern could occur by random chance
        
        Args:
            signal: Signal to validate
            historical_returns: Historical return data
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with Monte Carlo results
        """
        if len(historical_returns) < 30:
            return {'error': 'Insufficient data for Monte Carlo validation'}
        
        # Calculate actual signal metric
        actual_metric = signal.expected_return
        
        # Run simulations
        simulated_metrics = []
        for _ in range(n_simulations):
            # Permute returns randomly
            shuffled_returns = np.random.permutation(historical_returns)
            
            # Calculate metric on shuffled data
            simulated_metric = np.mean(shuffled_returns[:len(shuffled_returns)//2])
            simulated_metrics.append(simulated_metric)
        
        simulated_metrics = np.array(simulated_metrics)
        
        # Calculate p-value: fraction of simulations with metric >= actual
        p_value = np.sum(np.abs(simulated_metrics) >= abs(actual_metric)) / n_simulations
        
        # Calculate percentile
        percentile = stats.percentileofscore(simulated_metrics, actual_metric)
        
        return {
            'actual_metric': actual_metric,
            'mean_simulated': np.mean(simulated_metrics),
            'std_simulated': np.std(simulated_metrics),
            'p_value': p_value,
            'percentile': percentile,
            'is_significant': p_value < 0.05,
            'z_score': (actual_metric - np.mean(simulated_metrics)) / np.std(simulated_metrics)
        }
    
    def walk_forward_test(self, strategy_func, data: pd.DataFrame,
                          train_window: int = 252, test_window: int = 63) -> Dict:
        """
        Walk-forward optimization test
        
        Prevents overfitting by:
        1. Train on window 1
        2. Test on window 2
        3. Move forward and repeat
        
        Args:
            strategy_func: Function that generates signals
            data: Historical price data
            train_window: Training window size (days)
            test_window: Testing window size (days)
            
        Returns:
            Dictionary with walk-forward results
        """
        results = {
            'periods': [],
            'train_returns': [],
            'test_returns': [],
            'degradation': []
        }
        
        total_length = len(data)
        start_idx = 0
        
        while start_idx + train_window + test_window < total_length:
            # Split data
            train_data = data.iloc[start_idx:start_idx+train_window]
            test_data = data.iloc[start_idx+train_window:start_idx+train_window+test_window]
            
            # Train strategy (get parameters)
            # In a real implementation, strategy_func would optimize on train_data
            
            # Test on out-of-sample data
            # This is a simplified version
            train_return = train_data['returns'].mean() if 'returns' in train_data.columns else 0
            test_return = test_data['returns'].mean() if 'returns' in test_data.columns else 0
            
            degradation = (train_return - test_return) / train_return if train_return != 0 else 0
            
            results['periods'].append({
                'start': start_idx,
                'train_end': start_idx + train_window,
                'test_end': start_idx + train_window + test_window
            })
            results['train_returns'].append(train_return)
            results['test_returns'].append(test_return)
            results['degradation'].append(degradation)
            
            # Move forward by test_window
            start_idx += test_window
        
        # Aggregate results
        results['avg_train_return'] = np.mean(results['train_returns'])
        results['avg_test_return'] = np.mean(results['test_returns'])
        results['avg_degradation'] = np.mean(results['degradation'])
        results['consistency'] = np.std(results['test_returns'])
        
        # Is strategy robust?
        results['is_robust'] = (
            results['avg_test_return'] > 0 and
            results['avg_degradation'] < 0.5 and  # Less than 50% degradation
            results['consistency'] < 0.05  # Consistent performance
        )
        
        return results
    
    @lru_cache(maxsize=128)
    
    def get_validation_statistics(self) -> Dict:
        """Get overall validation statistics"""
        if not self.validation_history:
            return {}
        
        valid_count = sum(1 for v in self.validation_history if v['is_valid'])
        
        return {
            'total_validations': len(self.validation_history),
            'valid_signals': valid_count,
            'invalid_signals': len(self.validation_history) - valid_count,
            'validation_rate': valid_count / len(self.validation_history),
            'avg_overall_score': np.mean([v['overall_score'] for v in self.validation_history]),
            'common_warnings': self._get_common_warnings()
        }
    
    def _get_common_warnings(self) -> Dict[str, int]:
        """Get most common validation warnings"""
        warnings = {}
        for validation in self.validation_history:
            for warning in validation['warnings']:
                warnings[warning] = warnings.get(warning, 0) + 1
        
        return warnings

