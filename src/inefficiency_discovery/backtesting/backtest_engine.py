from ..base import BacktestResult, InefficiencySignal
from datetime import datetime, timedelta
from scipy import stats
from typing import Dict, List, Optional
import logging
import numpy as np
import pandas as pd

"""
Backtest engine for inefficiency strategies
"""



logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Fast vectorized backtesting engine for inefficiency strategies
    """
    
    def __init__(self, initial_capital: float = 100000, 
                 transaction_cost: float = 0.001):
        """
        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost per trade (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
    def backtest_strategy(self, signals: List[InefficiencySignal],
                         price_data: pd.DataFrame,
                         strategy_name: str = "Inefficiency Strategy") -> BacktestResult:
        """
        Backtest a list of signals
        
        Args:
            signals: List of trading signals
            price_data: DataFrame with columns: timestamp, symbol, price
            strategy_name: Name of strategy
            
        Returns:
            BacktestResult with performance metrics
        """
        if not signals:
            logger.warning("No signals to backtest")
            return None
        
        # Initialize tracking
        portfolio_value = [self.initial_capital]
        timestamps = [price_data['timestamp'].min()]
        trades = []
        
        capital = self.initial_capital
        positions = {}  # symbol -> (size, entry_price)
        
        # Sort signals by timestamp
        signals_sorted = sorted(signals, key=lambda s: s.timestamp)
        
        # Simulate trading
        for signal in signals_sorted:
            symbol = signal.symbols[0]  # For simplicity, use first symbol
            
            # Get price at signal time
            signal_data = price_data[
                (price_data['symbol'] == symbol) &
                (price_data['timestamp'] >= signal.timestamp)
            ].head(1)
            
            if signal_data.empty:
                continue
            
            entry_price = signal_data.iloc[0]['price']
            entry_time = signal_data.iloc[0]['timestamp']
            
            # Calculate position size (risk 2% of capital per trade)
            risk_per_trade = capital * 0.02
            position_size = risk_per_trade / entry_price if entry_price > 0 else 0
            
            if position_size == 0:
                continue
            
            # Enter trade
            trade_cost = position_size * entry_price * self.transaction_cost
            capital -= trade_cost
            
            positions[symbol] = (position_size, entry_price, entry_time, signal.direction)
            
            # Exit after expected duration or at stop loss/target
            if signal.expected_duration:
                exit_time = entry_time + timedelta(seconds=signal.expected_duration)
            else:
                exit_time = entry_time + timedelta(hours=24)  # Default 24 hours
            
            # Get exit price
            exit_data = price_data[
                (price_data['symbol'] == symbol) &
                (price_data['timestamp'] >= exit_time)
            ].head(1)
            
            if not exit_data.empty:
                exit_price = exit_data.iloc[0]['price']
                exit_time_actual = exit_data.iloc[0]['timestamp']
                
                # Calculate P&L
                if signal.direction == 'long':
                    pnl = position_size * (exit_price - entry_price)
                elif signal.direction == 'short':
                    pnl = position_size * (entry_price - exit_price)
                else:  # pair trade
                    pnl = position_size * abs(exit_price - entry_price) * 0.5
                
                # Subtract exit costs
                pnl -= position_size * exit_price * self.transaction_cost
                
                capital += pnl + (position_size * entry_price)  # Return capital
                
                # Record trade
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time_actual,
                    'symbol': symbol,
                    'direction': signal.direction,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'size': position_size,
                    'pnl': pnl,
                    'return': pnl / (position_size * entry_price) if entry_price > 0 else 0
                })
                
                # Update portfolio value
                portfolio_value.append(capital)
                timestamps.append(exit_time_actual)
                
                # Remove position
                del positions[symbol]
        
        # Calculate metrics
        if len(trades) < 2:
            logger.warning("Insufficient trades for backtesting")
            return None
        
        trades_df = pd.DataFrame(trades)
        
        # Performance metrics
        total_return = (capital - self.initial_capital) / self.initial_capital
        
        duration = (timestamps[-1] - timestamps[0]).days
        if duration > 0:
            annualized_return = (1 + total_return) ** (365 / duration) - 1
        else:
            annualized_return = 0
        
        # Trade statistics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        wins = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if t['pnl'] < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Risk metrics
        returns = trades_df['return'].values
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        if volatility > 0:
            sharpe_ratio = annualized_return / volatility
        else:
            sharpe_ratio = 0
        
        # Sortino ratio (only downside volatility)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        if downside_std > 0:
            sortino_ratio = annualized_return / downside_std
        else:
            sortino_ratio = 0
        
        # Max drawdown
        portfolio_series = pd.Series(portfolio_value)
        running_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Value at Risk (95%)
        var_95 = np.percentile(returns, 5)
        
        # Conditional VaR (95%)
        cvar_95 = np.mean(returns[returns <= var_95])
        
        # Statistical tests
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        
        # Calmar ratio
        if max_drawdown < 0:
            calmar_ratio = annualized_return / abs(max_drawdown)
        else:
            calmar_ratio = 0
        
        # Create result
        result = BacktestResult(
            strategy_name=strategy_name,
            start_date=timestamps[0],
            end_date=timestamps[-1],
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            volatility=volatility,
            var_95=var_95,
            cvar_95=cvar_95,
            t_statistic=t_stat,
            p_value=p_value,
            calmar_ratio=calmar_ratio
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Backtest Results: {strategy_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Annualized Return: {annualized_return:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"Win Rate: {win_rate:.1%}")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        logger.info(f"P-value: {p_value:.4f}")
        logger.info(f"Viable: {result.is_viable()}")
        
        return result
    
    def optimize_parameters(self, signals: List[InefficiencySignal],
                           price_data: pd.DataFrame,
                           param_ranges: Dict[str, List]) -> Dict:
        """
        Optimize strategy parameters using grid search
        
        Args:
            signals: Trading signals
            price_data: Price data
            param_ranges: Dictionary of parameters to optimize
            
        Returns:
            Best parameters and results
        """
        best_sharpe = -np.inf
        best_params = None
        best_result = None
        
        # This is a placeholder for parameter optimization
        # In a real implementation, would test different parameter combinations
        
        logger.info("Parameter optimization not yet implemented")
        
        return {
            'best_params': best_params,
            'best_result': best_result,
            'best_sharpe': best_sharpe
        }

