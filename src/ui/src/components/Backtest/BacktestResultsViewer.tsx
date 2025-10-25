import React, { useEffect, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState, AppDispatch } from '../../store/store';
import { fetchBacktestDetail, selectBacktest } from '../../store/slices/backtestSlice';
import EquityCurveChart from './EquityCurveChart';
import MonteCarloDistribution from './MonteCarloDistribution';
import ParameterSensitivity from './ParameterSensitivity';
import TradeList from './TradeList';
import { backtestService } from '../../services/backtestService';

const BacktestResultsViewer: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const { backtests, selectedBacktest, loading, comparisonBacktests } = useSelector(
    (state: RootState) => state.backtest
  );
  const [activeTab, setActiveTab] = useState<'overview' | 'equity' | 'monte_carlo' | 'sensitivity' | 'trades'>('overview');
  const [exportingPDF, setExportingPDF] = useState(false);

  const handleSelectBacktest = (backtestId: string) => {
    dispatch(fetchBacktestDetail(backtestId));
  };

  const handleExportPDF = async () => {
    if (!selectedBacktest) return;

    setExportingPDF(true);
    try {
      const blob = await backtestService.exportBacktestPDF(selectedBacktest.id);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `backtest-${selectedBacktest.name}-${Date.now()}.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Failed to export PDF:', error);
    } finally {
      setExportingPDF(false);
    }
  };

  const formatNumber = (num: number, decimals: number = 2): string => {
    return num.toFixed(decimals);
  };

  const formatPercent = (num: number): string => {
    return `${num >= 0 ? '+' : ''}${num.toFixed(2)}%`;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'text-bloomberg-green';
      case 'running':
        return 'text-bloomberg-amber';
      case 'failed':
        return 'text-bloomberg-red';
      default:
        return 'text-terminal-text';
    }
  };

  if (!selectedBacktest) {
    return (
      <div className="h-full flex flex-col">
        <div className="terminal-panel-header">
          <span className="text-terminal-green text-terminal-sm font-bold">BACKTEST RESULTS</span>
        </div>
        <div className="terminal-content flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="text-terminal-accent mb-4">Select a backtest to view results</div>
            <div className="grid grid-cols-1 gap-2 max-w-md">
              {backtests.map((backtest) => (
                <button
                  key={backtest.id}
                  onClick={() => handleSelectBacktest(backtest.id)}
                  className="bloomberg-button text-left p-3"
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="text-terminal-sm font-bold">{backtest.name}</div>
                      <div className="text-terminal-xs text-terminal-accent mt-1">
                        {new Date(backtest.created_at).toLocaleDateString()}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`text-terminal-sm ${backtest.performance.total_return >= 0 ? 'text-bloomberg-green' : 'text-bloomberg-red'}`}>
                        {formatPercent(backtest.performance.total_return_percent)}
                      </div>
                      <div className={`text-terminal-xs ${getStatusColor(backtest.status)}`}>
                        {backtest.status.toUpperCase()}
                      </div>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="terminal-panel-header flex justify-between items-center">
        <div className="flex items-center space-x-3">
          <button
            onClick={() => dispatch(selectBacktest(null))}
            className="text-terminal-accent hover:text-terminal-text"
          >
            ← Back
          </button>
          <span className="text-terminal-green text-terminal-sm font-bold">
            {selectedBacktest.name}
          </span>
          <span className={`text-terminal-xs ${getStatusColor(selectedBacktest.status)}`}>
            {selectedBacktest.status.toUpperCase()}
          </span>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={handleExportPDF}
            disabled={exportingPDF}
            className="bloomberg-button text-terminal-xs px-2 py-1 disabled:opacity-50"
          >
            {exportingPDF ? 'EXPORTING...' : 'EXPORT PDF'}
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-terminal-border">
        {['overview', 'equity', 'monte_carlo', 'sensitivity', 'trades'].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab as any)}
            className={`px-4 py-2 text-terminal-xs font-bold ${
              activeTab === tab
                ? 'text-bloomberg-green border-b-2 border-bloomberg-green'
                : 'text-terminal-accent hover:text-terminal-text'
            }`}
          >
            {tab.replace('_', ' ').toUpperCase()}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="terminal-content flex-1 overflow-y-auto terminal-scrollbar">
        {activeTab === 'overview' && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4">
            {/* Performance Metrics Grid */}
            <div className="terminal-panel p-3">
              <div className="text-terminal-accent text-terminal-xs mb-1">TOTAL RETURN</div>
              <div className={`text-terminal-lg font-bold ${selectedBacktest.performance.total_return >= 0 ? 'text-bloomberg-green' : 'text-bloomberg-red'}`}>
                {formatPercent(selectedBacktest.performance.total_return_percent)}
              </div>
              <div className="text-terminal-xs text-terminal-text mt-1">
                ${formatNumber(selectedBacktest.performance.final_equity - selectedBacktest.performance.initial_capital)}
              </div>
            </div>

            <div className="terminal-panel p-3">
              <div className="text-terminal-accent text-terminal-xs mb-1">SHARPE RATIO</div>
              <div className="text-terminal-lg font-bold text-terminal-text">
                {formatNumber(selectedBacktest.performance.sharpe_ratio)}
              </div>
              <div className="text-terminal-xs text-terminal-accent mt-1">Risk-Adjusted Return</div>
            </div>

            <div className="terminal-panel p-3">
              <div className="text-terminal-accent text-terminal-xs mb-1">MAX DRAWDOWN</div>
              <div className="text-terminal-lg font-bold text-bloomberg-red">
                {formatPercent(selectedBacktest.performance.max_drawdown_percent)}
              </div>
              <div className="text-terminal-xs text-terminal-text mt-1">
                ${formatNumber(Math.abs(selectedBacktest.performance.max_drawdown))}
              </div>
            </div>

            <div className="terminal-panel p-3">
              <div className="text-terminal-accent text-terminal-xs mb-1">WIN RATE</div>
              <div className="text-terminal-lg font-bold text-bloomberg-green">
                {formatNumber(selectedBacktest.performance.win_rate)}%
              </div>
              <div className="text-terminal-xs text-terminal-accent mt-1">
                {selectedBacktest.performance.winning_trades}/{selectedBacktest.performance.total_trades}
              </div>
            </div>

            <div className="terminal-panel p-3">
              <div className="text-terminal-accent text-terminal-xs mb-1">PROFIT FACTOR</div>
              <div className="text-terminal-lg font-bold text-terminal-text">
                {formatNumber(selectedBacktest.performance.profit_factor)}
              </div>
              <div className="text-terminal-xs text-terminal-accent mt-1">Gross Profit/Loss</div>
            </div>

            <div className="terminal-panel p-3">
              <div className="text-terminal-accent text-terminal-xs mb-1">SORTINO RATIO</div>
              <div className="text-terminal-lg font-bold text-terminal-text">
                {formatNumber(selectedBacktest.performance.sortino_ratio)}
              </div>
              <div className="text-terminal-xs text-terminal-accent mt-1">Downside Risk</div>
            </div>

            <div className="terminal-panel p-3">
              <div className="text-terminal-accent text-terminal-xs mb-1">CALMAR RATIO</div>
              <div className="text-terminal-lg font-bold text-terminal-text">
                {formatNumber(selectedBacktest.performance.calmar_ratio)}
              </div>
              <div className="text-terminal-xs text-terminal-accent mt-1">Return/Drawdown</div>
            </div>

            <div className="terminal-panel p-3">
              <div className="text-terminal-accent text-terminal-xs mb-1">TOTAL TRADES</div>
              <div className="text-terminal-lg font-bold text-terminal-text">
                {selectedBacktest.performance.total_trades}
              </div>
              <div className="text-terminal-xs text-bloomberg-green mt-1">
                {selectedBacktest.performance.winning_trades} wins
              </div>
            </div>

            {/* Period Information */}
            <div className="terminal-panel p-3 col-span-2">
              <div className="text-terminal-accent text-terminal-xs mb-2">BACKTEST PERIOD</div>
              <div className="text-terminal-sm text-terminal-text">
                {new Date(selectedBacktest.period.start).toLocaleDateString()} - {new Date(selectedBacktest.period.end).toLocaleDateString()}
              </div>
              <div className="text-terminal-xs text-terminal-accent mt-1">
                {selectedBacktest.period.days} days
              </div>
            </div>

            <div className="terminal-panel p-3 col-span-2">
              <div className="text-terminal-accent text-terminal-xs mb-2">CAPITAL</div>
              <div className="flex justify-between text-terminal-sm">
                <div>
                  <div className="text-terminal-accent">Initial:</div>
                  <div className="text-terminal-text font-bold">
                    ${formatNumber(selectedBacktest.performance.initial_capital)}
                  </div>
                </div>
                <div>
                  <div className="text-terminal-accent">Final:</div>
                  <div className={`font-bold ${selectedBacktest.performance.final_equity >= selectedBacktest.performance.initial_capital ? 'text-bloomberg-green' : 'text-bloomberg-red'}`}>
                    ${formatNumber(selectedBacktest.performance.final_equity)}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'equity' && selectedBacktest.equity_curve && (
          <EquityCurveChart
            equityCurve={selectedBacktest.equity_curve}
            drawdownCurve={selectedBacktest.drawdown_curve || []}
          />
        )}

        {activeTab === 'monte_carlo' && selectedBacktest.monte_carlo && (
          <MonteCarloDistribution monteCarloData={selectedBacktest.monte_carlo} />
        )}

        {activeTab === 'sensitivity' && selectedBacktest.parameter_sensitivity && (
          <ParameterSensitivity data={selectedBacktest.parameter_sensitivity} />
        )}

        {activeTab === 'trades' && selectedBacktest.trades && (
          <TradeList trades={selectedBacktest.trades} />
        )}
      </div>
    </div>
  );
};

export default BacktestResultsViewer;
