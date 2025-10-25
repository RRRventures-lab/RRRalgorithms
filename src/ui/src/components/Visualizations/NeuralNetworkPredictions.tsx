import React, { useEffect, useRef } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState, AppDispatch } from '../../store/store';
import { fetchPredictions, selectSymbol } from '../../store/slices/neuralNetworkSlice';
import { createChart, IChartApi } from 'lightweight-charts';

const NeuralNetworkPredictions: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const { predictions, selectedSymbol, loading, wsConnected } = useSelector(
    (state: RootState) => state.neuralNetwork
  );
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<IChartApi | null>(null);

  useEffect(() => {
    dispatch(fetchPredictions(selectedSymbol || undefined));
  }, [dispatch, selectedSymbol]);

  useEffect(() => {
    if (!chartRef.current || predictions.length === 0) return;

    // Clear previous chart
    if (chartInstance.current) {
      chartInstance.current.remove();
    }

    // Create chart
    chartInstance.current = createChart(chartRef.current, {
      width: chartRef.current.clientWidth,
      height: 300,
      layout: {
        background: { color: '#0a0e13' },
        textColor: '#a0aec0',
      },
      grid: {
        vertLines: { color: '#1a202c' },
        horzLines: { color: '#1a202c' },
      },
      timeScale: {
        borderColor: '#2d3748',
        timeVisible: true,
      },
      rightPriceScale: {
        borderColor: '#2d3748',
      },
    });

    const latestPrediction = predictions[0];
    if (latestPrediction) {
      // This is a simplified visualization - in production you'd show historical predictions vs actuals
      const baseTime = new Date(latestPrediction.timestamp).getTime() / 1000;

      // Mock data showing prediction confidence intervals
      const series = chartInstance.current.addAreaSeries({
        topColor: 'rgba(72, 187, 120, 0.4)',
        bottomColor: 'rgba(72, 187, 120, 0.0)',
        lineColor: '#48bb78',
        lineWidth: 2,
      });

      const confidenceData = [
        { time: baseTime, value: latestPrediction.confidence_interval.lower_1h },
        { time: baseTime + 3600, value: latestPrediction.prediction.price_next_1h },
        { time: baseTime + 14400, value: latestPrediction.confidence_interval.upper_4h },
      ];

      series.setData(confidenceData as any);
      chartInstance.current.timeScale().fitContent();
    }

    // Handle resize
    const handleResize = () => {
      if (chartRef.current && chartInstance.current) {
        chartInstance.current.applyOptions({
          width: chartRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartInstance.current) {
        chartInstance.current.remove();
      }
    };
  }, [predictions]);

  const symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD'];
  const latestPrediction = predictions.find(p => !selectedSymbol || p.symbol === selectedSymbol);

  return (
    <div className="h-full flex flex-col">
      <div className="terminal-panel-header flex justify-between items-center">
        <div className="flex items-center space-x-3">
          <span className="text-terminal-green text-terminal-sm font-bold">
            NEURAL NETWORK PREDICTIONS
          </span>
          <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-bloomberg-green' : 'bg-bloomberg-red'} animate-pulse`} />
        </div>
        <select
          value={selectedSymbol || 'all'}
          onChange={(e) => dispatch(selectSymbol(e.target.value === 'all' ? null : e.target.value))}
          className="bg-terminal-border border-terminal-border text-terminal-text px-2 py-1 text-terminal-xs rounded"
        >
          <option value="all">ALL SYMBOLS</option>
          {symbols.map(symbol => (
            <option key={symbol} value={symbol}>{symbol}</option>
          ))}
        </select>
      </div>

      <div className="terminal-content flex-1 overflow-y-auto terminal-scrollbar">
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-terminal-accent">Loading predictions...</div>
          </div>
        ) : latestPrediction ? (
          <div className="p-4 space-y-4">
            {/* Current Prediction Summary */}
            <div className="grid grid-cols-3 gap-4">
              <div className="terminal-panel p-3">
                <div className="text-terminal-accent text-terminal-xs mb-1">1H PREDICTION</div>
                <div className={`text-terminal-lg font-bold ${
                  latestPrediction.prediction.direction === 'up' ? 'text-bloomberg-green' :
                  latestPrediction.prediction.direction === 'down' ? 'text-bloomberg-red' :
                  'text-terminal-text'
                }`}>
                  ${latestPrediction.prediction.price_next_1h.toFixed(2)}
                </div>
                <div className="text-terminal-xs text-terminal-accent">
                  ±${(latestPrediction.confidence_interval.upper_1h - latestPrediction.confidence_interval.lower_1h).toFixed(2)}
                </div>
              </div>

              <div className="terminal-panel p-3">
                <div className="text-terminal-accent text-terminal-xs mb-1">4H PREDICTION</div>
                <div className={`text-terminal-lg font-bold ${
                  latestPrediction.prediction.direction === 'up' ? 'text-bloomberg-green' :
                  latestPrediction.prediction.direction === 'down' ? 'text-bloomberg-red' :
                  'text-terminal-text'
                }`}>
                  ${latestPrediction.prediction.price_next_4h.toFixed(2)}
                </div>
                <div className="text-terminal-xs text-terminal-accent">
                  ±${(latestPrediction.confidence_interval.upper_4h - latestPrediction.confidence_interval.lower_4h).toFixed(2)}
                </div>
              </div>

              <div className="terminal-panel p-3">
                <div className="text-terminal-accent text-terminal-xs mb-1">24H PREDICTION</div>
                <div className={`text-terminal-lg font-bold ${
                  latestPrediction.prediction.direction === 'up' ? 'text-bloomberg-green' :
                  latestPrediction.prediction.direction === 'down' ? 'text-bloomberg-red' :
                  'text-terminal-text'
                }`}>
                  ${latestPrediction.prediction.price_next_24h.toFixed(2)}
                </div>
                <div className="text-terminal-xs text-terminal-accent">
                  ±${(latestPrediction.confidence_interval.upper_24h - latestPrediction.confidence_interval.lower_24h).toFixed(2)}
                </div>
              </div>
            </div>

            {/* Direction & Confidence */}
            <div className="grid grid-cols-2 gap-4">
              <div className="terminal-panel p-3">
                <div className="text-terminal-accent text-terminal-xs mb-2">DIRECTION</div>
                <div className="flex items-center space-x-2">
                  <span className={`text-2xl ${
                    latestPrediction.prediction.direction === 'up' ? 'text-bloomberg-green' :
                    latestPrediction.prediction.direction === 'down' ? 'text-bloomberg-red' :
                    'text-terminal-accent'
                  }`}>
                    {latestPrediction.prediction.direction === 'up' ? '↑' :
                     latestPrediction.prediction.direction === 'down' ? '↓' : '→'}
                  </span>
                  <span className="text-terminal-text font-bold">
                    {latestPrediction.prediction.direction.toUpperCase()}
                  </span>
                </div>
              </div>

              <div className="terminal-panel p-3">
                <div className="text-terminal-accent text-terminal-xs mb-2">CONFIDENCE</div>
                <div className="flex items-center space-x-2">
                  <div className="flex-1 bg-terminal-border rounded-full h-4">
                    <div
                      className={`h-4 rounded-full ${
                        latestPrediction.prediction.confidence > 0.7 ? 'bg-bloomberg-green' :
                        latestPrediction.prediction.confidence > 0.5 ? 'bg-bloomberg-amber' :
                        'bg-bloomberg-red'
                      }`}
                      style={{ width: `${latestPrediction.prediction.confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-terminal-text font-bold">
                    {(latestPrediction.prediction.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Feature Analysis */}
            <div className="terminal-panel p-4">
              <div className="text-terminal-accent text-terminal-sm font-bold mb-3">
                FEATURE ANALYSIS
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <div className="flex justify-between text-terminal-xs mb-1">
                    <span className="text-terminal-accent">Momentum</span>
                    <span className="text-terminal-text">{latestPrediction.features.momentum.toFixed(2)}</span>
                  </div>
                  <div className="bg-terminal-border rounded-full h-2">
                    <div
                      className="bg-bloomberg-green h-2 rounded-full"
                      style={{ width: `${Math.abs(latestPrediction.features.momentum * 100)}%` }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-terminal-xs mb-1">
                    <span className="text-terminal-accent">Volatility</span>
                    <span className="text-terminal-text">{latestPrediction.features.volatility.toFixed(2)}</span>
                  </div>
                  <div className="bg-terminal-border rounded-full h-2">
                    <div
                      className="bg-bloomberg-amber h-2 rounded-full"
                      style={{ width: `${latestPrediction.features.volatility * 100}%` }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-terminal-xs mb-1">
                    <span className="text-terminal-accent">Volume Profile</span>
                    <span className="text-terminal-text">{latestPrediction.features.volume_profile.toFixed(2)}</span>
                  </div>
                  <div className="bg-terminal-border rounded-full h-2">
                    <div
                      className="bg-bloomberg-blue h-2 rounded-full"
                      style={{ width: `${latestPrediction.features.volume_profile * 100}%` }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-terminal-xs mb-1">
                    <span className="text-terminal-accent">Market Regime</span>
                    <span className="text-terminal-text font-bold">
                      {latestPrediction.features.market_regime}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Confidence Interval Chart */}
            <div className="terminal-panel p-4">
              <div className="text-terminal-accent text-terminal-sm font-bold mb-3">
                PREDICTION CONFIDENCE INTERVALS
              </div>
              <div ref={chartRef} className="w-full" />
            </div>

            {/* Metadata */}
            <div className="text-terminal-accent text-terminal-xs">
              Model: {latestPrediction.model} | Updated: {new Date(latestPrediction.timestamp).toLocaleString()}
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-terminal-accent">
              No predictions available for {selectedSymbol || 'any symbol'}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default NeuralNetworkPredictions;
