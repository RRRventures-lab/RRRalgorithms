import React, { useEffect, useRef } from 'react';
import { createChart, IChartApi, ISeriesApi } from 'lightweight-charts';
import { useSelector } from 'react-redux';
import { RootState } from '../../store/store';
import TechnicalIndicators from './TechnicalIndicators';

const Charts: React.FC = () => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const { prices } = useSelector((state: RootState) => state.marketData);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: chartContainerRef.current.clientHeight,
      layout: {
        background: { color: '#0A0A0A' },
        textColor: '#00FF41',
      },
      grid: {
        vertLines: {
          color: '#333333',
        },
        horzLines: {
          color: '#333333',
        },
      },
      crosshair: {
        mode: 0,
      },
      rightPriceScale: {
        borderColor: '#333333',
        textColor: '#00FF41',
      },
      timeScale: {
        borderColor: '#333333',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    // Create candlestick series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#00FF41',
      downColor: '#FF0000',
      borderDownColor: '#FF0000',
      borderUpColor: '#00FF41',
      wickDownColor: '#FF0000',
      wickUpColor: '#00FF41',
    });

    // Generate sample data
    const generateSampleData = () => {
      const data = [];
      const now = Date.now();
      let price = 45000;
      
      for (let i = 100; i >= 0; i--) {
        const time = now - (i * 60000); // 1 minute intervals
        const open = price;
        const change = (Math.random() - 0.5) * 100;
        const close = open + change;
        const high = Math.max(open, close) + Math.random() * 50;
        const low = Math.min(open, close) - Math.random() * 50;
        
        data.push({
          time: Math.floor(time / 1000) as any,
          open,
          high,
          low,
          close,
        });
        
        price = close;
      }
      
      return data;
    };

    // Set initial data
    candlestickSeries.setData(generateSampleData());

    chartRef.current = chart;
    seriesRef.current = candlestickSeries;

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chart) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);

  // Update chart with real-time data
  useEffect(() => {
    if (seriesRef.current && prices['BTC-USD']) {
      const btcPrice = prices['BTC-USD'].price;
      const now = Math.floor(Date.now() / 1000);
      
      // Add new candlestick data point
      seriesRef.current.update({
        time: now as any,
        open: btcPrice,
        high: btcPrice + 10,
        low: btcPrice - 10,
        close: btcPrice,
      });
    }
  }, [prices]);

  return (
    <div className="h-full flex flex-col">
      <div className="terminal-header">
        <span className="text-terminal-accent font-bold">CHARTS</span>
        <div className="flex space-x-2">
          <button className="bloomberg-button text-terminal-xs">1M</button>
          <button className="bloomberg-button text-terminal-xs">5M</button>
          <button className="bloomberg-button text-terminal-xs">1H</button>
          <button className="bloomberg-button text-terminal-xs">1D</button>
        </div>
      </div>
      <div className="terminal-content flex-1">
        <div ref={chartContainerRef} className="w-full h-full" />
        <TechnicalIndicators chart={chartRef.current} candlestickSeries={seriesRef.current} />
      </div>
    </div>
  );
};

export default Charts;
