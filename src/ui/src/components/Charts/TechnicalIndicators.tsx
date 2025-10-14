import React, { useEffect, useRef } from 'react';
import { IChartApi, ISeriesApi } from 'lightweight-charts';

interface TechnicalIndicatorsProps {
  chart: IChartApi | null;
  candlestickSeries: ISeriesApi<'Candlestick'> | null;
}

const TechnicalIndicators: React.FC<TechnicalIndicatorsProps> = ({ chart, candlestickSeries }) => {
  const indicatorsRef = useRef<{
    sma20?: ISeriesApi<'Line'>;
    sma50?: ISeriesApi<'Line'>;
    volume?: ISeriesApi<'Histogram'>;
    rsi?: ISeriesApi<'Line'>;
  }>({});

  useEffect(() => {
    if (!chart || !candlestickSeries) return;

    // Simple Moving Average 20
    const sma20 = chart.addLineSeries({
      color: '#FF8800',
      lineWidth: 2,
      title: 'SMA 20',
    });

    // Simple Moving Average 50
    const sma50 = chart.addLineSeries({
      color: '#0084FF',
      lineWidth: 2,
      title: 'SMA 50',
    });

    // Volume histogram
    const volume = chart.addHistogramSeries({
      color: '#00FF41',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: 'volume',
      title: 'Volume',
    });

    // RSI (simplified)
    const rsi = chart.addLineSeries({
      color: '#FF0000',
      lineWidth: 1,
      priceScaleId: 'rsi',
      title: 'RSI',
    });

    // Configure volume price scale
    chart.priceScale('volume').applyOptions({
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });

    // Configure RSI price scale
    chart.priceScale('rsi').applyOptions({
      scaleMargins: {
        top: 0.1,
        bottom: 0.1,
      },
    });

    indicatorsRef.current = { sma20, sma50, volume, rsi };

    // Generate sample indicator data
    const generateIndicatorData = () => {
      const data = [];
      const now = Date.now();
      let price = 45000;
      let volume = 1000;
      let rsiValue = 50;

      for (let i = 100; i >= 0; i--) {
        const time = now - (i * 60000);
        const change = (Math.random() - 0.5) * 100;
        price += change;
        volume = Math.floor(Math.random() * 2000) + 500;
        rsiValue = Math.max(0, Math.min(100, rsiValue + (Math.random() - 0.5) * 10));

        data.push({
          time: Math.floor(time / 1000) as any,
          price,
          volume,
          rsi: rsiValue,
        });
      }

      return data;
    };

    const indicatorData = generateIndicatorData();

    // Calculate and set SMA data
    const sma20Data = indicatorData.map((item, index) => {
      if (index < 19) return null;
      const sum = indicatorData.slice(index - 19, index + 1).reduce((acc, d) => acc + d.price, 0);
      return {
        time: item.time,
        value: sum / 20,
      };
    }).filter(Boolean) as any[];

    const sma50Data = indicatorData.map((item, index) => {
      if (index < 49) return null;
      const sum = indicatorData.slice(index - 49, index + 1).reduce((acc, d) => acc + d.price, 0);
      return {
        time: item.time,
        value: sum / 50,
      };
    }).filter(Boolean) as any[];

    // Set indicator data
    sma20.setData(sma20Data);
    sma50.setData(sma50Data);
    volume.setData(indicatorData.map(item => ({
      time: item.time,
      value: item.volume,
      color: item.volume > 1000 ? '#00FF41' : '#FF0000',
    })) as any);
    rsi.setData(indicatorData.map(item => ({
      time: item.time,
      value: item.rsi,
    })) as any);

    return () => {
      chart.removeSeries(sma20);
      chart.removeSeries(sma50);
      chart.removeSeries(volume);
      chart.removeSeries(rsi);
    };
  }, [chart, candlestickSeries]);

  return null; // This component only manages indicators, no UI
};

export default TechnicalIndicators;
