import React, { useEffect, useRef } from 'react';
import { createChart, IChartApi, ISeriesApi, LineData } from 'lightweight-charts';

interface EquityCurveChartProps {
  equityCurve: Array<{ timestamp: string; equity: number }>;
  drawdownCurve: Array<{ timestamp: string; drawdown: number }>;
}

const EquityCurveChart: React.FC<EquityCurveChartProps> = ({ equityCurve, drawdownCurve }) => {
  const equityChartRef = useRef<HTMLDivElement>(null);
  const drawdownChartRef = useRef<HTMLDivElement>(null);
  const equityChartInstance = useRef<IChartApi | null>(null);
  const drawdownChartInstance = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!equityChartRef.current || !drawdownChartRef.current) return;

    // Create equity curve chart
    equityChartInstance.current = createChart(equityChartRef.current, {
      width: equityChartRef.current.clientWidth,
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

    const equitySeries = equityChartInstance.current.addLineSeries({
      color: '#48bb78',
      lineWidth: 2,
      title: 'Equity',
    });

    // Transform data to LightweightCharts format
    const equityData: LineData[] = equityCurve.map(point => ({
      time: new Date(point.timestamp).getTime() / 1000,
      value: point.equity,
    }));

    equitySeries.setData(equityData);
    equityChartInstance.current.timeScale().fitContent();

    // Create drawdown chart
    drawdownChartInstance.current = createChart(drawdownChartRef.current, {
      width: drawdownChartRef.current.clientWidth,
      height: 150,
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

    const drawdownSeries = drawdownChartInstance.current.addAreaSeries({
      topColor: 'rgba(239, 68, 68, 0.4)',
      bottomColor: 'rgba(239, 68, 68, 0.0)',
      lineColor: '#ef4444',
      lineWidth: 2,
      title: 'Drawdown %',
    });

    const drawdownData: LineData[] = drawdownCurve.map(point => ({
      time: new Date(point.timestamp).getTime() / 1000,
      value: point.drawdown,
    }));

    drawdownSeries.setData(drawdownData);
    drawdownChartInstance.current.timeScale().fitContent();

    // Handle resize
    const handleResize = () => {
      if (equityChartRef.current && equityChartInstance.current) {
        equityChartInstance.current.applyOptions({
          width: equityChartRef.current.clientWidth,
        });
      }
      if (drawdownChartRef.current && drawdownChartInstance.current) {
        drawdownChartInstance.current.applyOptions({
          width: drawdownChartRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (equityChartInstance.current) {
        equityChartInstance.current.remove();
      }
      if (drawdownChartInstance.current) {
        drawdownChartInstance.current.remove();
      }
    };
  }, [equityCurve, drawdownCurve]);

  return (
    <div className="p-4 space-y-4">
      <div>
        <div className="text-terminal-accent text-terminal-sm font-bold mb-2">EQUITY CURVE</div>
        <div ref={equityChartRef} className="w-full" />
      </div>
      <div>
        <div className="text-terminal-accent text-terminal-sm font-bold mb-2">DRAWDOWN</div>
        <div ref={drawdownChartRef} className="w-full" />
      </div>
    </div>
  );
};

export default EquityCurveChart;
