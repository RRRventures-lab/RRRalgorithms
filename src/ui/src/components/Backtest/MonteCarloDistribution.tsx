import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface MonteCarloData {
  simulations: number;
  results: Array<{
    simulation_id: number;
    final_equity: number;
    total_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
  }>;
  statistics: {
    mean_return: number;
    median_return: number;
    std_deviation: number;
    percentile_5: number;
    percentile_95: number;
    probability_of_profit: number;
    value_at_risk_95: number;
  };
}

interface MonteCarloDistributionProps {
  monteCarloData: MonteCarloData;
}

const MonteCarloDistribution: React.FC<MonteCarloDistributionProps> = ({ monteCarloData }) => {
  const chartRef = useRef<SVGSVGElement>(null);
  const scatterRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!chartRef.current || !monteCarloData.results.length) return;

    // Clear previous chart
    d3.select(chartRef.current).selectAll('*').remove();

    const margin = { top: 20, right: 20, bottom: 40, left: 60 };
    const width = 600 - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;

    const svg = d3.select(chartRef.current)
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Create histogram
    const returns = monteCarloData.results.map(r => r.total_return);
    const bins = d3.bin()
      .domain([d3.min(returns)!, d3.max(returns)!])
      .thresholds(50)(returns);

    // Scales
    const x = d3.scaleLinear()
      .domain([d3.min(returns)!, d3.max(returns)!])
      .range([0, width]);

    const y = d3.scaleLinear()
      .domain([0, d3.max(bins, d => d.length)!])
      .range([height, 0]);

    // Draw histogram bars
    svg.selectAll('rect')
      .data(bins)
      .enter()
      .append('rect')
      .attr('x', d => x(d.x0!))
      .attr('width', d => Math.max(0, x(d.x1!) - x(d.x0!) - 1))
      .attr('y', d => y(d.length))
      .attr('height', d => height - y(d.length))
      .attr('fill', d => {
        const midpoint = (d.x0! + d.x1!) / 2;
        return midpoint >= 0 ? '#48bb78' : '#ef4444';
      })
      .attr('opacity', 0.7);

    // Add vertical lines for statistics
    const stats = [
      { value: monteCarloData.statistics.percentile_5, label: '5th %ile', color: '#ef4444' },
      { value: monteCarloData.statistics.median_return, label: 'Median', color: '#f59e0b' },
      { value: monteCarloData.statistics.percentile_95, label: '95th %ile', color: '#48bb78' },
    ];

    stats.forEach(stat => {
      svg.append('line')
        .attr('x1', x(stat.value))
        .attr('x2', x(stat.value))
        .attr('y1', 0)
        .attr('y2', height)
        .attr('stroke', stat.color)
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5,5');

      svg.append('text')
        .attr('x', x(stat.value))
        .attr('y', -5)
        .attr('text-anchor', 'middle')
        .attr('fill', stat.color)
        .attr('font-size', '10px')
        .text(stat.label);
    });

    // Add axes
    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x).ticks(10).tickFormat(d => `${d}%`))
      .attr('color', '#a0aec0');

    svg.append('g')
      .call(d3.axisLeft(y))
      .attr('color', '#a0aec0');

    // Add labels
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height + 35)
      .attr('text-anchor', 'middle')
      .attr('fill', '#a0aec0')
      .attr('font-size', '12px')
      .text('Total Return (%)');

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', -45)
      .attr('text-anchor', 'middle')
      .attr('fill', '#a0aec0')
      .attr('font-size', '12px')
      .text('Frequency');

  }, [monteCarloData]);

  useEffect(() => {
    if (!scatterRef.current || !monteCarloData.results.length) return;

    // Clear previous chart
    d3.select(scatterRef.current).selectAll('*').remove();

    const margin = { top: 20, right: 20, bottom: 40, left: 60 };
    const width = 600 - margin.left - margin.right;
    const height = 250 - margin.top - margin.bottom;

    const svg = d3.select(scatterRef.current)
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const x = d3.scaleLinear()
      .domain([0, d3.max(monteCarloData.results, d => Math.abs(d.max_drawdown))!])
      .range([0, width]);

    const y = d3.scaleLinear()
      .domain([d3.min(monteCarloData.results, d => d.total_return)!, d3.max(monteCarloData.results, d => d.total_return)!])
      .range([height, 0]);

    // Draw scatter points
    svg.selectAll('circle')
      .data(monteCarloData.results)
      .enter()
      .append('circle')
      .attr('cx', d => x(Math.abs(d.max_drawdown)))
      .attr('cy', d => y(d.total_return))
      .attr('r', 3)
      .attr('fill', d => d.total_return >= 0 ? '#48bb78' : '#ef4444')
      .attr('opacity', 0.5);

    // Add axes
    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x).tickFormat(d => `${d}%`))
      .attr('color', '#a0aec0');

    svg.append('g')
      .call(d3.axisLeft(y).tickFormat(d => `${d}%`))
      .attr('color', '#a0aec0');

    // Add labels
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height + 35)
      .attr('text-anchor', 'middle')
      .attr('fill', '#a0aec0')
      .attr('font-size', '12px')
      .text('Max Drawdown (%)');

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', -45)
      .attr('text-anchor', 'middle')
      .attr('fill', '#a0aec0')
      .attr('font-size', '12px')
      .text('Total Return (%)');

  }, [monteCarloData]);

  const formatPercent = (num: number): string => {
    return `${num >= 0 ? '+' : ''}${num.toFixed(2)}%`;
  };

  return (
    <div className="p-4 space-y-6">
      {/* Statistics Summary */}
      <div className="grid grid-cols-3 gap-4">
        <div className="terminal-panel p-3">
          <div className="text-terminal-accent text-terminal-xs mb-1">SIMULATIONS</div>
          <div className="text-terminal-lg font-bold text-terminal-text">
            {monteCarloData.simulations.toLocaleString()}
          </div>
        </div>

        <div className="terminal-panel p-3">
          <div className="text-terminal-accent text-terminal-xs mb-1">MEDIAN RETURN</div>
          <div className={`text-terminal-lg font-bold ${monteCarloData.statistics.median_return >= 0 ? 'text-bloomberg-green' : 'text-bloomberg-red'}`}>
            {formatPercent(monteCarloData.statistics.median_return)}
          </div>
        </div>

        <div className="terminal-panel p-3">
          <div className="text-terminal-accent text-terminal-xs mb-1">PROBABILITY OF PROFIT</div>
          <div className="text-terminal-lg font-bold text-bloomberg-green">
            {monteCarloData.statistics.probability_of_profit.toFixed(1)}%
          </div>
        </div>

        <div className="terminal-panel p-3">
          <div className="text-terminal-accent text-terminal-xs mb-1">5TH PERCENTILE</div>
          <div className={`text-terminal-sm font-bold ${monteCarloData.statistics.percentile_5 >= 0 ? 'text-bloomberg-green' : 'text-bloomberg-red'}`}>
            {formatPercent(monteCarloData.statistics.percentile_5)}
          </div>
        </div>

        <div className="terminal-panel p-3">
          <div className="text-terminal-accent text-terminal-xs mb-1">95TH PERCENTILE</div>
          <div className="text-terminal-sm font-bold text-bloomberg-green">
            {formatPercent(monteCarloData.statistics.percentile_95)}
          </div>
        </div>

        <div className="terminal-panel p-3">
          <div className="text-terminal-accent text-terminal-xs mb-1">VALUE AT RISK (95%)</div>
          <div className="text-terminal-sm font-bold text-bloomberg-red">
            {formatPercent(monteCarloData.statistics.value_at_risk_95)}
          </div>
        </div>
      </div>

      {/* Return Distribution */}
      <div className="terminal-panel p-4">
        <div className="text-terminal-accent text-terminal-sm font-bold mb-3">RETURN DISTRIBUTION</div>
        <svg ref={chartRef} className="w-full" />
      </div>

      {/* Return vs Drawdown Scatter */}
      <div className="terminal-panel p-4">
        <div className="text-terminal-accent text-terminal-sm font-bold mb-3">RETURN VS DRAWDOWN</div>
        <svg ref={scatterRef} className="w-full" />
      </div>
    </div>
  );
};

export default MonteCarloDistribution;
