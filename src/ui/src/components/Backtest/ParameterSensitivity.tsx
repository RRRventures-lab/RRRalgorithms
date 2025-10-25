import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface ParameterSensitivityData {
  parameter_name: string;
  values: number[];
  returns: number[];
  sharpe_ratios: number[];
}

interface ParameterSensitivityProps {
  data: ParameterSensitivityData[];
}

const ParameterSensitivity: React.FC<ParameterSensitivityProps> = ({ data }) => {
  const chartRefs = useRef<(SVGSVGElement | null)[]>([]);

  useEffect(() => {
    data.forEach((param, index) => {
      const chartRef = chartRefs.current[index];
      if (!chartRef) return;

      // Clear previous chart
      d3.select(chartRef).selectAll('*').remove();

      const margin = { top: 20, right: 60, bottom: 40, left: 60 };
      const width = 600 - margin.left - margin.right;
      const height = 200 - margin.top - margin.bottom;

      const svg = d3.select(chartRef)
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

      // Scales
      const x = d3.scaleLinear()
        .domain([d3.min(param.values)!, d3.max(param.values)!])
        .range([0, width]);

      const yReturn = d3.scaleLinear()
        .domain([d3.min(param.returns)!, d3.max(param.returns)!])
        .range([height, 0]);

      const ySharpe = d3.scaleLinear()
        .domain([d3.min(param.sharpe_ratios)!, d3.max(param.sharpe_ratios)!])
        .range([height, 0]);

      // Line generators
      const returnLine = d3.line<number>()
        .x((d, i) => x(param.values[i]))
        .y(d => yReturn(d));

      const sharpeLine = d3.line<number>()
        .x((d, i) => x(param.values[i]))
        .y(d => ySharpe(d));

      // Draw return line
      svg.append('path')
        .datum(param.returns)
        .attr('fill', 'none')
        .attr('stroke', '#48bb78')
        .attr('stroke-width', 2)
        .attr('d', returnLine);

      // Draw Sharpe ratio line
      svg.append('path')
        .datum(param.sharpe_ratios)
        .attr('fill', 'none')
        .attr('stroke', '#f59e0b')
        .attr('stroke-width', 2)
        .attr('d', sharpeLine);

      // Add dots for returns
      svg.selectAll('.dot-return')
        .data(param.returns)
        .enter()
        .append('circle')
        .attr('class', 'dot-return')
        .attr('cx', (d, i) => x(param.values[i]))
        .attr('cy', d => yReturn(d))
        .attr('r', 3)
        .attr('fill', '#48bb78');

      // Add dots for Sharpe
      svg.selectAll('.dot-sharpe')
        .data(param.sharpe_ratios)
        .enter()
        .append('circle')
        .attr('class', 'dot-sharpe')
        .attr('cx', (d, i) => x(param.values[i]))
        .attr('cy', d => ySharpe(d))
        .attr('r', 3)
        .attr('fill', '#f59e0b');

      // Add axes
      svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x))
        .attr('color', '#a0aec0');

      svg.append('g')
        .call(d3.axisLeft(yReturn).tickFormat(d => `${d}%`))
        .attr('color', '#a0aec0');

      svg.append('g')
        .attr('transform', `translate(${width},0)`)
        .call(d3.axisRight(ySharpe))
        .attr('color', '#a0aec0');

      // Add labels
      svg.append('text')
        .attr('x', width / 2)
        .attr('y', height + 35)
        .attr('text-anchor', 'middle')
        .attr('fill', '#a0aec0')
        .attr('font-size', '12px')
        .text(param.parameter_name);

      svg.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -height / 2)
        .attr('y', -45)
        .attr('text-anchor', 'middle')
        .attr('fill', '#48bb78')
        .attr('font-size', '12px')
        .text('Return (%)');

      svg.append('text')
        .attr('transform', 'rotate(90)')
        .attr('x', height / 2)
        .attr('y', -width - 45)
        .attr('text-anchor', 'middle')
        .attr('fill', '#f59e0b')
        .attr('font-size', '12px')
        .text('Sharpe Ratio');
    });
  }, [data]);

  return (
    <div className="p-4 space-y-6">
      <div className="text-terminal-accent text-terminal-sm font-bold mb-4">
        PARAMETER SENSITIVITY ANALYSIS
      </div>

      {data.map((param, index) => (
        <div key={param.parameter_name} className="terminal-panel p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="text-terminal-text font-bold">{param.parameter_name}</div>
            <div className="flex items-center space-x-4 text-terminal-xs">
              <div className="flex items-center space-x-1">
                <div className="w-3 h-3 bg-bloomberg-green rounded-full" />
                <span className="text-terminal-accent">Return</span>
              </div>
              <div className="flex items-center space-x-1">
                <div className="w-3 h-3 bg-bloomberg-amber rounded-full" />
                <span className="text-terminal-accent">Sharpe Ratio</span>
              </div>
            </div>
          </div>
          <svg
            ref={el => chartRefs.current[index] = el}
            className="w-full"
          />
        </div>
      ))}

      {data.length === 0 && (
        <div className="text-center text-terminal-accent py-8">
          No parameter sensitivity data available
        </div>
      )}
    </div>
  );
};

export default ParameterSensitivity;
