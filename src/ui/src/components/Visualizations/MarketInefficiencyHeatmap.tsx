import React, { useEffect, useRef } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState, AppDispatch } from '../../store/store';
import { fetchInefficiencies } from '../../store/slices/neuralNetworkSlice';
import * as d3 from 'd3';

const MarketInefficiencyHeatmap: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const { inefficiencies } = useSelector((state: RootState) => state.neuralNetwork);
  const heatmapRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    dispatch(fetchInefficiencies());
  }, [dispatch]);

  useEffect(() => {
    if (!heatmapRef.current || inefficiencies.length === 0) return;

    // Clear previous chart
    d3.select(heatmapRef.current).selectAll('*').remove();

    const margin = { top: 40, right: 100, bottom: 60, left: 100 };
    const width = 700 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const svg = d3.select(heatmapRef.current)
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Group inefficiencies by type and symbol
    const types = Array.from(new Set(inefficiencies.map(i => i.type)));
    const symbols = Array.from(new Set(inefficiencies.map(i => i.symbol)));

    const heatmapData = types.flatMap(type =>
      symbols.map(symbol => {
        const items = inefficiencies.filter(i => i.type === type && i.symbol === symbol);
        const avgSeverity = items.length > 0
          ? items.reduce((sum, i) => sum + i.severity, 0) / items.length
          : 0;
        const avgOpportunity = items.length > 0
          ? items.reduce((sum, i) => sum + i.opportunity_score, 0) / items.length
          : 0;
        return { type, symbol, severity: avgSeverity, opportunity: avgOpportunity, count: items.length };
      })
    );

    // Scales
    const xScale = d3.scaleBand()
      .domain(symbols)
      .range([0, width])
      .padding(0.05);

    const yScale = d3.scaleBand()
      .domain(types)
      .range([0, height])
      .padding(0.05);

    const colorScale = d3.scaleSequential(d3.interpolateRdYlGn)
      .domain([0, 100]);

    // Draw rectangles
    svg.selectAll('rect')
      .data(heatmapData)
      .enter()
      .append('rect')
      .attr('x', d => xScale(d.symbol)!)
      .attr('y', d => yScale(d.type)!)
      .attr('width', xScale.bandwidth())
      .attr('height', yScale.bandwidth())
      .attr('fill', d => colorScale(d.opportunity))
      .attr('stroke', '#1a202c')
      .attr('stroke-width', 1)
      .on('mouseenter', function(event, d) {
        d3.select(this).attr('opacity', 0.8);
        // Show tooltip
        svg.append('text')
          .attr('class', 'tooltip')
          .attr('x', xScale(d.symbol)! + xScale.bandwidth() / 2)
          .attr('y', yScale(d.type)! - 5)
          .attr('text-anchor', 'middle')
          .attr('fill', '#f59e0b')
          .attr('font-size', '12px')
          .attr('font-weight', 'bold')
          .text(`Score: ${d.opportunity.toFixed(1)}`);
      })
      .on('mouseleave', function() {
        d3.select(this).attr('opacity', 1);
        svg.selectAll('.tooltip').remove();
      });

    // Add cell text (count)
    svg.selectAll('.cell-text')
      .data(heatmapData)
      .enter()
      .append('text')
      .attr('class', 'cell-text')
      .attr('x', d => xScale(d.symbol)! + xScale.bandwidth() / 2)
      .attr('y', d => yScale(d.type)! + yScale.bandwidth() / 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('fill', '#0a0e13')
      .attr('font-size', '10px')
      .attr('font-weight', 'bold')
      .text(d => d.count > 0 ? d.count : '');

    // X axis
    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale))
      .selectAll('text')
      .attr('fill', '#a0aec0')
      .attr('transform', 'rotate(-45)')
      .attr('text-anchor', 'end');

    // Y axis
    svg.append('g')
      .call(d3.axisLeft(yScale))
      .selectAll('text')
      .attr('fill', '#a0aec0')
      .text(d => {
        // Shorten type names for better display
        const typeMap: Record<string, string> = {
          'orderbook_imbalance': 'Orderbook',
          'spread_anomaly': 'Spread',
          'volume_divergence': 'Volume',
          'momentum_shift': 'Momentum',
          'volatility_spike': 'Volatility',
          'correlation_break': 'Correlation',
        };
        return typeMap[d as string] || d;
      });

    // Color legend
    const legendWidth = 20;
    const legendHeight = height;
    const legendScale = d3.scaleLinear()
      .domain([0, 100])
      .range([legendHeight, 0]);

    const legend = svg.append('g')
      .attr('transform', `translate(${width + 20}, 0)`);

    const gradient = svg.append('defs')
      .append('linearGradient')
      .attr('id', 'legend-gradient')
      .attr('x1', '0%')
      .attr('x2', '0%')
      .attr('y1', '100%')
      .attr('y2', '0%');

    gradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', d3.interpolateRdYlGn(0));

    gradient.append('stop')
      .attr('offset', '50%')
      .attr('stop-color', d3.interpolateRdYlGn(0.5));

    gradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', d3.interpolateRdYlGn(1));

    legend.append('rect')
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', 'url(#legend-gradient)');

    legend.append('g')
      .attr('transform', `translate(${legendWidth}, 0)`)
      .call(d3.axisRight(legendScale).ticks(5))
      .selectAll('text')
      .attr('fill', '#a0aec0');

    legend.append('text')
      .attr('x', legendWidth / 2)
      .attr('y', -10)
      .attr('text-anchor', 'middle')
      .attr('fill', '#a0aec0')
      .attr('font-size', '10px')
      .text('Opportunity');

  }, [inefficiencies]);

  const recentInefficiencies = inefficiencies.slice(0, 5);

  return (
    <div className="h-full flex flex-col">
      <div className="terminal-panel-header">
        <span className="text-terminal-green text-terminal-sm font-bold">
          MARKET INEFFICIENCY DETECTOR
        </span>
      </div>

      <div className="terminal-content flex-1 overflow-y-auto terminal-scrollbar p-4">
        {inefficiencies.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-terminal-accent">No market inefficiencies detected</div>
          </div>
        ) : (
          <div className="space-y-4">
            {/* Heatmap */}
            <div className="terminal-panel p-4">
              <div className="text-terminal-accent text-terminal-sm font-bold mb-3">
                INEFFICIENCY HEATMAP
              </div>
              <svg ref={heatmapRef} className="w-full" />
            </div>

            {/* Recent Inefficiencies */}
            <div className="terminal-panel p-4">
              <div className="text-terminal-accent text-terminal-sm font-bold mb-3">
                RECENT DETECTIONS
              </div>
              <div className="space-y-2">
                {recentInefficiencies.map((inefficiency, index) => (
                  <div
                    key={`${inefficiency.timestamp}-${index}`}
                    className="bg-terminal-border p-3 rounded"
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <div className="flex items-center space-x-2 mb-1">
                          <span className="text-terminal-text font-bold text-terminal-sm">
                            {inefficiency.symbol}
                          </span>
                          <span className="text-terminal-accent text-terminal-xs">
                            {inefficiency.type.replace(/_/g, ' ').toUpperCase()}
                          </span>
                        </div>
                        <div className="text-terminal-xs text-terminal-text">
                          {inefficiency.description}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-terminal-xs text-terminal-accent mb-1">
                          OPPORTUNITY
                        </div>
                        <div className={`text-terminal-sm font-bold ${
                          inefficiency.opportunity_score > 70 ? 'text-bloomberg-green' :
                          inefficiency.opportunity_score > 40 ? 'text-bloomberg-amber' :
                          'text-bloomberg-red'
                        }`}>
                          {inefficiency.opportunity_score.toFixed(1)}
                        </div>
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-2 text-terminal-xs">
                      <div>
                        <span className="text-terminal-accent">Severity: </span>
                        <span className={`font-bold ${
                          inefficiency.severity > 0.7 ? 'text-bloomberg-red' :
                          inefficiency.severity > 0.4 ? 'text-bloomberg-amber' :
                          'text-bloomberg-green'
                        }`}>
                          {(inefficiency.severity * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div>
                        <span className="text-terminal-accent">Confidence: </span>
                        <span className="text-terminal-text font-bold">
                          {(inefficiency.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div>
                        <span className="text-terminal-accent">Action: </span>
                        <span className={`font-bold ${
                          inefficiency.recommended_action === 'buy' ? 'text-bloomberg-green' :
                          inefficiency.recommended_action === 'sell' ? 'text-bloomberg-red' :
                          'text-terminal-accent'
                        }`}>
                          {inefficiency.recommended_action?.toUpperCase() || 'N/A'}
                        </span>
                      </div>
                    </div>

                    <div className="text-terminal-xs text-terminal-accent mt-2">
                      {new Date(inefficiency.timestamp).toLocaleString()}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MarketInefficiencyHeatmap;
