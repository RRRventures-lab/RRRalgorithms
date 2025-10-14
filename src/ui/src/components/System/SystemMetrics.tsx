import React from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../../store/store';

const SystemMetrics: React.FC = () => {
  const { metrics, status } = useSelector((state: RootState) => state.system);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'text-bloomberg-green';
      case 'degraded':
        return 'text-bloomberg-amber';
      case 'critical':
        return 'text-bloomberg-red';
      default:
        return 'text-terminal-text';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return '●';
      case 'degraded':
        return '▲';
      case 'critical':
        return '▼';
      default:
        return '○';
    }
  };

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  return (
    <div className="h-full flex flex-col">
      <div className="terminal-header">
        <span className="text-terminal-accent font-bold">SYSTEM METRICS</span>
        <div className={`flex items-center space-x-1 ${getStatusColor(status)}`}>
          <span>{getStatusIcon(status)}</span>
          <span className="text-terminal-xs">{status.toUpperCase()}</span>
        </div>
      </div>
      <div className="terminal-content space-y-2">
        <div className="space-y-1 text-terminal-sm">
          <div className="flex justify-between">
            <span>CPU:</span>
            <span className="font-mono">{metrics.cpu}%</span>
          </div>
          
          <div className="flex justify-between">
            <span>MEM:</span>
            <span className="font-mono">{metrics.memory}GB</span>
          </div>
          
          <div className="flex justify-between">
            <span>LAT:</span>
            <span className="font-mono">{metrics.latency}ms</span>
          </div>
          
          <div className="flex justify-between">
            <span>TPS:</span>
            <span className="font-mono">{metrics.throughput}</span>
          </div>
          
          <div className="flex justify-between">
            <span>UPTIME:</span>
            <span className="font-mono">{formatUptime(metrics.uptime)}</span>
          </div>
        </div>
        
        {/* Simple progress bars */}
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <span className="text-terminal-xs w-8">CPU</span>
            <div className="flex-1 bg-terminal-border h-2">
              <div 
                className="bg-bloomberg-green h-full transition-all duration-300"
                style={{ width: `${Math.min(metrics.cpu, 100)}%` }}
              />
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <span className="text-terminal-xs w-8">MEM</span>
            <div className="flex-1 bg-terminal-border h-2">
              <div 
                className="bg-bloomberg-blue h-full transition-all duration-300"
                style={{ width: `${Math.min((metrics.memory / 8) * 100, 100)}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemMetrics;
