import React from 'react';

interface PerformanceProps {
  totalValue: number;
  cash: number;
  totalPnl: number;
  totalPnlPercent: number;
}

const Performance: React.FC<PerformanceProps> = ({
  totalValue,
  cash,
  totalPnl,
  totalPnlPercent,
}) => {
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const getPnlColor = (pnl: number) => {
    if (pnl > 0) return 'text-bloomberg-green';
    if (pnl < 0) return 'text-bloomberg-red';
    return 'text-terminal-text';
  };

  const getPnlIcon = (pnl: number) => {
    if (pnl > 0) return '▲';
    if (pnl < 0) return '▼';
    return '●';
  };

  return (
    <div className="space-y-2">
      <div className="text-terminal-accent text-terminal-xs font-bold border-b border-terminal-border pb-1">
        PERFORMANCE
      </div>
      
      <div className="space-y-1 text-terminal-sm">
        <div className="flex justify-between">
          <span>Value:</span>
          <span className="font-mono font-bold">{formatCurrency(totalValue)}</span>
        </div>
        
        <div className="flex justify-between">
          <span>Cash:</span>
          <span className="font-mono">{formatCurrency(cash)}</span>
        </div>
        
        <div className="flex justify-between">
          <span>P&L:</span>
          <div className={`font-mono ${getPnlColor(totalPnl)}`}>
            {getPnlIcon(totalPnl)} {formatCurrency(totalPnl)}
          </div>
        </div>
        
        <div className="flex justify-between">
          <span>P&L%:</span>
          <div className={`font-mono ${getPnlColor(totalPnl)}`}>
            {totalPnlPercent >= 0 ? '+' : ''}{totalPnlPercent.toFixed(2)}%
          </div>
        </div>
      </div>
    </div>
  );
};

export default Performance;
