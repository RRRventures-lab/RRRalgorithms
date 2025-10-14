import React, { memo } from 'react';
import { Position } from '../../store/slices/portfolioSlice';

interface HoldingsProps {
  positions: Position[];
}

const Holdings: React.FC<HoldingsProps> = memo(({ positions }) => {
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const formatQuantity = (qty: number) => {
    return qty.toFixed(4);
  };

  const getPnlColor = (pnl: number) => {
    if (pnl > 0) return 'text-bloomberg-green';
    if (pnl < 0) return 'text-bloomberg-red';
    return 'text-terminal-text';
  };

  return (
    <div className="space-y-1">
      <div className="text-terminal-accent text-terminal-xs font-bold border-b border-terminal-border pb-1">
        POSITIONS
      </div>
      
      {positions.length === 0 ? (
        <div className="text-terminal-text text-terminal-xs text-center py-2">
          No positions
        </div>
      ) : (
        <div className="space-y-1">
          {positions.map((position) => (
            <div key={position.symbol} className="text-terminal-xs space-y-1">
              <div className="flex justify-between items-center">
                <span className="font-bold">{position.symbol}</span>
                <span className="font-mono">{formatQuantity(position.quantity)}</span>
              </div>
              
              <div className="flex justify-between text-terminal-xs">
                <span>Value:</span>
                <span className="font-mono">{formatCurrency(position.marketValue)}</span>
              </div>
              
              <div className="flex justify-between text-terminal-xs">
                <span>P&L:</span>
                <div className={`font-mono ${getPnlColor(position.unrealizedPnl)}`}>
                  {position.unrealizedPnl >= 0 ? '+' : ''}{formatCurrency(position.unrealizedPnl)}
                </div>
              </div>
              
              <div className="flex justify-between text-terminal-xs">
                <span>P&L%:</span>
                <div className={`font-mono ${getPnlColor(position.unrealizedPnl)}`}>
                  {position.unrealizedPnlPercent >= 0 ? '+' : ''}{position.unrealizedPnlPercent.toFixed(2)}%
                </div>
              </div>
              
              <div className="border-b border-terminal-border"></div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
});

Holdings.displayName = 'Holdings';

export default Holdings;