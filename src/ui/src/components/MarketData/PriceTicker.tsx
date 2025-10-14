import React, { memo } from 'react';
import { PriceData } from '../../store/slices/marketDataSlice';

interface PriceTickerProps {
  prices: PriceData[];
}

const PriceTicker: React.FC<PriceTickerProps> = memo(({ prices }) => {
  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(price);
  };

  const formatChange = (change: number, changePercent: number) => {
    const sign = change >= 0 ? '+' : '';
    return `${sign}${change.toFixed(2)} (${sign}${changePercent.toFixed(2)}%)`;
  };

  const getChangeColor = (change: number) => {
    if (change > 0) return 'text-bloomberg-green';
    if (change < 0) return 'text-bloomberg-red';
    return 'text-terminal-text';
  };

  const getChangeIcon = (change: number) => {
    if (change > 0) return '↑';
    if (change < 0) return '↓';
    return '→';
  };

  return (
    <div className="space-y-1">
      <div className="text-terminal-accent text-terminal-xs font-bold border-b border-terminal-border pb-1">
        PRICES
      </div>
      {prices.map((price) => (
        <div key={price.symbol} className="flex justify-between items-center text-terminal-sm">
          <div className="flex items-center space-x-2">
            <span className="font-bold">{price.symbol}</span>
            <span className="text-terminal-accent">{getChangeIcon(price.change)}</span>
          </div>
          <div className="text-right">
            <div className="font-mono">{formatPrice(price.price)}</div>
            <div className={`text-terminal-xs ${getChangeColor(price.change)}`}>
              {formatChange(price.change, price.changePercent)}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
});

PriceTicker.displayName = 'PriceTicker';

export default PriceTicker;