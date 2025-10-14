import React from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../../store/store';
import PriceTicker from './PriceTicker';
import OrderBook from './OrderBook';

const MarketData: React.FC = () => {
  const { prices, orderBook } = useSelector((state: RootState) => state.marketData);

  return (
    <div className="h-full flex flex-col">
      <div className="terminal-header">
        <span className="text-terminal-accent font-bold">MARKET OVERVIEW</span>
      </div>
      <div className="terminal-content flex-1 flex flex-col space-y-2">
        <div className="flex-1">
          <PriceTicker prices={Object.values(prices)} />
        </div>
        <div className="flex-1">
          <OrderBook orderBook={orderBook} />
        </div>
      </div>
    </div>
  );
};

export default MarketData;
