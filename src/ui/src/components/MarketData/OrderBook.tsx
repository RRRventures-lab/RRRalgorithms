import React from 'react';
import { OrderBookEntry } from '../../store/slices/marketDataSlice';

interface OrderBookProps {
  orderBook: {
    bids: OrderBookEntry[];
    asks: OrderBookEntry[];
  };
}

const OrderBook: React.FC<OrderBookProps> = ({ orderBook }) => {
  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(price);
  };

  const formatSize = (size: number) => {
    return size.toFixed(2);
  };

  return (
    <div className="space-y-1">
      <div className="text-terminal-accent text-terminal-xs font-bold border-b border-terminal-border pb-1">
        ORDER BOOK
      </div>
      <div className="grid grid-cols-3 gap-1 text-terminal-xs">
        <div className="text-terminal-accent font-bold">BID</div>
        <div className="text-terminal-accent font-bold">SIZE</div>
        <div className="text-terminal-accent font-bold">ASK</div>
      </div>
      
      {/* Asks (highest first) */}
      {orderBook.asks.slice().reverse().map((ask, index) => (
        <div key={`ask-${index}`} className="grid grid-cols-3 gap-1 text-terminal-xs">
          <div></div>
          <div className="text-right">{formatSize(ask.size)}</div>
          <div className="text-bloomberg-red text-right">{formatPrice(ask.price)}</div>
        </div>
      ))}
      
      {/* Spread */}
      <div className="grid grid-cols-3 gap-1 text-terminal-xs border-t border-terminal-border pt-1">
        <div></div>
        <div className="text-terminal-accent text-center">SPREAD</div>
        <div className="text-terminal-accent text-right">
          {formatPrice(orderBook.asks[0]?.price - orderBook.bids[0]?.price || 0)}
        </div>
      </div>
      
      {/* Bids (highest first) */}
      {orderBook.bids.map((bid, index) => (
        <div key={`bid-${index}`} className="grid grid-cols-3 gap-1 text-terminal-xs">
          <div className="text-bloomberg-green">{formatPrice(bid.price)}</div>
          <div className="text-right">{formatSize(bid.size)}</div>
          <div></div>
        </div>
      ))}
    </div>
  );
};

export default OrderBook;
