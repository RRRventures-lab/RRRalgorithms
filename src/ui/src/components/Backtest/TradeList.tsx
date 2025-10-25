import React, { useState } from 'react';
import { FixedSizeList as List } from 'react-window';

interface Trade {
  timestamp: string;
  symbol: string;
  side: 'buy' | 'sell';
  price: number;
  quantity: number;
  pnl: number;
  commission: number;
}

interface TradeListProps {
  trades: Trade[];
}

const TradeList: React.FC<TradeListProps> = ({ trades }) => {
  const [sortBy, setSortBy] = useState<'timestamp' | 'pnl' | 'symbol'>('timestamp');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [filterSide, setFilterSide] = useState<'all' | 'buy' | 'sell'>('all');

  const handleSort = (field: 'timestamp' | 'pnl' | 'symbol') => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
  };

  const sortedTrades = [...trades]
    .filter(trade => filterSide === 'all' || trade.side === filterSide)
    .sort((a, b) => {
      let comparison = 0;
      if (sortBy === 'timestamp') {
        comparison = new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
      } else if (sortBy === 'pnl') {
        comparison = a.pnl - b.pnl;
      } else if (sortBy === 'symbol') {
        comparison = a.symbol.localeCompare(b.symbol);
      }
      return sortOrder === 'asc' ? comparison : -comparison;
    });

  const formatNumber = (num: number, decimals: number = 2): string => {
    return num.toFixed(decimals);
  };

  const formatCurrency = (num: number): string => {
    return `$${formatNumber(num)}`;
  };

  const Row = ({ index, style }: { index: number; style: React.CSSProperties }) => {
    const trade = sortedTrades[index];
    return (
      <div
        style={style}
        className={`grid grid-cols-7 gap-2 px-4 py-2 text-terminal-xs border-b border-terminal-border ${
          index % 2 === 0 ? 'bg-terminal-bg' : 'bg-terminal-border bg-opacity-30'
        }`}
      >
        <div className="text-terminal-accent">
          {new Date(trade.timestamp).toLocaleString()}
        </div>
        <div className="text-terminal-text font-bold">{trade.symbol}</div>
        <div className={trade.side === 'buy' ? 'text-bloomberg-green' : 'text-bloomberg-red'}>
          {trade.side.toUpperCase()}
        </div>
        <div className="text-terminal-text">{formatCurrency(trade.price)}</div>
        <div className="text-terminal-text">{formatNumber(trade.quantity, 4)}</div>
        <div className={trade.pnl >= 0 ? 'text-bloomberg-green font-bold' : 'text-bloomberg-red font-bold'}>
          {trade.pnl >= 0 ? '+' : ''}{formatCurrency(trade.pnl)}
        </div>
        <div className="text-terminal-accent">{formatCurrency(trade.commission)}</div>
      </div>
    );
  };

  const totalPnL = sortedTrades.reduce((sum, trade) => sum + trade.pnl, 0);
  const totalCommissions = sortedTrades.reduce((sum, trade) => sum + trade.commission, 0);
  const winningTrades = sortedTrades.filter(t => t.pnl > 0).length;
  const losingTrades = sortedTrades.filter(t => t.pnl < 0).length;

  return (
    <div className="h-full flex flex-col p-4">
      {/* Summary Stats */}
      <div className="grid grid-cols-5 gap-4 mb-4">
        <div className="terminal-panel p-3">
          <div className="text-terminal-accent text-terminal-xs mb-1">TOTAL TRADES</div>
          <div className="text-terminal-lg font-bold text-terminal-text">
            {sortedTrades.length}
          </div>
        </div>
        <div className="terminal-panel p-3">
          <div className="text-terminal-accent text-terminal-xs mb-1">TOTAL P&L</div>
          <div className={`text-terminal-lg font-bold ${totalPnL >= 0 ? 'text-bloomberg-green' : 'text-bloomberg-red'}`}>
            {totalPnL >= 0 ? '+' : ''}{formatCurrency(totalPnL)}
          </div>
        </div>
        <div className="terminal-panel p-3">
          <div className="text-terminal-accent text-terminal-xs mb-1">COMMISSIONS</div>
          <div className="text-terminal-lg font-bold text-terminal-text">
            {formatCurrency(totalCommissions)}
          </div>
        </div>
        <div className="terminal-panel p-3">
          <div className="text-terminal-accent text-terminal-xs mb-1">WINNING</div>
          <div className="text-terminal-lg font-bold text-bloomberg-green">
            {winningTrades}
          </div>
        </div>
        <div className="terminal-panel p-3">
          <div className="text-terminal-accent text-terminal-xs mb-1">LOSING</div>
          <div className="text-terminal-lg font-bold text-bloomberg-red">
            {losingTrades}
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <span className="text-terminal-accent text-terminal-xs">FILTER:</span>
          {(['all', 'buy', 'sell'] as const).map(side => (
            <button
              key={side}
              onClick={() => setFilterSide(side)}
              className={`bloomberg-button text-terminal-xs px-3 py-1 ${
                filterSide === side ? 'bg-bloomberg-green' : ''
              }`}
            >
              {side.toUpperCase()}
            </button>
          ))}
        </div>
        <div className="text-terminal-accent text-terminal-xs">
          Showing {sortedTrades.length} of {trades.length} trades
        </div>
      </div>

      {/* Table Header */}
      <div className="grid grid-cols-7 gap-2 px-4 py-2 bg-terminal-border text-terminal-xs font-bold">
        <button
          onClick={() => handleSort('timestamp')}
          className="text-left text-terminal-accent hover:text-terminal-text"
        >
          TIMESTAMP {sortBy === 'timestamp' && (sortOrder === 'asc' ? '↑' : '↓')}
        </button>
        <button
          onClick={() => handleSort('symbol')}
          className="text-left text-terminal-accent hover:text-terminal-text"
        >
          SYMBOL {sortBy === 'symbol' && (sortOrder === 'asc' ? '↑' : '↓')}
        </button>
        <div className="text-terminal-accent">SIDE</div>
        <div className="text-terminal-accent">PRICE</div>
        <div className="text-terminal-accent">QUANTITY</div>
        <button
          onClick={() => handleSort('pnl')}
          className="text-left text-terminal-accent hover:text-terminal-text"
        >
          P&L {sortBy === 'pnl' && (sortOrder === 'asc' ? '↑' : '↓')}
        </button>
        <div className="text-terminal-accent">COMMISSION</div>
      </div>

      {/* Virtualized Trade List */}
      <div className="flex-1 terminal-panel">
        <List
          height={400}
          itemCount={sortedTrades.length}
          itemSize={40}
          width="100%"
        >
          {Row}
        </List>
      </div>
    </div>
  );
};

export default TradeList;
