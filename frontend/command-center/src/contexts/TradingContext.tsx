import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { useWebSocket } from './WebSocketContext';
import toast from 'react-hot-toast';

interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercent: number;
  openTime: Date;
}

interface TradingContextType {
  tradingEnabled: boolean;
  positions: Position[];
  balance: number;
  equity: number;
  margin: number;
  freeMargin: number;
  toggleTrading: () => void;
  emergencyStop: () => void;
  closePosition: (id: string) => void;
  closeAllPositions: () => void;
  placeTrade: (symbol: string, side: 'buy' | 'sell', quantity: number) => void;
}

const TradingContext = createContext<TradingContextType | undefined>(undefined);

export function TradingProvider({ children }: { children: React.ReactNode }) {
  const { sendMessage, lastMessage, connected } = useWebSocket();
  const [tradingEnabled, setTradingEnabled] = useState(false);
  const [positions, setPositions] = useState<Position[]>([]);
  const [balance, setBalance] = useState(100000);
  const [equity, setEquity] = useState(100000);
  const [margin, setMargin] = useState(0);
  const [freeMargin, setFreeMargin] = useState(100000);

  // Handle incoming WebSocket messages
  useEffect(() => {
    if (!lastMessage) return;

    switch (lastMessage.type) {
      case 'portfolio_update':
        const { portfolio } = lastMessage.data;
        setBalance(portfolio.balance);
        setEquity(portfolio.equity);
        setMargin(portfolio.margin);
        setFreeMargin(portfolio.freeMargin);
        break;

      case 'position_update':
        const { positions: newPositions } = lastMessage.data;
        setPositions(newPositions);
        break;

      case 'trade_update':
        const { trade } = lastMessage.data;
        if (trade.status === 'executed') {
          toast.success(`Trade executed: ${trade.symbol}`);
        } else if (trade.status === 'rejected') {
          toast.error(`Trade rejected: ${trade.reason}`);
        }
        break;
    }
  }, [lastMessage]);

  const toggleTrading = useCallback(() => {
    const newState = !tradingEnabled;
    setTradingEnabled(newState);
    sendMessage('toggle_trading', { enabled: newState });
    
    if (newState) {
      toast.success('Trading enabled');
    } else {
      toast.warning('Trading paused');
    }
  }, [tradingEnabled, sendMessage]);

  const emergencyStop = useCallback(() => {
    if (!connected) {
      toast.error('Not connected to trading system');
      return;
    }

    if (window.confirm('⚠️ EMERGENCY STOP\n\nThis will:\n- Close all open positions immediately\n- Stop all trading activities\n- May result in losses\n\nAre you absolutely sure?')) {
      setTradingEnabled(false);
      sendMessage('emergency_stop', {});
      toast.error('EMERGENCY STOP ACTIVATED');
      
      // Clear all positions locally
      setPositions([]);
    }
  }, [connected, sendMessage]);

  const closePosition = useCallback((id: string) => {
    if (!connected) {
      toast.error('Not connected to trading system');
      return;
    }

    sendMessage('close_position', { id });
    toast.info('Closing position...');
  }, [connected, sendMessage]);

  const closeAllPositions = useCallback(() => {
    if (!connected) {
      toast.error('Not connected to trading system');
      return;
    }

    if (window.confirm('Close all positions?')) {
      sendMessage('close_all_positions', {});
      toast.warning('Closing all positions...');
    }
  }, [connected, sendMessage]);

  const placeTrade = useCallback((symbol: string, side: 'buy' | 'sell', quantity: number) => {
    if (!connected) {
      toast.error('Not connected to trading system');
      return;
    }

    if (!tradingEnabled) {
      toast.error('Trading is disabled');
      return;
    }

    sendMessage('place_trade', {
      symbol,
      side,
      quantity,
      type: 'market',
    });
    
    toast.info(`Placing ${side} order for ${quantity} ${symbol}...`);
  }, [connected, tradingEnabled, sendMessage]);

  const value: TradingContextType = {
    tradingEnabled,
    positions,
    balance,
    equity,
    margin,
    freeMargin,
    toggleTrading,
    emergencyStop,
    closePosition,
    closeAllPositions,
    placeTrade,
  };

  return (
    <TradingContext.Provider value={value}>
      {children}
    </TradingContext.Provider>
  );
}

export function useTrading() {
  const context = useContext(TradingContext);
  if (context === undefined) {
    throw new Error('useTrading must be used within a TradingProvider');
  }
  return context;
}