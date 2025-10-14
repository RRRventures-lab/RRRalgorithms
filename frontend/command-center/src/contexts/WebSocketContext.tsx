import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react';
import io, { Socket } from 'socket.io-client';
import toast from 'react-hot-toast';

interface MarketData {
  symbol: string;
  price: number;
  timestamp: number;
  volume?: number;
  change24h?: number;
}

interface WebSocketContextType {
  connected: boolean;
  lastMessage: any;
  marketData: Map<string, MarketData>;
  sendMessage: (event: string, data: any) => void;
  subscribe: (symbols: string[]) => void;
  unsubscribe: (symbols: string[]) => void;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

export function WebSocketProvider({ children }: { children: React.ReactNode }) {
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const [marketData, setMarketData] = useState<Map<string, MarketData>>(new Map());
  const socketRef = useRef<Socket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const reconnectAttempts = useRef(0);

  const connect = useCallback(() => {
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:8000';
    
    const socket = io(wsUrl, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
    });

    socket.on('connect', () => {
      console.log('WebSocket connected');
      setConnected(true);
      reconnectAttempts.current = 0;
      toast.success('Connected to trading system');
    });

    socket.on('disconnect', () => {
      console.log('WebSocket disconnected');
      setConnected(false);
      toast.error('Disconnected from trading system');
    });

    socket.on('error', (error) => {
      console.error('WebSocket error:', error);
      toast.error('Connection error');
    });

    // Market data updates
    socket.on('market_data', (data: MarketData) => {
      setMarketData((prev) => {
        const newData = new Map(prev);
        newData.set(data.symbol, data);
        return newData;
      });
      setLastMessage({ type: 'market_data', data });
    });

    // Trade updates
    socket.on('trade_update', (data: any) => {
      setLastMessage({ type: 'trade_update', data });
      toast.success(`Trade executed: ${data.symbol} ${data.side} ${data.quantity}`);
    });

    // System alerts
    socket.on('alert', (data: any) => {
      setLastMessage({ type: 'alert', data });
      
      switch (data.severity) {
        case 'error':
          toast.error(data.message);
          break;
        case 'warning':
          toast(data.message, { icon: '⚠️' });
          break;
        case 'info':
          toast(data.message, { icon: 'ℹ️' });
          break;
        default:
          toast.success(data.message);
      }
    });

    // Portfolio updates
    socket.on('portfolio_update', (data: any) => {
      setLastMessage({ type: 'portfolio_update', data });
    });

    // ML predictions
    socket.on('prediction', (data: any) => {
      setLastMessage({ type: 'prediction', data });
    });

    socketRef.current = socket;
  }, []);

  const sendMessage = useCallback((event: string, data: any) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit(event, data);
    } else {
      console.warn('WebSocket not connected');
      toast.error('Not connected to trading system');
    }
  }, []);

  const subscribe = useCallback((symbols: string[]) => {
    sendMessage('subscribe', { symbols });
  }, [sendMessage]);

  const unsubscribe = useCallback((symbols: string[]) => {
    sendMessage('unsubscribe', { symbols });
  }, [sendMessage]);

  useEffect(() => {
    connect();

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [connect]);

  // Auto-reconnect logic
  useEffect(() => {
    if (!connected && reconnectAttempts.current < 5) {
      reconnectTimeoutRef.current = setTimeout(() => {
        reconnectAttempts.current++;
        console.log(`Reconnection attempt ${reconnectAttempts.current}/5`);
        connect();
      }, Math.min(1000 * Math.pow(2, reconnectAttempts.current), 10000));
    }
  }, [connected, connect]);

  const value: WebSocketContextType = {
    connected,
    lastMessage,
    marketData,
    sendMessage,
    subscribe,
    unsubscribe,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
}

export function useWebSocket() {
  const context = useContext(WebSocketContext);
  if (context === undefined) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
}