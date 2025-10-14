import { store } from '../store/store';
import { updatePrice, updateOrderBook } from '../store/slices/marketDataSlice';
import { updatePosition } from '../store/slices/portfolioSlice';
import { updateMetrics, addAlert } from '../store/slices/systemSlice';

class TradingWebSocket {
  private connections: Map<string, WebSocket> = new Map();
  private reconnectIntervals: Map<string, number> = new Map();
  private reconnectAttempts: Map<string, number> = new Map();
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  streams = {
    prices: 'ws://localhost:8080/stream/prices',
    orders: 'ws://localhost:8080/stream/orders',
    positions: 'ws://localhost:8080/stream/positions',
    metrics: 'ws://localhost:8080/stream/metrics'
  };

  subscribeToAll() {
    Object.entries(this.streams).forEach(([key, url]) => {
      this.connect(key, url);
    });
  }

  private connect(streamName: string, url: string) {
    try {
      const ws = new WebSocket(url);
      
      ws.onopen = () => {
        console.log(`Connected to ${streamName} stream`);
        this.connections.set(streamName, ws);
        this.reconnectAttempts.set(streamName, 0);
        
        // Clear any existing reconnect interval
        const interval = this.reconnectIntervals.get(streamName);
        if (interval) {
          clearInterval(interval);
          this.reconnectIntervals.delete(streamName);
        }
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleMessage(streamName, data);
        } catch (error) {
          console.error(`Error parsing message from ${streamName}:`, error);
        }
      };

      ws.onclose = () => {
        console.log(`Disconnected from ${streamName} stream`);
        this.connections.delete(streamName);
        this.scheduleReconnect(streamName, url);
      };

      ws.onerror = (error) => {
        console.error(`WebSocket error for ${streamName}:`, error);
        store.dispatch(addAlert(`WebSocket error: ${streamName}`));
      };

    } catch (error) {
      console.error(`Failed to connect to ${streamName}:`, error);
      this.scheduleReconnect(streamName, url);
    }
  }

  private handleMessage(streamName: string, data: any) {
    switch (streamName) {
      case 'prices':
        if (data.symbol && data.price) {
          store.dispatch(updatePrice({
            symbol: data.symbol,
            price: data.price,
            change: data.change || 0,
            changePercent: data.changePercent || 0,
            volume: data.volume || 0,
            timestamp: Date.now()
          }));
        }
        break;

      case 'orders':
        if (data.bids && data.asks) {
          store.dispatch(updateOrderBook({
            bids: data.bids,
            asks: data.asks
          }));
        }
        break;

      case 'positions':
        if (data.symbol && data.quantity !== undefined) {
          store.dispatch(updatePosition({
            symbol: data.symbol,
            quantity: data.quantity,
            avgPrice: data.avgPrice || 0,
            currentPrice: data.currentPrice || 0,
            unrealizedPnl: data.unrealizedPnl || 0,
            unrealizedPnlPercent: data.unrealizedPnlPercent || 0,
            marketValue: data.marketValue || 0,
            costBasis: data.costBasis || (data.quantity * (data.avgPrice || 0))
          }));
        }
        break;

      case 'metrics':
        if (data.cpu !== undefined) {
          store.dispatch(updateMetrics({
            cpu: data.cpu,
            memory: data.memory || 0,
            latency: data.latency || 0,
            throughput: data.throughput || 0,
            uptime: data.uptime || 0
          }));
        }
        break;

      default:
        console.warn(`Unknown stream: ${streamName}`);
    }
  }

  private scheduleReconnect(streamName: string, url: string) {
    const attempts = this.reconnectAttempts.get(streamName) || 0;
    
    if (attempts >= this.maxReconnectAttempts) {
      console.error(`Max reconnection attempts reached for ${streamName}`);
      store.dispatch(addAlert(`Connection failed: ${streamName}`));
      return;
    }

    this.reconnectAttempts.set(streamName, attempts + 1);
    
    const delay = this.reconnectDelay * Math.pow(2, attempts);
    console.log(`Reconnecting to ${streamName} in ${delay}ms (attempt ${attempts + 1})`);

    const interval = window.setTimeout(() => {
      this.connect(streamName, url);
    }, delay);

    this.reconnectIntervals.set(streamName, interval);
  }

  disconnect() {
    // Close all connections
    this.connections.forEach((ws, streamName) => {
      ws.close();
      console.log(`Disconnected from ${streamName}`);
    });
    this.connections.clear();

    // Clear all reconnect intervals
    this.reconnectIntervals.forEach((interval) => {
      window.clearTimeout(interval);
    });
    this.reconnectIntervals.clear();
    this.reconnectAttempts.clear();
  }

  getConnectionStatus() {
    const status: Record<string, boolean> = {};
    Object.keys(this.streams).forEach(streamName => {
      status[streamName] = this.connections.has(streamName);
    });
    return status;
  }
}

// Create singleton instance
export const tradingWebSocket = new TradingWebSocket();

// Auto-connect on import
if (typeof window !== 'undefined') {
  tradingWebSocket.subscribeToAll();
}
