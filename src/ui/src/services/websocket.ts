import { io, Socket } from 'socket.io-client';
import { store } from '../store/store';
import { updatePrice, updateOrderBook } from '../store/slices/marketDataSlice';
import { updatePosition } from '../store/slices/portfolioSlice';
import { updateMetrics, addAlert } from '../store/slices/systemSlice';

interface TradeFeedData {
  id: string;
  timestamp: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  total_value: number;
  fee: number;
  status: string;
  type: string;
}

interface PortfolioUpdateData {
  timestamp: string;
  total_equity: number;
  cash_balance: number;
  invested: number;
  total_pnl: number;
  total_pnl_percent: number;
  day_pnl: number;
  day_pnl_percent: number;
  positions_count: number;
  open_orders: number;
}

interface AIDecisionData {
  id: string;
  timestamp: string;
  model_name: string;
  symbol: string;
  prediction: {
    direction: 'up' | 'down';
    confidence: number;
    price_target: number;
    time_horizon: string;
  };
  reasoning: string;
  outcome: string;
}

interface PerformanceMetricsData {
  timestamp: string;
  period: string;
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  profit_factor: number;
  total_trades: number;
  api_latency_ms: number;
  websocket_connections: number;
}

class TransparencyWebSocket {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private serverUrl = 'http://localhost:8000';
  private connected = false;

  // Event handlers storage
  private eventHandlers: Map<string, Function[]> = new Map();

  constructor() {
    this.initializeEventHandlers();
  }

  private initializeEventHandlers() {
    // Initialize empty handler arrays for each event
    ['trade_feed', 'portfolio_update', 'ai_decision', 'performance_metrics'].forEach(event => {
      this.eventHandlers.set(event, []);
    });
  }

  connect() {
    if (this.socket?.connected) {
      console.log('Socket.IO already connected');
      return;
    }

    console.log(`Connecting to transparency API at ${this.serverUrl}...`);

    this.socket = io(this.serverUrl, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: this.reconnectDelay,
      reconnectionDelayMax: 5000,
      timeout: 10000,
      path: '/socket.io'
    });

    this.setupEventListeners();
  }

  private setupEventListeners() {
    if (!this.socket) return;

    // Connection events
    this.socket.on('connect', () => {
      console.log('Connected to transparency API');
      this.connected = true;
      this.reconnectAttempts = 0;

      store.dispatch(addAlert('Connected to real-time transparency feed'));

      // Subscribe to all streams by default
      this.subscribe(['trade_feed', 'portfolio_update', 'ai_decision', 'performance_metrics']);
    });

    this.socket.on('disconnect', (reason) => {
      console.log('Disconnected from transparency API:', reason);
      this.connected = false;
      store.dispatch(addAlert('Disconnected from transparency feed'));
    });

    this.socket.on('connect_error', (error) => {
      console.error('Connection error:', error);
      this.reconnectAttempts++;

      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        console.error('Max reconnection attempts reached');
        store.dispatch(addAlert('Failed to connect to transparency API'));
      }
    });

    // Server messages
    this.socket.on('connected', (data) => {
      console.log('Server welcome message:', data);
      console.log('API Version:', data.api_version);
      console.log('Available features:', data.features);
    });

    this.socket.on('subscription_confirmed', (data) => {
      console.log('Subscription confirmed:', data);
    });

    // Data streams
    this.socket.on('trade_feed', (data: TradeFeedData) => {
      this.handleTradeFeed(data);
      this.triggerHandlers('trade_feed', data);
    });

    this.socket.on('portfolio_update', (data: PortfolioUpdateData) => {
      this.handlePortfolioUpdate(data);
      this.triggerHandlers('portfolio_update', data);
    });

    this.socket.on('ai_decision', (data: AIDecisionData) => {
      this.handleAIDecision(data);
      this.triggerHandlers('ai_decision', data);
    });

    this.socket.on('performance_metrics', (data: PerformanceMetricsData) => {
      this.handlePerformanceMetrics(data);
      this.triggerHandlers('performance_metrics', data);
    });

    // Ping/pong for latency testing
    this.socket.on('pong', (data) => {
      const latency = Date.now() - new Date(data.client_timestamp).getTime();
      console.log(`Latency: ${latency}ms`);
    });
  }

  private handleTradeFeed(data: TradeFeedData) {
    console.log('Trade feed update:', data);

    // Update Redux store with trade data
    // This can be expanded based on your Redux slice structure
    store.dispatch(updatePrice({
      symbol: data.symbol,
      price: data.price,
      change: 0, // Calculate based on previous price
      changePercent: 0,
      volume: data.total_value,
      timestamp: new Date(data.timestamp).getTime()
    }));
  }

  private handlePortfolioUpdate(data: PortfolioUpdateData) {
    console.log('Portfolio update:', data);

    // Update Redux store with portfolio data
    store.dispatch(updateMetrics({
      cpu: 0, // These would come from performance_metrics
      memory: 0,
      latency: 0,
      throughput: 0,
      uptime: 0
    }));
  }

  private handleAIDecision(data: AIDecisionData) {
    console.log('AI decision:', data);

    // Dispatch alert for new AI decision
    store.dispatch(addAlert(
      `${data.model_name} predicts ${data.prediction.direction} for ${data.symbol} ` +
      `(confidence: ${(data.prediction.confidence * 100).toFixed(0)}%)`
    ));
  }

  private handlePerformanceMetrics(data: PerformanceMetricsData) {
    console.log('Performance metrics:', data);

    store.dispatch(updateMetrics({
      cpu: 0,
      memory: 0,
      latency: data.api_latency_ms,
      throughput: 0,
      uptime: 0
    }));
  }

  private triggerHandlers(event: string, data: any) {
    const handlers = this.eventHandlers.get(event) || [];
    handlers.forEach(handler => {
      try {
        handler(data);
      } catch (error) {
        console.error(`Error in ${event} handler:`, error);
      }
    });
  }

  // Public API methods
  subscribe(streams: string[]) {
    if (!this.socket?.connected) {
      console.warn('Cannot subscribe: not connected');
      return;
    }

    console.log('Subscribing to streams:', streams);
    this.socket.emit('subscribe', { streams });
  }

  unsubscribe(streams: string[]) {
    if (!this.socket?.connected) {
      console.warn('Cannot unsubscribe: not connected');
      return;
    }

    console.log('Unsubscribing from streams:', streams);
    this.socket.emit('unsubscribe', { streams });
  }

  ping() {
    if (!this.socket?.connected) {
      console.warn('Cannot ping: not connected');
      return;
    }

    this.socket.emit('ping', { timestamp: new Date().toISOString() });
  }

  disconnect() {
    if (this.socket) {
      console.log('Disconnecting from transparency API...');
      this.socket.disconnect();
      this.socket = null;
      this.connected = false;
    }
  }

  isConnected(): boolean {
    return this.connected && this.socket?.connected === true;
  }

  // Allow external components to register event handlers
  on(event: string, handler: Function) {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, []);
    }
    this.eventHandlers.get(event)?.push(handler);
  }

  off(event: string, handler: Function) {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  getConnectionStatus() {
    return {
      connected: this.isConnected(),
      reconnectAttempts: this.reconnectAttempts,
      serverUrl: this.serverUrl
    };
  }
}

// Create singleton instance
export const transparencyWebSocket = new TransparencyWebSocket();

// Auto-connect on import
if (typeof window !== 'undefined') {
  transparencyWebSocket.connect();
}

// Export for use in components
export default transparencyWebSocket;
