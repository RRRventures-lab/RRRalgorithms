// Real Market Data Service - Connects to your existing Polygon.io integration
import { store } from '../store/store';
import { updatePrice } from '../store/slices/marketDataSlice';

interface PriceData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  timestamp: number;
}

interface OrderBookData {
  symbol: string;
  bids: Array<{ price: number; size: number }>;
  asks: Array<{ price: number; size: number }>;
  timestamp: number;
}

class MarketDataService {
  private polygonApiKey: string;
  private baseUrl = 'https://api.polygon.io';
  private websocketUrl = 'wss://socket.polygon.io/crypto';
  private ws: WebSocket | null = null;
  private reconnectInterval: number | null = null;
  private isConnected = false;
  private subscribedSymbols: string[] = [];
  private priceCache: Map<string, PriceData> = new Map();

  constructor() {
    // Get API key from environment or config
    this.polygonApiKey = import.meta.env.VITE_POLYGON_API_KEY || '';
    
    if (!this.polygonApiKey) {
      console.warn('Polygon.io API key not found. Using mock data.');
    }
  }

  // Connect to Polygon.io WebSocket for real-time data
  async connectWebSocket(symbols: string[] = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD']) {
    if (!this.polygonApiKey) {
      console.log('Using mock data - no API key provided');
      this.startMockDataStream(symbols);
      return;
    }

    try {
      this.ws = new WebSocket(`${this.websocketUrl}?apiKey=${this.polygonApiKey}`);
      this.subscribedSymbols = symbols;

      this.ws.onopen = () => {
        console.log('Connected to Polygon.io WebSocket');
        this.isConnected = true;
        
        // Subscribe to crypto trades
        const subscribeMessage = {
          action: 'subscribe',
          params: symbols.map(symbol => `XT.${symbol}`)
        };
        
        this.ws?.send(JSON.stringify(subscribeMessage));
        console.log(`Subscribed to: ${symbols.join(', ')}`);
      };

      this.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.handleWebSocketMessage(data);
      };

      this.ws.onclose = () => {
        console.log('Polygon.io WebSocket disconnected');
        this.isConnected = false;
        this.scheduleReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('Polygon.io WebSocket error:', error);
        this.isConnected = false;
      };

    } catch (error) {
      console.error('Failed to connect to Polygon.io WebSocket:', error);
      this.startMockDataStream(symbols);
    }
  }

  private handleWebSocketMessage(data: any) {
    if (data.ev === 'XT') { // Trade event
      const trade = data;
      const symbol = trade.pair.replace('X:', '');
      
      const priceData: PriceData = {
        symbol,
        price: trade.p,
        change: 0, // Will be calculated
        changePercent: 0, // Will be calculated
        volume: trade.s,
        timestamp: trade.t
      };

      // Calculate change from previous price
      const previousPrice = this.priceCache.get(symbol);
      if (previousPrice) {
        priceData.change = priceData.price - previousPrice.price;
        priceData.changePercent = (priceData.change / previousPrice.price) * 100;
      }

      this.priceCache.set(symbol, priceData);
      store.dispatch(updatePrice({
        symbol,
        price: priceData.price,
        change: priceData.change,
        changePercent: priceData.changePercent,
        volume: priceData.volume,
        timestamp: priceData.timestamp
      }));
    }
  }

  private scheduleReconnect() {
    if (this.reconnectInterval) {
      clearTimeout(this.reconnectInterval);
    }

    this.reconnectInterval = window.setTimeout(() => {
      if (!this.isConnected) {
        console.log('Attempting to reconnect to Polygon.io...');
        this.connectWebSocket(this.subscribedSymbols);
      }
    }, 5000);
  }

  // Mock data stream for development/testing
  private startMockDataStream(symbols: string[]) {
    console.log('Starting mock data stream for:', symbols);
    
    const generateMockPrice = (symbol: string, basePrice: number) => {
      const change = (Math.random() - 0.5) * basePrice * 0.02; // ±2% change
      const newPrice = basePrice + change;
      const changePercent = (change / basePrice) * 100;
      const volume = Math.floor(Math.random() * 1000) + 100;

      return {
        symbol,
        price: newPrice,
        change,
        changePercent,
        volume,
        timestamp: Date.now()
      };
    };

    // Base prices for top 5 cryptocurrencies
    const basePrices: Record<string, number> = {
      'BTC-USD': 45234.12,
      'ETH-USD': 2812.45,
      'SOL-USD': 102.34,
      'ADA-USD': 0.45,
      'DOT-USD': 7.89
    };

    // Update prices every 2 seconds
    const interval = setInterval(() => {
      symbols.forEach(symbol => {
        const basePrice = basePrices[symbol] || 100;
        const priceData = generateMockPrice(symbol, basePrice);
        
        store.dispatch(updatePrice({
          symbol: priceData.symbol,
          price: priceData.price,
          change: priceData.change,
          changePercent: priceData.changePercent,
          volume: priceData.volume,
          timestamp: priceData.timestamp
        }));
      });
    }, 2000);

    // Store interval for cleanup
    (this as any).mockInterval = interval;
  }

  // Get historical data from Polygon.io REST API
  async getHistoricalData(symbol: string, timespan: string = 'hour', limit: number = 100) {
    if (!this.polygonApiKey) {
      console.log('No API key - returning mock historical data');
      return this.generateMockHistoricalData(symbol, limit);
    }

    try {
      const response = await fetch(
        `${this.baseUrl}/v2/aggs/ticker/X:${symbol}/range/1/${timespan}/2024-01-01/2024-12-31?adjusted=true&sort=asc&limit=${limit}&apikey=${this.polygonApiKey}`
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.results) {
        return data.results.map((bar: any) => ({
          timestamp: bar.t,
          open: bar.o,
          high: bar.h,
          low: bar.l,
          close: bar.c,
          volume: bar.v
        }));
      }

      return [];
    } catch (error) {
      console.error('Error fetching historical data:', error);
      return this.generateMockHistoricalData(symbol, limit);
    }
  }

  private generateMockHistoricalData(_symbol: string, limit: number) {
    const data = [];
    let price = 10000; // Base price
    const now = Date.now();

    for (let i = 0; i < limit; i++) {
      const timestamp = now - (limit - i) * 60 * 60 * 1000; // Hourly data
      const change = (Math.random() - 0.5) * price * 0.05; // ±5% change
      const open = price;
      const close = open + change;
      const high = Math.max(open, close) + Math.random() * price * 0.02;
      const low = Math.min(open, close) - Math.random() * price * 0.02;
      const volume = Math.floor(Math.random() * 1000) + 100;

      data.push({
        timestamp,
        open,
        high,
        low,
        close,
        volume
      });

      price = close;
    }

    return data;
  }

  // Get order book data
  async getOrderBook(_symbol: string): Promise<OrderBookData> {
    if (!this.polygonApiKey) {
      return this.generateMockOrderBook(_symbol);
    }

    try {
      const response = await fetch(
        `${this.baseUrl}/v2/snapshot/locale/global/markets/crypto/tickers/X:${_symbol}?apikey=${this.polygonApiKey}`
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.ticker?.book) {
        return {
          symbol: _symbol,
          bids: data.ticker.book.bids?.slice(0, 10) || [],
          asks: data.ticker.book.asks?.slice(0, 10) || [],
          timestamp: Date.now()
        };
      }

      return this.generateMockOrderBook(_symbol);
    } catch (error) {
      console.error('Error fetching order book:', error);
      return this.generateMockOrderBook(_symbol);
    }
  }

  private generateMockOrderBook(symbol: string): OrderBookData {
    const basePrice = 45000; // Mock base price
    const bids = [];
    const asks = [];

    for (let i = 0; i < 10; i++) {
      const bidPrice = basePrice - (i + 1) * 10;
      const askPrice = basePrice + (i + 1) * 10;
      const size = Math.random() * 10 + 1;

      bids.push({ price: bidPrice, size });
      asks.push({ price: askPrice, size });
    }

    return {
      symbol,
      bids,
      asks,
      timestamp: Date.now()
    };
  }

  // Disconnect WebSocket
  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    if (this.reconnectInterval) {
      clearTimeout(this.reconnectInterval);
      this.reconnectInterval = null;
    }

    if ((this as any).mockInterval) {
      clearInterval((this as any).mockInterval);
    }

    this.isConnected = false;
  }

  // Get current connection status
  getConnectionStatus() {
    return {
      isConnected: this.isConnected,
      subscribedSymbols: this.subscribedSymbols,
      hasApiKey: !!this.polygonApiKey
    };
  }
}

export const marketDataService = new MarketDataService();
