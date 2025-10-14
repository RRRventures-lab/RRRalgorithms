import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface PriceData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  timestamp: number;
}

export interface OrderBookEntry {
  price: number;
  size: number;
}

export interface MarketDataState {
  prices: Record<string, PriceData>;
  orderBook: {
    bids: OrderBookEntry[];
    asks: OrderBookEntry[];
  };
  isLoading: boolean;
  lastUpdate: number;
}

const initialState: MarketDataState = {
  prices: {
    'BTC-USD': {
      symbol: 'BTC-USD',
      price: 45234.12,
      change: 1234.56,
      changePercent: 2.81,
      volume: 1234567,
      timestamp: Date.now(),
    },
    'ETH-USD': {
      symbol: 'ETH-USD',
      price: 2812.45,
      change: -45.67,
      changePercent: -1.60,
      volume: 987654,
      timestamp: Date.now(),
    },
    'SOL-USD': {
      symbol: 'SOL-USD',
      price: 102.34,
      change: 0.00,
      changePercent: 0.00,
      volume: 543210,
      timestamp: Date.now(),
    },
  },
  orderBook: {
    bids: [
      { price: 45230, size: 0.5 },
      { price: 45225, size: 1.2 },
      { price: 45220, size: 0.8 },
    ],
    asks: [
      { price: 45235, size: 0.3 },
      { price: 45240, size: 1.1 },
      { price: 45245, size: 0.7 },
    ],
  },
  isLoading: false,
  lastUpdate: Date.now(),
};

const marketDataSlice = createSlice({
  name: 'marketData',
  initialState,
  reducers: {
    updatePrice: (state, action: PayloadAction<PriceData>) => {
      const { symbol } = action.payload;
      state.prices[symbol] = action.payload;
      state.lastUpdate = Date.now();
    },
    updateOrderBook: (state, action: PayloadAction<{ bids: OrderBookEntry[]; asks: OrderBookEntry[] }>) => {
      state.orderBook = action.payload;
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
  },
});

export const { updatePrice, updateOrderBook, setLoading } = marketDataSlice.actions;
export default marketDataSlice.reducer;
