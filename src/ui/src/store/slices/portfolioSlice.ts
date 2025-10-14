import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface Position {
  symbol: string;
  quantity: number;
  avgPrice: number;
  currentPrice: number;
  unrealizedPnl: number;
  unrealizedPnlPercent: number;
  marketValue: number;
  costBasis: number;
}

export interface PortfolioState {
  totalValue: number;
  cash: number;
  totalPnl: number;
  totalPnlPercent: number;
  positions: Position[];
  isLoading: boolean;
}

const initialState: PortfolioState = {
  totalValue: 100000,
  cash: 50000,
  totalPnl: 1234.56,
  totalPnlPercent: 1.23,
  positions: [
    {
      symbol: 'BTC-USD',
      quantity: 0.5,
      avgPrice: 44000,
      currentPrice: 45234.12,
      unrealizedPnl: 617.06,
      unrealizedPnlPercent: 2.81,
      marketValue: 22617.06,
      costBasis: 22000,
    },
    {
      symbol: 'ETH-USD',
      quantity: 2.0,
      avgPrice: 2800,
      currentPrice: 2812.45,
      unrealizedPnl: 24.90,
      unrealizedPnlPercent: 0.44,
      marketValue: 5624.90,
      costBasis: 5600,
    },
  ],
  isLoading: false,
};

const portfolioSlice = createSlice({
  name: 'portfolio',
  initialState,
  reducers: {
    updatePosition: (state, action: PayloadAction<Position>) => {
      const index = state.positions.findIndex(p => p.symbol === action.payload.symbol);
      if (index >= 0) {
        state.positions[index] = action.payload;
      } else {
        state.positions.push(action.payload);
      }
      
      // Recalculate totals
      state.totalValue = state.cash + state.positions.reduce((sum, pos) => sum + pos.marketValue, 0);
      state.totalPnl = state.positions.reduce((sum, pos) => sum + pos.unrealizedPnl, 0);
      state.totalPnlPercent = (state.totalPnl / (state.totalValue - state.totalPnl)) * 100;
    },
    updateCash: (state, action: PayloadAction<number>) => {
      state.cash = action.payload;
      state.totalValue = state.cash + state.positions.reduce((sum, pos) => sum + pos.marketValue, 0);
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
    updatePortfolio: (state, action: PayloadAction<Partial<PortfolioState>>) => {
      return { ...state, ...action.payload };
    },
  },
});

export const { updatePosition, updateCash, setLoading, updatePortfolio } = portfolioSlice.actions;
export default portfolioSlice.reducer;
