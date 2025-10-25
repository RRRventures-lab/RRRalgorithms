import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';

// Types
export interface BacktestTrade {
  timestamp: string;
  symbol: string;
  side: 'buy' | 'sell';
  price: number;
  quantity: number;
  pnl: number;
  commission: number;
}

export interface MonteCarloResult {
  simulation_id: number;
  final_equity: number;
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
}

export interface BacktestResult {
  id: string;
  name: string;
  description: string;
  created_at: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress?: number;
  period: {
    start: string;
    end: string;
    days: number;
  };
  performance: {
    initial_capital: number;
    final_equity: number;
    total_return: number;
    total_return_percent: number;
    sharpe_ratio: number;
    sortino_ratio: number;
    max_drawdown: number;
    max_drawdown_percent: number;
    calmar_ratio: number;
    win_rate: number;
    profit_factor: number;
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    avg_win: number;
    avg_loss: number;
    largest_win: number;
    largest_loss: number;
  };
  equity_curve?: Array<{ timestamp: string; equity: number }>;
  drawdown_curve?: Array<{ timestamp: string; drawdown: number }>;
  trades?: BacktestTrade[];
  monte_carlo?: {
    simulations: number;
    results: MonteCarloResult[];
    statistics: {
      mean_return: number;
      median_return: number;
      std_deviation: number;
      percentile_5: number;
      percentile_95: number;
      probability_of_profit: number;
      value_at_risk_95: number;
    };
  };
  parameter_sensitivity?: {
    parameter_name: string;
    values: number[];
    returns: number[];
    sharpe_ratios: number[];
  }[];
}

export interface BacktestState {
  backtests: BacktestResult[];
  selectedBacktest: BacktestResult | null;
  loading: boolean;
  error: string | null;
  comparisonBacktests: BacktestResult[];
}

const initialState: BacktestState = {
  backtests: [],
  selectedBacktest: null,
  loading: false,
  error: null,
  comparisonBacktests: [],
};

// Async thunks
export const fetchBacktests = createAsyncThunk(
  'backtest/fetchBacktests',
  async (limit: number = 20) => {
    const response = await fetch(`http://localhost:8000/api/backtests?limit=${limit}`);
    if (!response.ok) throw new Error('Failed to fetch backtests');
    const data = await response.json();
    return data.backtests;
  }
);

export const fetchBacktestDetail = createAsyncThunk(
  'backtest/fetchBacktestDetail',
  async (backtestId: string) => {
    const response = await fetch(`http://localhost:8000/api/backtests/${backtestId}`);
    if (!response.ok) throw new Error('Failed to fetch backtest details');
    return await response.json();
  }
);

export const runBacktest = createAsyncThunk(
  'backtest/runBacktest',
  async (config: {
    name: string;
    strategy: string;
    parameters: Record<string, any>;
    start_date: string;
    end_date: string;
    initial_capital: number;
    monte_carlo_simulations?: number;
  }) => {
    const response = await fetch('http://localhost:8000/api/backtests', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (!response.ok) throw new Error('Failed to start backtest');
    return await response.json();
  }
);

const backtestSlice = createSlice({
  name: 'backtest',
  initialState,
  reducers: {
    selectBacktest: (state, action: PayloadAction<BacktestResult | null>) => {
      state.selectedBacktest = action.payload;
    },
    addToComparison: (state, action: PayloadAction<BacktestResult>) => {
      if (!state.comparisonBacktests.find(bt => bt.id === action.payload.id)) {
        state.comparisonBacktests.push(action.payload);
      }
    },
    removeFromComparison: (state, action: PayloadAction<string>) => {
      state.comparisonBacktests = state.comparisonBacktests.filter(
        bt => bt.id !== action.payload
      );
    },
    clearComparison: (state) => {
      state.comparisonBacktests = [];
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    // Fetch backtests
    builder.addCase(fetchBacktests.pending, (state) => {
      state.loading = true;
      state.error = null;
    });
    builder.addCase(fetchBacktests.fulfilled, (state, action) => {
      state.loading = false;
      state.backtests = action.payload;
    });
    builder.addCase(fetchBacktests.rejected, (state, action) => {
      state.loading = false;
      state.error = action.error.message || 'Failed to fetch backtests';
    });

    // Fetch backtest detail
    builder.addCase(fetchBacktestDetail.pending, (state) => {
      state.loading = true;
      state.error = null;
    });
    builder.addCase(fetchBacktestDetail.fulfilled, (state, action) => {
      state.loading = false;
      state.selectedBacktest = action.payload;
      // Update in list if exists
      const index = state.backtests.findIndex(bt => bt.id === action.payload.id);
      if (index !== -1) {
        state.backtests[index] = action.payload;
      }
    });
    builder.addCase(fetchBacktestDetail.rejected, (state, action) => {
      state.loading = false;
      state.error = action.error.message || 'Failed to fetch backtest details';
    });

    // Run backtest
    builder.addCase(runBacktest.pending, (state) => {
      state.loading = true;
      state.error = null;
    });
    builder.addCase(runBacktest.fulfilled, (state, action) => {
      state.loading = false;
      state.backtests.unshift(action.payload);
      state.selectedBacktest = action.payload;
    });
    builder.addCase(runBacktest.rejected, (state, action) => {
      state.loading = false;
      state.error = action.error.message || 'Failed to run backtest';
    });
  },
});

export const {
  selectBacktest,
  addToComparison,
  removeFromComparison,
  clearComparison,
  clearError,
} = backtestSlice.actions;

export default backtestSlice.reducer;
