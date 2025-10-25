import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';

// Types
export interface StrategyCondition {
  id: string;
  type: 'indicator' | 'price' | 'volume' | 'time';
  indicator?: string;
  operator: '>' | '<' | '=' | '>=' | '<=' | 'crosses_above' | 'crosses_below';
  value: number | string;
  timeframe?: string;
}

export interface StrategyAction {
  id: string;
  type: 'buy' | 'sell' | 'close_long' | 'close_short';
  size_type: 'fixed' | 'percent_equity' | 'percent_position';
  size_value: number;
}

export interface StrategyRule {
  id: string;
  name: string;
  conditions: StrategyCondition[];
  logic: 'and' | 'or';
  actions: StrategyAction[];
  enabled: boolean;
}

export interface Strategy {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  type: 'momentum' | 'mean_reversion' | 'breakout' | 'arbitrage' | 'custom';
  status: 'draft' | 'active' | 'paused' | 'archived';
  entry_rules: StrategyRule[];
  exit_rules: StrategyRule[];
  risk_management: {
    max_position_size: number;
    max_portfolio_risk: number;
    stop_loss_percent?: number;
    take_profit_percent?: number;
    trailing_stop?: boolean;
    max_drawdown_stop?: number;
  };
  parameters: Record<string, any>;
  performance?: {
    total_trades: number;
    win_rate: number;
    avg_return: number;
    sharpe_ratio: number;
    last_updated: string;
  };
}

export interface StrategyState {
  strategies: Strategy[];
  selectedStrategy: Strategy | null;
  builderStrategy: Strategy | null;
  loading: boolean;
  error: string | null;
  availableIndicators: string[];
}

const initialState: StrategyState = {
  strategies: [],
  selectedStrategy: null,
  builderStrategy: null,
  loading: false,
  error: null,
  availableIndicators: [
    'RSI',
    'MACD',
    'SMA',
    'EMA',
    'Bollinger Bands',
    'ATR',
    'Stochastic',
    'ADX',
    'OBV',
    'VWAP',
  ],
};

// Async thunks
export const fetchStrategies = createAsyncThunk(
  'strategy/fetchStrategies',
  async () => {
    const response = await fetch('http://localhost:8000/api/strategies');
    if (!response.ok) throw new Error('Failed to fetch strategies');
    return await response.json();
  }
);

export const saveStrategy = createAsyncThunk(
  'strategy/saveStrategy',
  async (strategy: Partial<Strategy>) => {
    const method = strategy.id ? 'PUT' : 'POST';
    const url = strategy.id
      ? `http://localhost:8000/api/strategies/${strategy.id}`
      : 'http://localhost:8000/api/strategies';

    const response = await fetch(url, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(strategy),
    });
    if (!response.ok) throw new Error('Failed to save strategy');
    return await response.json();
  }
);

export const deleteStrategy = createAsyncThunk(
  'strategy/deleteStrategy',
  async (strategyId: string) => {
    const response = await fetch(`http://localhost:8000/api/strategies/${strategyId}`, {
      method: 'DELETE',
    });
    if (!response.ok) throw new Error('Failed to delete strategy');
    return strategyId;
  }
);

export const activateStrategy = createAsyncThunk(
  'strategy/activateStrategy',
  async (strategyId: string) => {
    const response = await fetch(`http://localhost:8000/api/strategies/${strategyId}/activate`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to activate strategy');
    return await response.json();
  }
);

const strategySlice = createSlice({
  name: 'strategy',
  initialState,
  reducers: {
    selectStrategy: (state, action: PayloadAction<Strategy | null>) => {
      state.selectedStrategy = action.payload;
    },
    createNewStrategy: (state) => {
      state.builderStrategy = {
        id: '',
        name: 'New Strategy',
        description: '',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        type: 'custom',
        status: 'draft',
        entry_rules: [],
        exit_rules: [],
        risk_management: {
          max_position_size: 10,
          max_portfolio_risk: 2,
        },
        parameters: {},
      };
    },
    updateBuilderStrategy: (state, action: PayloadAction<Partial<Strategy>>) => {
      if (state.builderStrategy) {
        state.builderStrategy = { ...state.builderStrategy, ...action.payload };
      }
    },
    addEntryRule: (state, action: PayloadAction<StrategyRule>) => {
      if (state.builderStrategy) {
        state.builderStrategy.entry_rules.push(action.payload);
      }
    },
    addExitRule: (state, action: PayloadAction<StrategyRule>) => {
      if (state.builderStrategy) {
        state.builderStrategy.exit_rules.push(action.payload);
      }
    },
    updateRule: (state, action: PayloadAction<{ type: 'entry' | 'exit'; rule: StrategyRule }>) => {
      if (state.builderStrategy) {
        const rules = action.payload.type === 'entry'
          ? state.builderStrategy.entry_rules
          : state.builderStrategy.exit_rules;
        const index = rules.findIndex(r => r.id === action.payload.rule.id);
        if (index !== -1) {
          rules[index] = action.payload.rule;
        }
      }
    },
    removeRule: (state, action: PayloadAction<{ type: 'entry' | 'exit'; ruleId: string }>) => {
      if (state.builderStrategy) {
        if (action.payload.type === 'entry') {
          state.builderStrategy.entry_rules = state.builderStrategy.entry_rules.filter(
            r => r.id !== action.payload.ruleId
          );
        } else {
          state.builderStrategy.exit_rules = state.builderStrategy.exit_rules.filter(
            r => r.id !== action.payload.ruleId
          );
        }
      }
    },
    clearBuilder: (state) => {
      state.builderStrategy = null;
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    // Fetch strategies
    builder.addCase(fetchStrategies.pending, (state) => {
      state.loading = true;
      state.error = null;
    });
    builder.addCase(fetchStrategies.fulfilled, (state, action) => {
      state.loading = false;
      state.strategies = action.payload;
    });
    builder.addCase(fetchStrategies.rejected, (state, action) => {
      state.loading = false;
      state.error = action.error.message || 'Failed to fetch strategies';
    });

    // Save strategy
    builder.addCase(saveStrategy.pending, (state) => {
      state.loading = true;
      state.error = null;
    });
    builder.addCase(saveStrategy.fulfilled, (state, action) => {
      state.loading = false;
      const index = state.strategies.findIndex(s => s.id === action.payload.id);
      if (index !== -1) {
        state.strategies[index] = action.payload;
      } else {
        state.strategies.push(action.payload);
      }
      state.builderStrategy = null;
    });
    builder.addCase(saveStrategy.rejected, (state, action) => {
      state.loading = false;
      state.error = action.error.message || 'Failed to save strategy';
    });

    // Delete strategy
    builder.addCase(deleteStrategy.fulfilled, (state, action) => {
      state.strategies = state.strategies.filter(s => s.id !== action.payload);
      if (state.selectedStrategy?.id === action.payload) {
        state.selectedStrategy = null;
      }
    });

    // Activate strategy
    builder.addCase(activateStrategy.fulfilled, (state, action) => {
      const index = state.strategies.findIndex(s => s.id === action.payload.id);
      if (index !== -1) {
        state.strategies[index] = action.payload;
      }
    });
  },
});

export const {
  selectStrategy,
  createNewStrategy,
  updateBuilderStrategy,
  addEntryRule,
  addExitRule,
  updateRule,
  removeRule,
  clearBuilder,
  clearError,
} = strategySlice.actions;

export default strategySlice.reducer;
