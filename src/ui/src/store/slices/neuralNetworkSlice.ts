import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';

// Types
export interface NeuralNetworkPrediction {
  timestamp: string;
  symbol: string;
  model: string;
  prediction: {
    price_next_1h: number;
    price_next_4h: number;
    price_next_24h: number;
    direction: 'up' | 'down' | 'neutral';
    confidence: number;
  };
  confidence_interval: {
    lower_1h: number;
    upper_1h: number;
    lower_4h: number;
    upper_4h: number;
    lower_24h: number;
    upper_24h: number;
  };
  features: {
    momentum: number;
    volatility: number;
    volume_profile: number;
    market_regime: string;
  };
}

export interface MarketInefficiency {
  timestamp: string;
  type: 'orderbook_imbalance' | 'spread_anomaly' | 'volume_divergence' | 'momentum_shift' | 'volatility_spike' | 'correlation_break';
  symbol: string;
  severity: number;
  description: string;
  confidence: number;
  opportunity_score: number;
  recommended_action?: 'buy' | 'sell' | 'hold' | 'close';
}

export interface ModelPerformance {
  model_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  sharpe_ratio: number;
  total_predictions: number;
  correct_predictions: number;
  last_updated: string;
}

export interface NeuralNetworkState {
  predictions: NeuralNetworkPrediction[];
  inefficiencies: MarketInefficiency[];
  modelPerformance: ModelPerformance[];
  selectedSymbol: string | null;
  loading: boolean;
  error: string | null;
  wsConnected: boolean;
}

const initialState: NeuralNetworkState = {
  predictions: [],
  inefficiencies: [],
  modelPerformance: [],
  selectedSymbol: null,
  loading: false,
  error: null,
  wsConnected: false,
};

// Async thunks
export const fetchPredictions = createAsyncThunk(
  'neuralNetwork/fetchPredictions',
  async (symbol?: string) => {
    const url = symbol
      ? `http://localhost:8000/api/ai/predictions?symbol=${symbol}`
      : 'http://localhost:8000/api/ai/predictions';
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch predictions');
    return await response.json();
  }
);

export const fetchInefficiencies = createAsyncThunk(
  'neuralNetwork/fetchInefficiencies',
  async () => {
    const response = await fetch('http://localhost:8000/api/ai/inefficiencies');
    if (!response.ok) throw new Error('Failed to fetch market inefficiencies');
    return await response.json();
  }
);

export const fetchModelPerformance = createAsyncThunk(
  'neuralNetwork/fetchModelPerformance',
  async () => {
    const response = await fetch('http://localhost:8000/api/ai/models');
    if (!response.ok) throw new Error('Failed to fetch model performance');
    return await response.json();
  }
);

const neuralNetworkSlice = createSlice({
  name: 'neuralNetwork',
  initialState,
  reducers: {
    addPrediction: (state, action: PayloadAction<NeuralNetworkPrediction>) => {
      // Keep only latest prediction per symbol+model
      state.predictions = state.predictions.filter(
        p => !(p.symbol === action.payload.symbol && p.model === action.payload.model)
      );
      state.predictions.unshift(action.payload);
      // Keep last 100 predictions
      if (state.predictions.length > 100) {
        state.predictions = state.predictions.slice(0, 100);
      }
    },
    addInefficiency: (state, action: PayloadAction<MarketInefficiency>) => {
      state.inefficiencies.unshift(action.payload);
      // Keep last 50 inefficiencies
      if (state.inefficiencies.length > 50) {
        state.inefficiencies = state.inefficiencies.slice(0, 50);
      }
    },
    updateModelPerformance: (state, action: PayloadAction<ModelPerformance>) => {
      const index = state.modelPerformance.findIndex(m => m.model_name === action.payload.model_name);
      if (index !== -1) {
        state.modelPerformance[index] = action.payload;
      } else {
        state.modelPerformance.push(action.payload);
      }
    },
    selectSymbol: (state, action: PayloadAction<string | null>) => {
      state.selectedSymbol = action.payload;
    },
    setWsConnected: (state, action: PayloadAction<boolean>) => {
      state.wsConnected = action.payload;
    },
    clearPredictions: (state) => {
      state.predictions = [];
    },
    clearInefficiencies: (state) => {
      state.inefficiencies = [];
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    // Fetch predictions
    builder.addCase(fetchPredictions.pending, (state) => {
      state.loading = true;
      state.error = null;
    });
    builder.addCase(fetchPredictions.fulfilled, (state, action) => {
      state.loading = false;
      state.predictions = action.payload;
    });
    builder.addCase(fetchPredictions.rejected, (state, action) => {
      state.loading = false;
      state.error = action.error.message || 'Failed to fetch predictions';
    });

    // Fetch inefficiencies
    builder.addCase(fetchInefficiencies.fulfilled, (state, action) => {
      state.inefficiencies = action.payload;
    });
    builder.addCase(fetchInefficiencies.rejected, (state, action) => {
      state.error = action.error.message || 'Failed to fetch inefficiencies';
    });

    // Fetch model performance
    builder.addCase(fetchModelPerformance.fulfilled, (state, action) => {
      state.modelPerformance = action.payload;
    });
    builder.addCase(fetchModelPerformance.rejected, (state, action) => {
      state.error = action.error.message || 'Failed to fetch model performance';
    });
  },
});

export const {
  addPrediction,
  addInefficiency,
  updateModelPerformance,
  selectSymbol,
  setWsConnected,
  clearPredictions,
  clearInefficiencies,
  clearError,
} = neuralNetworkSlice.actions;

export default neuralNetworkSlice.reducer;
