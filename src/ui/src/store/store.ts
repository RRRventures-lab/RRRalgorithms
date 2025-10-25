import { configureStore } from '@reduxjs/toolkit';
import marketDataReducer from './slices/marketDataSlice';
import portfolioReducer from './slices/portfolioSlice';
import systemReducer from './slices/systemSlice';
import backtestReducer from './slices/backtestSlice';
import strategyReducer from './slices/strategySlice';
import alertReducer from './slices/alertSlice';
import neuralNetworkReducer from './slices/neuralNetworkSlice';

export const store = configureStore({
  reducer: {
    marketData: marketDataReducer,
    portfolio: portfolioReducer,
    system: systemReducer,
    backtest: backtestReducer,
    strategy: strategyReducer,
    alert: alertReducer,
    neuralNetwork: neuralNetworkReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: [
          'marketData/updatePrice',
          'portfolio/updatePosition',
          'neuralNetwork/addPrediction',
          'alert/addNotification',
        ],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
