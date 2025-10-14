import { configureStore } from '@reduxjs/toolkit';
import marketDataReducer from './slices/marketDataSlice';
import portfolioReducer from './slices/portfolioSlice';
import systemReducer from './slices/systemSlice';

export const store = configureStore({
  reducer: {
    marketData: marketDataReducer,
    portfolio: portfolioReducer,
    system: systemReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['marketData/updatePrice', 'portfolio/updatePosition'],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
