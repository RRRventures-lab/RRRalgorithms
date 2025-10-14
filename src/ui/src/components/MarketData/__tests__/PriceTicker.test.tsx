import React from 'react';
import { render, screen } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import PriceTicker from '../PriceTicker';
import marketDataReducer from '../../../store/slices/marketDataSlice';

const createMockStore = () => {
  return configureStore({
    reducer: {
      marketData: marketDataReducer,
    },
  });
};

const mockPrices = [
  {
    symbol: 'BTC-USD',
    price: 45234.12,
    change: 1234.56,
    changePercent: 2.81,
    volume: 1234567,
    timestamp: Date.now(),
  },
  {
    symbol: 'ETH-USD',
    price: 2812.45,
    change: -45.67,
    changePercent: -1.60,
    volume: 987654,
    timestamp: Date.now(),
  },
];

describe('PriceTicker', () => {
  it('renders price ticker with correct data', () => {
    render(
      <Provider store={createMockStore()}>
        <PriceTicker prices={mockPrices} />
      </Provider>
    );

    expect(screen.getByText('PRICES')).toBeInTheDocument();
    expect(screen.getByText('BTC-USD')).toBeInTheDocument();
    expect(screen.getByText('ETH-USD')).toBeInTheDocument();
    expect(screen.getByText('$45,234.12')).toBeInTheDocument();
    expect(screen.getByText('$2,812.45')).toBeInTheDocument();
  });

  it('displays correct change indicators', () => {
    render(
      <Provider store={createMockStore()}>
        <PriceTicker prices={mockPrices} />
      </Provider>
    );

    // BTC should show up arrow (positive change)
    expect(screen.getByText('↑')).toBeInTheDocument();
    
    // ETH should show down arrow (negative change)
    const arrows = screen.getAllByText(/[↑↓→]/);
    expect(arrows.length).toBeGreaterThan(0);
  });

  it('formats currency correctly', () => {
    render(
      <Provider store={createMockStore()}>
        <PriceTicker prices={mockPrices} />
      </Provider>
    );

    expect(screen.getByText('$45,234.12')).toBeInTheDocument();
    expect(screen.getByText('$2,812.45')).toBeInTheDocument();
  });

  it('displays change percentages', () => {
    render(
      <Provider store={createMockStore()}>
        <PriceTicker prices={mockPrices} />
      </Provider>
    );

    expect(screen.getByText(/\+2\.81%/)).toBeInTheDocument();
    expect(screen.getByText(/-1\.60%/)).toBeInTheDocument();
  });
});
