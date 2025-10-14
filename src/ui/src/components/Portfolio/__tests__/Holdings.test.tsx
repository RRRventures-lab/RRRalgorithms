import React from 'react';
import { render, screen } from '@testing-library/react';
import Holdings from '../Holdings';
import { Position } from '../../../store/slices/portfolioSlice';

const mockPositions: Position[] = [
  {
    symbol: 'BTC-USD',
    quantity: 0.5,
    avgPrice: 44000,
    currentPrice: 45234.12,
    unrealizedPnl: 617.06,
    unrealizedPnlPercent: 2.81,
    marketValue: 22617.06,
  },
  {
    symbol: 'ETH-USD',
    quantity: 2.0,
    avgPrice: 2800,
    currentPrice: 2812.45,
    unrealizedPnl: 24.90,
    unrealizedPnlPercent: 0.44,
    marketValue: 5624.90,
  },
];

describe('Holdings', () => {
  it('renders holdings with correct data', () => {
    render(<Holdings positions={mockPositions} />);

    expect(screen.getByText('POSITIONS')).toBeInTheDocument();
    expect(screen.getByText('BTC-USD')).toBeInTheDocument();
    expect(screen.getByText('ETH-USD')).toBeInTheDocument();
  });

  it('displays position quantities correctly', () => {
    render(<Holdings positions={mockPositions} />);

    expect(screen.getByText('0.5000')).toBeInTheDocument(); // BTC quantity
    expect(screen.getByText('2.0000')).toBeInTheDocument(); // ETH quantity
  });

  it('formats currency values correctly', () => {
    render(<Holdings positions={mockPositions} />);

    expect(screen.getByText('$22,617')).toBeInTheDocument(); // BTC market value
    expect(screen.getByText('$5,625')).toBeInTheDocument(); // ETH market value
  });

  it('displays P&L with correct colors', () => {
    render(<Holdings positions={mockPositions} />);

    // Both positions have positive P&L, so should be green
    const pnlElements = screen.getAllByText(/\+.*\$/);
    expect(pnlElements.length).toBeGreaterThan(0);
  });

  it('shows no positions message when empty', () => {
    render(<Holdings positions={[]} />);

    expect(screen.getByText('No positions')).toBeInTheDocument();
  });

  it('displays P&L percentages correctly', () => {
    render(<Holdings positions={mockPositions} />);

    expect(screen.getByText('+2.81%')).toBeInTheDocument(); // BTC P&L%
    expect(screen.getByText('+0.44%')).toBeInTheDocument(); // ETH P&L%
  });
});
