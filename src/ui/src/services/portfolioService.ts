// Real Portfolio Service - Connects to your existing paper trading system
import { store } from '../store/store';
import { updatePortfolio } from '../store/slices/portfolioSlice';

interface Position {
  symbol: string;
  quantity: number;
  avgPrice: number;
  currentPrice: number;
  unrealizedPnl: number;
  unrealizedPnlPercent: number;
  marketValue: number;
  costBasis: number;
}

interface Portfolio {
  totalValue: number;
  cash: number;
  positions: Position[];
  totalPnl: number;
  totalPnlPercent: number;
  dayPnl: number;
  dayPnlPercent: number;
}

class PortfolioService {
  private apiBaseUrl = 'http://localhost:8000'; // Your FastAPI backend
  private updateInterval: number | null = null;
  private isConnected = false;

  constructor() {
    // Initialize with mock data if backend is not available
    this.initializeMockData();
  }

  // Connect to your existing paper trading system
  async connectToBackend() {
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/portfolio/status`);
      
      if (response.ok) {
        this.isConnected = true;
        console.log('Connected to paper trading backend');
        this.startRealTimeUpdates();
        return true;
      } else {
        throw new Error(`Backend not available: ${response.status}`);
      }
    } catch (error) {
      console.warn('Backend not available, using mock data:', error);
      this.isConnected = false;
      this.startMockUpdates();
      return false;
    }
  }

  // Get real portfolio data from your backend
  async getRealPortfolio(): Promise<Portfolio> {
    if (!this.isConnected) {
      return this.getMockPortfolio();
    }

    try {
      const response = await fetch(`${this.apiBaseUrl}/api/portfolio/positions`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Transform backend data to UI format
      const positions: Position[] = data.positions?.map((pos: any) => ({
        symbol: pos.symbol,
        quantity: pos.quantity,
        avgPrice: pos.avg_price,
        currentPrice: pos.current_price,
        unrealizedPnl: pos.unrealized_pnl,
        unrealizedPnlPercent: pos.unrealized_pnl_percent,
        marketValue: pos.market_value,
        costBasis: pos.cost_basis
      })) || [];

      const portfolio: Portfolio = {
        totalValue: data.total_value || 0,
        cash: data.cash || 0,
        positions,
        totalPnl: data.total_pnl || 0,
        totalPnlPercent: data.total_pnl_percent || 0,
        dayPnl: data.day_pnl || 0,
        dayPnlPercent: data.day_pnl_percent || 0
      };

      // Update Redux store
      store.dispatch(updatePortfolio(portfolio));
      
      return portfolio;
    } catch (error) {
      console.error('Error fetching portfolio:', error);
      return this.getMockPortfolio();
    }
  }

  // Get mock portfolio data for development
  private getMockPortfolio(): Portfolio {
    const positions: Position[] = [
      {
        symbol: 'BTC-USD',
        quantity: 0.5,
        avgPrice: 42000,
        currentPrice: 45234.12,
        unrealizedPnl: 1617.06,
        unrealizedPnlPercent: 7.7,
        marketValue: 22617.06,
        costBasis: 21000
      },
      {
        symbol: 'ETH-USD',
        quantity: 2.0,
        avgPrice: 2600,
        currentPrice: 2812.45,
        unrealizedPnl: 424.90,
        unrealizedPnlPercent: 8.2,
        marketValue: 5624.90,
        costBasis: 5200
      },
      {
        symbol: 'SOL-USD',
        quantity: 50,
        avgPrice: 95,
        currentPrice: 102.34,
        unrealizedPnl: 367.00,
        unrealizedPnlPercent: 7.7,
        marketValue: 5117.00,
        costBasis: 4750
      }
    ];

    const totalMarketValue = positions.reduce((sum, pos) => sum + pos.marketValue, 0);
    const totalCostBasis = positions.reduce((sum, pos) => sum + pos.costBasis, 0);
    const totalPnl = totalMarketValue - totalCostBasis;
    const totalPnlPercent = (totalPnl / totalCostBasis) * 100;

    const portfolio: Portfolio = {
      totalValue: totalMarketValue + 50000, // $50k cash
      cash: 50000,
      positions,
      totalPnl,
      totalPnlPercent,
      dayPnl: totalPnl * 0.1, // Mock day P&L
      dayPnlPercent: totalPnlPercent * 0.1
    };

    // Update Redux store
    store.dispatch(updatePortfolio(portfolio));
    
    return portfolio;
  }

  // Initialize with mock data
  private initializeMockData() {
    this.getMockPortfolio();
  }

  // Start real-time updates from backend
  private startRealTimeUpdates() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }

    // Update portfolio every 5 seconds
    this.updateInterval = window.setInterval(() => {
      this.getRealPortfolio();
    }, 5000);
  }

  // Start mock updates for development
  private startMockUpdates() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }

    // Update mock portfolio every 10 seconds
    this.updateInterval = window.setInterval(() => {
      this.updateMockPortfolio();
    }, 10000);
  }

  // Update mock portfolio with realistic changes
  private updateMockPortfolio() {
    const state = store.getState();
    const currentPortfolio = state.portfolio;

    if (currentPortfolio.positions) {
      const updatedPositions = currentPortfolio.positions.map(position => {
        // Simulate price changes
        const priceChange = (Math.random() - 0.5) * position.currentPrice * 0.02; // ±2%
        const newPrice = position.currentPrice + priceChange;
        const newMarketValue = position.quantity * newPrice;
        const newUnrealizedPnl = newMarketValue - position.costBasis;
        const newUnrealizedPnlPercent = (newUnrealizedPnl / position.costBasis) * 100;

        return {
          ...position,
          currentPrice: newPrice,
          marketValue: newMarketValue,
          unrealizedPnl: newUnrealizedPnl,
          unrealizedPnlPercent: newUnrealizedPnlPercent
        };
      });

      const totalMarketValue = updatedPositions.reduce((sum, pos) => sum + pos.marketValue, 0);
      const totalCostBasis = updatedPositions.reduce((sum, pos) => sum + pos.costBasis, 0);
      const totalPnl = totalMarketValue - totalCostBasis;
      const totalPnlPercent = (totalPnl / totalCostBasis) * 100;

      const updatedPortfolio: Portfolio = {
        ...currentPortfolio,
        totalValue: totalMarketValue + currentPortfolio.cash,
        positions: updatedPositions,
        totalPnl,
        totalPnlPercent,
        dayPnl: totalPnl * 0.1,
        dayPnlPercent: totalPnlPercent * 0.1
      };

      store.dispatch(updatePortfolio(updatedPortfolio));
    }
  }

  // Execute a trade (connect to your trading engine)
  async executeTrade(symbol: string, side: 'buy' | 'sell', quantity: number, price?: number) {
    if (!this.isConnected) {
      console.log('Mock trade execution:', { symbol, side, quantity, price });
      return this.executeMockTrade(symbol, side, quantity, price);
    }

    try {
      const response = await fetch(`${this.apiBaseUrl}/api/trading/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbol,
          side,
          quantity,
          price,
          type: price ? 'limit' : 'market'
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log('Trade executed:', result);
      
      // Refresh portfolio after trade
      await this.getRealPortfolio();
      
      return result;
    } catch (error) {
      console.error('Error executing trade:', error);
      return this.executeMockTrade(symbol, side, quantity, price);
    }
  }

  // Execute mock trade for development
  private async executeMockTrade(symbol: string, side: 'buy' | 'sell', quantity: number, price?: number) {
    // Simulate trade execution delay
    await new Promise(resolve => setTimeout(resolve, 1000));

    const mockResult = {
      id: `mock_${Date.now()}`,
      symbol,
      side,
      quantity,
      price: price || 45000, // Mock price
      status: 'filled',
      timestamp: Date.now(),
      fees: quantity * (price || 45000) * 0.001 // 0.1% fee
    };

    console.log('Mock trade executed:', mockResult);
    
    // Update portfolio after mock trade
    this.updatePortfolioAfterTrade(mockResult);
    
    return mockResult;
  }

  // Update portfolio after trade execution
  private updatePortfolioAfterTrade(trade: any) {
    const state = store.getState();
    const currentPortfolio = state.portfolio;
    
    // Find existing position or create new one
    const existingPositionIndex = currentPortfolio.positions?.findIndex(
      pos => pos.symbol === trade.symbol
    ) ?? -1;

    let updatedPositions = [...(currentPortfolio.positions || [])];

    if (existingPositionIndex >= 0) {
      // Update existing position
      const existingPosition = updatedPositions[existingPositionIndex];
      const newQuantity = trade.side === 'buy' 
        ? existingPosition.quantity + trade.quantity
        : existingPosition.quantity - trade.quantity;

      if (newQuantity <= 0) {
        // Remove position if quantity is 0 or negative
        updatedPositions.splice(existingPositionIndex, 1);
      } else {
        // Update position
        const totalCost = existingPosition.costBasis + (trade.quantity * trade.price);
        const newAvgPrice = totalCost / newQuantity;
        
        updatedPositions[existingPositionIndex] = {
          ...existingPosition,
          quantity: newQuantity,
          avgPrice: newAvgPrice,
          costBasis: totalCost,
          marketValue: newQuantity * existingPosition.currentPrice,
          unrealizedPnl: (newQuantity * existingPosition.currentPrice) - totalCost,
          unrealizedPnlPercent: ((newQuantity * existingPosition.currentPrice) - totalCost) / totalCost * 100
        };
      }
    } else if (trade.side === 'buy') {
      // Add new position
      const newPosition: Position = {
        symbol: trade.symbol,
        quantity: trade.quantity,
        avgPrice: trade.price,
        currentPrice: trade.price,
        unrealizedPnl: 0,
        unrealizedPnlPercent: 0,
        marketValue: trade.quantity * trade.price,
        costBasis: trade.quantity * trade.price
      };
      
      updatedPositions.push(newPosition);
    }

    // Update cash
    const cashChange = trade.side === 'buy' 
      ? -(trade.quantity * trade.price + trade.fees)
      : (trade.quantity * trade.price - trade.fees);
    
    const newCash = currentPortfolio.cash + cashChange;
    const totalMarketValue = updatedPositions.reduce((sum, pos) => sum + pos.marketValue, 0);
    const totalCostBasis = updatedPositions.reduce((sum, pos) => sum + pos.costBasis, 0);
    const totalPnl = totalMarketValue - totalCostBasis;
    const totalPnlPercent = (totalPnl / totalCostBasis) * 100;

    const updatedPortfolio: Portfolio = {
      ...currentPortfolio,
      cash: newCash,
      totalValue: totalMarketValue + newCash,
      positions: updatedPositions,
      totalPnl,
      totalPnlPercent,
      dayPnl: totalPnl * 0.1,
      dayPnlPercent: totalPnlPercent * 0.1
    };

    store.dispatch(updatePortfolio(updatedPortfolio));
  }

  // Get trading history
  async getTradingHistory(limit: number = 50) {
    if (!this.isConnected) {
      return this.getMockTradingHistory(limit);
    }

    try {
      const response = await fetch(`${this.apiBaseUrl}/api/trading/history?limit=${limit}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data.trades || [];
    } catch (error) {
      console.error('Error fetching trading history:', error);
      return this.getMockTradingHistory(limit);
    }
  }

  // Get mock trading history
  private getMockTradingHistory(limit: number) {
    const trades = [];
    const symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD'];
    const sides = ['buy', 'sell'];
    
    for (let i = 0; i < limit; i++) {
      const symbol = symbols[Math.floor(Math.random() * symbols.length)];
      const side = sides[Math.floor(Math.random() * sides.length)];
      const quantity = Math.random() * 2 + 0.1;
      const price = 40000 + Math.random() * 10000;
      
      trades.push({
        id: `mock_${Date.now()}_${i}`,
        symbol,
        side,
        quantity,
        price,
        status: 'filled',
        timestamp: Date.now() - (i * 60 * 60 * 1000), // Spread over hours
        fees: quantity * price * 0.001
      });
    }
    
    return trades.sort((a, b) => b.timestamp - a.timestamp);
  }

  // Disconnect and cleanup
  disconnect() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
    this.isConnected = false;
  }

  // Get connection status
  getConnectionStatus() {
    return {
      isConnected: this.isConnected,
      apiBaseUrl: this.apiBaseUrl
    };
  }
}

export const portfolioService = new PortfolioService();
