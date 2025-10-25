/**
 * Backtest Service
 * Handles API calls for backtesting operations
 */

const API_BASE_URL = 'http://localhost:8000/api';

export interface BacktestConfig {
  name: string;
  description?: string;
  strategy: string;
  parameters: Record<string, any>;
  start_date: string;
  end_date: string;
  initial_capital: number;
  symbols?: string[];
  monte_carlo_simulations?: number;
  parameter_sensitivity?: {
    parameter_name: string;
    min_value: number;
    max_value: number;
    step: number;
  }[];
}

export interface BacktestProgress {
  backtest_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  current_date?: string;
  estimated_completion?: string;
}

class BacktestService {
  private wsConnection: WebSocket | null = null;

  /**
   * Get list of all backtests
   */
  async getBacktests(limit: number = 20): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/backtests?limit=${limit}`);
    if (!response.ok) {
      throw new Error('Failed to fetch backtests');
    }
    return await response.json();
  }

  /**
   * Get detailed backtest results
   */
  async getBacktestDetail(backtestId: string): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/backtests/${backtestId}`);
    if (!response.ok) {
      throw new Error('Failed to fetch backtest details');
    }
    return await response.json();
  }

  /**
   * Run a new backtest
   */
  async runBacktest(config: BacktestConfig): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/backtests`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to start backtest');
    }

    return await response.json();
  }

  /**
   * Get backtest progress via WebSocket
   */
  subscribeToBacktestProgress(
    backtestId: string,
    onProgress: (progress: BacktestProgress) => void,
    onError: (error: Error) => void
  ): () => void {
    const wsUrl = `ws://localhost:8000/ws/backtest/${backtestId}`;
    this.wsConnection = new WebSocket(wsUrl);

    this.wsConnection.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onProgress(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.wsConnection.onerror = (event) => {
      onError(new Error('WebSocket error occurred'));
    };

    this.wsConnection.onclose = () => {
      console.log('Backtest progress WebSocket closed');
    };

    // Return cleanup function
    return () => {
      if (this.wsConnection) {
        this.wsConnection.close();
        this.wsConnection = null;
      }
    };
  }

  /**
   * Cancel a running backtest
   */
  async cancelBacktest(backtestId: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/backtests/${backtestId}/cancel`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error('Failed to cancel backtest');
    }
  }

  /**
   * Delete a backtest
   */
  async deleteBacktest(backtestId: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/backtests/${backtestId}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      throw new Error('Failed to delete backtest');
    }
  }

  /**
   * Export backtest results as PDF
   */
  async exportBacktestPDF(backtestId: string): Promise<Blob> {
    const response = await fetch(`${API_BASE_URL}/backtests/${backtestId}/export/pdf`);

    if (!response.ok) {
      throw new Error('Failed to export backtest PDF');
    }

    return await response.blob();
  }

  /**
   * Export backtest results as CSV
   */
  async exportBacktestCSV(backtestId: string): Promise<Blob> {
    const response = await fetch(`${API_BASE_URL}/backtests/${backtestId}/export/csv`);

    if (!response.ok) {
      throw new Error('Failed to export backtest CSV');
    }

    return await response.blob();
  }

  /**
   * Compare multiple backtests
   */
  async compareBacktests(backtestIds: string[]): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/backtests/compare`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ backtest_ids: backtestIds }),
    });

    if (!response.ok) {
      throw new Error('Failed to compare backtests');
    }

    return await response.json();
  }

  /**
   * Get available strategies for backtesting
   */
  async getAvailableStrategies(): Promise<string[]> {
    const response = await fetch(`${API_BASE_URL}/backtests/strategies`);

    if (!response.ok) {
      throw new Error('Failed to fetch available strategies');
    }

    const data = await response.json();
    return data.strategies;
  }
}

export const backtestService = new BacktestService();
