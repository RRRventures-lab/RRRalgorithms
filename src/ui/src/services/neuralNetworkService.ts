/**
 * Neural Network Service
 * Handles API calls for AI predictions and market inefficiency detection
 */

import { NeuralNetworkPrediction, MarketInefficiency, ModelPerformance } from '../store/slices/neuralNetworkSlice';

const API_BASE_URL = 'http://localhost:8000/api';

class NeuralNetworkService {
  private wsConnection: WebSocket | null = null;
  private predictionCallbacks: ((prediction: NeuralNetworkPrediction) => void)[] = [];
  private inefficiencyCallbacks: ((inefficiency: MarketInefficiency) => void)[] = [];

  /**
   * Get latest neural network predictions
   */
  async getPredictions(symbol?: string): Promise<NeuralNetworkPrediction[]> {
    const url = symbol
      ? `${API_BASE_URL}/ai/predictions?symbol=${symbol}`
      : `${API_BASE_URL}/ai/predictions`;

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('Failed to fetch predictions');
    }

    const data = await response.json();
    return data.predictions || [];
  }

  /**
   * Get prediction for specific symbol
   */
  async getPredictionForSymbol(symbol: string, model?: string): Promise<NeuralNetworkPrediction> {
    const url = model
      ? `${API_BASE_URL}/ai/predictions/${symbol}?model=${model}`
      : `${API_BASE_URL}/ai/predictions/${symbol}`;

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('Failed to fetch prediction');
    }

    return await response.json();
  }

  /**
   * Get market inefficiencies detected by AI
   */
  async getInefficiencies(limit: number = 50): Promise<MarketInefficiency[]> {
    const response = await fetch(`${API_BASE_URL}/ai/inefficiencies?limit=${limit}`);
    if (!response.ok) {
      throw new Error('Failed to fetch market inefficiencies');
    }

    const data = await response.json();
    return data.inefficiencies || [];
  }

  /**
   * Get AI model performance metrics
   */
  async getModelPerformance(): Promise<ModelPerformance[]> {
    const response = await fetch(`${API_BASE_URL}/ai/models`);
    if (!response.ok) {
      throw new Error('Failed to fetch model performance');
    }

    const data = await response.json();
    return data.models || [];
  }

  /**
   * Get AI decisions and their outcomes
   */
  async getAIDecisions(limit: number = 50, model?: string): Promise<any[]> {
    const url = model
      ? `${API_BASE_URL}/ai/decisions?limit=${limit}&model=${model}`
      : `${API_BASE_URL}/ai/decisions?limit=${limit}`;

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('Failed to fetch AI decisions');
    }

    const data = await response.json();
    return data.decisions || [];
  }

  /**
   * Subscribe to real-time neural network predictions via WebSocket
   */
  subscribeToPredictions(
    onPrediction: (prediction: NeuralNetworkPrediction) => void,
    symbols?: string[]
  ): () => void {
    this.predictionCallbacks.push(onPrediction);

    if (!this.wsConnection || this.wsConnection.readyState !== WebSocket.OPEN) {
      this.connectWebSocket(symbols);
    }

    // Return cleanup function
    return () => {
      const index = this.predictionCallbacks.indexOf(onPrediction);
      if (index > -1) {
        this.predictionCallbacks.splice(index, 1);
      }

      if (this.predictionCallbacks.length === 0 && this.inefficiencyCallbacks.length === 0) {
        this.disconnectWebSocket();
      }
    };
  }

  /**
   * Subscribe to real-time market inefficiency detections
   */
  subscribeToInefficiencies(
    onInefficiency: (inefficiency: MarketInefficiency) => void
  ): () => void {
    this.inefficiencyCallbacks.push(onInefficiency);

    if (!this.wsConnection || this.wsConnection.readyState !== WebSocket.OPEN) {
      this.connectWebSocket();
    }

    // Return cleanup function
    return () => {
      const index = this.inefficiencyCallbacks.indexOf(onInefficiency);
      if (index > -1) {
        this.inefficiencyCallbacks.splice(index, 1);
      }

      if (this.predictionCallbacks.length === 0 && this.inefficiencyCallbacks.length === 0) {
        this.disconnectWebSocket();
      }
    };
  }

  /**
   * Connect to WebSocket for real-time updates
   */
  private connectWebSocket(symbols?: string[]): void {
    const symbolParam = symbols ? `?symbols=${symbols.join(',')}` : '';
    const wsUrl = `ws://localhost:8000/ws/ai${symbolParam}`;

    this.wsConnection = new WebSocket(wsUrl);

    this.wsConnection.onopen = () => {
      console.log('Neural Network WebSocket connected');
    };

    this.wsConnection.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'prediction') {
          this.predictionCallbacks.forEach(callback => callback(data.data));
        } else if (data.type === 'inefficiency') {
          this.inefficiencyCallbacks.forEach(callback => callback(data.data));
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.wsConnection.onerror = (error) => {
      console.error('Neural Network WebSocket error:', error);
    };

    this.wsConnection.onclose = () => {
      console.log('Neural Network WebSocket closed');
      // Attempt to reconnect after 5 seconds if there are active callbacks
      if (this.predictionCallbacks.length > 0 || this.inefficiencyCallbacks.length > 0) {
        setTimeout(() => this.connectWebSocket(symbols), 5000);
      }
    };
  }

  /**
   * Disconnect WebSocket
   */
  private disconnectWebSocket(): void {
    if (this.wsConnection) {
      this.wsConnection.close();
      this.wsConnection = null;
    }
  }

  /**
   * Get prediction accuracy for a specific symbol
   */
  async getPredictionAccuracy(symbol: string, period: string = '7d'): Promise<any> {
    const response = await fetch(
      `${API_BASE_URL}/ai/accuracy/${symbol}?period=${period}`
    );
    if (!response.ok) {
      throw new Error('Failed to fetch prediction accuracy');
    }

    return await response.json();
  }

  /**
   * Get feature importance for predictions
   */
  async getFeatureImportance(model: string): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/ai/models/${model}/features`);
    if (!response.ok) {
      throw new Error('Failed to fetch feature importance');
    }

    return await response.json();
  }
}

export const neuralNetworkService = new NeuralNetworkService();
