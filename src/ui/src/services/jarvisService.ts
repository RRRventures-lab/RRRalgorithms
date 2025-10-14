// Jarvis AI Service - Local LLM integration with Ollama
import { store } from '../store/store';
import { marketDataService } from './marketDataService';
import { portfolioService } from './portfolioService';

interface JarvisContext {
  user: string;
  system: string;
  capabilities: string[];
  currentState: {
    portfolio: any;
    marketData: any;
    systemHealth: any;
  };
  conversationHistory: Array<{
    role: 'user' | 'assistant';
    content: string;
    timestamp: number;
  }>;
}

interface ToolCall {
  name: string;
  parameters: any;
  result?: any;
}

class JarvisService {
  private ollamaUrl = 'http://localhost:11434';
  private context: JarvisContext;
  private isProcessing = false;
  private maxHistoryLength = 10;

  constructor() {
    this.context = {
      user: 'Trading Professional',
      system: 'RRRalgorithms Trading Terminal',
      capabilities: [
        'Real-time market analysis',
        'Portfolio management',
        'Trade execution',
        'Risk assessment',
        'Market news analysis',
        'Technical analysis',
        'Performance tracking'
      ],
      currentState: {
        portfolio: null,
        marketData: null,
        systemHealth: null
      },
      conversationHistory: []
    };

    this.initializeContext();
  }

  private async initializeContext() {
    // Get initial system state
    try {
      const portfolio = await portfolioService.getRealPortfolio();
      const marketStatus = marketDataService.getConnectionStatus();
      
      this.context.currentState = {
        portfolio,
        marketData: marketStatus,
        systemHealth: {
          status: 'operational',
          uptime: Date.now(),
          lastUpdate: Date.now()
        }
      };
    } catch (error) {
      console.error('Error initializing Jarvis context:', error);
    }
  }

  // Process text message with Jarvis
  async processTextCommand(userInput: string): Promise<string> {
    if (this.isProcessing) {
      return "I'm still processing your previous request. Please wait a moment.";
    }

    this.isProcessing = true;

    try {
      // Add user message to history
      this.addToHistory('user', userInput);

      // Check if this is a tool call request
      const toolCall = this.parseToolCall(userInput);
      if (toolCall) {
        const result = await this.executeToolCall(toolCall);
        const response = await this.generateResponseWithToolResult(userInput, toolCall, result);
        this.addToHistory('assistant', response);
        return response;
      }

      // Generate regular response
      const response = await this.generateResponse(userInput);
      this.addToHistory('assistant', response);
      return response;

    } catch (error) {
      console.error('Jarvis processing error:', error);
      const errorResponse = "I'm having trouble processing that request right now. Please try again or rephrase your question.";
      this.addToHistory('assistant', errorResponse);
      return errorResponse;
    } finally {
      this.isProcessing = false;
    }
  }

  // Process voice command (will be used by voice interface)
  async processVoiceCommand(_audioBlob: Blob): Promise<string> {
    try {
      // For now, we'll use Web Speech API for STT
      // In production, you'd want to use a more robust solution
      const transcription = await this.speechToText(_audioBlob);
      const response = await this.processTextCommand(transcription);
      
      // Convert response to speech
      await this.textToSpeech(response);
      
      return response;
    } catch (error) {
      console.error('Voice processing error:', error);
      return "I'm having trouble with voice processing. Please try typing your request.";
    }
  }

  // Parse tool calls from user input
  private parseToolCall(input: string): ToolCall | null {
    const lowerInput = input.toLowerCase();

    // Buy/Sell commands
    if (lowerInput.includes('buy') || lowerInput.includes('sell')) {
      const match = input.match(/(buy|sell)\s+(\d+(?:\.\d+)?)\s+(\w+-\w+)/i);
      if (match) {
        return {
          name: 'execute_trade',
          parameters: {
            side: match[1].toLowerCase(),
            quantity: parseFloat(match[2]),
            symbol: match[3].toUpperCase()
          }
        };
      }
    }

    // Portfolio status
    if (lowerInput.includes('portfolio') || lowerInput.includes('positions')) {
      return {
        name: 'get_portfolio_status',
        parameters: {}
      };
    }

    // Market data
    if (lowerInput.includes('price') || lowerInput.includes('market')) {
      const symbolMatch = input.match(/(\w+-\w+)/i);
      return {
        name: 'get_market_data',
        parameters: {
          symbol: symbolMatch ? symbolMatch[1].toUpperCase() : 'BTC-USD'
        }
      };
    }

    // Trading history
    if (lowerInput.includes('history') || lowerInput.includes('trades')) {
      return {
        name: 'get_trading_history',
        parameters: {
          limit: 10
        }
      };
    }

    return null;
  }

  // Execute tool calls
  private async executeToolCall(toolCall: ToolCall): Promise<any> {
    switch (toolCall.name) {
      case 'execute_trade':
        return await portfolioService.executeTrade(
          toolCall.parameters.symbol,
          toolCall.parameters.side,
          toolCall.parameters.quantity
        );

      case 'get_portfolio_status':
        return await portfolioService.getRealPortfolio();

      case 'get_market_data':
        // Get current market data from Redux store
        const state = store.getState();
        return state.marketData.prices[toolCall.parameters.symbol] || null;

      case 'get_trading_history':
        return await portfolioService.getTradingHistory(toolCall.parameters.limit);

      default:
        throw new Error(`Unknown tool: ${toolCall.name}`);
    }
  }

  // Generate response with tool result
  private async generateResponseWithToolResult(
    userInput: string, 
    toolCall: ToolCall, 
    result: any
  ): Promise<string> {
    const systemPrompt = this.buildSystemPrompt();
    const toolResultText = this.formatToolResult(toolCall, result);
    
    const prompt = `${systemPrompt}

User: ${userInput}

Tool Call: ${toolCall.name}(${JSON.stringify(toolCall.parameters)})
Tool Result: ${toolResultText}

Jarvis:`;

    return await this.callOllama(prompt);
  }

  // Generate regular response
  private async generateResponse(userInput: string): Promise<string> {
    const systemPrompt = this.buildSystemPrompt();
    const conversationContext = this.getConversationContext();
    
    const prompt = `${systemPrompt}

${conversationContext}

User: ${userInput}

Jarvis:`;

    return await this.callOllama(prompt);
  }

  // Build system prompt for Jarvis
  private buildSystemPrompt(): string {
    const portfolio = this.context.currentState.portfolio;
    const marketData = this.context.currentState.marketData;
    
    return `You are Jarvis, an AI trading assistant for the RRRalgorithms trading terminal. You are conversational, helpful, and knowledgeable about cryptocurrency trading.

PERSONALITY:
- Professional but friendly and informal
- Confident in your analysis
- Always confirm before executing trades
- Provide clear, actionable insights
- Use trading terminology naturally

CURRENT SYSTEM STATE:
- Portfolio Value: ${portfolio?.totalValue ? `$${portfolio.totalValue.toLocaleString()}` : 'Loading...'}
- Cash Available: ${portfolio?.cash ? `$${portfolio.cash.toLocaleString()}` : 'Loading...'}
- Active Positions: ${portfolio?.positions?.length || 0}
- Market Data: ${marketData?.isConnected ? 'Connected' : 'Mock Data'}
- System Status: ${this.context.currentState.systemHealth?.status || 'Unknown'}

CAPABILITIES:
${this.context.capabilities.map(cap => `- ${cap}`).join('\n')}

RESPONSE GUIDELINES:
- Keep responses concise but informative
- Always provide context for your recommendations
- Use specific numbers and percentages when available
- Confirm trade details before execution
- Explain your reasoning for analysis
- Be proactive in suggesting actions

Remember: You're helping a professional trader make informed decisions. Be confident but always emphasize risk management.`;
  }

  // Get conversation context
  private getConversationContext(): string {
    if (this.context.conversationHistory.length === 0) {
      return '';
    }

    const recentHistory = this.context.conversationHistory
      .slice(-this.maxHistoryLength)
      .map(msg => `${msg.role === 'user' ? 'User' : 'Jarvis'}: ${msg.content}`)
      .join('\n');

    return `Recent conversation:\n${recentHistory}\n`;
  }

  // Format tool result for LLM
  private formatToolResult(toolCall: ToolCall, result: any): string {
    switch (toolCall.name) {
      case 'execute_trade':
        return `Trade executed successfully: ${toolCall.parameters.side.toUpperCase()} ${toolCall.parameters.quantity} ${toolCall.parameters.symbol} at $${result.price}. Status: ${result.status}`;

      case 'get_portfolio_status':
        return `Portfolio Status: Total Value: $${result.totalValue?.toLocaleString()}, Cash: $${result.cash?.toLocaleString()}, Positions: ${result.positions?.length || 0}, Total P&L: $${result.totalPnl?.toFixed(2)} (${result.totalPnlPercent?.toFixed(2)}%)`;

      case 'get_market_data':
        return result ? `Current ${toolCall.parameters.symbol} Price: $${result.price}, Change: ${result.changePercent?.toFixed(2)}%, Volume: ${result.volume}` : 'Market data not available';

      case 'get_trading_history':
        return `Recent Trades: ${result.length} trades found. Latest: ${result[0]?.side?.toUpperCase()} ${result[0]?.quantity} ${result[0]?.symbol} at $${result[0]?.price}`;

      default:
        return JSON.stringify(result);
    }
  }

  // Call Ollama API
  private async callOllama(prompt: string): Promise<string> {
    try {
      const response = await fetch(`${this.ollamaUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'llama3.1:8b',
          prompt: prompt,
          stream: false,
          options: {
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 500,
            stop: ['User:', 'Jarvis:']
          }
        })
      });

      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.status}`);
      }

      const data = await response.json();
      return data.response?.trim() || "I'm having trouble generating a response right now.";
    } catch (error) {
      console.error('Ollama API error:', error);
      return "I'm having trouble connecting to my AI brain right now. Please try again.";
    }
  }

  // Speech-to-text using Web Speech API
  private async speechToText(_audioBlob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      // For now, we'll use a simple approach
      // In production, you'd want to use a more robust STT service
      const recognition = new (window as any).webkitSpeechRecognition();
      
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        resolve(transcript);
      };

      recognition.onerror = (event: any) => {
        reject(new Error(`Speech recognition error: ${event.error}`));
      };

      recognition.start();
    });
  }

  // Text-to-speech using Web Speech API
  private async textToSpeech(text: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.pitch = 0.8;
      utterance.volume = 0.8;

      // Use a male voice
      const voices = speechSynthesis.getVoices();
      const maleVoice = voices.find(voice => 
        voice.name.includes('Daniel') || 
        voice.name.includes('Alex') ||
        voice.name.includes('Male') ||
        voice.name.includes('Google UK English Male')
      );

      if (maleVoice) {
        utterance.voice = maleVoice;
      }

      utterance.onend = () => resolve();
      utterance.onerror = (event) => reject(new Error(`TTS error: ${event.error}`));

      speechSynthesis.speak(utterance);
    });
  }

  // Add message to conversation history
  private addToHistory(role: 'user' | 'assistant', content: string) {
    this.context.conversationHistory.push({
      role,
      content,
      timestamp: Date.now()
    });

    // Keep history within limit
    if (this.context.conversationHistory.length > this.maxHistoryLength * 2) {
      this.context.conversationHistory = this.context.conversationHistory.slice(-this.maxHistoryLength);
    }
  }

  // Update context with current system state
  async updateContext() {
    try {
      const portfolio = await portfolioService.getRealPortfolio();
      const marketStatus = marketDataService.getConnectionStatus();
      
      this.context.currentState = {
        ...this.context.currentState,
        portfolio,
        marketData: marketStatus,
        systemHealth: {
          ...this.context.currentState.systemHealth,
          lastUpdate: Date.now()
        }
      };
    } catch (error) {
      console.error('Error updating Jarvis context:', error);
    }
  }

  // Get current context
  getContext(): JarvisContext {
    return { ...this.context };
  }

  // Clear conversation history
  clearHistory() {
    this.context.conversationHistory = [];
  }

  // Check if Jarvis is ready
  isReady(): boolean {
    return !this.isProcessing;
  }

  // Get processing status
  getStatus() {
    return {
      isReady: this.isReady(),
      isProcessing: this.isProcessing,
      ollamaUrl: this.ollamaUrl,
      conversationLength: this.context.conversationHistory.length,
      systemState: this.context.currentState
    };
  }
}

export const jarvisService = new JarvisService();
