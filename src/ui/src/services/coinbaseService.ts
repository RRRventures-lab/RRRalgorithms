// Coinbase Advanced Trade API Service

interface CoinbaseConfig {
  apiKey: string;
  apiSecret: string;
  passphrase: string;
  sandbox: boolean;
}

interface CoinbaseAccount {
  uuid: string;
  name: string;
  currency: string;
  available_balance: {
    value: string;
    currency: string;
  };
  hold: {
    value: string;
    currency: string;
  };
}

interface CoinbaseOrder {
  order_id: string;
  product_id: string;
  side: 'BUY' | 'SELL';
  order_configuration: {
    market_market_ioc?: {
      quote_size: string;
    };
    limit_limit_gtc?: {
      base_size: string;
      limit_price: string;
    };
  };
  client_order_id: string;
  status: string;
  time_in_force: string;
  created_time: string;
  completion_percentage: string;
  filled_size: string;
  average_filled_price: string;
  fee: string;
  number_of_fills: string;
  filled_value: string;
  pending_cancel: boolean;
  size_in_quote: boolean;
  total_fees: string;
  size_inclusive_of_fees: boolean;
  total_value_after_fees: string;
  trigger_status: string;
  order_type: string;
  reject_reason: string;
  settled: boolean;
  product_type: string;
  reject_message: string;
  cancel_message: string;
  order_placement_source: string;
  outstanding_hold_amount: string;
  is_liquidation: boolean;
  last_fill_time: string;
  edit_history: any[];
  leverage: string;
  margin_type: string;
  retail_portfolio_id: string;
}

interface CoinbaseProduct {
  product_id: string;
  price: string;
  price_percentage_change_24h: string;
  volume_24h: string;
  volume_percentage_change_24h: string;
  base_increment: string;
  quote_increment: string;
  quote_min_size: string;
  quote_max_size: string;
  base_min_size: string;
  base_max_size: string;
  base_name: string;
  quote_name: string;
  watched: boolean;
  is_disabled: boolean;
  new: boolean;
  status: string;
  cancel_only: boolean;
  limit_only: boolean;
  post_only: boolean;
  trading_disabled: boolean;
  auction_mode: boolean;
  product_type: string;
  quote_currency_id: string;
  base_currency_id: string;
  fcm_trading_session_details: any;
  mid_market_price: string;
  alias: string;
  alias_to: string[];
  base_display_symbol: string;
  quote_display_symbol: string;
  view_only: boolean;
  price_increment: string;
  display_name: string;
  product_venue: string;
  approval_required: boolean;
  max_slippage_percentage: string;
  post_only_enabled: boolean;
  limit_only_enabled: boolean;
  cancel_only_enabled: boolean;
  trading_disabled_reason: string;
  auction_mode_enabled: boolean;
  [key: string]: any; // Allow additional properties
}

class CoinbaseService {
  private config: CoinbaseConfig | null = null;
  private baseUrl = 'https://api.coinbase.com';
  private sandboxUrl = 'https://api-public.sandbox.coinbase.com';
  private isConnected = false;
  private accounts: CoinbaseAccount[] = [];
  private products: CoinbaseProduct[] = [];

  constructor() {
    this.loadConfig();
  }

  private loadConfig() {
    // Load Coinbase API credentials from environment or config
    const apiKey = import.meta.env.VITE_COINBASE_API_KEY || '';
    const apiSecret = import.meta.env.VITE_COINBASE_API_SECRET || '';
    const passphrase = import.meta.env.VITE_COINBASE_PASSPHRASE || '';
    const sandbox = import.meta.env.VITE_COINBASE_SANDBOX === 'true';

    if (apiKey && apiSecret && passphrase) {
      this.config = {
        apiKey,
        apiSecret,
        passphrase,
        sandbox
      };
      console.log('Coinbase API configured:', { sandbox, hasCredentials: true });
    } else {
      console.warn('Coinbase API credentials not found. Using mock data.');
    }
  }

  // Generate Coinbase API signature
  private generateSignature(
    timestamp: string,
    method: string,
    requestPath: string,
    body: string = ''
  ): string {
    if (!this.config) {
      throw new Error('Coinbase API not configured');
    }

    const message = timestamp + method + requestPath + body;
    // For browser environment, we'll use a simplified approach
    // In production, you'd want to use a proper crypto library
    const key = atob(this.config.apiSecret);
    const encoder = new TextEncoder();
    const data = encoder.encode(message);
    const keyData = encoder.encode(key);
    
    // Simple XOR for demo purposes - in production use proper HMAC
    let result = '';
    for (let i = 0; i < data.length; i++) {
      result += String.fromCharCode(data[i] ^ keyData[i % keyData.length]);
    }
    
    return btoa(result);
  }

  // Make authenticated API request
  private async makeRequest(
    method: string,
    endpoint: string,
    body: any = null
  ): Promise<any> {
    if (!this.config) {
      throw new Error('Coinbase API not configured');
    }

    const timestamp = Math.floor(Date.now() / 1000).toString();
    const requestPath = endpoint;
    const bodyString = body ? JSON.stringify(body) : '';

    const signature = this.generateSignature(timestamp, method, requestPath, bodyString);

    const url = `${this.config.sandbox ? this.sandboxUrl : this.baseUrl}${endpoint}`;

    const headers: Record<string, string> = {
      'CB-ACCESS-KEY': this.config.apiKey,
      'CB-ACCESS-SIGN': signature,
      'CB-ACCESS-TIMESTAMP': timestamp,
      'CB-ACCESS-PASSPHRASE': this.config.passphrase,
      'Content-Type': 'application/json'
    };

    try {
      const response = await fetch(url, {
        method,
        headers,
        body: bodyString || undefined
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Coinbase API error: ${response.status} - ${errorText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Coinbase API request failed:', error);
      throw error;
    }
  }

  // Connect to Coinbase API
  async connect(): Promise<boolean> {
    if (!this.config) {
      console.log('Coinbase API not configured, using mock data');
      this.isConnected = false;
      return false;
    }

    try {
      // Test connection by getting accounts
      const accounts = await this.getAccounts();
      this.accounts = accounts;
      this.isConnected = true;
      console.log('Connected to Coinbase API');
      return true;
    } catch (error) {
      console.error('Failed to connect to Coinbase API:', error);
      this.isConnected = false;
      return false;
    }
  }

  // Get account information
  async getAccounts(): Promise<CoinbaseAccount[]> {
    if (!this.config) {
      return this.getMockAccounts();
    }

    try {
      const response = await this.makeRequest('GET', '/api/v3/brokerage/accounts');
      return response.accounts || [];
    } catch (error) {
      console.error('Error fetching accounts:', error);
      return this.getMockAccounts();
    }
  }

  // Get mock accounts for development
  private getMockAccounts(): CoinbaseAccount[] {
    return [
      {
        uuid: 'mock-usd-account',
        name: 'USD Wallet',
        currency: 'USD',
        available_balance: {
          value: '50000.00',
          currency: 'USD'
        },
        hold: {
          value: '0.00',
          currency: 'USD'
        }
      },
      {
        uuid: 'mock-btc-account',
        name: 'BTC Wallet',
        currency: 'BTC',
        available_balance: {
          value: '0.50000000',
          currency: 'BTC'
        },
        hold: {
          value: '0.00000000',
          currency: 'BTC'
        }
      },
      {
        uuid: 'mock-eth-account',
        name: 'ETH Wallet',
        currency: 'ETH',
        available_balance: {
          value: '2.00000000',
          currency: 'ETH'
        },
        hold: {
          value: '0.00000000',
          currency: 'ETH'
        }
      }
    ];
  }

  // Get product information
  async getProducts(): Promise<CoinbaseProduct[]> {
    if (!this.config) {
      return this.getMockProducts();
    }

    try {
      const response = await this.makeRequest('GET', '/api/v3/brokerage/products');
      return response.products || [];
    } catch (error) {
      console.error('Error fetching products:', error);
      return this.getMockProducts();
    }
  }

  // Get mock products for development
  private getMockProducts(): CoinbaseProduct[] {
    return [
      {
        product_id: 'BTC-USD',
        price: '45234.12',
        price_percentage_change_24h: '2.34',
        volume_24h: '1234567.89',
        volume_percentage_change_24h: '5.67',
        base_increment: '0.00000001',
        quote_increment: '0.01',
        quote_min_size: '1.00',
        quote_max_size: '1000000.00',
        base_min_size: '0.00000001',
        base_max_size: '1000.00000000',
        base_name: 'Bitcoin',
        quote_name: 'US Dollar',
        watched: false,
        is_disabled: false,
        new: false,
        status: 'online',
        cancel_only: false,
        limit_only: false,
        post_only: false,
        trading_disabled: false,
        auction_mode: false,
        product_type: 'SPOT',
        quote_currency_id: 'USD',
        base_currency_id: 'BTC',
        fcm_trading_session_details: null,
        mid_market_price: '45234.12',
        alias: '',
        alias_to: [],
        base_display_symbol: 'BTC',
        quote_display_symbol: 'USD',
        view_only: false,
        price_increment: '0.01',
        display_name: 'BTC-USD',
        product_venue: 'coinbase',
        approval_required: false,
        max_slippage_percentage: '0.05',
        post_only_enabled: true,
        limit_only_enabled: true,
        cancel_only_enabled: true,
        trading_disabled_reason: '',
        auction_mode_enabled: false,
        auction_mode_slippage_tolerance: '0.05',
        auction_mode_max_slippage_percentage: '0.05',
        auction_mode_price_band_percentage: '0.05',
        auction_mode_order_limit: '100',
        auction_mode_time_limit: '300',
        auction_mode_block_trade: false,
        auction_mode_crossing: false,
        auction_mode_imbalance_threshold: '0.05',
        auction_mode_min_order_size: '0.00000001',
        auction_mode_max_order_size: '1000.00000000',
        auction_mode_min_order_value: '1.00',
        auction_mode_max_order_value: '1000000.00',
        auction_mode_min_increment: '0.00000001',
        auction_mode_max_increment: '0.00000001',
        auction_mode_min_price: '0.01',
        auction_mode_max_price: '1000000.00',
        auction_mode_min_quantity: '0.00000001',
        auction_mode_max_quantity: '1000.00000000',
        auction_mode_min_value: '1.00',
        auction_mode_max_value: '1000000.00',
        auction_mode_min_time: '60',
        auction_mode_max_time: '300',
        auction_mode_min_block_trade: '0',
        auction_mode_max_block_trade: '0',
        auction_mode_min_crossing: '0',
        auction_mode_max_crossing: '0',
        auction_mode_min_imbalance_threshold: '0.01',
        auction_mode_max_imbalance_threshold: '0.10',
        auction_mode_min_order_limit: '10',
        auction_mode_max_order_limit: '1000',
        auction_mode_min_slippage_tolerance: '0.01',
        auction_mode_max_slippage_tolerance: '0.10',
        auction_mode_min_price_band_percentage: '0.01',
        auction_mode_max_price_band_percentage: '0.10',
        auction_mode_min_time_limit: '60',
        auction_mode_max_time_limit: '300',
        auction_mode_min_block_trade_enabled: 'false',
        auction_mode_max_block_trade_enabled: 'false',
        auction_mode_min_crossing_enabled: 'false',
        auction_mode_max_crossing_enabled: 'false',
        auction_mode_min_imbalance_threshold_enabled: 'false',
        auction_mode_max_imbalance_threshold_enabled: 'false',
        auction_mode_min_order_limit_enabled: 'false',
        auction_mode_max_order_limit_enabled: 'false',
        auction_mode_min_slippage_tolerance_enabled: 'false',
        auction_mode_max_slippage_tolerance_enabled: 'false',
        auction_mode_min_price_band_percentage_enabled: 'false',
        auction_mode_max_price_band_percentage_enabled: 'false',
        auction_mode_min_time_limit_enabled: 'false',
        auction_mode_max_time_limit_enabled: 'false'
      }
    ];
  }

  // Place a market order
  async placeMarketOrder(
    productId: string,
    side: 'BUY' | 'SELL',
    quoteSize?: string,
    baseSize?: string
  ): Promise<CoinbaseOrder> {
    if (!this.config) {
      return this.placeMockOrder(productId, side, 'market', quoteSize, baseSize);
    }

    const orderConfiguration: any = {};

    if (side === 'BUY' && quoteSize) {
      orderConfiguration.market_market_ioc = {
        quote_size: quoteSize
      };
    } else if (side === 'SELL' && baseSize) {
      orderConfiguration.market_market_ioc = {
        base_size: baseSize
      };
    } else {
      throw new Error('Invalid market order parameters');
    }

    const orderData = {
      product_id: productId,
      side,
      order_configuration: orderConfiguration,
      client_order_id: `jarvis-${Date.now()}`
    };

    try {
      const response = await this.makeRequest('POST', '/api/v3/brokerage/orders', orderData);
      return response.order;
    } catch (error) {
      console.error('Error placing market order:', error);
      return this.placeMockOrder(productId, side, 'market', quoteSize, baseSize);
    }
  }

  // Place a limit order
  async placeLimitOrder(
    productId: string,
    side: 'BUY' | 'SELL',
    baseSize: string,
    limitPrice: string
  ): Promise<CoinbaseOrder> {
    if (!this.config) {
      return this.placeMockOrder(productId, side, 'limit', undefined, baseSize, limitPrice);
    }

    const orderData = {
      product_id: productId,
      side,
      order_configuration: {
        limit_limit_gtc: {
          base_size: baseSize,
          limit_price: limitPrice
        }
      },
      client_order_id: `jarvis-${Date.now()}`
    };

    try {
      const response = await this.makeRequest('POST', '/api/v3/brokerage/orders', orderData);
      return response.order;
    } catch (error) {
      console.error('Error placing limit order:', error);
      return this.placeMockOrder(productId, side, 'limit', undefined, baseSize, limitPrice);
    }
  }

  // Place mock order for development
  private placeMockOrder(
    productId: string,
    side: 'BUY' | 'SELL',
    orderType: 'market' | 'limit',
    quoteSize?: string,
    baseSize?: string,
    limitPrice?: string
  ): CoinbaseOrder {
    const mockOrder: CoinbaseOrder = {
      order_id: `mock-${Date.now()}`,
      product_id: productId,
      side,
      order_configuration: orderType === 'market' 
        ? { market_market_ioc: { quote_size: quoteSize || '100.00' } }
        : { limit_limit_gtc: { base_size: baseSize || '0.001', limit_price: limitPrice || '45000.00' } },
      client_order_id: `jarvis-${Date.now()}`,
      status: 'FILLED',
      time_in_force: 'IOC',
      created_time: new Date().toISOString(),
      completion_percentage: '100',
      filled_size: baseSize || '0.001',
      average_filled_price: limitPrice || '45000.00',
      fee: '0.50',
      number_of_fills: '1',
      filled_value: quoteSize || '100.00',
      pending_cancel: false,
      size_in_quote: orderType === 'market' && side === 'BUY',
      total_fees: '0.50',
      size_inclusive_of_fees: false,
      total_value_after_fees: (parseFloat(quoteSize || '100.00') - 0.50).toString(),
      trigger_status: 'INVALID_ORDER_TYPE',
      order_type: orderType.toUpperCase(),
      reject_reason: '',
      settled: true,
      product_type: 'SPOT',
      reject_message: '',
      cancel_message: '',
      order_placement_source: 'RETAIL_ADVANCED',
      outstanding_hold_amount: '0.00',
      is_liquidation: false,
      last_fill_time: new Date().toISOString(),
      edit_history: [],
      leverage: '',
      margin_type: '',
      retail_portfolio_id: 'mock-portfolio'
    };

    console.log('Mock order placed:', mockOrder);
    return mockOrder;
  }

  // Get order status
  async getOrderStatus(orderId: string): Promise<CoinbaseOrder> {
    if (!this.config) {
      return this.getMockOrderStatus(orderId);
    }

    try {
      const response = await this.makeRequest('GET', `/api/v3/brokerage/orders/historical/${orderId}`);
      return response.order;
    } catch (error) {
      console.error('Error fetching order status:', error);
      return this.getMockOrderStatus(orderId);
    }
  }

  // Get mock order status
  private getMockOrderStatus(orderId: string): CoinbaseOrder {
    return {
      order_id: orderId,
      product_id: 'BTC-USD',
      side: 'BUY',
      order_configuration: { market_market_ioc: { quote_size: '100.00' } },
      client_order_id: `jarvis-${Date.now()}`,
      status: 'FILLED',
      time_in_force: 'IOC',
      created_time: new Date().toISOString(),
      completion_percentage: '100',
      filled_size: '0.002',
      average_filled_price: '45000.00',
      fee: '0.50',
      number_of_fills: '1',
      filled_value: '100.00',
      pending_cancel: false,
      size_in_quote: true,
      total_fees: '0.50',
      size_inclusive_of_fees: false,
      total_value_after_fees: '99.50',
      trigger_status: 'INVALID_ORDER_TYPE',
      order_type: 'MARKET',
      reject_reason: '',
      settled: true,
      product_type: 'SPOT',
      reject_message: '',
      cancel_message: '',
      order_placement_source: 'RETAIL_ADVANCED',
      outstanding_hold_amount: '0.00',
      is_liquidation: false,
      last_fill_time: new Date().toISOString(),
      edit_history: [],
      leverage: '',
      margin_type: '',
      retail_portfolio_id: 'mock-portfolio'
    };
  }

  // Cancel an order
  async cancelOrder(orderId: string): Promise<{ success: boolean; message: string }> {
    if (!this.config) {
      console.log('Mock order cancelled:', orderId);
      return { success: true, message: 'Order cancelled successfully' };
    }

    try {
      await this.makeRequest('POST', `/api/v3/brokerage/orders/${orderId}/cancel`);
      return { success: true, message: 'Order cancelled successfully' };
    } catch (error) {
      console.error('Error cancelling order:', error);
      return { success: false, message: 'Failed to cancel order' };
    }
  }

  // Get connection status
  getConnectionStatus() {
    return {
      isConnected: this.isConnected,
      hasConfig: !!this.config,
      sandbox: this.config?.sandbox || false,
      accountsCount: this.accounts.length,
      productsCount: this.products.length
    };
  }

  // Get account balance for a specific currency
  async getAccountBalance(currency: string): Promise<number> {
    const accounts = await this.getAccounts();
    const account = accounts.find(acc => acc.currency === currency);
    return account ? parseFloat(account.available_balance.value) : 0;
  }

  // Get current price for a product
  async getCurrentPrice(productId: string): Promise<number> {
    const products = await this.getProducts();
    const product = products.find(prod => prod.product_id === productId);
    return product ? parseFloat(product.price) : 0;
  }
}

export const coinbaseService = new CoinbaseService();
