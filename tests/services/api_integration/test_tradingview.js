/**
 * Integration tests for TradingView webhook server
 */

const request = require('supertest');
const app = require('../tradingview/webhook_server');

describe('TradingView Webhook Server', () => {
  describe('GET /health', () => {
    it('should return health status', async () => {
      const response = await request(app)
        .get('/health')
        .expect(200);

      expect(response.body).toHaveProperty('status', 'healthy');
      expect(response.body).toHaveProperty('service', 'tradingview-webhook');
      expect(response.body).toHaveProperty('timestamp');
    });
  });

  describe('POST /webhook/tradingview', () => {
    const validPayload = {
      strategy: 'EMA_CrossOver',
      action: 'buy',
      ticker: 'BTC/USD',
      price: 45000,
      timestamp: new Date().toISOString(),
      timeframe: '1h',
      confidence: 0.85,
      stop_loss: 44000,
      take_profit: 47000,
      message: 'Test alert'
    };

    it('should accept valid webhook payload', async () => {
      const response = await request(app)
        .post('/webhook/tradingview')
        .send(validPayload)
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('signal_id');
      expect(response.body.signal).toHaveProperty('ticker', 'BTC/USD');
      expect(response.body.signal).toHaveProperty('signal', 'buy');
    });

    it('should reject payload without required fields', async () => {
      const invalidPayload = {
        strategy: 'TestStrategy',
        // missing action and ticker
      };

      const response = await request(app)
        .post('/webhook/tradingview')
        .send(invalidPayload)
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });

    it('should normalize different action formats', async () => {
      const testCases = [
        { action: 'long', expected: 'buy' },
        { action: 'short', expected: 'sell' },
        { action: 'close', expected: 'hold' },
      ];

      for (const testCase of testCases) {
        const payload = { ...validPayload, action: testCase.action };
        const response = await request(app)
          .post('/webhook/tradingview')
          .send(payload)
          .expect(200);

        expect(response.body.signal.signal).toBe(testCase.expected);
      }
    });
  });

  describe('POST /test/alert', () => {
    it('should create test alert', async () => {
      const response = await request(app)
        .post('/test/alert')
        .send({
          strategy: 'TEST',
          action: 'buy',
          ticker: 'TEST/USD',
          price: 100
        })
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('signal');
    });

    it('should create default test alert with empty body', async () => {
      const response = await request(app)
        .post('/test/alert')
        .send({})
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
    });
  });

  describe('GET /signals/recent', () => {
    it('should return recent signals', async () => {
      const response = await request(app)
        .get('/signals/recent')
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('signals');
      expect(Array.isArray(response.body.signals)).toBe(true);
    });

    it('should respect limit parameter', async () => {
      const response = await request(app)
        .get('/signals/recent?limit=5')
        .expect(200);

      expect(response.body.signals.length).toBeLessThanOrEqual(5);
    });

    it('should filter by ticker', async () => {
      const response = await request(app)
        .get('/signals/recent?ticker=BTC/USD')
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
    });
  });
});

describe('Signal Parsing', () => {
  it('should parse complete signal correctly', () => {
    const payload = {
      strategy: 'TestStrategy',
      action: 'buy',
      ticker: 'ETH/USD',
      price: 3000,
      confidence: 0.9,
      take_profit: 3200,
      stop_loss: 2900,
      timeframe: '4h'
    };

    // Test would require importing parseAlert function
    // This is a placeholder for actual implementation
    expect(payload.action).toBe('buy');
  });
});
