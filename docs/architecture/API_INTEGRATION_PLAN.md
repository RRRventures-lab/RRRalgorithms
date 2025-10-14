# API & MCP Integration Implementation Plan

## Priority Matrix

| API/MCP | Priority | Complexity | Value | Status | ETA |
|---------|----------|------------|-------|--------|-----|
| **Polygon.io REST** | 🔥 Critical | Medium | ⭐⭐⭐⭐⭐ | ⏳ In Progress | 2 days |
| **Polygon.io WebSocket** | 🔥 Critical | Medium | ⭐⭐⭐⭐⭐ | ⏳ In Progress | 2 days |
| **PostgreSQL MCP** | 🔥 Critical | Low | ⭐⭐⭐⭐⭐ | ⏳ In Progress | 1 hour |
| **Perplexity AI** | ⭐ High | Low | ⭐⭐⭐⭐ | ⏸️ Planned | 1 day |
| **Perplexity MCP** | ⭐ High | Low | ⭐⭐⭐ | ⏸️ Planned | 4 hours |
| **TradingView Webhook** | 📊 Medium | Medium | ⭐⭐⭐ | 📝 Future | 2 days |
| **GitHub MCP** | 📊 Low | Low | ⭐⭐ | 📝 Future | 1 hour |

## Detailed Implementation Plan

### Phase 1: Core Market Data (Days 1-3)

#### 1.1 Polygon.io REST Client
**Worktree**: `worktrees/data-pipeline/`
**Files to create**:
- `src/data_pipeline/polygon/rest_client.py`
- `src/data_pipeline/polygon/models.py` (data models)
- `tests/integration/test_polygon_rest.py`

**Endpoints to implement**:
```python
# Historical data
get_aggregates(ticker, timespan, from, to)
get_daily_bars(ticker, date)
get_trades(ticker, timestamp)
get_quotes(ticker, timestamp)

# Reference data
get_ticker_details(ticker)
get_market_status()
list_crypto_tickers()

# Real-time (last)
get_last_trade(ticker)
get_last_quote(ticker)
```

**Features**:
- Rate limiting (5 requests/sec free tier, 100/sec paid)
- Automatic retry with exponential backoff
- Response caching in Redis
- Error handling and logging

#### 1.2 Polygon.io WebSocket Client
**Worktree**: `worktrees/data-pipeline/`
**Files to create**:
- `src/data_pipeline/polygon/websocket_client.py`
- `src/data_pipeline/polygon/stream_processor.py`
- `tests/integration/test_polygon_ws.py`

**Channels to subscribe**:
```python
# Crypto streams
"XA.*"  # All crypto aggregates (OHLCV bars)
"XT.*"  # All crypto trades
"XQ.*"  # All crypto quotes
"XA.BTC-USD"  # BTC specific aggregates
```

**Features**:
- Automatic reconnection on disconnect
- Message queue (Redis Pub/Sub) for distribution
- Backpressure handling
- Heartbeat monitoring

#### 1.3 PostgreSQL MCP Setup
**Worktree**: Main repository
**Files to create**:
- `config/database/schema.sql`
- `scripts/setup/init-databases.sh`

**Schema**:
```sql
-- Market data tables
CREATE TABLE crypto_aggregates (
    ticker VARCHAR(20),
    timestamp TIMESTAMPTZ,
    open DECIMAL,
    high DECIMAL,
    low DECIMAL,
    close DECIMAL,
    volume DECIMAL,
    vwap DECIMAL,
    PRIMARY KEY (ticker, timestamp)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('crypto_aggregates', 'timestamp');

-- Trades table
CREATE TABLE crypto_trades (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(20),
    timestamp TIMESTAMPTZ,
    price DECIMAL,
    size DECIMAL,
    exchange VARCHAR(50)
);

-- Sentiment data
CREATE TABLE market_sentiment (
    id BIGSERIAL PRIMARY KEY,
    asset VARCHAR(20),
    timestamp TIMESTAMPTZ,
    source VARCHAR(50),
    sentiment_score DECIMAL,
    confidence DECIMAL,
    text TEXT
);
```

**MCP Configuration**:
Already in `config/mcp-servers/mcp-config.json` - just needs DATABASE_URL in .env

---

### Phase 2: Sentiment Analysis (Days 4-5)

#### 2.1 Perplexity AI Client
**Worktree**: `worktrees/api-integration/`
**Files to create**:
- `perplexity/client.py`
- `perplexity/sentiment_analyzer.py`
- `perplexity/models.py`
- `tests/integration/test_perplexity.py`

**Methods to implement**:
```python
class PerplexityClient:
    def query(prompt, model, search_recency)
    def get_market_sentiment(asset, timeframe)
    def detect_market_events(keywords)
    def research_strategy(strategy_type)
    def analyze_news_impact(news_text)
```

**Features**:
- Response caching (5-minute TTL)
- Rate limiting (usage tracking)
- Cost monitoring (API calls are paid)
- Sentiment score extraction from responses

#### 2.2 Perplexity MCP Server
**Worktree**: `worktrees/api-integration/`
**Files to create**:
- `perplexity/mcp-server.ts`
- `perplexity/package.json`

**MCP Tools**:
```typescript
- query_market_intelligence(prompt, recency)
- get_sentiment_analysis(asset, timeframe)
- detect_events(keywords)
- research_topic(topic)
```

---

### Phase 3: Enhanced Features (Days 6-8)

#### 3.1 TradingView Webhook Receiver
**Worktree**: `worktrees/api-integration/`
**Files to create**:
- `tradingview/webhook_server.py`
- `tradingview/alert_processor.py`
- `tradingview/mcp-server.ts`

#### 3.2 GitHub MCP
**Configuration only** - already available via npm

---

## Implementation Sequence

### **RIGHT NOW: Let's Start with Polygon.io** ✅

I'll implement in this order:
1. ✅ Create database schema
2. ✅ Implement Polygon REST client
3. ✅ Implement Polygon WebSocket client
4. ✅ Create basic MCP server for Polygon
5. ✅ Test with sample data

### Day 1 (Today): Polygon.io Foundation
- [x] Database schema
- [ ] REST client implementation
- [ ] Basic WebSocket client
- [ ] Integration tests

### Day 2: Real-Time Data Pipeline
- [ ] Complete WebSocket client
- [ ] Redis pub/sub integration
- [ ] Data storage pipeline
- [ ] Monitoring and logging

### Day 3: MCP Integration
- [ ] Polygon MCP server
- [ ] Test MCP tools in Claude Code
- [ ] Documentation

### Days 4-5: Perplexity Integration
- [ ] Client implementation
- [ ] Sentiment analysis pipeline
- [ ] MCP server
- [ ] Integration with trading signals

---

## Cost Estimation

### Polygon.io
- **Free Tier**: 5 requests/sec, delayed data
- **Starter ($29/mo)**: 100 req/sec, real-time
- **Developer ($99/mo)**: Unlimited, WebSocket
- **Recommendation**: Start free, upgrade to Starter when testing

### Perplexity AI
- **Free**: 5 queries/day (not enough)
- **Standard ($20/mo)**: 300 queries/day
- **Max ($200/mo)**: Unlimited (what you have)
- **Recommendation**: Use your Max plan, monitor costs

### PostgreSQL/TimescaleDB
- **Docker (Local)**: Free
- **Cloud**: $25-100/mo depending on scale
- **Recommendation**: Start with Docker locally

**Total Monthly Cost (Development)**: ~$49/mo (Polygon Starter + Perplexity Standard)
**Total Monthly Cost (Production)**: ~$299/mo (Polygon Developer + Perplexity Max)

---

## Testing Strategy

### Unit Tests
- Test each API client method
- Mock external API responses
- Test error handling

### Integration Tests
- Test against real APIs (sandbox/test data)
- Test WebSocket connection and reconnection
- Test database insertion

### End-to-End Tests
- Test full data pipeline: API → Processing → Storage → Retrieval
- Test MCP server tools
- Test with Claude Code

---

## Success Criteria

### Phase 1 Complete When:
- ✅ Can fetch BTC price from Polygon REST API
- ✅ Can receive real-time trades via WebSocket
- ✅ Data stored in TimescaleDB
- ✅ Can query data via PostgreSQL MCP from Claude Code
- ✅ All integration tests passing

### Phase 2 Complete When:
- ✅ Can get sentiment score for BTC from Perplexity
- ✅ Sentiment stored in database
- ✅ Can query sentiment via Perplexity MCP
- ✅ Sentiment integrated with price data

---

**Let's start implementing Phase 1 right now!**

Next steps:
1. Create database schema
2. Implement Polygon REST client
3. Implement WebSocket client
4. Test everything

Ready to proceed?
