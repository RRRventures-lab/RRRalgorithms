# API & MCP Integration Architecture

## Overview

This document defines the integration architecture for external APIs and Model Context Protocol (MCP) servers that power the RRRalgorithms trading system.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude Code (Sonnet/Opus)                 │
│                      Multi-Agent System                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ MCP Protocol
                     │
    ┌────────────────┼────────────────┬────────────────────────┐
    │                │                │                        │
    │                │                │                        │
┌───▼─────┐   ┌─────▼──────┐  ┌─────▼──────┐   ┌────────▼────────┐
│ TradingView│   │Polygon.io   │  │ Perplexity │   │ PostgreSQL   │
│ MCP Server │   │MCP Server   │  │ MCP Server │   │ MCP Server   │
└───┬─────┘   └─────┬──────┘  └─────┬──────┘   └────────┬────────┘
    │                │                │                   │
    │ REST/Webhook   │ REST/WS        │ API               │ SQL
    │                │                │                   │
┌───▼─────┐   ┌─────▼──────┐  ┌─────▼──────┐   ┌────────▼────────┐
│TradingView│   │Polygon.io   │  │ Perplexity │   │  TimescaleDB  │
│  Service  │   │   API       │  │    API     │   │   Database    │
└───────────┘   └─────────────┘  └────────────┘   └───────────────┘
```

## Primary Data Sources

### 1. TradingView Integration

#### Overview
- **Purpose**: Technical analysis, chart patterns, custom indicators, alert webhooks
- **Integration Type**: Webhooks (inbound) + Pine Script strategies
- **Worktree**: `worktrees/api-integration/tradingview/`

#### TradingView Webhook Architecture

```
TradingView Alert → Webhook → FastAPI Server → Message Queue → Trading Engine
```

#### Webhook Payload Example
```json
{
  "timestamp": "2025-10-11T14:30:00Z",
  "symbol": "BTCUSD",
  "interval": "15m",
  "alert_name": "RSI_Divergence_Bull",
  "signal": "BUY",
  "price": 67450.00,
  "indicator_values": {
    "rsi": 32.5,
    "macd": -150.3,
    "volume": 1234567
  },
  "strategy": "mean_reversion_v2"
}
```

#### Implementation Plan

**Phase 1: Webhook Receiver**
```python
# worktrees/api-integration/tradingview/webhook_server.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import hmac
import hashlib

app = FastAPI()

class TradingViewAlert(BaseModel):
    timestamp: str
    symbol: str
    interval: str
    alert_name: str
    signal: str
    price: float
    indicator_values: dict

def verify_webhook_signature(payload: bytes, signature: str) -> bool:
    """Verify webhook authenticity using HMAC"""
    secret = os.getenv("TRADINGVIEW_WEBHOOK_SECRET")
    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)

@app.post("/webhook/tradingview")
async def receive_tradingview_alert(
    alert: TradingViewAlert,
    signature: str = Header(None)
):
    # Verify signature
    if not verify_webhook_signature(alert.json().encode(), signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Process alert
    await process_trading_signal(alert)
    return {"status": "received"}
```

**Phase 2: Custom MCP Server**
```typescript
// worktrees/api-integration/tradingview/mcp-server.ts
import { Server } from "@modelcontextprotocol/sdk/server";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio";

const server = new Server({
  name: "tradingview-mcp",
  version: "1.0.0"
});

// Tool: Send TradingView Alert
server.setRequestHandler("tools/call", async (request) => {
  if (request.params.name === "send_tradingview_alert") {
    const { symbol, timeframe, strategy } = request.params.arguments;
    // Logic to fetch recent alerts from webhook database
    const alerts = await getRecentAlerts(symbol, timeframe, strategy);
    return { content: [{ type: "text", text: JSON.stringify(alerts) }] };
  }
});

// Resource: TradingView Strategies
server.setRequestHandler("resources/list", async () => {
  return {
    resources: [
      {
        uri: "tradingview://strategies/all",
        name: "All TradingView Strategies",
        mimeType: "application/json"
      }
    ]
  };
});

const transport = new StdioServerTransport();
server.connect(transport);
```

**MCP Configuration** (`config/mcp-servers/tradingview-mcp.json`):
```json
{
  "mcpServers": {
    "tradingview": {
      "command": "node",
      "args": [
        "/Volumes/Lexar/RRRVentures/RRRalgorithms/worktrees/api-integration/tradingview/mcp-server.js"
      ],
      "env": {
        "TRADINGVIEW_WEBHOOK_URL": "http://localhost:8080/webhook/tradingview",
        "TRADINGVIEW_WEBHOOK_SECRET": "${TRADINGVIEW_WEBHOOK_SECRET}"
      }
    }
  }
}
```

---

### 2. Polygon.io Integration

#### Overview
- **Purpose**: Real-time and historical market data (stocks, crypto, forex, options)
- **Integration Type**: REST API + WebSocket Streams
- **Worktree**: `worktrees/data-pipeline/polygon/`
- **Official Docs**: https://polygon.io/docs

#### Data Types Available
1. **Real-time Data**: Trades, Quotes, Aggregates (bars)
2. **Historical Data**: Daily bars, minute bars, tick data
3. **Reference Data**: Tickers, exchanges, market status
4. **Options Data**: Options chains, Greeks
5. **Crypto Data**: All major cryptocurrencies

#### Polygon.io WebSocket Architecture

```
Polygon WebSocket → Stream Handler → Redis Pub/Sub → Multiple Consumers
                                     ├─> Neural Network (Feature Generation)
                                     ├─> Trading Engine (Order Execution)
                                     └─> Data Pipeline (Storage)
```

#### WebSocket Implementation

```python
# worktrees/data-pipeline/polygon/websocket_client.py
import asyncio
import websockets
import json
import os
from typing import Callable, List

class PolygonWebSocketClient:
    """Real-time data streaming from Polygon.io"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.ws_url = "wss://socket.polygon.io/crypto"  # or /stocks, /forex
        self.subscriptions: List[str] = []

    async def connect(self):
        """Establish WebSocket connection"""
        async with websockets.connect(self.ws_url) as websocket:
            # Authenticate
            auth_msg = {"action": "auth", "params": self.api_key}
            await websocket.send(json.dumps(auth_msg))

            # Subscribe to channels
            for sub in self.subscriptions:
                subscribe_msg = {"action": "subscribe", "params": sub}
                await websocket.send(json.dumps(subscribe_msg))

            # Listen for messages
            async for message in websocket:
                data = json.loads(message)
                await self.handle_message(data)

    async def handle_message(self, data: dict):
        """Process incoming market data"""
        msg_type = data[0].get("ev")  # Event type

        if msg_type == "XA":  # Aggregate (OHLCV bar)
            await self.handle_aggregate(data[0])
        elif msg_type == "XT":  # Trade
            await self.handle_trade(data[0])
        elif msg_type == "XQ":  # Quote
            await self.handle_quote(data[0])

    async def handle_aggregate(self, agg: dict):
        """Handle aggregate bar data"""
        # agg = {
        #   "ev": "XA",
        #   "pair": "BTC-USD",
        #   "o": 67450.0,
        #   "h": 67500.0,
        #   "l": 67400.0,
        #   "c": 67475.0,
        #   "v": 123.45,
        #   "s": 1697031000000  # Start timestamp
        # }
        await self.publish_to_redis("polygon:aggregates", agg)

    def subscribe(self, channels: List[str]):
        """Subscribe to data channels"""
        # Examples:
        # "XA.BTC-USD" - Aggregates for BTC-USD
        # "XT.*" - All trades
        # "XQ.ETH-USD" - Quotes for ETH-USD
        self.subscriptions.extend(channels)
```

#### REST API Implementation

```python
# worktrees/data-pipeline/polygon/rest_client.py
import requests
from typing import Optional, List
from datetime import datetime, timedelta

class PolygonRESTClient:
    """Historical and reference data from Polygon.io"""

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def get_aggregates(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,  # "minute", "hour", "day"
        from_date: str,
        to_date: str,
        limit: int = 50000
    ) -> List[dict]:
        """Get historical aggregate bars"""
        url = f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {"limit": limit, "adjusted": "true", "sort": "asc"}

        response = self.session.get(url, params=params)
        response.raise_for_status()

        return response.json()["results"]

    def get_last_trade(self, ticker: str) -> dict:
        """Get the most recent trade"""
        url = f"{self.BASE_URL}/v2/last/trade/{ticker}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()["results"]

    def get_options_chain(self, underlying: str, expiration: str) -> List[dict]:
        """Get options chain for a given expiration"""
        url = f"{self.BASE_URL}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": underlying,
            "expiration_date": expiration,
            "limit": 1000
        }
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()["results"]
```

#### Polygon MCP Server (Custom)

```typescript
// worktrees/api-integration/polygon/mcp-server.ts
import { Server } from "@modelcontextprotocol/sdk/server";
import axios from "axios";

const server = new Server({
  name: "polygon-mcp",
  version: "1.0.0"
});

// Tool: Get market data
server.setRequestHandler("tools/call", async (request) => {
  const apiKey = process.env.POLYGON_API_KEY;

  if (request.params.name === "get_crypto_price") {
    const { symbol } = request.params.arguments;
    const url = `https://api.polygon.io/v2/last/trade/${symbol}`;
    const response = await axios.get(url, {
      headers: { Authorization: `Bearer ${apiKey}` }
    });
    return { content: [{ type: "text", text: JSON.stringify(response.data) }] };
  }

  if (request.params.name === "get_historical_bars") {
    const { ticker, from, to, timespan } = request.params.arguments;
    const url = `https://api.polygon.io/v2/aggs/ticker/${ticker}/range/1/${timespan}/${from}/${to}`;
    const response = await axios.get(url, {
      headers: { Authorization: `Bearer ${apiKey}` }
    });
    return { content: [{ type: "text", text: JSON.stringify(response.data) }] };
  }
});

// Resources: Market status, tickers
server.setRequestHandler("resources/list", async () => {
  return {
    resources: [
      { uri: "polygon://market/status", name: "Market Status" },
      { uri: "polygon://tickers/crypto", name: "Crypto Tickers" }
    ]
  };
});
```

**MCP Configuration** (`config/mcp-servers/polygon-mcp.json`):
```json
{
  "mcpServers": {
    "polygon": {
      "command": "node",
      "args": ["/Volumes/Lexar/RRRVentures/RRRalgorithms/worktrees/api-integration/polygon/mcp-server.js"],
      "env": {
        "POLYGON_API_KEY": "${POLYGON_API_KEY}"
      }
    }
  }
}
```

---

### 3. Perplexity AI Integration

#### Overview
- **Purpose**: Market sentiment analysis, news aggregation, research intelligence
- **Integration Type**: REST API (Perplexity Max plan)
- **Worktree**: `worktrees/api-integration/perplexity/`
- **API Docs**: https://docs.perplexity.ai/

#### Use Cases

1. **News Sentiment Analysis**
   - Query: "What is the current market sentiment on Bitcoin based on the latest news?"
   - Use: Feed sentiment scores into neural network

2. **Event Detection**
   - Query: "Has there been any major regulatory news about cryptocurrency in the last 24 hours?"
   - Use: Trigger risk management protocols

3. **Research Synthesis**
   - Query: "Summarize the latest academic research on transformer models for time series forecasting"
   - Use: Inform model architecture decisions

4. **Competitor Analysis**
   - Query: "What trading strategies are discussed in the latest quantitative finance papers?"
   - Use: Strategy research and validation

#### Perplexity API Implementation

```python
# worktrees/api-integration/perplexity/client.py
import requests
from typing import Optional, List

class PerplexityClient:
    """Perplexity AI API client for market intelligence"""

    BASE_URL = "https://api.perplexity.ai"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })

    def query(
        self,
        prompt: str,
        model: str = "llama-3.1-sonar-large-128k-online",  # Max plan model
        search_recency: str = "day",  # "hour", "day", "week", "month"
        return_citations: bool = True
    ) -> dict:
        """Query Perplexity for research and analysis"""
        url = f"{self.BASE_URL}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "search_recency_filter": search_recency,
            "return_citations": return_citations,
            "temperature": 0.2  # Low temperature for factual responses
        }

        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def get_market_sentiment(self, asset: str, timeframe: str = "day") -> dict:
        """Analyze market sentiment for an asset"""
        prompt = f"""
        Analyze the market sentiment for {asset} based on news from the last {timeframe}.
        Provide:
        1. Overall sentiment (bullish/neutral/bearish) with confidence score (0-100)
        2. Key news events driving sentiment
        3. Potential risks and catalysts
        4. Sentiment from different sources (institutional, retail, social media)

        Format response as JSON.
        """
        response = self.query(prompt, search_recency=timeframe)
        return self.parse_sentiment_response(response)

    def detect_market_events(self, keywords: List[str]) -> dict:
        """Detect significant market events"""
        prompt = f"""
        Search for significant events related to: {', '.join(keywords)}.
        Focus on the last 24 hours.
        Identify:
        1. Regulatory announcements
        2. Major price movements or market crashes
        3. Institutional adoption news
        4. Security incidents or hacks
        5. Technical developments

        Return as structured JSON with event type, timestamp, and impact level.
        """
        return self.query(prompt, search_recency="day")

    def research_strategy(self, strategy_type: str) -> dict:
        """Research trading strategy information"""
        prompt = f"""
        Research the latest information about {strategy_type} trading strategies.
        Include:
        1. Recent academic papers (last 6 months)
        2. Industry best practices
        3. Risk considerations
        4. Performance benchmarks
        5. Implementation challenges

        Provide citations for all sources.
        """
        return self.query(prompt, search_recency="month")
```

#### Perplexity MCP Server

```typescript
// worktrees/api-integration/perplexity/mcp-server.ts
import { Server } from "@modelcontextprotocol/sdk/server";
import axios from "axios";

const server = new Server({
  name: "perplexity-mcp",
  version: "1.0.0"
});

server.setRequestHandler("tools/call", async (request) => {
  const apiKey = process.env.PERPLEXITY_API_KEY;

  if (request.params.name === "query_market_intelligence") {
    const { prompt, recency } = request.params.arguments;

    const response = await axios.post(
      "https://api.perplexity.ai/chat/completions",
      {
        model: "llama-3.1-sonar-large-128k-online",
        messages: [{ role: "user", content: prompt }],
        search_recency_filter: recency || "day",
        return_citations: true
      },
      {
        headers: {
          "Authorization": `Bearer ${apiKey}`,
          "Content-Type": "application/json"
        }
      }
    );

    return {
      content: [{
        type: "text",
        text: JSON.stringify(response.data)
      }]
    };
  }
});

server.setRequestHandler("resources/list", async () => {
  return {
    resources: [
      { uri: "perplexity://sentiment/crypto", name: "Crypto Market Sentiment" },
      { uri: "perplexity://events/regulatory", name: "Regulatory Events" }
    ]
  };
});
```

**MCP Configuration** (`config/mcp-servers/perplexity-mcp.json`):
```json
{
  "mcpServers": {
    "perplexity": {
      "command": "node",
      "args": ["/Volumes/Lexar/RRRVentures/RRRalgorithms/worktrees/api-integration/perplexity/mcp-server.js"],
      "env": {
        "PERPLEXITY_API_KEY": "${PERPLEXITY_API_KEY}"
      }
    }
  }
}
```

---

## Additional Recommended MCP Servers

### 4. PostgreSQL MCP (Database Access)

**Install**:
```bash
npm install @modelcontextprotocol/server-postgres
```

**Configuration** (`config/mcp-servers/postgres-mcp.json`):
```json
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://user:pass@localhost:5432/trading_db"]
    }
  }
}
```

**Usage**: Direct SQL queries from Claude Code for data analysis

---

### 5. GitHub MCP (Code Management)

**Install**:
```bash
npm install @modelcontextprotocol/server-github
```

**Configuration**:
```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

**Usage**: Automated PR creation, issue tracking, code reviews

---

## API Key Management

### Environment Variables

Create `config/api-keys/.env`:
```bash
# Polygon.io
POLYGON_API_KEY=your_polygon_api_key_here

# Perplexity AI
PERPLEXITY_API_KEY=your_perplexity_api_key_here

# TradingView Webhook
TRADINGVIEW_WEBHOOK_SECRET=your_webhook_secret_here

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/trading_db
TIMESCALE_DB_URL=postgresql://user:password@localhost:5432/timeseries_db

# Redis
REDIS_URL=redis://localhost:6379

# GitHub
GITHUB_TOKEN=your_github_token_here

# Exchange APIs (for live trading)
COINBASE_API_KEY=your_coinbase_api_key
COINBASE_API_SECRET=your_coinbase_secret
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_secret
```

### Security Best Practices

1. **Never Commit `.env` Files**
   ```bash
   echo "config/api-keys/.env" >> .gitignore
   ```

2. **Use Secret Management** (Production)
   - HashiCorp Vault
   - AWS Secrets Manager
   - Google Cloud Secret Manager

3. **Rotate Keys Regularly**
   - Schedule: Monthly for API keys
   - Audit access logs

4. **Rate Limiting**
   - Implement rate limiters for all API calls
   - Cache responses when appropriate

5. **Error Handling**
   - Never expose API keys in error messages
   - Log API failures securely

---

## Integration Testing

### Test Polygon.io Connection

```python
# tests/integration/test_polygon.py
import pytest
from worktrees.data_pipeline.polygon.rest_client import PolygonRESTClient

def test_polygon_connection():
    client = PolygonRESTClient(api_key=os.getenv("POLYGON_API_KEY"))
    data = client.get_last_trade("X:BTCUSD")
    assert data is not None
    assert "price" in data
```

### Test Perplexity Connection

```python
# tests/integration/test_perplexity.py
import pytest
from worktrees.api_integration.perplexity.client import PerplexityClient

def test_perplexity_sentiment():
    client = PerplexityClient(api_key=os.getenv("PERPLEXITY_API_KEY"))
    result = client.get_market_sentiment("Bitcoin", timeframe="day")
    assert "sentiment" in result
    assert result["sentiment"] in ["bullish", "neutral", "bearish"]
```

---

## Monitoring & Observability

### API Metrics to Track

1. **Latency**: P50, P95, P99 response times
2. **Error Rate**: 4xx, 5xx errors per endpoint
3. **Rate Limit Usage**: % of rate limit consumed
4. **Data Freshness**: Time since last successful update
5. **Cost**: API call costs (especially Perplexity)

### Alerting Rules

```yaml
# config/alerts/api-monitoring.yml
- alert: PolygonAPIDown
  expr: polygon_api_success_rate < 0.95
  for: 5m
  annotations:
    summary: "Polygon API success rate below 95%"

- alert: PerplexityRateLimitReached
  expr: perplexity_rate_limit_remaining < 10
  annotations:
    summary: "Perplexity API rate limit nearly exhausted"

- alert: TradingViewWebhookDelay
  expr: (time() - tradingview_last_webhook_timestamp) > 300
  annotations:
    summary: "No TradingView webhooks received in 5 minutes"
```

---

**Last Updated**: 2025-10-11
**Maintained By**: API Integration Team
