# Cryptocurrency Data Pipeline

Enterprise-grade real-time data ingestion system for cryptocurrency trading algorithms.

## Overview

This data pipeline provides:

- **Real-time WebSocket Streaming**: Live trades, quotes, and aggregates from Polygon.io
- **Sentiment Analysis**: AI-powered market sentiment using Perplexity AI
- **Data Quality Monitoring**: Automated validation and anomaly detection
- **Historical Backfill**: Load months of historical data for backtesting
- **Supabase Integration**: Automatic storage in PostgreSQL database

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Data Pipeline Orchestrator                 │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  WebSocket    │   │  Sentiment    │   │   Quality     │
│  Streaming    │   │  Analysis     │   │  Validator    │
│  (Real-time)  │   │  (15 min)     │   │  (5 min)      │
└───────────────┘   └───────────────┘   └───────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                    ┌───────────────┐
                    │   Supabase    │
                    │  (PostgreSQL) │
                    └───────────────┘
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Edit `/Volumes/Lexar/RRRVentures/RRRalgorithms/config/api-keys/.env`:

```bash
# Required
POLYGON_API_KEY=your_polygon_api_key
PERPLEXITY_API_KEY=your_perplexity_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_service_key
```

### 3. Initialize Database

The database schema is already set up in Supabase with these tables:

- `crypto_aggregates`: OHLCV bars (1-minute)
- `crypto_trades`: Individual trades
- `crypto_quotes`: Bid/ask quotes
- `market_sentiment`: Sentiment analysis results
- `system_events`: Quality issues and logs

## Usage

### Run the Complete Pipeline

Start all components (WebSocket, sentiment, quality monitoring):

```bash
python src/data_pipeline/main.py
```

### Custom Configuration

```bash
# Track specific pairs only
python src/data_pipeline/main.py --tickers X:BTCUSD X:ETHUSD

# Disable sentiment analysis
python src/data_pipeline/main.py --no-sentiment

# Custom intervals
python src/data_pipeline/main.py \
  --sentiment-interval 1800 \
  --quality-interval 600
```

### Historical Backfill

Before running the pipeline for the first time, backfill historical data:

```bash
# Backfill 6 months of data
python src/data_pipeline/backfill/historical.py --months 6

# Backfill specific tickers
python src/data_pipeline/backfill/historical.py \
  --tickers X:BTCUSD X:ETHUSD \
  --months 3

# 5-minute bars instead of 1-minute
python src/data_pipeline/backfill/historical.py \
  --timespan minute \
  --multiplier 5 \
  --months 6
```

**Note**: Backfill is resumable. If interrupted, run the same command again to continue.

## Components

### 1. Polygon WebSocket Client

Real-time streaming from Polygon.io.

**File**: `src/data_pipeline/polygon/websocket_client.py`

**Streams**:
- `XT`: Crypto trades
- `XQ`: Crypto quotes (bid/ask)
- `XA`: Crypto aggregates (1-minute bars)

**Features**:
- Auto-reconnection with exponential backoff
- Concurrent streaming for multiple pairs
- Direct Supabase integration

**Example**:
```python
from data_pipeline.polygon.websocket_client import PolygonWebSocketClient
from data_pipeline.supabase_client import SupabaseClient

supabase = SupabaseClient()
ws_client = PolygonWebSocketClient(
    supabase_client=supabase,
    pairs=["X:BTCUSD", "X:ETHUSD"]
)

await ws_client.run()
```

### 2. Perplexity Sentiment Analyzer

AI-powered market sentiment analysis.

**File**: `src/data_pipeline/perplexity/sentiment_analyzer.py`

**Features**:
- Real-time news analysis
- Sentiment scoring (-1.0 to +1.0)
- Confidence levels
- Scheduled updates (default: every 15 minutes)

**Output**:
- `sentiment_label`: bullish/neutral/bearish
- `sentiment_score`: -1.0 (bearish) to +1.0 (bullish)
- `confidence`: 0.0 to 1.0
- `reasoning`: Brief explanation

**Example**:
```python
from data_pipeline.perplexity.sentiment_analyzer import PerplexitySentimentAnalyzer

analyzer = PerplexitySentimentAnalyzer(supabase_client=supabase)
sentiment = await analyzer.analyze_sentiment("BTC")

print(f"Sentiment: {sentiment.sentiment_label}")
print(f"Score: {sentiment.sentiment_score}")
```

### 3. Data Quality Validator

Automated data quality monitoring.

**File**: `src/data_pipeline/quality/validator.py`

**Checks**:
- Missing data gaps
- Price outliers (>20% spikes)
- Volume anomalies (>5x average)
- Null values in critical fields
- Statistical outliers (Z-score > 4)

**Example**:
```python
from data_pipeline.quality.validator import DataQualityValidator

validator = DataQualityValidator(supabase_client=supabase)
issues = await validator.validate_recent_data(lookback_hours=24)

for issue in issues:
    print(f"[{issue.severity}] {issue.message}")
```

### 4. Historical Backfill

Load historical data for backtesting.

**File**: `src/data_pipeline/backfill/historical.py`

**Features**:
- Resumable (tracks progress)
- Bulk inserts for efficiency
- Configurable timeframes
- Rate-limited

## Data Schema

### crypto_aggregates

| Column       | Type      | Description              |
|--------------|-----------|--------------------------|
| id           | UUID      | Primary key              |
| ticker       | TEXT      | Symbol (e.g., X:BTCUSD) |
| event_time   | TIMESTAMP | Bar timestamp            |
| open         | NUMERIC   | Open price               |
| high         | NUMERIC   | High price               |
| low          | NUMERIC   | Low price                |
| close        | NUMERIC   | Close price              |
| volume       | NUMERIC   | Trading volume           |
| vwap         | NUMERIC   | Volume-weighted avg price|
| trade_count  | INTEGER   | Number of trades         |
| created_at   | TIMESTAMP | Insert timestamp         |

### crypto_trades

| Column       | Type      | Description              |
|--------------|-----------|--------------------------|
| id           | UUID      | Primary key              |
| ticker       | TEXT      | Symbol                   |
| event_time   | TIMESTAMP | Trade timestamp          |
| price        | NUMERIC   | Trade price              |
| size         | NUMERIC   | Trade size               |
| exchange_id  | INTEGER   | Exchange identifier      |
| conditions   | INTEGER[] | Trade conditions         |
| trade_id     | TEXT      | Unique trade ID          |
| created_at   | TIMESTAMP | Insert timestamp         |

### market_sentiment

| Column           | Type      | Description                    |
|------------------|-----------|--------------------------------|
| id               | UUID      | Primary key                    |
| asset            | TEXT      | Asset symbol (e.g., BTC)      |
| source           | TEXT      | Data source (perplexity)      |
| sentiment_label  | TEXT      | bullish/neutral/bearish       |
| sentiment_score  | NUMERIC   | -1.0 to +1.0                  |
| confidence       | NUMERIC   | 0.0 to 1.0                    |
| text             | TEXT      | Source text                   |
| metadata         | JSONB     | Additional data               |
| event_time       | TIMESTAMP | Analysis timestamp            |
| created_at       | TIMESTAMP | Insert timestamp              |

## Monitoring & Logging

### Logs

Application logs are written to:
- **Console**: `stdout` (INFO level)
- **File**: `data_pipeline.log` (INFO level)

### System Events

All quality issues and system events are logged to `system_events` table:

```sql
SELECT * FROM system_events
WHERE component = 'data_quality_validator'
ORDER BY event_time DESC
LIMIT 10;
```

### Pipeline Statistics

Get real-time stats:

```python
orchestrator = DataPipelineOrchestrator(...)
stats = orchestrator.get_stats()

print(f"WebSocket messages: {stats['websocket']['message_count']}")
print(f"Sentiment analyses: {stats['sentiment']['analysis_count']}")
print(f"Quality issues: {stats['quality']['issues_found']}")
```

## Performance

### WebSocket Throughput

- **Typical**: 100-500 messages/second
- **Peak**: 2,000+ messages/second
- **Latency**: <100ms from market to database

### Resource Usage

- **CPU**: ~10-20% (single core)
- **Memory**: ~200-500 MB
- **Network**: ~1-5 MB/min

### Database Storage

Approximate storage per ticker:
- **1-minute bars**: ~1.5 MB/day
- **Trades**: Varies (10-100 MB/day)
- **Quotes**: Varies (5-50 MB/day)

## Troubleshooting

### Connection Issues

**Problem**: WebSocket keeps disconnecting

**Solutions**:
1. Check API key validity
2. Verify network connectivity
3. Check rate limits (upgrade Polygon plan if needed)
4. Review logs for specific errors

### Missing Data

**Problem**: Gaps in data

**Solutions**:
1. Check quality validator logs
2. Verify Polygon API status
3. Run backfill for missing periods
4. Check Supabase connection

### Sentiment Analysis Fails

**Problem**: Perplexity API errors

**Solutions**:
1. Verify API key
2. Check rate limits
3. Reduce update frequency
4. Review Perplexity API status

## Development

### Project Structure

```
src/data_pipeline/
├── main.py                      # Main orchestrator
├── supabase_client.py           # Supabase integration
├── polygon/
│   ├── rest_client.py           # REST API client
│   ├── websocket_client.py      # WebSocket streaming
│   └── models.py                # Pydantic models
├── perplexity/
│   └── sentiment_analyzer.py    # Sentiment analysis
├── quality/
│   └── validator.py             # Data quality checks
└── backfill/
    └── historical.py            # Historical data backfill
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src/data_pipeline tests/

# Run specific test
pytest tests/test_websocket.py
```

### Code Quality

```bash
# Format code
black src/

# Lint
ruff check src/

# Type checking
mypy src/
```

## API Rate Limits

### Polygon.io

**Free Tier**:
- 5 requests/second
- Limited historical data

**Currencies Starter** (Current):
- 100 requests/second
- Unlimited WebSocket connections
- 2 years historical data

### Perplexity AI

**Max Plan** (Current):
- Unlimited requests
- No rate limiting

## Production Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

CMD ["python", "src/data_pipeline/main.py"]
```

### systemd Service

```ini
[Unit]
Description=Cryptocurrency Data Pipeline
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/opt/rrralgorithms/worktrees/data-pipeline
ExecStart=/usr/bin/python3 src/data_pipeline/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Support

For issues or questions:

1. Check logs: `data_pipeline.log`
2. Review system events in Supabase
3. Check component statistics
4. Consult project documentation

## License

Proprietary - RRRVentures

---

**Version**: 1.0.0
**Last Updated**: 2025-10-11
**Status**: Production Ready
# RRRalgorithms
