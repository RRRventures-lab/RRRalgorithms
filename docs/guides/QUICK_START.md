# Data Pipeline Quick Start Guide

Get the cryptocurrency data pipeline running in 5 minutes.

## Prerequisites

- Python 3.9+
- API keys configured in `/Volumes/Lexar/RRRVentures/RRRalgorithms/config/api-keys/.env`
- Supabase database initialized

## Step 1: Install Dependencies

```bash
cd /Volumes/Lexar/RRRVentures/RRRalgorithms/worktrees/data-pipeline
pip install -r requirements.txt
```

## Step 2: Verify Configuration

Check that your API keys are set:

```bash
# Should show your API keys
cat /Volumes/Lexar/RRRVentures/RRRalgorithms/config/api-keys/.env | grep -E "POLYGON|PERPLEXITY|SUPABASE"
```

Required variables:
- `POLYGON_API_KEY`
- `PERPLEXITY_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`

## Step 3: Test Connection

Test that you can connect to Supabase:

```bash
cd /Volumes/Lexar/RRRVentures/RRRalgorithms/worktrees/data-pipeline
python3 src/data_pipeline/supabase_client.py
```

You should see:
```
INFO:data_pipeline.supabase_client:Supabase client initialized
✅ Inserted: [...]
✅ Found 5 price records
```

## Step 4: Backfill Historical Data (Optional but Recommended)

Load 6 months of historical data for backtesting:

```bash
cd /Volumes/Lexar/RRRVentures/RRRalgorithms/worktrees/data-pipeline

# Backfill major cryptocurrencies
python3 src/data_pipeline/backfill/historical.py --months 6

# Or backfill just BTC and ETH for faster testing
python3 src/data_pipeline/backfill/historical.py \
  --tickers X:BTCUSD X:ETHUSD \
  --months 1
```

This will take 10-30 minutes depending on how many tickers you're backfilling.

**Note**: Backfill is resumable. If interrupted, just run the command again.

## Step 5: Start the Pipeline

Run all components (WebSocket streaming, sentiment analysis, quality monitoring):

```bash
cd /Volumes/Lexar/RRRVentures/RRRalgorithms/worktrees/data-pipeline
python3 src/data_pipeline/main.py
```

You should see:
```
╔═══════════════════════════════════════════════════════════════╗
║  RRRalgorithms - Cryptocurrency Data Pipeline                ║
║  Real-time Streaming | Sentiment Analysis | Quality Control  ║
╚═══════════════════════════════════════════════════════════════╝

======================================================================
STARTING DATA PIPELINE
======================================================================
WebSocket streaming: True
Sentiment analysis: True
Quality monitoring: True
Monitored tickers: None
======================================================================
INFO:data_pipeline.orchestrator:WebSocket streaming started
INFO:data_pipeline.orchestrator:Sentiment analysis started
INFO:data_pipeline.orchestrator:Quality monitoring started
```

## Step 6: Verify Data is Flowing

Open another terminal and check your Supabase database:

### Check Real-time Aggregates (OHLCV bars)

```sql
SELECT
  ticker,
  event_time,
  close as price,
  volume
FROM crypto_aggregates
ORDER BY event_time DESC
LIMIT 10;
```

### Check Sentiment Data

```sql
SELECT
  asset,
  sentiment_label,
  sentiment_score,
  confidence,
  event_time
FROM market_sentiment
ORDER BY event_time DESC
LIMIT 5;
```

### Check System Health

```sql
SELECT
  event_type,
  severity,
  message,
  event_time
FROM system_events
ORDER BY event_time DESC
LIMIT 10;
```

## Alternative: Run Individual Components

### WebSocket Only (No Sentiment or Quality)

```bash
python3 src/data_pipeline/main.py --no-sentiment --no-quality
```

### Track Specific Pairs Only

```bash
python3 src/data_pipeline/main.py --tickers X:BTCUSD X:ETHUSD
```

### Test Sentiment Analyzer Standalone

```bash
cd /Volumes/Lexar/RRRVentures/RRRalgorithms/worktrees/data-pipeline
python3 src/data_pipeline/perplexity/sentiment_analyzer.py
```

### Test Quality Validator Standalone

```bash
cd /Volumes/Lexar/RRRVentures/RRRalgorithms/worktrees/data-pipeline
python3 src/data_pipeline/quality/validator.py
```

## Troubleshooting

### Import Error: No module named 'websockets'

Install missing dependencies:
```bash
pip install -r requirements.txt
```

### Connection Error: Supabase

Check your `.env` file has the correct `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`.

### WebSocket Disconnects Frequently

1. Check your network connection
2. Verify Polygon API key is valid
3. Check rate limits (you may need to upgrade your Polygon plan)

### No Data Appearing in Database

1. Check logs for errors: `data_pipeline.log`
2. Verify Supabase connection works
3. Check system_events table for issues

## Next Steps

Once the pipeline is running:

1. **Monitor Performance**: Check `data_pipeline.log` for metrics
2. **Review Data Quality**: Query `system_events` for issues
3. **Build Trading Strategies**: Use the data in `crypto_aggregates` and `market_sentiment`
4. **Set Up Alerts**: Configure notifications for quality issues

## Production Tips

### Run as Background Service

```bash
# Using nohup
nohup python3 src/data_pipeline/main.py > pipeline.log 2>&1 &

# Check if running
ps aux | grep main.py
```

### Monitor Logs in Real-time

```bash
tail -f data_pipeline.log
```

### Stop the Pipeline

```bash
# Find process ID
ps aux | grep main.py

# Kill process
kill <PID>

# Or use Ctrl+C if running in foreground
```

## Performance Expectations

- **WebSocket Messages**: 100-500/second typical
- **Database Inserts**: 50-200/second
- **Sentiment Updates**: Every 15 minutes
- **Quality Checks**: Every 5 minutes
- **CPU Usage**: 10-20% of one core
- **Memory Usage**: 200-500 MB

## Support

For help:
1. Check `README.md` for full documentation
2. Review logs in `data_pipeline.log`
3. Query `system_events` table for issues
4. Check Supabase dashboard for database metrics

---

**Happy Trading!**
