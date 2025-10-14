# Supabase Integration Guide

## Overview

This guide walks you through integrating Supabase as the primary database for the RRRalgorithms trading system. Supabase provides:

- **PostgreSQL Database**: Full-featured relational database with TimescaleDB capabilities
- **Real-time Subscriptions**: Listen to database changes in real-time
- **Row Level Security**: Built-in security policies
- **RESTful API**: Automatic REST API for all tables
- **Edge Functions**: Serverless functions for custom logic
- **MCP Integration**: Connect via Model Context Protocol for AI workflows

## Prerequisites

- [ ] Supabase account (create at https://supabase.com)
- [ ] Project created in Supabase
- [ ] PostgreSQL client (`psql`) installed locally

## Step 1: Get Supabase Credentials

### 1.1 Get Project URL and Anon Key

1. Go to your Supabase project dashboard
2. Navigate to **Settings** → **API**
3. You'll see:
   - **Project URL**: `https://your-project-id.supabase.co`
   - **anon/public key**: Long JWT token starting with `eyJ...`

### 1.2 Get Service Role Key

⚠️ **IMPORTANT**: This key has admin privileges - keep it secure!

1. Same page as above (Settings → API)
2. Scroll down to find **service_role** key
3. Click "Reveal" and copy the key
4. This key starts with `eyJ...` but is different from the anon key

Direct link: `https://supabase.com/dashboard/project/isqznbvfmjmghxvctguh/settings/api`

### 1.3 Get Database Password

1. Navigate to **Settings** → **Database**
2. Look for **Connection Info** section
3. Find or reset your database password
4. Copy the password (you'll need it for connection strings)

Direct link: `https://supabase.com/dashboard/project/isqznbvfmjmghxvctguh/settings/database`

## Step 2: Configure Environment Variables

### 2.1 Edit the .env file

```bash
cd /Volumes/Lexar/RRRVentures/RRRalgorithms
nano config/api-keys/.env
```

### 2.2 Update these values:

```bash
# Replace this placeholder
SUPABASE_SERVICE_KEY=your_service_role_key_here
# With your actual service role key from Step 1.2

# Replace YOUR_DB_PASSWORD in these URLs
SUPABASE_DB_URL=postgresql://postgres.isqznbvfmjmghxvctguh:YOUR_DB_PASSWORD@aws-0-us-west-1.pooler.supabase.com:6543/postgres
DATABASE_URL=postgresql://postgres.isqznbvfmjmghxvctguh:YOUR_DB_PASSWORD@aws-0-us-west-1.pooler.supabase.com:6543/postgres
# With your actual database password from Step 1.3
```

### 2.3 Save and verify

```bash
# Save file (Ctrl+O, Enter, Ctrl+X in nano)

# Verify credentials
./scripts/setup/check-credentials.sh
```

You should see:
```
✅ All critical credentials configured!
You can now run: ./scripts/setup/init-supabase.sh
```

## Step 3: Initialize Database Schema

### 3.1 Run the initialization script

```bash
./scripts/setup/init-supabase.sh
```

This script will:
1. ✅ Validate your credentials
2. ✅ Test database connection
3. ✅ Apply the schema (create all tables)
4. ✅ Enable real-time subscriptions
5. ✅ Create helper functions
6. ✅ Set up Row Level Security policies

### 3.2 Expected Output

```
=============================================================================
Supabase Database Initialization
=============================================================================

Step 1: Loading environment variables...
✅ Environment variables loaded

Step 2: Validating Supabase credentials...
✅ SUPABASE_URL: https://isqznbvfmjmghxvctguh.supabase.co
✅ SUPABASE_ANON_KEY: eyJhbGciOiJIUzI1NiIs...
✅ SUPABASE_DB_URL: postgresql://postgres.***

Step 3: Testing database connection...
✅ Database connection successful

Step 4: Applying Supabase schema...
✅ Schema applied successfully

Step 5: Verifying tables...
✅ Found 12 tables

Step 6: Verifying real-time subscriptions...
✅ Real-time enabled on 8 tables

=============================================================================
✅ Supabase Setup Complete!
=============================================================================
```

## Step 4: Verify Tables in Supabase Dashboard

1. Go to **Table Editor** in your Supabase dashboard
2. You should see these tables:
   - `crypto_aggregates` - OHLCV bars
   - `crypto_trades` - Individual trades
   - `crypto_quotes` - Bid/ask quotes
   - `market_sentiment` - Sentiment data
   - `orders` - Trading orders
   - `positions` - Portfolio positions
   - `portfolio_snapshots` - Portfolio history
   - `trading_signals` - Trading signals
   - `ml_models` - ML model registry
   - `model_predictions` - Model predictions
   - `system_events` - System logs
   - `api_usage` - API usage tracking

## Step 5: Test MCP Connection

### 5.1 Verify MCP Configuration

The MCP server is already configured in `config/mcp-servers/mcp-config.json`:

```json
{
  "mcpServers": {
    "supabase": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-postgres",
        "${SUPABASE_DB_URL}"
      ],
      "description": "Supabase (PostgreSQL) database"
    }
  }
}
```

### 5.2 Test Connection via Script

```bash
./scripts/setup/verify-supabase-mcp.sh
```

### 5.3 Test in Claude Code

If you're using Claude Code with MCP support, you can now query the database:

```
User: "Show me the structure of the crypto_aggregates table"
Claude: [Uses MCP to query the database and shows the schema]

User: "Insert a test record for BTC"
Claude: [Uses MCP to insert test data]
```

## Step 6: Using Supabase in Your Application

### 6.1 Python Client Example

Create a file: `worktrees/data-pipeline/src/data_pipeline/supabase_client.py`

```python
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv("config/api-keys/.env")

class SupabaseClient:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_SERVICE_KEY")  # Use service key for backend
        self.client: Client = create_client(self.url, self.key)

    def insert_crypto_aggregate(self, data):
        """Insert OHLCV data"""
        return self.client.table("crypto_aggregates").insert(data).execute()

    def get_latest_prices(self, ticker: str, limit: int = 10):
        """Get latest prices for a ticker"""
        return (
            self.client
            .table("crypto_aggregates")
            .select("*")
            .eq("ticker", ticker)
            .order("event_time", desc=True)
            .limit(limit)
            .execute()
        )

    def subscribe_to_trades(self, callback):
        """Subscribe to real-time trades"""
        return (
            self.client
            .table("crypto_trades")
            .on("INSERT", callback)
            .subscribe()
        )

# Usage
if __name__ == "__main__":
    client = SupabaseClient()

    # Insert data
    result = client.insert_crypto_aggregate({
        "ticker": "X:BTCUSD",
        "event_time": "2025-10-11T14:00:00Z",
        "open": 67000,
        "high": 67500,
        "low": 66800,
        "close": 67200,
        "volume": 150.5,
        "vwap": 67100
    })

    print("Inserted:", result)

    # Query data
    prices = client.get_latest_prices("X:BTCUSD")
    print("Latest prices:", prices)
```

### 6.2 Install Supabase Python Client

```bash
cd worktrees/data-pipeline
pip install supabase
```

## Step 7: Enable Real-time Subscriptions

### 7.1 In Supabase Dashboard

1. Go to **Database** → **Publications**
2. Verify that `supabase_realtime` publication exists
3. Check that these tables are included:
   - crypto_aggregates
   - crypto_trades
   - crypto_quotes
   - orders
   - positions
   - trading_signals

### 7.2 Test Real-time in Python

```python
from supabase_client import SupabaseClient

client = SupabaseClient()

def handle_new_trade(payload):
    print("New trade:", payload)

# Subscribe to trades
subscription = client.subscribe_to_trades(handle_new_trade)

print("Listening for new trades... Press Ctrl+C to stop")
# Keep running to receive updates
import time
while True:
    time.sleep(1)
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    RRRalgorithms System                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              │
                              ▼
    ┌─────────────────────────────────────────────────────┐
    │                  MCP Integration                     │
    │  (Model Context Protocol for AI Workflows)          │
    └─────────────────────────────────────────────────────┘
                              │
                              │
                              ▼
    ┌─────────────────────────────────────────────────────┐
    │                  Supabase Cloud                      │
    │  ┌───────────────────────────────────────────────┐  │
    │  │     PostgreSQL Database (TimescaleDB)         │  │
    │  │  - OHLCV data                                 │  │
    │  │  - Trading orders                             │  │
    │  │  - Positions & Portfolio                      │  │
    │  │  - ML models & predictions                    │  │
    │  │  - System logs                                │  │
    │  └───────────────────────────────────────────────┘  │
    │  ┌───────────────────────────────────────────────┐  │
    │  │     Real-time Subscriptions                   │  │
    │  │  - Live price updates                         │  │
    │  │  - Order status changes                       │  │
    │  │  - Trading signals                            │  │
    │  └───────────────────────────────────────────────┘  │
    │  ┌───────────────────────────────────────────────┐  │
    │  │     RESTful API & Row Level Security          │  │
    │  └───────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────┘
                              │
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │   Trading   │  │    Data     │  │   Neural    │
    │   Engine    │  │   Pipeline  │  │   Network   │
    │             │  │             │  │   Training  │
    └─────────────┘  └─────────────┘  └─────────────┘
```

## Troubleshooting

### Issue: "Connection refused"

**Solution**: Check your database password and connection URL
```bash
# Verify .env file
cat config/api-keys/.env | grep SUPABASE_DB_URL

# Test connection manually
psql "postgresql://postgres.isqznbvfmjmghxvctguh:YOUR_PASSWORD@aws-0-us-west-1.pooler.supabase.com:6543/postgres" -c "SELECT 1"
```

### Issue: "Permission denied"

**Solution**: Verify you're using the service_role key, not the anon key
```bash
cat config/api-keys/.env | grep SUPABASE_SERVICE_KEY
```

### Issue: "Table does not exist"

**Solution**: Re-run the schema initialization
```bash
./scripts/setup/init-supabase.sh
```

### Issue: "Real-time not working"

**Solution**: Check publications in Supabase dashboard
1. Go to Database → Publications
2. Verify `supabase_realtime` includes your tables
3. If missing, run this SQL in SQL Editor:
```sql
ALTER PUBLICATION supabase_realtime ADD TABLE crypto_aggregates;
ALTER PUBLICATION supabase_realtime ADD TABLE crypto_trades;
-- Repeat for all tables
```

## Next Steps

After completing Supabase integration:

1. **Set up data pipeline**
   ```bash
   cd worktrees/data-pipeline
   python src/data_pipeline/ingest.py
   ```

2. **Configure Polygon.io WebSocket**
   - Real-time data streaming into Supabase
   - Location: `worktrees/api-integration/`

3. **Build Perplexity sentiment engine**
   - Store sentiment data in `market_sentiment` table
   - Location: `worktrees/api-integration/`

4. **Start neural network training**
   - Use data from Supabase for training
   - Store model metadata in `ml_models` table

## Resources

- **Supabase Documentation**: https://supabase.com/docs
- **PostgreSQL MCP Server**: https://github.com/modelcontextprotocol/servers/tree/main/src/postgres
- **Supabase Python Client**: https://supabase.com/docs/reference/python/introduction
- **Real-time Subscriptions**: https://supabase.com/docs/guides/realtime

## Security Best Practices

✅ **DO:**
- Use service_role key only in backend/server code
- Use anon key for frontend/client code
- Enable Row Level Security (RLS) on all tables
- Rotate service_role key monthly
- Keep `.env` file secure (chmod 600)

❌ **DON'T:**
- Commit `.env` file to git
- Share service_role key publicly
- Disable RLS without good reason
- Use service_role key in frontend code
- Store API keys in code files

---

**Status**: Setup complete ✅
**Last Updated**: 2025-10-11
**Version**: 1.0.0
