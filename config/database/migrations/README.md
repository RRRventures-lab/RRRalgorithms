# Database Migrations

This directory contains SQL migration files for the RRRalgorithms database schema.

## Migration Files

Migrations are numbered sequentially and should be applied in order:

1. `001_initial_schema.sql` - Initial database schema with all core tables
2. (Future migrations will be added here)

## Running Migrations

### Using Supabase CLI

```bash
# Initialize Supabase (if not already done)
supabase init

# Link to your remote project
supabase link --project-ref isqznbvfmjmghxvctguh

# Run migrations
supabase db push
```

### Manual Application via psql

```bash
# Set environment variables
source config/api-keys/.env

# Apply migration
psql "$SUPABASE_DB_URL" -f config/database/migrations/001_initial_schema.sql
```

### Using Python Script

```bash
# Run the migration script
python scripts/setup/init-supabase.sh
```

## Schema Overview

### Market Data Tables
- `crypto_aggregates` - OHLCV candlestick data
- `crypto_trades` - Individual trade tick data
- `crypto_quotes` - Bid/ask quotes

### Analysis Tables
- `market_sentiment` - Sentiment analysis results
- `trading_signals` - Generated trading signals

### Trading Tables
- `orders` - Order history and status
- `positions` - Current and historical positions
- `portfolio_snapshots` - Portfolio value over time

### Monitoring Tables
- `system_events` - System event logs
- `api_usage` - API call tracking for monitoring

## Creating New Migrations

When creating a new migration:

1. Number it sequentially (e.g., `002_add_user_preferences.sql`)
2. Include both UP and DOWN migrations
3. Test thoroughly in development first
4. Document the changes in this README

### Migration Template

```sql
-- =============================================================================
-- Migration: XXX_description
-- Created: YYYY-MM-DD
-- Description: Brief description of changes
-- =============================================================================

-- UP Migration
-- (Your schema changes here)

-- DOWN Migration (Rollback)
-- DROP TABLE IF EXISTS ...;
-- ALTER TABLE ... DROP COLUMN ...;
```

## Best Practices

1. **Never modify existing migrations** - Create new ones instead
2. **Always test rollbacks** - Ensure DOWN migrations work
3. **Use transactions** - Wrap migrations in BEGIN/COMMIT
4. **Add indexes carefully** - Consider performance impact
5. **Document breaking changes** - Note any API changes
6. **Backup before migrating** - Especially in production

## Troubleshooting

### Migration fails with "relation already exists"
The table may already exist. Check if previous migrations were partially applied:
```sql
SELECT tablename FROM pg_tables WHERE schemaname = 'public';
```

### Permission denied errors
Ensure you're using the service role key, not the anon key:
```bash
export SUPABASE_SERVICE_KEY="your_service_role_key"
```

### Connection timeout
Check your database password and connection string:
```bash
psql "$SUPABASE_DB_URL" -c "SELECT 1"
```
