# Integration Tests for RRRalgorithms

## Overview

This directory contains comprehensive integration tests that validate the end-to-end functionality of the RRRalgorithms cryptocurrency trading system. These tests ensure that all components work correctly together through the Supabase database layer.

## Test Suites

### 1. MCP Connection Tests (`test_mcp_connections.py`)

Validates that all Model Context Protocol servers and external services are properly configured:

- ✅ Supabase database connection
- ✅ Real-time subscriptions enabled
- ✅ Polygon.io API key validity
- ✅ Perplexity AI API key validity
- ✅ Environment variables loaded
- ✅ Database query/write performance
- ✅ Paper trading mode enabled (safety check)

**Total Tests**: 15

### 2. Real-time Subscription Tests (`test_realtime_subscriptions.py`)

Tests the real-time data propagation between worktrees via Supabase subscriptions:

- ✅ Signal insertion detection
- ✅ Order execution flow
- ✅ Position updates
- ✅ Portfolio snapshot creation
- ✅ Market data streaming
- ✅ Sentiment updates
- ✅ System event logging
- ✅ Concurrent writes
- ✅ Performance under load
- ✅ Cross-worktree communication

**Total Tests**: 14

### 3. End-to-End Pipeline Tests (`test_end_to_end_pipeline.py`)

Tests the complete trading pipeline from market data to order execution:

- ✅ Data pipeline writes to database
- ✅ Neural network signal generation
- ✅ Risk management validation
- ✅ Order execution flow
- ✅ Position P&L calculation
- ✅ Stop loss triggers
- ✅ Sentiment integration
- ✅ Portfolio snapshots
- ✅ System event logging
- ✅ Complete trading cycle (signal → open → close)

**Total Tests**: 10

**Total Integration Tests**: 39

## Prerequisites

### 1. Environment Setup

Ensure your environment is configured:

```bash
# Environment variables must be loaded
source config/api-keys/.env

# Verify Supabase is configured
echo $SUPABASE_URL
echo $SUPABASE_ANON_KEY
```

### 2. Database Setup

Supabase database must be initialized:

```bash
# Run database initialization
./scripts/setup/init-supabase.sh
```

### 3. Python Dependencies

Install test dependencies:

```bash
# Create virtual environment (if not exists)
python3 -m venv .venv
source .venv/bin/activate

# Install test requirements
pip install -r tests/integration/requirements.txt
```

## Running Tests

### Quick Start (Recommended)

Run all integration tests with the provided script:

```bash
# From project root
./tests/integration/run_integration_tests.sh
```

This script will:
1. Load environment variables
2. Check Python version
3. Install dependencies
4. Run all 3 test suites
5. Display comprehensive results

### Run Individual Test Suites

```bash
# MCP Connection Tests
pytest tests/integration/test_mcp_connections.py -v

# Real-time Subscription Tests
pytest tests/integration/test_realtime_subscriptions.py -v

# End-to-End Pipeline Tests
pytest tests/integration/test_end_to_end_pipeline.py -v
```

### Run Specific Tests

```bash
# Run a specific test
pytest tests/integration/test_mcp_connections.py::TestMCPConnections::test_supabase_mcp_connection -v

# Run tests matching a pattern
pytest tests/integration/ -k "realtime" -v
```

### Run with Coverage

```bash
# Install coverage
pip install pytest-cov

# Run with coverage report
pytest tests/integration/ --cov=. --cov-report=html

# View coverage
open htmlcov/index.html
```

## Test Output

### Success Output

```
============================================================================
RRRalgorithms Integration Test Suite
============================================================================

Loading environment variables...
✓ Environment loaded

Checking Python version...
✓ Python 3.11.0

✓ Dependencies installed

Verifying Supabase connection...
✓ Supabase configured

============================================================================
Running Integration Tests
============================================================================

[1/3] Testing MCP Connections...
test_mcp_connections.py::TestMCPConnections::test_supabase_mcp_connection PASSED
test_mcp_connections.py::TestMCPConnections::test_supabase_realtime_enabled PASSED
... (13 more tests)
✅ MCP Connection Tests PASSED

[2/3] Testing Real-time Subscriptions...
test_realtime_subscriptions.py::TestRealtimeSubscriptions::test_signal_insertion_detected PASSED
... (13 more tests)
✅ Real-time Subscription Tests PASSED

[3/3] Testing End-to-End Pipeline...
test_end_to_end_pipeline.py::TestEndToEndPipeline::test_data_pipeline_to_database PASSED
... (9 more tests)
✅ End-to-End Pipeline Tests PASSED

============================================================================
ALL INTEGRATION TESTS PASSED ✅
============================================================================

System is ready for paper trading deployment!
```

## Test Data Cleanup

All tests clean up their data automatically:

- Test data uses unique identifiers (e.g., `TEST-USD`, `COMM-TEST`)
- Cleanup runs after each test via `pytest` fixtures
- Manual cleanup (if needed):

```python
from supabase import create_client
import os

supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_ANON_KEY'))

# Remove test data
supabase.table('trading_signals').delete().like('symbol', '%TEST%').execute()
supabase.table('orders').delete().like('symbol', '%TEST%').execute()
supabase.table('positions').delete().like('symbol', '%TEST%').execute()
```

## Performance Benchmarks

Expected performance metrics:

| Metric | Target | Typical |
|--------|--------|---------|
| Database write latency | <1000ms | ~200-400ms |
| Database query latency (100 records) | <500ms | ~100-200ms |
| Signal → Order flow | <200ms | ~100ms |
| Position update propagation | <100ms | ~50ms |
| Concurrent writes (5 threads) | All succeed | ✅ |

## Troubleshooting

### Connection Errors

```bash
# Verify Supabase credentials
./scripts/setup/check-credentials.sh

# Test direct connection
psql "$SUPABASE_DB_URL" -c "SELECT 1;"
```

### Import Errors

```bash
# Reinstall dependencies
pip install --force-reinstall -r tests/integration/requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

### API Rate Limits

If you hit Polygon.io or Perplexity rate limits during testing:

```bash
# Wait 60 seconds between test runs
sleep 60
pytest tests/integration/
```

### Database Connection Limits

If you get "too many connections" errors:

```python
# Close existing connections
supabase._client.close()
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r tests/integration/requirements.txt
      - name: Run integration tests
        env:
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_ANON_KEY: ${{ secrets.SUPABASE_ANON_KEY }}
          POLYGON_API_KEY: ${{ secrets.POLYGON_API_KEY }}
          PERPLEXITY_API_KEY: ${{ secrets.PERPLEXITY_API_KEY }}
        run: |
          pytest tests/integration/ -v
```

## Next Steps After Tests Pass

Once all integration tests pass:

1. **Start Data Pipeline**
   ```bash
   cd worktrees/data-pipeline
   python src/main.py
   ```

2. **Start Neural Network**
   ```bash
   cd worktrees/neural-network
   python src/main.py
   ```

3. **Start Trading Engine**
   ```bash
   cd worktrees/trading-engine
   python src/main.py --mode paper
   ```

4. **Start Monitoring Dashboard**
   ```bash
   cd worktrees/monitoring
   streamlit run src/dashboard/app.py
   ```

5. **Monitor System Health**
   - Open dashboard at http://localhost:8501
   - Watch for signals, orders, positions
   - Monitor risk metrics and P&L

## Support

If tests fail unexpectedly:

1. Check `docs/integration/SYSTEM_INTEGRATION_SUMMARY.md` for architecture details
2. Review worktree-specific READMEs for component documentation
3. Check Supabase dashboard for database status
4. Review system logs in `logs/` directory

## Contributing

When adding new features, update integration tests:

1. Add tests for new database tables
2. Add tests for new MCP servers
3. Add tests for new worktree communication patterns
4. Update this README with new test descriptions

---

**Last Updated**: 2025-10-11
**Total Tests**: 39
**Test Coverage**: End-to-end system integration
