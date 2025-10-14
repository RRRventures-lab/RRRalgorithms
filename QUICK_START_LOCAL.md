# Quick Start - Local Development

Get RRRalgorithms running on your laptop in **5 minutes** with no Docker, no cloud services, and minimal resources.

## Prerequisites

- **Python 3.11+** (check with `python3 --version`)
- **2-4GB free RAM**
- **2-5GB disk space**
- **macOS, Linux, or Windows with WSL**

No Docker, no cloud accounts, no API keys required!

## One-Command Setup

```bash
# Run the setup script
./scripts/setup/setup-local.sh
```

That's it! The script will:
1. Create a Python virtual environment
2. Install minimal dependencies (~300MB)
3. Initialize SQLite database with sample data
4. Configure everything for local development

## Start Trading System

```bash
# Activate virtual environment
source venv/bin/activate

# Start the system
./scripts/dev/start-local.sh
```

You should see:
```
✓ Activating virtual environment...
✓ Starting trading system...

[INFO] Initializing RRRalgorithms Trading System...
[INFO] Environment: local
[INFO] Database: sqlite
[SUCCESS] ✓ Data pipeline initialized (mock mode, 2 symbols)
[SUCCESS] ✓ Trading engine initialized (paper mode, $10,000)
[SUCCESS] ✓ Risk management initialized
[SUCCESS] ✓ Trading system started

Press Ctrl+C to stop
```

## What's Running?

The system is now:
- ✅ Generating mock market data (BTC-USD, ETH-USD)
- ✅ Making ML predictions (mock predictor)
- ✅ Storing data in SQLite database
- ✅ Running in paper trading mode
- ✅ All in a single Python process

## View Live Dashboard

In another terminal:

```bash
source venv/bin/activate
python -m src.main --service monitor
```

You'll see real-time:
- Portfolio value and P&L
- Trading statistics
- Win rate
- Service status

## Run Specific Services

```bash
# Data pipeline only
./scripts/dev/run-service.sh data_pipeline

# Trading engine only
./scripts/dev/run-service.sh trading_engine

# Monitoring dashboard
./scripts/dev/run-service.sh monitor
```

## Check Status

```bash
python -m src.main --status
```

## View Logs

```bash
# Show available logs
./scripts/dev/show-logs.sh

# Tail specific log
./scripts/dev/show-logs.sh trading
```

## Run Tests

```bash
# All tests
pytest tests/

# Unit tests only (fast)
pytest tests/unit/ -v

# With coverage
pytest --cov=src tests/
```

## Configuration

### Main Config

Edit `config/local.yml` to customize:
- Trading parameters (capital, risk limits)
- Data sources (mock, historical, live)
- ML model mode (mock, lightweight, full)
- Monitoring settings

### Environment Variables

Edit `.env.local` to set:
- `NN_MODE` - Neural network mode (mock/lightweight/full)
- `DATA_MODE` - Data source (mock/historical/live)
- `LOG_LEVEL` - Logging verbosity (INFO/DEBUG)

### Using Real APIs (Optional)

To use real market data instead of mock:

1. Edit `.env.local`:
```bash
DATA_MODE=live
COINBASE_API_KEY=your_key_here
COINBASE_API_SECRET=your_secret_here
```

2. Update `config/local.yml`:
```yaml
data_pipeline:
  mode: live
  live:
    enabled: true
```

## Database

Your trading data is stored in `data/local.db` (SQLite).

### View Database

```bash
# Install SQLite browser (optional)
brew install sqlite  # macOS
apt-get install sqlite3  # Linux

# Query database
sqlite3 data/local.db "SELECT * FROM trades LIMIT 10;"

# Or use Python
python -c "from src.core.database.local_db import get_db; db = get_db(); print(db.get_trades(limit=10))"
```

### Reset Database

```bash
python scripts/setup/init-local-db.py --reset
```

## Project Structure

```
RRRalgorithms/
├── src/                      # All source code
│   ├── main.py              # Entry point
│   ├── core/
│   │   ├── config/          # Configuration loader
│   │   └── database/        # SQLite database
│   ├── data-pipeline/       # Mock data source
│   ├── neural-network/      # Mock predictor
│   ├── trading-engine/      # Trading logic
│   ├── risk-management/     # Risk controls
│   └── monitoring/          # Local monitor
├── config/
│   ├── local.yml           # Local config (default)
│   └── production.yml      # Production config
├── scripts/
│   ├── setup/              # Setup scripts
│   └── dev/                # Development scripts
├── tests/                  # Test suite
├── data/                   # SQLite database
├── logs/                   # Log files
└── venv/                   # Virtual environment
```

## Upgrading Features

### Add Lightweight ML

```bash
pip install scikit-learn scipy ta
```

Then set in `.env.local`:
```bash
NN_MODE=lightweight
```

### Add Full ML (Heavy!)

⚠️ Requires 8GB+ RAM and 2GB+ disk space

```bash
pip install -r requirements-full.txt
```

Then set in `.env.local`:
```bash
NN_MODE=full
```

### Add Better Terminal UI

```bash
pip install rich
```

The monitor will automatically use rich for beautiful output.

## Common Tasks

### Stop the System

Press `Ctrl+C` in the terminal, or:

```bash
./scripts/dev/stop-local.sh
```

### View Portfolio

```python
from src.core.database.local_db import get_db

db = get_db()
metrics = db.get_latest_portfolio_metrics()
print(f"Portfolio Value: ${metrics['total_value']:,.2f}")
print(f"P&L: ${metrics['total_pnl']:+,.2f}")
```

### Get Historical Data

```python
from src.data_pipeline.mock_data_source import MockDataSource

source = MockDataSource()
df = source.get_historical_data('BTC-USD', periods=100)
print(df.tail())
```

### Make Prediction

```python
from src.neural_network.mock_predictor import MockPredictor

predictor = MockPredictor()
pred = predictor.predict('BTC-USD', 50000)
print(f"Predicted: ${pred['predicted_price']:,.2f}")
print(f"Direction: {pred['direction']}")
print(f"Confidence: {pred['confidence']:.1%}")
```

## Troubleshooting

### Python Version Error

```bash
# Check version
python3 --version

# Install Python 3.11+ if needed
# macOS: brew install python@3.11
# Ubuntu: apt-get install python3.11
```

### Module Not Found

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements-local.txt
```

### Database Locked

```bash
# Stop all processes
./scripts/dev/stop-local.sh

# Reset database
python scripts/setup/init-local-db.py --reset
```

### Port Already in Use

The local system doesn't use ports by default. If you're running services separately, check for conflicts:

```bash
lsof -i :8000  # Check what's using port 8000
```

## Next Steps

1. **Explore the code**: Start with `src/main.py`
2. **Write a strategy**: Create your trading logic
3. **Add indicators**: Implement technical indicators
4. **Run backtests**: Test strategies on historical data
5. **Deploy to production**: See `deployment/README.md`

## Getting Help

- **Documentation**: See `docs/` directory
- **Examples**: Check `notebooks/` for Jupyter notebooks
- **Tests**: Look at `tests/` for usage examples
- **Config**: Review `config/local.yml` for all options

## What's Different from Production?

| Feature | Local | Production |
|---------|-------|------------|
| Database | SQLite | PostgreSQL |
| Cache | In-memory | Redis |
| Data | Mock generator | Real APIs |
| ML Models | Mock predictor | Transformers |
| Monitoring | Console | Prometheus/Grafana |
| Deployment | Native Python | Docker/K8s |
| Resources | 2-4GB RAM | 8GB+ RAM |

## Benefits of Local Development

✅ **Fast**: No Docker overhead, instant startup
✅ **Lightweight**: 2-4GB RAM vs 8GB+ for Docker
✅ **Simple**: Single Python process, easy debugging
✅ **Offline**: No API keys or internet required
✅ **Cheap**: Free, no cloud costs
✅ **Portable**: Works on any laptop

## Ready for Production?

When your strategy is ready, see:
- `deployment/README.md` - Docker deployment
- `config/production.yml` - Production configuration
- `DEPLOYMENT_STATUS_REPORT.md` - Production checklist

---

**Happy Trading! 📈**

For questions or issues, check the main README.md or open an issue.

