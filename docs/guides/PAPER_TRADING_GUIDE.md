# Paper Trading Guide

## Overview

Paper trading mode allows you to test the RRR Trading System with live market data **without risking real capital**. All orders are simulated, but the system behaves exactly as it would in production.

## Safety Features

### Multi-Layer Protection

1. **Environment Variable**: `PAPER_TRADING_MODE=true` in all services
2. **Exchange Mode**: `EXCHANGE_MODE=paper` in trading engine
3. **Live Trading Disabled**: `ENABLE_LIVE_TRADING=false` flag
4. **Separate Docker Compose**: `docker-compose.paper-trading.yml` override

### What's Simulated

- ✅ Order execution (instant fills at market price)
- ✅ Position tracking
- ✅ P&L calculation
- ✅ Risk management
- ✅ ML predictions
- ✅ All monitoring and metrics

### What's Real

- ✅ Market data (live prices from exchanges)
- ✅ Technical indicators
- ✅ ML model predictions
- ✅ Database storage
- ✅ Monitoring dashboards

## Quick Start

### 1. Prerequisites

```bash
# Check Docker is running
docker --version
docker-compose --version

# Check environment file exists
ls -la config/api-keys/.env
```

### 2. Start Paper Trading

```bash
./scripts/paper-trading/start-paper-trading.sh
```

This script will:
- ✅ Verify Docker is running
- ✅ Check all required files exist
- ✅ Confirm paper trading mode is enabled
- ✅ Start all services
- ✅ Verify health of all services

### 3. Monitor Performance

```bash
./scripts/paper-trading/monitor-paper-trading.sh
```

Interactive menu with options:
- View service status
- Check trading metrics
- Tail logs
- Open Grafana dashboard

### 4. Stop Paper Trading

```bash
./scripts/paper-trading/stop-paper-trading.sh
```

Gracefully stops all services and exports final metrics.

## Configuration

### Initial Capital

Default: $100,000 (simulated)

Edit in `docker-compose.paper-trading.yml`:
```yaml
services:
  trading-engine:
    environment:
      - INITIAL_CAPITAL=100000  # Change this
```

### Risk Limits

Configure in `docker-compose.paper-trading.yml`:
```yaml
services:
  risk-management:
    environment:
      - DAILY_LOSS_LIMIT=-2000      # Max $2k loss/day
      - MAX_PORTFOLIO_RISK=0.05      # 5% portfolio risk
      - MAX_POSITION_SIZE=0.20       # Max 20% per position
      - KELLY_FRACTION=0.25          # Conservative Kelly
```

### Symbols to Trade

Edit in `worktrees/trading-engine/config/symbols.json`:
```json
{
  "symbols": [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD"
  ]
}
```

## Monitoring

### Grafana Dashboards

Access: http://localhost:3000 (default: admin/admin)

**Available Dashboards:**
1. **System Overview** - Service health, CPU, memory
2. **Trading Performance** - P&L, win rate, Sharpe ratio
3. **ML Model Performance** - Accuracy, confidence, inference latency
4. **Risk Management** - VaR, drawdown, position allocation
5. **Data Pipeline** - Data freshness, WebSocket status

### Prometheus Metrics

Access: http://localhost:9090

**Key Metrics:**
- `trading_orders_total` - Total orders placed
- `trading_daily_pnl` - Daily profit/loss
- `trading_win_rate` - Percentage of winning trades
- `ml_prediction_accuracy` - Model accuracy
- `risk_current_var` - Current value at risk

### Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f trading-engine

# Search for errors
docker-compose logs | grep -i error
```

## Testing Scenarios

### 1. Basic System Test (1 Hour)

**Goal**: Verify all services work together

```bash
# Start system
./scripts/paper-trading/start-paper-trading.sh

# Monitor for 1 hour
./scripts/paper-trading/monitor-paper-trading.sh

# Check results
open http://localhost:3000

# Stop
./scripts/paper-trading/stop-paper-trading.sh
```

**Expected Results:**
- All services healthy
- No errors in logs
- Some orders placed (depends on signals)
- Dashboards show real-time data

### 2. Model Validation Test (24 Hours)

**Goal**: Validate ML model performance

```bash
# Start with validation enabled
docker-compose -f docker-compose.yml \
  -f docker-compose.paper-trading.yml \
  -e MODEL_VALIDATION=true \
  up -d

# Let run for 24 hours
# Monitor accuracy in Grafana
```

**Expected Results:**
- Model accuracy > 55%
- Confidence scores reasonable (0.6-0.8)
- No overfitting (train/val gap < 15%)

### 3. Risk Management Test

**Goal**: Test risk limits and stop-losses

```bash
# Set aggressive limits for testing
# Edit docker-compose.paper-trading.yml:
#   DAILY_LOSS_LIMIT=-500  # Low limit
#   MAX_POSITION_SIZE=0.10  # Small positions

# Start system
./scripts/paper-trading/start-paper-trading.sh

# Monitor risk metrics
# Verify stop-losses trigger correctly
```

### 4. High Volatility Test

**Goal**: Test system during market volatility

**Best times:**
- During major news events
- Market open/close
- Options expiry days

Monitor for:
- Order execution speed
- Risk limit enforcement
- Model behavior under stress

## Troubleshooting

### Services Won't Start

```bash
# Check Docker resources
docker system df

# Clean up old containers
docker system prune -a

# Restart Docker Desktop
```

### No Trading Activity

**Possible causes:**
1. **No signals**: Model not confident enough
   - Check `ml_prediction_confidence` in Grafana
   - Lower confidence threshold in config

2. **Risk limits**: Hitting risk limits
   - Check `risk_current_var` in Prometheus
   - Review risk limit configuration

3. **Data issues**: Not receiving market data
   - Check `data_websocket_connected` metric
   - Review data-pipeline logs

### High Latency

```bash
# Check system resources
docker stats

# Reduce services if needed
docker-compose -f docker-compose.yml \
  -f docker-compose.paper-trading.yml \
  up -d neural-network data-pipeline trading-engine
```

### Database Connection Issues

```bash
# Check Supabase connection
curl -I $SUPABASE_URL

# Test database connectivity
docker-compose exec trading-engine python -c "import psycopg2; print('OK')"
```

## Performance Benchmarks

### Minimum Requirements

- **CPU**: 4 cores
- **RAM**: 8GB
- **Disk**: 20GB free
- **Network**: Stable internet connection

### Expected Performance

- **Order Latency**: < 100ms (p95)
- **ML Inference**: < 500ms (p95)
- **Data Latency**: < 5s
- **System Uptime**: > 99% (during market hours)

## Best Practices

### Before Starting

1. ✅ Review all configuration files
2. ✅ Verify paper trading mode is enabled
3. ✅ Check API keys are set (even for paper trading)
4. ✅ Ensure sufficient disk space
5. ✅ Review risk limits

### During Paper Trading

1. ✅ Monitor dashboards regularly
2. ✅ Check for errors daily
3. ✅ Review trading decisions
4. ✅ Validate ML predictions
5. ✅ Track performance metrics

### After Paper Trading

1. ✅ Export final metrics
2. ✅ Analyze performance
3. ✅ Review all trades
4. ✅ Identify issues or improvements
5. ✅ Update documentation

## Metrics to Track

### Daily

- Total orders placed
- Win rate
- Daily P&L
- Sharpe ratio
- Max drawdown

### Weekly

- Model accuracy trend
- Risk-adjusted returns
- System uptime
- Error frequency
- Latency percentiles

### Before Production

- Minimum 30 days paper trading
- Win rate > 55%
- Sharpe ratio > 1.5
- Max drawdown < 15%
- No critical errors
- System uptime > 99%

## Transition to Live Trading

### Requirements Checklist

- [ ] 30+ days successful paper trading
- [ ] All metrics meet thresholds
- [ ] No critical errors in logs
- [ ] Risk management verified
- [ ] Stop-losses tested
- [ ] Monitoring dashboards reviewed
- [ ] Team approval obtained
- [ ] Real API keys configured
- [ ] Backup procedures tested
- [ ] Emergency shutdown plan ready

### Gradual Rollout

1. **Phase 1**: Live trading with minimum capital (1-5% of target)
2. **Phase 2**: Increase to 10-25% if Phase 1 successful
3. **Phase 3**: Full deployment after 2+ weeks of Phase 2

### Live Trading Differences

- Real money at risk
- Order fills not guaranteed
- Slippage and fees apply
- Exchange rate limits
- Market impact on large orders

## Support

### Issues

- **GitHub Issues**: https://github.com/RRRVentures/RRRalgorithms/issues
- **Discord**: [Link to community]
- **Email**: support@rrrventures.com

### Resources

- **Documentation**: `docs/`
- **Examples**: `notebooks/`
- **Tests**: `tests/`
- **Monitoring**: http://localhost:3000

---

**Remember**: Paper trading is essential but not perfect. Real trading involves additional complexities. Always start with minimal capital when transitioning to live trading.
