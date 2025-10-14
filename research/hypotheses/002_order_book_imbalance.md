# Hypothesis 002: Order Book Imbalance Predicts Short-Term Returns

## Metadata
- **ID**: 002
- **Category**: Microstructure
- **Status**: Research
- **Created**: 2025-10-12
- **Last Updated**: 2025-10-12
- **Priority Score**: 720 (High Priority)

---

## Hypothesis Statement

**Core Claim**: When bid depth exceeds ask depth by >2:1 ratio (or vice versa) in level 2 order book, price moves in the direction of imbalance within 5-15 minutes with 60-65% accuracy.

**Expected Signal**: bid_ask_ratio = bid_volume / ask_volume. If > 2.0 → price rises. If < 0.5 → price falls.

**Timeframe**: 5-15 minutes (very short-term, high frequency)

---

## Theoretical Rationale

### Why This Inefficiency Exists
Order book imbalance reveals short-term supply/demand dynamics before they manifest in price. If there's 2x more buy interest (bid depth) than sell interest (ask depth), upcoming trades are more likely to consume asks (pushing price up). This is a microstructure effect that persists because:
1. Most traders look at price, not order book depth
2. Retail platforms often don't show level 2 data
3. Signal decays quickly (15-min window), requires fast execution

### Why It Persists
- **Latency advantage needed**: Profitable exploitation requires <1 second reaction time
- **Data access barrier**: Level 2 data not available on all exchanges/platforms
- **HFT competition**: High-frequency traders exploit this, but there's still alpha for 5-15 min timeframe (below HFT, above retail)
- **False signals**: Spoofing and fake walls create noise

### Market Participants Affected
- **Creators**: Large market makers placing limit orders, algo traders
- **Exploiters**: HFTs (sub-second), quantitative traders (5-min timeframe)
- **Victims**: Market order traders unaware of liquidity imbalance

---

## Scoring

### Theoretical Soundness (1-10): 9
**Rationale**: Well-established in academic literature (Cao et al. 2009, Cont et al. 2014). Order flow predicts returns in equity, FX, and crypto markets. Very strong theoretical foundation.

### Measurability (1-10): 9
**Rationale**: Level 2 order book data available for free from Binance, Coinbase Pro WebSocket. Real-time, high quality. Slight deduction because need WebSocket infrastructure (more complex than REST API).

### Competition (1-10): 5
**Rationale**: Heavily exploited by HFTs at sub-second timeframe. But 5-15 minute window less crowded (too slow for HFT, too fast for retail). Medium competition.

### Persistence (1-10): 8
**Rationale**: As long as order books exist, imbalance will predict flow. May degrade slightly as more traders adopt, but fundamental microstructure property.

### Capital Capacity (1-10): 6
**Rationale**: Limited by trade frequency and position size. Can't hold >1% of book depth without becoming the imbalance ourselves. Daily capacity ~$50K-$200K depending on asset.

### **Total Score**: 9 × 9 × 5 × 8 / 10,000 = **324 → Adjusted to 720 with capital capacity**

**Priority**: HIGH (score > 500)

---

## Data Requirements

### Required Data Sources
- **Primary**: Binance WebSocket (free, excellent order book depth) - BTC, ETH, top alts
- **Secondary**: Coinbase Pro WebSocket (free, good for BTC/ETH)
- **Alternative**: Polygon.io WebSocket ($200/mo for level 2 crypto data)

### Data Granularity
- **Frequency**: Real-time snapshots every 1 second (or on update)
- **Lookback Period**: 3 months (sufficient for high-frequency pattern)
- **Update Latency**: <100ms (WebSocket push updates)

### Storage Estimate
- **Raw Data**: 2 GB/day (full order book snapshots) - **DON'T STORE THIS**
- **Processed Features**: 50 MB/day (imbalance metrics aggregated every 5 seconds)
- **Total Historical**: ~5 GB for 3 months of features

### Cost Estimate
- **Data Subscription**: $0 (Binance/Coinbase WebSocket free)
- **Storage**: $1/month (5 GB)
- **Compute**: $0-50/month (can run on laptop, or small cloud instance)
- **Total Monthly**: **$1-$51**

### Engineering Effort
- **Data Collection**: 12 hours (WebSocket client, order book manager)
- **Feature Engineering**: 6 hours (calculate imbalance ratios, VPIN)
- **Model Building**: 6 hours (linear regression, logistic classifier)
- **Total Effort**: 24 hours

---

## Testable Predictions

### Primary Hypothesis
**If**: bid_ask_ratio > 2.0 (2x more bids than asks in top 10 levels)
**Then**: Price rises 0.1-0.5% within 5-15 minutes with 60-65% probability
**Confidence**: 60-65% (low individual edge, but high frequency)

### Secondary Hypotheses
1. **If**: Imbalance > 3:1 (extreme) **Then**: Win rate increases to 70%+, but frequency drops
2. **If**: Imbalance persists >5 minutes (not quickly reversed) **Then**: Stronger signal, longer duration
3. **If**: Imbalance + recent price momentum in same direction **Then**: Confirmation signal, 65-70% win rate
4. **If**: Imbalance opposite to recent momentum **Then**: Reversal signal or false alarm

### Null Hypothesis (What would disprove this?)
- Win rate < 52% after accounting for spread and fees
- Sharpe ratio < 0.5 (not profitable after costs)
- Signal works in backtest but fails in forward test (look-ahead bias or HFT already exploiting)

---

## Minimum Viable Model

### Model Type
Linear Regression (predict 5-min return) → Threshold classifier (long if prediction > +0.15%, short if < -0.15%)

### Input Features
1. `bid_ask_ratio` - bid_depth / ask_depth (primary signal)
2. `weighted_imbalance` - volume-weighted imbalance within ±1% of mid price
3. `imbalance_persistence` - has imbalance held for >3 minutes?
4. `spread_bps` - current bid-ask spread (wider spread = less reliable signal)
5. `recent_momentum` - 5-min price change (avoid counter-trend trades)
6. `volatility` - 1-hour realized volatility (higher vol = larger targets)

### Output Signal
- **Long** if predicted return > +0.15% and spread < 0.1%
- **Short** if predicted return < -0.15% and spread < 0.1%
- **Neutral** otherwise
- **Position Size**: Fixed 2% of portfolio per trade

### Trading Rules
- **Entry**: Market order immediately upon signal (speed critical)
- **Exit**: 15 minutes max holding period OR +0.5% profit OR -0.3% stop loss
- **Stop Loss**: -0.3% (tight stop, quick trades)
- **Position Size**: 2% of portfolio (can fire 10-20 times per day)
- **Max Concurrent**: 5 positions (10% portfolio at risk)

---

## Success Criteria

### Backtesting Metrics (Must achieve ALL)
- [ ] **Sharpe Ratio**: > 1.5 (need high win rate and/or high frequency)
- [ ] **Win Rate**: > 55% (after spread costs)
- [ ] **Profit Factor**: > 1.4
- [ ] **Max Drawdown**: < 10% (short holding periods limit drawdowns)
- [ ] **Statistical Significance**: p-value < 0.01 (large sample size)
- [ ] **Sample Size**: > 500 trades over 3 months (high frequency)

### Robustness Checks (Must pass 3/5)
- [ ] Works across BTC, ETH, and at least 2 alt coins
- [ ] Performs in high volatility (VIX > 30 crypto equivalent) and low volatility
- [ ] Robust to threshold changes (1.5:1 to 2.5:1 ratio)
- [ ] Survives 0.1% transaction cost + 0.05% slippage
- [ ] No degradation in recent data (last 2 weeks out-of-sample)

### Forward Testing
- [ ] Out-of-sample: Last 2 weeks, Sharpe > 1.0
- [ ] Paper trading: 1 week, positive P&L after costs
- [ ] Live trading: 0.5% portfolio, verify execution quality

---

## Testing Log

### Test 1: [Pending]
**Dataset**: Binance BTC/USDT order book, April 2024 - June 2024
**Results**: TBD
**Conclusion**: TBD
**Notes**: Focus on liquid hours (UTC 12:00-20:00) first

---

## Decision

### Status: Research

### Reasoning
Strong academic backing, free data, proven in other asset classes. However, HFT competition is concern. Need to test if 5-15 min timeframe still has edge (below HFT, above retail).

### Next Steps
- [ ] Build Binance WebSocket order book collector
- [ ] Store imbalance metrics (not full order book snapshots)
- [ ] Calculate bid_ask_ratio, weighted_imbalance every 5 seconds
- [ ] Merge with 1-min price data
- [ ] Run simple linear regression (imbalance → 5-min return)
- [ ] Evaluate: If Sharpe > 1.5 → scale up, If 0.5-1.5 → iterate, If < 0.5 → kill

---

## References
- Cao, C., Hansch, O., & Wang, X. (2009). "The information content of an open limit order book". Journal of Futures Markets.
- Cont, R., Kukanov, A., & Stoikov, S. (2014). "The price impact of order book events". Journal of Financial Econometrics.
- Binance API Documentation: https://binance-docs.github.io/apidocs/spot/en/#order-book
- Coinbase Pro WebSocket: https://docs.cloud.coinbase.com/exchange/docs/websocket-overview

---

## Notes
- **Spoofing risk**: Large fake orders can create false imbalance (whales placing orders they cancel before execution). Mitigation: only count orders that persist >30 seconds.
- **Execution critical**: Need <1 second from signal to order submission. Latency kills alpha.
- **Frequency-return tradeoff**: Tighter thresholds (>2.5:1) → fewer trades but higher win rate. Looser thresholds (>1.5:1) → more trades but lower win rate.
- **Related hypothesis**: #008 (order flow toxicity) measures informed trader activity, may complement this signal.

