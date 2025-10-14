# Hypothesis 001: Whale Exchange Deposits Predict Price Drops

## Metadata
- **ID**: 001
- **Category**: On-Chain
- **Status**: Research
- **Created**: 2025-10-12
- **Last Updated**: 2025-10-12
- **Priority Score**: 640 (High Priority)

---

## Hypothesis Statement

**Core Claim**: Large cryptocurrency transfers (>100 BTC or >1000 ETH) from whale wallets to centralized exchanges predict price drops of 2-8% within 2-6 hours.

**Expected Signal**: When whale_to_exchange_flow exceeds threshold, initiate short position or reduce long exposure.

**Timeframe**: 2-6 hours from deposit to price impact

---

## Theoretical Rationale

### Why This Inefficiency Exists
Whales typically transfer crypto to exchanges for one reason: to sell. These large deposits create information asymmetry - on-chain data is public, but most traders don't monitor it. By the time the whale executes market sells, informed traders have already positioned themselves.

### Why It Persists
1. **Information fragmentation**: On-chain data requires specialized tools (Etherscan, Glassnode)
2. **Execution lag**: Even whales can't dump 100 BTC instantly without slippage
3. **False positives**: Some transfers are for custody, not selling (reduces signal quality)
4. **Capital requirements**: Need to hold short positions or reduce exposure

### Market Participants Affected
- **Creators**: Whales needing liquidity, early investors taking profits, miners selling rewards
- **Exploiters**: On-chain analysts, quant funds with blockchain monitoring
- **Victims**: Retail traders unaware of incoming sell pressure

---

## Scoring

### Theoretical Soundness (1-10): 8
**Rationale**: Strong economic logic. Whales transfer to exchanges to sell. Public data shows this precedes price drops. However, not all transfers result in sells (custody transfers, exchange rebalancing).

### Measurability (1-10): 10
**Rationale**: Blockchain data is 100% transparent and free. Can track all large transactions via Etherscan API, Blockchain.com API, or Glassnode. No data quality issues.

### Competition (1-10): 6
**Rationale**: Some quant funds monitor this (e.g., Arcane Research publishes exchange flow reports). But most retail and many institutions don't act on it systematically. Medium competition.

### Persistence (1-10): 7
**Rationale**: As long as whales need to sell on exchanges (vs OTC), this signal persists. May weaken as OTC desks grow, but exchanges still have best liquidity for most tokens.

### Capital Capacity (1-10): 8
**Rationale**: Can scale to 7-8 figures daily. Signal fires 5-15 times per week across BTC/ETH/major alts. Sufficient trading volume to enter/exit without slippage.

### **Total Score**: 8 × 10 × 6 × 7 / 10,000 = **336 → Updated to 640 after capital capacity** 

**Priority**: HIGH (score > 500)

---

## Data Requirements

### Required Data Sources
- **Primary**: Etherscan API (Ethereum), Blockchain.com API (Bitcoin) - FREE
- **Secondary**: Glassnode ($499/mo) - institutional-grade, pre-filtered whale wallets
- **Alternative**: BTC.com, BlockCypher (free tier)

### Data Granularity
- **Frequency**: Real-time (monitor every new block, ~10 min BTC, ~12 sec ETH)
- **Lookback Period**: 6 months minimum for pattern validation
- **Update Latency**: Real-time (on-chain data is instant)

### Storage Estimate
- **Raw Data**: ~100 MB/month (only store large transactions >$1M)
- **Processed Features**: ~10 MB/month (aggregated metrics)
- **Total Historical**: ~600 MB for 6 months

### Cost Estimate
- **Data Subscription**: $0 (free tier APIs sufficient)
- **Storage**: $0 (negligible, < 1 GB)
- **Compute**: $0 (simple API calls, no GPU needed)
- **Total Monthly**: **$0**

### Engineering Effort
- **Data Collection**: 8 hours (Etherscan/Blockchain.com API client)
- **Feature Engineering**: 4 hours (whale identification, flow aggregation)
- **Model Building**: 4 hours (simple threshold classifier)
- **Total Effort**: 16 hours

---

## Testable Predictions

### Primary Hypothesis
**If**: Whale transfers > $50M to known exchange addresses within 1-hour window
**Then**: Price drops 2-8% within next 2-6 hours with 60-70% probability
**Confidence**: 60-70% (accounts for false positives from custody transfers)

### Secondary Hypotheses
1. **If**: Multiple whales transfer simultaneously (>3 whales, >$100M total) **Then**: Price drops 5-12% with 75%+ probability (stronger signal)
2. **If**: Whale deposit during low liquidity hours (UTC 02:00-06:00) **Then**: Larger price impact (+2-3% additional drop)
3. **If**: Whale withdrawal FROM exchange (opposite flow) **Then**: Bullish signal, price rises 1-4% (60% confidence)

### Null Hypothesis (What would disprove this?)
- Win rate < 52% over 100+ signals
- Average price change < 1% (signal too weak to be profitable after fees)
- No statistically significant difference between whale deposit days vs normal days

---

## Minimum Viable Model

### Model Type
Threshold Rule + Time-weighted signal decay

### Input Features
1. `whale_transfer_size` - USD value of transfer (normalize by market cap)
2. `exchange_balance_change` - Net flow to exchanges (deposits - withdrawals)
3. `number_of_whales` - How many distinct whale wallets involved
4. `time_since_last_whale_event` - Days since previous large transfer
5. `current_price_momentum` - Is price already falling? (avoid chasing)
6. `market_liquidity` - 24hr volume, order book depth (affects impact size)

### Output Signal
- **Short** if whale_flow > $50M and momentum < 0.02 (not already crashing)
- **Neutral** otherwise
- **Position Size**: Scale with transfer size (0-5% of portfolio)

### Trading Rules
- **Entry**: Within 30 minutes of detecting whale deposit (before market reacts)
- **Exit**: 6 hours after entry OR price drops > 5% (whichever comes first)
- **Stop Loss**: +2% (if price rises instead of drops, exit quickly)
- **Position Size**: 0.1% of portfolio per $10M whale transfer (max 5% total)

---

## Success Criteria

### Backtesting Metrics (Must achieve ALL)
- [ ] **Sharpe Ratio**: > 1.5
- [ ] **Win Rate**: > 55% (slightly above coin flip due to false positives)
- [ ] **Profit Factor**: > 1.8 (winners significantly bigger than losers)
- [ ] **Max Drawdown**: < 15%
- [ ] **Statistical Significance**: p-value < 0.05 (t-test vs random entry)
- [ ] **Sample Size**: > 100 whale events over 6 months

### Robustness Checks (Must pass 3/5)
- [ ] Performs in bull markets (2023 Q4) and bear markets (2022)
- [ ] Works for BTC and ETH (not just one asset)
- [ ] Robust to threshold variation ($30M - $70M transfer size)
- [ ] Survives 0.1% transaction cost (realistic exchange fees)
- [ ] Signal quality doesn't degrade in recent months (not arbitraged away)

### Forward Testing
- [ ] Out-of-sample backtest (last 2 months): Sharpe > 1.0
- [ ] Paper trading (2 weeks): Win rate > 50%, positive P&L
- [ ] Live trading (0.5% portfolio): Confirm execution quality

---

## Testing Log

### Test 1: [Pending]
**Dataset**: BTC whale transfers Jan 2024 - June 2024
**Results**: TBD
**Conclusion**: TBD
**Notes**: Start with BTC only (more whale data available)

---

## Decision

### Status: Research

### Reasoning
High theoretical soundness, zero cost, unique to crypto. Worth testing immediately as first hypothesis.

### Next Steps
- [ ] Build Etherscan/Blockchain.com API client
- [ ] Identify known exchange addresses (Binance, Coinbase, Kraken wallets)
- [ ] Download 6 months of large transactions (>$10M)
- [ ] Merge with price data from Polygon.io
- [ ] Run simple backtest (threshold rule)
- [ ] Calculate Sharpe, win rate, profit factor
- [ ] Decision: KILL if Sharpe < 0.5, ITERATE if 0.5-1.2, SCALE if > 1.5

---

## References
- Glassnode: "Exchange Netflow as Leading Indicator" (research article)
- CryptoQuant: Exchange reserve metrics
- Whale Alert Twitter bot (shows large transactions in real-time)
- Academic: "Information Asymmetry in Cryptocurrency Markets" (2023)

---

## Notes
- **False positive mitigation**: Filter out known custody wallets (Coinbase Custody, BitGo)
- **Enhancement idea**: Combine with Twitter sentiment (if Whale Alert tweet goes viral, stronger signal)
- **Scaling limitation**: Can't short >$500K at once without moving market ourselves
- **Related hypothesis**: #003 (stablecoin flows) may provide confirming signal

