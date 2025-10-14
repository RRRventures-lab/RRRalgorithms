# Hypothesis [ID]: [Short Title]

## Metadata
- **ID**: [3-digit number, e.g., 001]
- **Category**: [On-Chain | Microstructure | Arbitrage | Sentiment | Regime-Dependent]
- **Status**: [Research | Testing | Validated | Killed | Production]
- **Created**: [YYYY-MM-DD]
- **Last Updated**: [YYYY-MM-DD]
- **Priority Score**: [Calculated score, target > 500]

---

## Hypothesis Statement

**Core Claim**: [One sentence describing the inefficiency]

**Expected Signal**: [What we expect to observe if hypothesis is true]

**Timeframe**: [How long after trigger until price impact occurs]

---

## Theoretical Rationale

### Why This Inefficiency Exists
[Explain the economic/behavioral reason this pattern should exist]

### Why It Persists
[Why hasn't this been arbitraged away? Information asymmetry? Execution difficulty? Capital constraints?]

### Market Participants Affected
[Who creates this inefficiency? Who could exploit it?]

---

## Scoring

### Theoretical Soundness (1-10): [Score]
**Rationale**: [Why this score?]

### Measurability (1-10): [Score]
**Rationale**: [Can we get the required data? Cost? Quality?]

### Competition (1-10): [Score]
**Rationale**: [How crowded is this trade? 10 = uncrowded, 1 = everyone does this]

### Persistence (1-10): [Score]
**Rationale**: [Will this last? Or get arbitraged away quickly?]

### Capital Capacity (1-10): [Score]
**Rationale**: [Can we scale this? Daily trading volume capacity?]

### **Total Score**: [Soundness × Measurability × Competition × Persistence / 10,000]

---

## Data Requirements

### Required Data Sources
- **Primary**: [Main data source, e.g., Glassnode, Coinbase WS]
- **Secondary**: [Backup or complementary sources]
- **Alternative**: [Free alternatives if primary unavailable]

### Data Granularity
- **Frequency**: [Tick | 1-second | 1-minute | 5-minute | 1-hour | Daily]
- **Lookback Period**: [Minimum history needed for testing, e.g., 6 months]
- **Update Latency**: [Real-time | 5-min delay | Daily]

### Storage Estimate
- **Raw Data**: [GB per month]
- **Processed Features**: [GB per month]
- **Total Historical**: [GB for full lookback]

### Cost Estimate
- **Data Subscription**: [$X/month or Free]
- **Storage**: [$X/month]
- **Compute**: [$X/month]
- **Total Monthly**: [$X/month]

### Engineering Effort
- **Data Collection**: [Hours to implement]
- **Feature Engineering**: [Hours to implement]
- **Model Building**: [Hours to implement]
- **Total Effort**: [Hours]

---

## Testable Predictions

### Primary Hypothesis
**If**: [Condition, e.g., "Whale transfers > $50M BTC to exchanges"]
**Then**: [Expected outcome, e.g., "Price drops 2-5% within 2 hours"]
**Confidence**: [Expected win rate, e.g., 60-70%]

### Secondary Hypotheses
1. **If**: [Alternative condition] **Then**: [Expected outcome]
2. **If**: [Edge case] **Then**: [Expected outcome]

### Null Hypothesis (What would disprove this?)
[What results would make us kill this hypothesis?]

---

## Minimum Viable Model

### Model Type
[Threshold Rule | Linear Regression | Logistic Classifier | Random Forest | Neural Network | HMM]

### Input Features
1. [Feature 1, e.g., "whale_transfer_size"]
2. [Feature 2, e.g., "exchange_balance_change"]
3. [Feature 3, e.g., "current_price_momentum"]
4. [Feature 4...]

### Output Signal
[Long | Short | Neutral | Position Size (0-100%)]

### Trading Rules
- **Entry**: [When to enter position]
- **Exit**: [When to close position]
- **Stop Loss**: [Risk management]
- **Position Size**: [% of portfolio]

---

## Success Criteria

### Backtesting Metrics (Must achieve ALL)
- [ ] **Sharpe Ratio**: > 1.5
- [ ] **Win Rate**: > 52% (mean-reversion) or > 45% (momentum)
- [ ] **Profit Factor**: > 1.5 (gross profit / gross loss)
- [ ] **Max Drawdown**: < 20%
- [ ] **Statistical Significance**: p-value < 0.05
- [ ] **Sample Size**: > 100 trades

### Robustness Checks (Must pass 3/5)
- [ ] Performs across different time periods (train, validation, test)
- [ ] Performs across different market regimes (bull, bear, sideways)
- [ ] Robust to parameter changes (±20% parameter variation)
- [ ] Works on multiple assets (not just BTC)
- [ ] Survives transaction cost sensitivity analysis

### Forward Testing
- [ ] Out-of-sample backtest: Sharpe > 1.0
- [ ] Paper trading (2 weeks): Positive P&L
- [ ] Live trading (small size): Positive P&L

---

## Testing Log

### Test 1: [Date]
**Dataset**: [Description]
**Results**: [Sharpe, Win Rate, etc.]
**Conclusion**: [Continue | Iterate | Kill]
**Notes**: [Key learnings]

### Test 2: [Date]
**Dataset**: [Description]
**Results**: [Sharpe, Win Rate, etc.]
**Conclusion**: [Continue | Iterate | Kill]
**Notes**: [Key learnings]

---

## Decision

### Status: [Research | Testing | Validated | Killed | Production]

### Reasoning
[Why did we reach this conclusion?]

### Next Steps
- [ ] [Action item 1]
- [ ] [Action item 2]
- [ ] [Action item 3]

### Production Deployment (If validated)
- **Capital Allocation**: [$X or Y% of portfolio]
- **Risk Limits**: [Max position size, stop loss]
- **Monitoring**: [What metrics to track daily]
- **Kill Switch**: [Under what conditions do we shut this down]

---

## References
- [Academic paper, blog post, or source of inspiration]
- [Data source documentation]
- [Related hypotheses in our database]

---

## Notes
[Any additional observations, edge cases, or ideas for future research]

