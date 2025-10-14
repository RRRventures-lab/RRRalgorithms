# Hypothesis Database & Tracking

This directory contains all trading inefficiency hypotheses being researched and tested. Each hypothesis is scored on 5 dimensions to determine testing priority.

## Priority Scoring Formula

```
Base Score = (Soundness × Measurability × Competition × Persistence) / 10,000
Final Score = Base Score × 1000 × (Capital Capacity / 10)

Priority Tiers:
- CRITICAL: Score ≥ 700
- HIGH: Score ≥ 500
- MEDIUM: Score ≥ 300
- LOW: Score < 300
```

## Scoring Dimensions

| Dimension | Description | Scale |
|-----------|-------------|-------|
| **Theoretical Soundness** | Is there a logical reason this inefficiency exists? | 1-10 |
| **Measurability** | Can we get the required data at reasonable cost/quality? | 1-10 |
| **Competition** | How uncrowded is this trade? (10 = no one, 1 = everyone) | 1-10 |
| **Persistence** | Will this inefficiency last or get arbitraged away? | 1-10 |
| **Capital Capacity** | Can we scale this strategy to meaningful size? | 1-10 |

## Current Hypothesis Status

Last Updated: 2025-10-12

### By Priority Tier

| Tier | Count | Hypotheses |
|------|-------|------------|
| 🔴 CRITICAL (≥700) | 1 | 003 |
| 🟠 HIGH (≥500) | 2 | 001, 002 |
| 🟡 MEDIUM (≥300) | 0 | - |
| ⚪ LOW (<300) | 0 | - |

### By Category

| Category | Count | Status |
|----------|-------|--------|
| 🔗 On-Chain | 1 | 001 (Research) |
| 📊 Microstructure | 1 | 002 (Research) |
| 🔄 Arbitrage | 1 | 003 (Research) |
| 📰 Sentiment | 0 | Coming soon |
| 📈 Regime-Dependent | 0 | Coming soon |

### By Status

| Status | Count | IDs |
|--------|-------|-----|
| 📚 Research | 3 | 001, 002, 003 |
| 🧪 Testing | 0 | - |
| ✅ Validated | 0 | - |
| ❌ Killed | 0 | - |
| 🚀 Production | 0 | - |

## Top 5 Hypotheses (By Priority Score)

| Rank | ID | Title | Score | Tier | Data Cost | Eng Hours |
|------|----|----|-------|------|-----------|-----------|
| 1 | 003 | CEX-DEX Price Dislocations Create Arbitrage | 810 | CRITICAL | $0 | 20 |
| 2 | 002 | Order Book Imbalance Predicts Short-Term Returns | 720 | HIGH | $0 | 24 |
| 3 | 001 | Whale Exchange Deposits Predict Price Drops | 640 | HIGH | $0 | 16 |

**Total Data Cost**: $0/month (all free tier APIs)
**Total Engineering Effort**: 60 hours (for top 3)

## Testing Priority & Timeline

### Week 1 (Current): Foundation
- [x] Create hypothesis template
- [x] Document first 3 hypotheses (001-003)
- [x] Build scoring model
- [ ] Complete 22 remaining hypotheses (004-025)
- [ ] Run full prioritization

### Week 2: Data Infrastructure
Focus on top 3 hypotheses:
- [ ] **003 (CRITICAL)**: Build CEX-DEX arbitrage scanner
  - CEX price monitor (Coinbase, Binance)
  - DEX price monitor (Uniswap, Sushiswap)
  - Gas cost tracker
  - Profitability calculator
- [ ] **002 (HIGH)**: Build order book collector
  - Binance WebSocket level 2 data
  - Order book imbalance calculator
- [ ] **001 (HIGH)**: Build on-chain whale tracker
  - Etherscan API for large transactions
  - Exchange address labeling
  - Whale flow metrics

### Week 3: Initial Testing
- [ ] Test hypothesis 003: Monitor CEX-DEX spreads for 1 week
- [ ] Test hypothesis 002: Collect order book data, run linear regression
- [ ] Test hypothesis 001: Download 6 months whale transfers, backtest

### Week 4-5: Expand Testing
- [ ] Test hypotheses 004-008 (remaining high-priority)
- [ ] Apply KILL/ITERATE/SCALE framework
- [ ] Select top 3-5 for production

## Hypothesis Lifecycle

```
Research → Testing → Validated → Production
             ↓
           Killed
```

### Research Phase
- Document hypothesis in detail
- Calculate priority score
- Identify data requirements
- Estimate engineering effort

### Testing Phase
- Collect minimum viable data
- Build simplest possible model
- Run backtest
- Calculate Sharpe, win rate, statistical significance

### Decision Framework
- **KILL** if Sharpe < 0.5 or win rate < 50% or p-value > 0.1
- **ITERATE** if Sharpe 0.5-1.2 (some signal, needs improvement)
- **VALIDATE** if Sharpe > 1.5 and passes robustness checks

### Production Phase
- Scale up data collection
- Implement real-time execution
- Paper trade for 2 weeks
- Deploy with appropriate capital allocation

## Hypothesis Categories Explained

### 🔗 On-Chain (Unique to Crypto)
Exploit blockchain transparency to detect informed trading signals before they impact price. Examples:
- Whale wallet movements
- Exchange flow imbalances
- Stablecoin supply changes
- Smart contract liquidations

**Edge**: Traditional finance has no equivalent. Stock traders can't see inside Goldman Sachs' wallet.

### 📊 Microstructure (Academic Foundation)
Exploit order book dynamics and market microstructure properties. Examples:
- Order book imbalance
- Quote stuffing detection
- Spread dynamics
- Order flow toxicity

**Edge**: Proven in academic literature. Retail traders don't have level 2 data access.

### 🔄 Arbitrage (Near Risk-Free)
Exploit price dislocations between venues or assets. Examples:
- CEX-DEX arbitrage
- Triangular arbitrage
- Funding rate arbitrage
- Cross-chain arbitrage

**Edge**: Fragmented liquidity in crypto creates more opportunities than traditional markets.

### 📰 Sentiment (Information Edge)
Detect news and sentiment signals before broad market awareness. Examples:
- Perplexity AI real-time news analysis
- Social sentiment divergence
- GitHub activity for altcoins
- Regulatory announcement detection

**Edge**: Speed advantage (5-10 minute window before retail reacts).

### 📈 Regime-Dependent (Adaptive Strategies)
Detect market regime changes and rotate strategies accordingly. Examples:
- Volatility regime switching
- Correlation breakdown detection
- Trend/range transition detection
- Risk-on/risk-off regimes

**Edge**: Most traders use static strategies. Adaptive systems perform better across market cycles.

## Data Cost Summary

| Hypothesis | Data Source | Cost/Month | Alternative |
|------------|-------------|------------|-------------|
| 001 | Etherscan, Blockchain.com | $0 | Glassnode ($499) |
| 002 | Binance, Coinbase WS | $0 | Polygon.io ($200) |
| 003 | CEX APIs, DEX subgraphs | $0 | None needed |

**Current Commitment**: $0/month (100% free tier)

**Optional Upgrades**:
- Glassnode ($499/mo): Pre-filtered whale wallets, institutional-grade on-chain data
- Polygon.io Pro ($200/mo): Level 2 market data with lower latency

**Recommendation**: Start with free tier, prove edge, then upgrade if needed.

## Engineering Roadmap

Total effort to test top 10 hypotheses: ~150 hours (3-4 weeks full-time)

### Infrastructure (Reusable Across Hypotheses)
- [ ] Polygon.io REST client (DONE - already exists)
- [ ] PostgreSQL/TimescaleDB schema (DONE - already exists)
- [ ] Backtesting engine with realistic costs (20 hours)
- [ ] Statistical validation framework (10 hours)
- [ ] Hypothesis testing automation (15 hours)

### Hypothesis-Specific (Per Top-5)
- [ ] 001: On-chain data collector (16 hours)
- [ ] 002: Order book analyzer (24 hours)
- [ ] 003: Arbitrage scanner (20 hours)
- [ ] 004-005: TBD based on prioritization

## Success Metrics

### Research Phase (Weeks 1-2)
- ✅ 3 hypotheses documented
- [ ] 25+ total hypotheses documented
- [ ] Top 5 prioritized by score

### Testing Phase (Weeks 3-5)
- [ ] Top 5 hypotheses tested
- [ ] At least 2 achieve Sharpe > 1.5
- [ ] 1-2 ready for paper trading

### Production Phase (Weeks 6-8)
- [ ] 2-3 strategies in paper trading
- [ ] Positive P&L over 2 weeks
- [ ] Ready for live deployment (small size)

## Key Principles

1. **Simple models first**: Start with threshold rules and linear regression. Only use neural networks if simple models prove edge.
2. **Kill losers fast**: If Sharpe < 0.5 after proper test, move on immediately.
3. **Scale winners aggressively**: If Sharpe > 1.5 and robust, deploy capital and scale data collection.
4. **Alternative data > more price data**: On-chain whale movements are more valuable than 5 years of daily OHLC bars.
5. **Measure everything**: Track win rate, Sharpe, statistical significance, parameter sensitivity.

## Files in This Directory

```
research/hypotheses/
├── README.md                              (this file)
├── hypothesis_template.md                 (template for new hypotheses)
├── priority_scores.json                   (generated by scoring_model.py)
│
├── 001_whale_transaction_impact.md        ✅ Documented
├── 002_order_book_imbalance.md            ✅ Documented
├── 003_cex_dex_arbitrage.md               ✅ Documented
├── 004_stablecoin_flows.md                ⏳ To be created
├── 005_smart_contract_events.md           ⏳ To be created
├── ...
└── 025_macro_regime.md                    ⏳ To be created
```

## How to Add a New Hypothesis

1. Copy `hypothesis_template.md` → `0XX_your_hypothesis.md`
2. Fill in all sections
3. Calculate priority score (use `scoring_model.py`)
4. Add to `scoring_model.py` `initialize_database()` function
5. Run `python research/prioritization/scoring_model.py` to regenerate priorities
6. Update this README with new hypothesis

## How to Update Hypothesis Status

When test results are available:
1. Open hypothesis markdown file
2. Update "Testing Log" section with results
3. Update "Decision" section (KILL/ITERATE/SCALE)
4. Change "Status" in metadata
5. Update `scoring_model.py` with backtest results
6. Run scoring model to regenerate report

## Questions?

- See `hypothesis_template.md` for detailed structure
- See `001_whale_transaction_impact.md` for complete example
- See `../prioritization/scoring_model.py` for scoring logic
- See main project README for overall system architecture

---

**Last Updated**: 2025-10-12
**Maintained By**: Research Team
**Next Review**: After completing 25 hypotheses (target: 2025-10-19)

