# Phase 2A: Parallel Hypothesis Testing (Research Factory)
## Claude Code Max Activation Prompt

**Copy and paste this entire prompt into Claude Code Max:**

---

```
SUPERTHINK MODE ACTIVATED: Phase 2A - Parallel Hypothesis Testing

Context: RRRalgorithms Cryptocurrency Trading System - Alpha Discovery
Repository: /Volumes/Lexar/RRRVentures/RRRalgorithms/research/hypotheses/
Phase: 2A of 6
Priority: CRITICAL
Duration: 4-6 hours
Subagents: 10 independent research agents

OBJECTIVE:
Test 10 market inefficiency hypotheses in parallel. Each agent independently documents hypothesis, collects 6-12 months of data, engineers features, runs statistical tests, backtests strategy, and outputs KILL/ITERATE/SCALE decision. Aggregate results to identify top 3 strategies for production.

EXECUTION MODEL:
Deploy 10 completely independent research agents. Each agent executes full research pipeline from hypothesis documentation through backtesting. No dependencies between agents - 100% parallel execution.

═══════════════════════════════════════════════════════════════
INFRASTRUCTURE SETUP (Build First)
═══════════════════════════════════════════════════════════════

Before deploying research agents, create reusable testing framework:

Location: research/testing/
Files to create:
├── hypothesis_tester.py - Base class for all hypothesis testing
├── data_collectors.py - Unified interface for all data sources
├── feature_engineering.py - Common feature engineering utilities
├── backtesting_engine.py - Simple backtest with realistic costs
├── statistical_validator.py - Sharpe, p-value, correlation tests
├── decision_framework.py - KILL/ITERATE/SCALE logic
└── report_generator.py - Auto-generate markdown reports

═══════════════════════════════════════════════════════════════
RESEARCH AGENT TEMPLATE
═══════════════════════════════════════════════════════════════

Each agent executes:

class HypothesisResearchAgent:
    def __init__(self, hypothesis_id, hypothesis_title):
        self.id = hypothesis_id
        self.title = hypothesis_title
        
    async def execute_full_pipeline(self):
        # Step 1: Document hypothesis (2 hours)
        self.document_hypothesis()
        
        # Step 2: Collect data (1-2 hours)  
        data = await self.collect_historical_data()
        
        # Step 3: Feature engineering (1 hour)
        features = self.engineer_features(data)
        
        # Step 4: Statistical validation (30 min)
        stats = self.validate_hypothesis(features)
        
        # Step 5: Backtest strategy (1 hour)
        results = self.backtest_simple_strategy(features)
        
        # Step 6: Make decision (15 min)
        decision = self.decide_kill_iterate_scale(stats, results)
        
        return HypothesisReport(
            id=self.id,
            title=self.title,
            sharpe=results.sharpe,
            win_rate=results.win_rate,
            p_value=stats.p_value,
            decision=decision,
            reasoning=self.explain_decision()
        )

═══════════════════════════════════════════════════════════════
AGENT H1: WHALE TRANSACTION IMPACT
═══════════════════════════════════════════════════════════════

Priority: HIGH (Score: 640)
Hypothesis: Large whale deposits to exchanges predict 2-8 hour price drops

Data Sources:
- Etherscan API (free) - Ethereum transaction data
- Known whale addresses database
- Exchange deposit addresses

Feature Engineering:
- Transfer size ($USD equivalent)
- Destination exchange
- Historical price impact correlation
- Transfer timing (day/night, weekday/weekend)

Backtest Strategy:
IF transfer > $50M to exchange THEN SHORT for 6 hours

Files to Create:
├── research/hypotheses/004_whale_transaction_impact.md
├── research/testing/whale_backtest.py
└── research/data/whale_transfers_historical.csv

Target Metrics:
- Sharpe Ratio: > 1.5
- Win Rate: > 60%
- P-value: < 0.05

═══════════════════════════════════════════════════════════════
AGENT H2: ORDER BOOK IMBALANCE
═══════════════════════════════════════════════════════════════

Priority: HIGH (Score: 720)
Hypothesis: Bid/ask imbalance predicts 5-15 minute returns

Data Sources:
- Binance WebSocket (free) - Level 2 order book
- Historical order book snapshots

Feature Engineering:
- Simple imbalance ratio: (bid_volume - ask_volume) / total_volume
- Depth-weighted imbalance (weighted by price distance)
- Time decay factor

Backtest Strategy:
IF imbalance > 70% bids THEN LONG for 10 minutes

Files to Create:
├── research/hypotheses/005_order_book_imbalance.md
├── research/testing/orderbook_backtest.py
└── research/data/orderbook_snapshots.parquet

Target Metrics:
- Sharpe Ratio: > 1.2
- Win Rate: > 58%
- P-value: < 0.05

═══════════════════════════════════════════════════════════════
AGENT H3: CEX-DEX ARBITRAGE
═══════════════════════════════════════════════════════════════

Priority: CRITICAL (Score: 810)
Hypothesis: CEX-DEX price dislocations create profitable arbitrage

Data Sources:
- Coinbase API (free) - CEX prices
- Uniswap subgraph (free) - DEX prices
- Etherscan Gas Tracker - Gas costs

Feature Engineering:
- Price spread: (CEX_price - DEX_price) / CEX_price
- Gas cost in USD
- Net profit after all costs
- Liquidity depth on both sides

Backtest Strategy:
IF spread > 0.5% after costs THEN execute arbitrage

Files to Create:
├── research/hypotheses/006_cex_dex_arbitrage.md
├── research/testing/arbitrage_backtest.py
└── research/data/cex_dex_spreads.csv

Target Metrics:
- Sharpe Ratio: > 2.0
- Win Rate: > 70%
- Daily Opportunities: > 5

═══════════════════════════════════════════════════════════════
AGENT H4: FUNDING RATE DIVERGENCE
═══════════════════════════════════════════════════════════════

Hypothesis: Extreme funding rates predict mean reversion

Data Sources:
- Binance perpetual futures API
- Bybit perpetual futures API
- Deribit perpetual futures API

Feature Engineering:
- Funding rate (8-hour rate)
- Historical percentile (where is it vs history?)
- Cross-exchange divergence

Backtest Strategy:
IF funding rate > 0.1% (8-hour) THEN SHORT perpetual

Files to Create:
├── research/hypotheses/007_funding_rate.md
├── research/testing/funding_backtest.py
└── research/data/funding_rates_historical.csv

Target: Sharpe > 1.3

═══════════════════════════════════════════════════════════════
AGENT H5: STABLECOIN SUPPLY CHANGES
═══════════════════════════════════════════════════════════════

Hypothesis: USDT/USDC supply changes predict BTC price 24-48 hours later

Data Sources:
- Etherscan API - USDT contract total supply
- Etherscan API - USDC contract total supply
- Historical BTC price data

Feature Engineering:
- Daily supply change (absolute)
- Daily supply change (%)
- Cross-correlation with BTC price (lag 24-72 hours)

Backtest Strategy:
IF supply increases > $500M THEN LONG BTC 48 hours later

Files to Create:
├── research/hypotheses/008_stablecoin_flows.md
├── research/testing/stablecoin_backtest.py
└── research/data/stablecoin_supply_daily.csv

Target: Sharpe > 1.0

═══════════════════════════════════════════════════════════════
AGENT H6: LIQUIDATION CASCADE PREDICTION
═══════════════════════════════════════════════════════════════

Hypothesis: Approaching liquidation levels create cascade risk

Data Sources:
- Exchange order books
- Coinglass liquidation heatmaps (free tier)
- Historical liquidation events

Feature Engineering:
- Distance to major liquidation cluster (%)
- Liquidation volume at risk
- Current leverage ratio (open interest / market cap)

Backtest Strategy:
IF price within 2% of major liquidation level THEN avoid leverage / reduce position

Files to Create:
├── research/hypotheses/009_liquidation_cascade.md
├── research/testing/liquidation_backtest.py
└── research/data/liquidation_clusters.csv

Target: Sharpe > 0.8 (defensive strategy)

═══════════════════════════════════════════════════════════════
AGENT H7: OPTIONS IMPLIED VOLATILITY SKEW
═══════════════════════════════════════════════════════════════

Hypothesis: IV skew predicts directional moves

Data Sources:
- Deribit API (free for recent data)
- Options pricing data (calls and puts)

Feature Engineering:
- 25-delta put/call skew
- Term structure (front month vs back month)
- Skew change velocity

Backtest Strategy:
IF skew > 10 vol points THEN LONG volatility (straddles)

Files to Create:
├── research/hypotheses/010_options_iv_skew.md
├── research/testing/options_backtest.py
└── research/data/options_skew_historical.csv

Target: Sharpe > 1.1

═══════════════════════════════════════════════════════════════
AGENT H8: CROSS-EXCHANGE SENTIMENT DIVERGENCE
═══════════════════════════════════════════════════════════════

Hypothesis: Sentiment divergence between exchanges predicts arbitrage

Data Sources:
- Perplexity AI - News sentiment
- Exchange-specific social sentiment (if available)

Feature Engineering:
- Sentiment score difference (US vs Asia)
- Sentiment magnitude
- Persistence (how long has divergence lasted?)

Backtest Strategy:
IF US sentiment bullish but Asia sentiment bearish THEN arbitrage opportunity

Files to Create:
├── research/hypotheses/011_sentiment_divergence.md
├── research/testing/sentiment_backtest.py
└── research/data/sentiment_by_region.csv

Target: Sharpe > 0.9

═══════════════════════════════════════════════════════════════
AGENT H9: MINER CAPITULATION SIGNAL
═══════════════════════════════════════════════════════════════

Hypothesis: Bitcoin miner selling predicts short-term bottoms

Data Sources:
- Blockchain.com API (free)
- Known miner addresses
- Hash rate data
- BTC price data

Feature Engineering:
- Miner outflow (BTC/day)
- Hash rate changes
- Miner revenue (USD/day)
- Historical capitulation events

Backtest Strategy:
IF miner outflow > 2 std dev THEN LONG BTC (capitulation = bottom)

Files to Create:
├── research/hypotheses/012_miner_capitulation.md
├── research/testing/miner_backtest.py
└── research/data/miner_flows_historical.csv

Target: Sharpe > 1.4

═══════════════════════════════════════════════════════════════
AGENT H10: DEFI PROTOCOL TVL CHANGES
═══════════════════════════════════════════════════════════════

Hypothesis: Sudden TVL changes predict token price moves

Data Sources:
- DeFiLlama API (free)
- Protocol token prices
- Cross-protocol flow data

Feature Engineering:
- TVL change % (24 hours)
- TVL velocity (rate of change)
- Cross-protocol flow (money moving where?)

Backtest Strategy:
IF TVL increases > 20% in 24h THEN LONG protocol token

Files to Create:
├── research/hypotheses/013_defi_tvl.md
├── research/testing/tvl_backtest.py
└── research/data/defi_tvl_daily.csv

Target: Sharpe > 1.2

═══════════════════════════════════════════════════════════════
SUCCESS CRITERIA
═══════════════════════════════════════════════════════════════

phase_2a_complete = {
    "hypotheses_documented": 10,
    "hypotheses_tested": 10,
    "decisions": {
        "KILL": "> 0",  # At least some should fail
        "ITERATE": "> 0",  # Some show promise but need work
        "SCALE": "> 2"  # At least 2 ready for production
    },
    "top_3_identified": True,
    "testing_framework_reusable": True,
    "data_cost": "$0/month"  # All free tier APIs
}

═══════════════════════════════════════════════════════════════
DELIVERABLES
═══════════════════════════════════════════════════════════════

1. 10 fully documented hypotheses (research/hypotheses/004-013.md)
2. Reusable hypothesis testing framework (research/testing/)
3. Backtest results for all 10 hypotheses
4. Priority matrix with top 3 strategies identified
5. Decision report: Which to KILL, ITERATE, SCALE
6. Production deployment plan for top 3

═══════════════════════════════════════════════════════════════
DECISION FRAMEWORK
═══════════════════════════════════════════════════════════════

After testing each hypothesis:

KILL if:
- Sharpe Ratio < 0.5
- Win Rate < 50%
- P-value > 0.1
- Not statistically significant

ITERATE if:
- Sharpe Ratio 0.5-1.2
- Shows promise but needs refinement
- Data quality issues can be fixed
- Feature engineering can improve results

SCALE if:
- Sharpe Ratio > 1.5
- Win Rate > 60%
- P-value < 0.05
- Passes robustness checks
- Ready for paper trading

═══════════════════════════════════════════════════════════════
EXECUTION INSTRUCTIONS
═══════════════════════════════════════════════════════════════

1. Build testing framework first (research/testing/)
2. Deploy all 10 research agents in parallel
3. Each agent works independently on their hypothesis
4. Aggregate results into priority matrix
5. Generate final report with top 3 recommendations
6. Update tracker: docs/superthink-execution/tracker.md

BEGIN EXECUTION NOW.
```

---

## Post-Execution Validation

```bash
cd /Volumes/Lexar/RRRVentures/RRRalgorithms

# Run all hypothesis tests
python research/testing/hypothesis_tester.py --validate-all

# Generate priority report
python research/testing/decision_framework.py --generate-report

# Check data collection status
ls -lh research/data/

# Verify all hypotheses documented
ls research/hypotheses/00*.md | wc -l  # Should be 13 (003 existing + 10 new)
```

## Expected Outcomes

- **KILL**: 3-4 hypotheses (not statistically significant)
- **ITERATE**: 3-4 hypotheses (promising but need work)
- **SCALE**: 2-3 hypotheses (ready for production)

## Next Phase

After successful completion:
→ Proceed to Phase 2B: Strategy Implementation (top 3 strategies)

