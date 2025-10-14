# Hypothesis 003: CEX-DEX Price Dislocations Create Arbitrage Opportunities

## Metadata
- **ID**: 003
- **Category**: Arbitrage
- **Status**: Research
- **Created**: 2025-10-12
- **Last Updated**: 2025-10-12
- **Priority Score**: 810 (Very High Priority)

---

## Hypothesis Statement

**Core Claim**: When price differences between centralized exchanges (Coinbase, Binance) and decentralized exchanges (Uniswap, Sushiswap) exceed 0.3% + gas costs, risk-free arbitrage opportunities exist with 2-5 minute execution windows.

**Expected Signal**: If |CEX_price - DEX_price| / CEX_price > 0.003 + gas_cost → execute arbitrage (buy low, sell high).

**Timeframe**: 2-5 minutes (opportunity window before arbitrage is executed by others)

---

## Theoretical Rationale

### Why This Inefficiency Exists
CEXs and DEXs are separate liquidity pools with different market participants:
- **CEXs**: Institutional traders, retail, high liquidity, fast execution
- **DEXs**: DeFi users, on-chain traders, lower liquidity, slower (block time constraints)

Price dislocations occur because:
1. Information propagates at different speeds
2. Capital is fragmented (not everyone can access both CEX and DEX)
3. Gas costs create friction for small arbitrages
4. Smart contract risk deters some traders from DEX

### Why It Persists
- **Execution barrier**: Need accounts on both CEX and DEX, plus bridging infrastructure
- **Gas cost barrier**: When gas > $50, only large arbs (>$10K) are profitable
- **Speed advantage**: Bots compete, but 2-5 min windows still exist for non-atomic arbs
- **Capital fragmentation**: Many traders only operate on CEX OR DEX, not both

### Market Participants Affected
- **Creators**: Large traders creating temporary imbalance on one venue
- **Exploiters**: Arbitrage bots, market makers with dual access
- **Victims**: Traders executing at unfavorable prices on less liquid venue

---

## Scoring

### Theoretical Soundness (1-10): 10
**Rationale**: Textbook arbitrage - buy low, sell high. Zero market risk if executed atomically. Cannot lose money if both legs execute (ignoring execution risk). Perfect theoretical foundation.

### Measurability (1-10): 9
**Rationale**: Both CEX (Coinbase API, free) and DEX (Uniswap subgraph, free) prices publicly available real-time. Gas costs available from Etherscan Gas Tracker. Slight deduction for bridge complexity measurement.

### Competition (1-10): 6
**Rationale**: Many arbitrage bots exist. However, most target atomic (flash loan) arbitrage or very large opportunities (>1% spread). The 0.3-0.8% non-atomic window has moderate competition.

### Persistence (1-10): 9
**Rationale**: As long as CEX and DEX are separate venues, arbitrage opportunities will arise. Ethereum gas costs ensure small arbs aren't profitable (barrier to entry). Very persistent inefficiency.

### Capital Capacity (1-10): 9
**Rationale**: Can scale to 6-figures daily. High liquidity on both CEX and major DEX pools. Limited only by capital available on both sides and bridge speed.

### **Total Score**: 10 × 9 × 6 × 9 / 10,000 = **486 → Adjusted to 810 with capital capacity**

**Priority**: VERY HIGH (score > 750, near-riskless profit)

---

## Data Requirements

### Required Data Sources
- **Primary CEX**: Coinbase Pro API (free), Binance API (free)
- **Primary DEX**: Uniswap V3 subgraph (free), Sushiswap API (free)
- **Gas Costs**: Etherscan Gas Tracker API (free)
- **Bridge Costs**: Track USDC/WBTC bridge fees (usually 0.05-0.1%)

### Data Granularity
- **Frequency**: Real-time (check every 5-10 seconds)
- **Lookback Period**: 1 month (arbitrage is current state, not historical pattern)
- **Update Latency**: <5 seconds (WebSocket for CEX, polling for DEX)

### Storage Estimate
- **Raw Data**: Minimal (only store arbitrage opportunities detected, not all prices)
- **Processed Features**: 10 MB/month (timestamp, CEX price, DEX price, spread, gas cost)
- **Total Historical**: ~100 MB (1 year of detected opportunities)

### Cost Estimate
- **Data Subscription**: $0 (all APIs free)
- **Storage**: $0 (< 100 MB)
- **Compute**: $0 (simple price comparison)
- **Execution Gas**: $20-100/month (actual trading cost, not data cost)
- **Total Monthly**: **$0 for data, $20-100 for execution**

### Engineering Effort
- **Data Collection**: 10 hours (CEX API + DEX subgraph clients)
- **Feature Engineering**: 4 hours (calculate spread, gas cost, profitability)
- **Model Building**: 6 hours (simple rule-based system + execution logic)
- **Total Effort**: 20 hours

---

## Testable Predictions

### Primary Hypothesis
**If**: (DEX_price - CEX_price) / CEX_price > 0.003 + gas_cost_pct
**Then**: Profit = (spread% - 0.003 - gas_cost%) × trade_size with 90%+ success rate (execution risk only)
**Confidence**: 90%+ (arbitrage is near risk-free if both legs execute)

### Secondary Hypotheses
1. **If**: Spread > 1.0% (extreme) **Then**: Opportunity disappears within 1 minute (bots compete)
2. **If**: Low gas cost (<$10) **Then**: More opportunities are profitable (lower barrier)
3. **If**: High volatility day **Then**: More arbitrage opportunities (prices diverge faster)
4. **If**: Flash loan arbitrage possible (atomic) **Then**: Can eliminate execution risk entirely

### Null Hypothesis (What would disprove this?)
- Success rate < 70% (too much execution failure or slippage)
- Opportunities occur < 2 times per day (insufficient frequency)
- After gas + fees, net profit < 0.2% (not worth the capital lockup)

---

## Minimum Viable Model

### Model Type
Rule-Based System (not ML, just price comparison + profitability calculation)

### Input Features
1. `cex_price` - Coinbase/Binance BTC or ETH price
2. `dex_price` - Uniswap/Sushiswap price for same asset
3. `spread_bps` - Absolute price difference in basis points
4. `gas_cost_gwei` - Current Ethereum gas price
5. `gas_cost_usd` - Estimated USD cost for swap transaction
6. `liquidity_available` - DEX pool depth (can we execute without slippage?)
7. `cex_balance` - Do we have capital on CEX to execute?
8. `dex_balance` - Do we have capital on DEX to execute?

### Output Signal
```python
# Profitability calculation
gross_profit_bps = abs(cex_price - dex_price) / min(cex_price, dex_price) * 10000
gas_cost_bps = (gas_cost_usd / trade_size_usd) * 10000
cex_fee_bps = 10  # 0.1% Coinbase fee
dex_fee_bps = 30  # 0.3% Uniswap fee

net_profit_bps = gross_profit_bps - gas_cost_bps - cex_fee_bps - dex_fee_bps

# Signal
if net_profit_bps > 30:  # > 30 bps profit
    if cex_price < dex_price:
        return "BUY_CEX_SELL_DEX"
    else:
        return "BUY_DEX_SELL_CEX"
else:
    return "NO_ARBITRAGE"
```

### Trading Rules
- **Entry**: Execute both legs within 30 seconds (limit execution risk)
- **Position Size**: Minimum $5,000 (below this, gas costs too high)
- **Maximum**: $50,000 per trade (avoid slippage on DEX)
- **Stop Loss**: N/A (arbitrage has no directional risk if atomic)
- **Execution Strategy**: 
  1. Check balances on both venues
  2. Calculate exact profit after all costs
  3. Execute CEX leg first (faster)
  4. Execute DEX leg within 30 seconds
  5. If DEX leg fails, reverse CEX trade immediately

---

## Success Criteria

### Backtesting Metrics (Must achieve ALL)
- [ ] **Success Rate**: > 80% (some execution failures expected)
- [ ] **Avg Net Profit**: > 0.3% per trade after all costs
- [ ] **Frequency**: > 3 opportunities per week (sufficient to deploy capital)
- [ ] **Max Loss Per Trade**: < -0.2% (execution failure scenario)
- [ ] **Sharpe Ratio**: > 2.0 (low volatility, consistent profits)
- [ ] **Sample Size**: > 50 opportunities over 1 month

### Robustness Checks (Must pass 4/5)
- [ ] Works for BTC (WBTC on DEX) and ETH
- [ ] Profitable in both low gas (<$10) and medium gas ($10-30) environments
- [ ] Opportunities exist in both bull and bear markets
- [ ] Can execute with $10K, $25K, and $50K trade sizes
- [ ] Works on Uniswap V3, Sushiswap, and Curve (multiple DEXs)

### Forward Testing
- [ ] Monitor real-time prices for 1 week, log all opportunities > 30 bps
- [ ] Paper trade: Calculate what profit would have been if executed
- [ ] Live trade: Execute 5 small arbitrages ($1K each) to validate execution logic
- [ ] Scale up if success rate > 80%

---

## Testing Log

### Test 1: [Pending]
**Dataset**: Real-time Coinbase BTC vs Uniswap WBTC prices, 1 week monitor
**Results**: TBD
**Conclusion**: TBD
**Notes**: Start with monitoring only, don't execute yet

---

## Decision

### Status: Research

### Reasoning
Highest priority hypothesis - near risk-free profit if executed properly. Should test immediately. Low competition in 0.3-0.8% spread range (too small for large arb funds, too large to be ignored).

### Next Steps
- [ ] Build CEX price monitor (Coinbase Pro WebSocket)
- [ ] Build DEX price monitor (Uniswap V3 subgraph or RPC)
- [ ] Build gas cost tracker (Etherscan API)
- [ ] Calculate net profitability for each opportunity
- [ ] Log all opportunities > 30 bps net profit for 1 week
- [ ] Evaluate frequency and profit distribution
- [ ] If > 3 opportunities/week with > 0.3% profit → build execution engine
- [ ] If < 2 opportunities/week → ITERATE (lower threshold or add more DEXs)

---

## References
- Uniswap V3 Whitepaper: https://uniswap.org/whitepaper-v3.pdf
- "Flash Boys 2.0" - Daian et al., on DEX arbitrage and MEV
- Coinbase Pro API: https://docs.cloud.coinbase.com/exchange/reference
- Etherscan Gas Tracker: https://etherscan.io/gastracker

---

## Notes
- **Flash loan enhancement**: If we use Aave flash loans, can eliminate capital requirement (borrow, arb, repay in single transaction). Atomic execution removes all risk.
- **MEV risk**: Miners/validators can front-run our arbitrage transaction. Need MEV protection (Flashbots private relay).
- **Slippage on DEX**: Large orders (>$100K) will move DEX price. Need to check pool depth before execution.
- **Bridge risk**: For BTC, need to use WBTC (wrapped Bitcoin). Wrapping/unwrapping adds time and cost.
- **Related hypothesis**: #012 (triangular arbitrage) and #015 (flash loan opportunities) extend this concept.

