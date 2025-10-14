# AI Psychology Team Quick Start Guide

## Get Started in 5 Minutes

This guide shows you how to use the AI Psychology Team validation system in your trading code.

---

## 1. Basic Usage (Synchronous)

```python
from src.validation.ai_validator_integration import (
    AIValidatorIntegration,
    ValidationRequest,
    UrgencyLevel
)
from datetime import datetime

# Initialize validator
validator_integration = AIValidatorIntegration()

# Create validation request
request = ValidationRequest(
    request_id="req_001",
    timestamp=datetime.utcnow(),
    request_type="TRADE_DECISION",

    # Model info
    model_name="transformer_v2",
    model_version="2.1.0",

    # Decision details
    decision_id="dec_001",
    decision_type="BUY",
    symbol="BTC-USD",
    quantity=0.001,
    price=51000,
    confidence=0.75,
    urgency=UrgencyLevel.NORMAL,
    timeout_ms=50,

    # Input data
    features=[0.23, -0.45, 0.67],
    feature_names=["rsi", "macd", "volume"],
    current_price=50000,
    historical_prices=[49800, 49900, 50000],
    market_context={"volatility": 0.025, "trend": "upward"},

    # Reasoning
    reasoning={
        "primary_signal": "RSI oversold",
        "supporting_signals": ["Bullish divergence"],
        "feature_importance": {"rsi": 0.5, "macd": 0.3, "volume": 0.2}
    },

    # Optional: ensemble predictions
    ensemble_predictions=[
        {"model": "model1", "prediction": 51000},
        {"model": "model2", "prediction": 51200}
    ],

    # Optional: risk assessment
    risk_assessment={
        "expected_value": 150,
        "max_loss": 2000,
        "probability_success": 0.75
    }
)

# Validate (synchronous)
response = validator_integration.validate_decision_sync(request)

# Check result
if response.execution_allowed:
    print(f"✅ Decision APPROVED (confidence: {response.confidence:.2%})")
    # Execute trade
    execute_trade(...)
else:
    print(f"❌ Decision REJECTED: {response.validations}")
    # Don't execute
```

---

## 2. Advanced Usage (Async with Superthink Agents)

```python
import asyncio
from src.validation.agent_coordinator import SuperthinkCoordinator

# Initialize superthink coordinator
coordinator = SuperthinkCoordinator()

# Your decision object
decision = {
    "symbol": "BTC-USD",
    "action": "BUY",
    "price": 51000,
    "confidence": 0.75
}

# Validate with all 6 agents in parallel
async def validate_with_agents():
    consensus = await coordinator.validate_async(decision)

    print(f"Consensus: {consensus.status}")
    print(f"Execution allowed: {consensus.execution_allowed}")
    print(f"Confidence: {consensus.confidence:.2%}")
    print(f"Latency: {consensus.total_latency_ms:.1f}ms")

    # Show individual agent votes
    for opinion in consensus.agent_votes:
        print(f"  {opinion.agent_name}: {opinion.vote.value} ({opinion.confidence:.2%})")

    return consensus

# Run
consensus = asyncio.run(validate_with_agents())
```

---

## 3. Monte Carlo Simulation

```python
from src.validation.monte_carlo_engine import MonteCarloEngine

# Create engine
engine = MonteCarloEngine(
    num_market_scenarios=10000,
    num_microstructure_scenarios=5000,
    num_risk_scenarios=3000,
    num_adversarial_scenarios=2000
)

# Generate scenarios
scenarios = engine.generate_all_scenarios()
print(f"Generated {len(scenarios)} scenarios")

# Define your system under test
def my_trading_system(scenario):
    # Your trading logic here
    # Return SimulationResult
    pass

# Run simulations
results = engine.run_all_scenarios(my_trading_system, parallel=True)

# Get summary
stats = engine.get_summary_statistics()
print(f"Pass rate: {stats['pass_rate']*100:.1f}%")
```

---

## 4. View Real-Time Monitoring

```bash
# Open Grafana dashboard
open http://localhost:3000

# Navigate to:
# Dashboards → AI Validation & Hallucination Detection

# Key metrics to watch:
# - Hallucination detection rate (should be <1%)
# - Validation latency (p95 should be <10ms)
# - Decision approval rate
# - Agent vote distribution
```

---

## 5. Generate Reports

```bash
# Generate Monte Carlo report
cd /Volumes/Lexar/RRRVentures/RRRalgorithms

python3 worktrees/monitoring/src/validation/monte_carlo_reporter.py \
    --input simulation_stats.json \
    --format markdown \
    --output report.md \
    --detailed

# View report
cat report.md
```

---

## 6. Run Tests

```bash
# Run all validation tests
pytest worktrees/monitoring/tests/test_ai_validator.py -v

# Run Monte Carlo tests
pytest worktrees/monitoring/tests/test_monte_carlo_suite.py -v

# Run with coverage
pytest worktrees/monitoring/tests/ --cov=worktrees/monitoring/src/validation
```

---

## 7. Check Audit Trail

```python
from src.validation.decision_auditor import DecisionAuditor

auditor = DecisionAuditor()

# Get decision history
history = auditor.get_decision_history("dec_001")

print(f"Found {len(history)} entries for decision")

for entry in history:
    print(f"  {entry['timestamp']}: {entry['validation_status']}")

# Verify chain integrity
verification = auditor.verify_chain_integrity()

if verification['verified']:
    print("✅ Audit trail verified")
else:
    print(f"❌ Audit trail compromised: {verification['tampered_entries']}")
```

---

## 8. Performance Statistics

```python
# Get validator statistics
stats = validator_integration.get_performance_stats()

print(f"Total validations: {stats['total_validations']}")
print(f"Approval rate: {stats['approval_rate']*100:.1f}%")
print(f"p95 latency: {stats['latency_stats']['p95_ms']:.1f}ms")

# Get agent statistics
agent_stats = coordinator.get_agent_statistics()

for agent_name, stats in agent_stats.items():
    print(f"\n{agent_name}:")
    print(f"  Evaluations: {stats['total_evaluations']}")
    print(f"  Votes: {stats['votes_by_type']}")
```

---

## Common Patterns

### Pattern 1: Pre-Trade Validation

```python
async def execute_trade_with_validation(trade_params):
    # 1. Create validation request
    request = create_validation_request(trade_params)

    # 2. Validate
    response = await validator.validate_decision_async(request)

    # 3. Check approval
    if not response.execution_allowed:
        logger.warning(f"Trade rejected: {response.concerns}")
        return None

    # 4. Execute if approved
    result = await execute_trade(trade_params)

    # 5. Log execution result
    auditor.update_execution_result(
        decision_id=request.decision_id,
        execution_status="SUCCESS",
        actual_outcome=result
    )

    return result
```

### Pattern 2: Batch Validation

```python
async def validate_portfolio_decisions(decisions):
    # Validate all decisions in parallel
    tasks = [
        validator.validate_decision_async(create_request(d))
        for d in decisions
    ]

    responses = await asyncio.gather(*tasks)

    # Filter approved only
    approved = [
        d for d, r in zip(decisions, responses)
        if r.execution_allowed
    ]

    return approved
```

### Pattern 3: Monte Carlo Optimization

```python
from src.validation.monte_carlo_optimizer import MonteCarloOptimizer

optimizer = MonteCarloOptimizer()

# Optimize stop-loss
result = optimizer.optimize_stop_loss_levels(
    historical_returns=returns_data,
    initial_stop_loss=0.02
)

print(f"Optimal stop-loss: {result.optimized_value*100:.1f}%")
print(f"Improvement: {result.improvement_percent:.1f}%")

# Apply to system
system.update_parameter("stop_loss", result.optimized_value)
```

---

## Troubleshooting

### Issue: High validation latency

```python
# Check validator stats
stats = validator.get_performance_stats()
if stats['latency_stats']['p95_ms'] > 10:
    # Investigate slow agents
    agent_stats = coordinator.get_agent_statistics()
    # Check for outliers
```

### Issue: High rejection rate

```python
# Analyze rejection reasons
for concern in response.concerns:
    print(f"Concern: {concern['type']} (severity: {concern['severity']})")

# Review hallucination reports
if response.validations['hallucination_check']['passed'] == False:
    print("Check model output for hallucinations")
```

### Issue: Audit trail integrity failed

```python
# Verify chain
report = auditor.verify_chain_integrity()

if not report['verified']:
    # Check tampered entries
    for entry in report['tampered_entries']:
        print(f"Tampered: line {entry['line_num']}")

    # Alert security team
    alert_security("Audit trail compromised")
```

---

## Best Practices

1. **Always validate before executing** - Never skip validation in production

2. **Use async validation** - Better performance for high-frequency trading

3. **Monitor latency** - Alert if p95 >10ms

4. **Review rejections** - Understand why decisions are rejected

5. **Run Monte Carlo regularly** - Monthly stress testing recommended

6. **Verify audit trail** - Daily integrity checks

7. **Tune thresholds** - Based on production data

8. **Update agents** - Retrain quarterly

---

## Next Steps

1. Read [AI_PSYCHOLOGY_TEAM.md](../teams/AI_PSYCHOLOGY_TEAM.md) for full details
2. Review [SUPERTHINK_SUBAGENTS.md](../architecture/SUPERTHINK_SUBAGENTS.md) for agent architecture
3. Check [AI_VALIDATION_PROTOCOL.md](../protocols/AI_VALIDATION_PROTOCOL.md) for API specs
4. Run test suite to verify installation

---

**Questions?** Check the full documentation in `/docs/teams/AI_PSYCHOLOGY_TEAM.md`
