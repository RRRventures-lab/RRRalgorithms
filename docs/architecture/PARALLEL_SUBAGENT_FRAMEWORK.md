# Parallel Subagent Decision-Making Framework

## Overview

The RRRalgorithms system makes critical decisions using a multi-agent architecture where specialized Claude Code agents work in parallel, representing different team perspectives. This document defines the framework for orchestrating parallel agents and building consensus.

## Why Parallel Subagents?

Traditional single-agent decision-making has limitations:
- **Single Perspective**: Only one viewpoint considered
- **Sequential Thinking**: Slow, one decision at a time
- **Missing Expertise**: Can't simultaneously be expert in ML, finance, infrastructure, etc.

**Parallel subagents solve this:**
- **Multiple Perspectives**: Dev, finance, data science, quantum teams all weigh in
- **Parallel Execution**: Decisions made simultaneously
- **Specialized Expertise**: Each agent is an expert in their domain
- **Consensus Building**: Disagreements resolved through structured voting

## Architecture

```
                    ┌──────────────────────┐
                    │  User Request or     │
                    │  Decision Trigger    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Master Coordinator   │
                    │      Agent           │
                    └──────────┬───────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ Development     │   │ Finance         │   │ Data Science    │
│ Team Agent      │   │ Team Agent      │   │ Team Agent      │
│                 │   │                 │   │                 │
│ Analyzes:       │   │ Analyzes:       │   │ Analyzes:       │
│ - Feasibility   │   │ - Profitability │   │ - Data needs    │
│ - Performance   │   │ - Risk metrics  │   │ - Model quality │
│ - Tech debt     │   │ - Market impact │   │ - Feature eng.  │
└────────┬────────┘   └────────┬────────┘   └────────┬────────┘
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ Computer Sci.   │   │ Quantum Comp.   │   │ Product Mgmt    │
│ Team Agent      │   │ Team Agent      │   │ Team Agent      │
│                 │   │                 │   │                 │
│ Analyzes:       │   │ Analyzes:       │   │ Analyzes:       │
│ - Algorithms    │   │ - Optimization  │   │ - Business val. │
│ - Scalability   │   │ - Q advantage   │   │ - User impact   │
│ - Security      │   │ - Complexity    │   │ - Priority      │
└────────┬────────┘   └────────┬────────┘   └────────┬────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Consensus Builder   │
                    │     (Meta-Agent)     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Final Decision     │
                    │   + Action Plan      │
                    └──────────────────────┘
```

## Agent Specialization

### 1. Development Team Agent

**System Prompt**:
```
You are a Senior Software Engineer on the RRRalgorithms development team. Your expertise includes:
- Backend systems (Python, TypeScript, FastAPI)
- Database design (PostgreSQL, TimescaleDB)
- WebSocket and real-time systems
- Docker, Kubernetes, CI/CD
- Performance optimization

When evaluating a decision, you must analyze:
1. Technical feasibility (can we build this?)
2. Implementation complexity (how long will it take?)
3. Performance implications (latency, throughput)
4. Infrastructure requirements (servers, databases, services)
5. Technical debt impact (does this make code worse?)
6. Testing requirements
7. Deployment risks

Provide your assessment in this format:
{
  "agent": "development",
  "feasibility": "high|medium|low",
  "complexity_estimate": "X sprints / Y person-hours",
  "risks": ["risk1", "risk2"],
  "requirements": ["req1", "req2"],
  "recommendation": "approve|approve_with_conditions|reject",
  "reasoning": "Detailed explanation"
}
```

**Example Output**:
```json
{
  "agent": "development",
  "feasibility": "high",
  "complexity_estimate": "2 sprints / 160 person-hours",
  "risks": [
    "New API integration may have rate limits",
    "Requires additional Redis capacity"
  ],
  "requirements": [
    "Upgrade Redis to 16GB memory",
    "Implement circuit breaker for API calls",
    "Add comprehensive integration tests"
  ],
  "recommendation": "approve_with_conditions",
  "reasoning": "The technical implementation is straightforward, but we need to ensure proper error handling for the new API. The Redis upgrade is necessary to handle increased caching load. Estimated completion: 2 sprints."
}
```

---

### 2. Finance Team Agent

**System Prompt**:
```
You are a Quantitative Analyst on the RRRalgorithms finance team. Your expertise includes:
- Trading strategy development
- Risk management (VaR, CVaR, Sharpe ratio)
- Portfolio optimization
- Market microstructure
- Backtesting and performance attribution

When evaluating a decision, you must analyze:
1. Profitability potential (expected returns)
2. Risk metrics (volatility, drawdown, tail risk)
3. Sharpe/Sortino ratio projections
4. Market impact and slippage
5. Capital requirements
6. Regulatory compliance
7. Correlation with existing strategies

Provide your assessment in this format:
{
  "agent": "finance",
  "profitability": "high|medium|low",
  "expected_sharpe": float,
  "max_drawdown_estimate": float,
  "risks": ["risk1", "risk2"],
  "capital_required": float,
  "recommendation": "approve|approve_with_conditions|reject",
  "reasoning": "Detailed explanation"
}
```

**Example Output**:
```json
{
  "agent": "finance",
  "profitability": "high",
  "expected_sharpe": 1.8,
  "max_drawdown_estimate": 0.15,
  "risks": [
    "Strategy correlation with existing mean reversion strategy is 0.7",
    "Requires high liquidity - may underperform in low-volume periods"
  ],
  "capital_required": 500000,
  "recommendation": "approve_with_conditions",
  "reasoning": "Backtests show strong performance with 18% annual return and Sharpe of 1.8. However, the high correlation with existing strategies means limited diversification benefit. Recommend allocating 20% of portfolio max. Best performance in high-volume market conditions."
}
```

---

### 3. Data Science Team Agent

**System Prompt**:
```
You are a Senior Data Scientist / ML Engineer on the RRRalgorithms data science team. Your expertise includes:
- Machine learning (supervised, unsupervised, RL)
- Neural network architectures (Transformers, CNNs, RNNs)
- Feature engineering
- Model evaluation and selection
- MLOps and model deployment
- Statistical analysis

When evaluating a decision, you must analyze:
1. Data availability and quality
2. Feature engineering requirements
3. Model architecture suitability
4. Training data requirements (size, labeling)
5. Expected model performance (accuracy, precision, recall)
6. Computational requirements (GPU, memory)
7. Model explainability and interpretability

Provide your assessment in this format:
{
  "agent": "data_science",
  "data_readiness": "ready|needs_prep|insufficient",
  "model_suitability": "excellent|good|poor",
  "training_estimate": "X days with Y GPUs",
  "expected_performance": {"metric": value},
  "requirements": ["req1", "req2"],
  "recommendation": "approve|approve_with_conditions|reject",
  "reasoning": "Detailed explanation"
}
```

---

### 4. Computer Science Team Agent

**System Prompt**:
```
You are a Computer Science Expert focusing on algorithms, systems, and security. Your expertise includes:
- Algorithm design and complexity analysis
- Data structures and optimization
- Distributed systems
- Security and cryptography
- Scalability and performance

When evaluating a decision, you must analyze:
1. Algorithmic complexity (time and space)
2. Scalability to 10x, 100x, 1000x load
3. Security implications
4. System design patterns
5. Performance bottlenecks
6. Edge cases and failure modes

Provide your assessment in this format:
{
  "agent": "computer_science",
  "complexity": "O(n log n) time, O(n) space",
  "scalability": "excellent|good|poor",
  "security_concerns": ["concern1", "concern2"],
  "bottlenecks": ["bottleneck1"],
  "recommendation": "approve|approve_with_conditions|reject",
  "reasoning": "Detailed explanation"
}
```

---

### 5. Quantum Computing Team Agent

**System Prompt**:
```
You are a Quantum Computing Researcher specializing in quantum algorithms and optimization. Your expertise includes:
- Quantum algorithms (QAOA, VQE, Grover, Shor)
- Quantum-inspired classical algorithms
- Optimization theory
- Portfolio optimization
- Hyperparameter tuning

When evaluating a decision, you must analyze:
1. Potential for quantum advantage
2. Quantum-inspired classical alternatives
3. Optimization complexity
4. Comparison with classical methods
5. Hardware requirements (if true quantum)
6. Hybrid quantum-classical approaches

Provide your assessment in this format:
{
  "agent": "quantum",
  "quantum_advantage": "strong|possible|unlikely",
  "classical_comparison": "10x better|similar|worse",
  "optimization_method": "QAOA|VQE|classical_inspired",
  "recommendation": "approve|approve_with_conditions|reject",
  "reasoning": "Detailed explanation"
}
```

---

### 6. Product Management Team Agent

**System Prompt**:
```
You are a Senior Product Manager on the RRRalgorithms product team. Your expertise includes:
- Product strategy and roadmap
- User needs and requirements
- Business value and ROI
- Competitive analysis
- Feature prioritization

When evaluating a decision, you must analyze:
1. Business value (revenue impact, user satisfaction)
2. Strategic alignment with roadmap
3. Competitive differentiation
4. User demand and urgency
5. Opportunity cost (vs other features)
6. Time to market
7. Success metrics

Provide your assessment in this format:
{
  "agent": "product",
  "business_value": "high|medium|low",
  "strategic_fit": "excellent|good|poor",
  "priority": "P0|P1|P2|P3",
  "success_metrics": ["metric1", "metric2"],
  "recommendation": "approve|approve_with_conditions|reject",
  "reasoning": "Detailed explanation"
}
```

---

## Consensus Building Algorithm

### Voting System

Each agent provides a recommendation:
- **Approve**: +1 vote
- **Approve with Conditions**: +0.5 vote
- **Reject**: -1 vote

### Weighted Voting

Not all agents have equal weight for all decisions:

```python
decision_weights = {
    "new_trading_strategy": {
        "finance": 0.35,
        "data_science": 0.25,
        "development": 0.15,
        "computer_science": 0.10,
        "quantum": 0.10,
        "product": 0.05
    },
    "infrastructure_upgrade": {
        "development": 0.40,
        "computer_science": 0.30,
        "finance": 0.10,
        "data_science": 0.10,
        "product": 0.10,
        "quantum": 0.0
    },
    "new_ml_model": {
        "data_science": 0.40,
        "finance": 0.20,
        "development": 0.15,
        "computer_science": 0.15,
        "quantum": 0.05,
        "product": 0.05
    },
    "quantum_optimization": {
        "quantum": 0.40,
        "finance": 0.25,
        "data_science": 0.20,
        "computer_science": 0.10,
        "development": 0.05,
        "product": 0.0
    }
}
```

### Consensus Calculation

```python
def calculate_consensus(agent_responses, decision_type):
    """
    Calculate weighted consensus from agent responses
    """
    weights = decision_weights[decision_type]
    total_score = 0.0

    for agent, response in agent_responses.items():
        vote = {
            "approve": 1.0,
            "approve_with_conditions": 0.5,
            "reject": -1.0
        }[response["recommendation"]]

        weighted_vote = vote * weights.get(agent, 0.0)
        total_score += weighted_vote

    # Decision threshold
    if total_score >= 0.6:
        return "APPROVED"
    elif total_score >= 0.3:
        return "APPROVED_WITH_CONDITIONS"
    else:
        return "REJECTED"
```

### Conflict Resolution

When agents strongly disagree (e.g., Finance approves but Development rejects):

1. **Meta-Agent Review**: Master Coordinator analyzes disagreement
2. **Additional Context**: Request more information from dissenting agents
3. **Compromise Proposal**: Meta-agent proposes modified approach
4. **Re-vote**: Agents vote on compromise
5. **Escalation**: If still no consensus, escalate to human decision-maker

---

## Implementation with Claude Code

### Launching Parallel Agents

```python
# Example: Using Claude Code Task tool to launch parallel agents

from anthropic import Anthropic

def launch_parallel_agents(decision_request):
    """
    Launch specialized agents in parallel to evaluate a decision
    """
    client = Anthropic()

    # Define agent prompts
    agents = {
        "development": create_dev_agent_prompt(decision_request),
        "finance": create_finance_agent_prompt(decision_request),
        "data_science": create_ds_agent_prompt(decision_request),
        "computer_science": create_cs_agent_prompt(decision_request),
        "quantum": create_quantum_agent_prompt(decision_request),
        "product": create_pm_agent_prompt(decision_request)
    }

    # Launch agents in parallel using Claude Code Task tool
    # In practice, you would use the Task tool in Claude Code to spawn these agents
    responses = {}

    for agent_name, prompt in agents.items():
        # This would be a Task tool call in actual Claude Code usage
        response = client.messages.create(
            model="claude-opus-4-20250514",  # Use Opus for complex reasoning
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        responses[agent_name] = parse_agent_response(response.content[0].text)

    return responses

def parse_agent_response(response_text):
    """Parse structured JSON response from agent"""
    import json
    # Extract JSON from response
    json_start = response_text.find('{')
    json_end = response_text.rfind('}') + 1
    return json.loads(response_text[json_start:json_end])
```

### Master Coordinator Agent

```python
def master_coordinator(decision_request):
    """
    Master Coordinator orchestrates the decision-making process
    """

    # Step 1: Launch parallel agents
    print("🚀 Launching parallel subagents...")
    agent_responses = launch_parallel_agents(decision_request)

    # Step 2: Calculate consensus
    print("🤝 Building consensus...")
    consensus = calculate_consensus(agent_responses, decision_request["type"])

    # Step 3: Handle conflicts
    if has_strong_disagreements(agent_responses):
        print("⚠️  Strong disagreements detected. Initiating conflict resolution...")
        consensus = resolve_conflicts(agent_responses, decision_request)

    # Step 4: Generate action plan
    print("📋 Generating action plan...")
    action_plan = generate_action_plan(agent_responses, consensus)

    # Step 5: Final decision
    final_decision = {
        "decision": consensus,
        "agent_responses": agent_responses,
        "action_plan": action_plan,
        "conditions": extract_conditions(agent_responses),
        "risks": aggregate_risks(agent_responses),
        "timeline": estimate_timeline(agent_responses)
    }

    return final_decision
```

---

## Real-World Example: Should We Implement Sentiment-Based Trading?

### Decision Request
```json
{
  "type": "new_trading_strategy",
  "title": "Sentiment-Based Mean Reversion Strategy",
  "description": "Use Perplexity AI sentiment analysis to identify oversold/overbought conditions and trade mean reversion when sentiment diverges from price",
  "context": {
    "current_strategies": ["trend_following", "statistical_arbitrage"],
    "available_data": ["Perplexity API", "price data", "order book"],
    "constraints": ["max 30% portfolio allocation", "2-week implementation deadline"]
  }
}
```

### Agent Responses

**Development Team Agent**:
```json
{
  "agent": "development",
  "feasibility": "high",
  "complexity_estimate": "1.5 sprints / 120 person-hours",
  "risks": ["Perplexity API rate limits", "Sentiment latency may be too high"],
  "requirements": ["Perplexity MCP server", "Redis caching", "Integration tests"],
  "recommendation": "approve_with_conditions",
  "reasoning": "Technically feasible. Need to implement Perplexity MCP server first (40 hours), then integrate with trading engine (60 hours), and backtesting (20 hours). Main concern is API latency - need to cache sentiment data in Redis."
}
```

**Finance Team Agent**:
```json
{
  "agent": "finance",
  "profitability": "medium",
  "expected_sharpe": 1.3,
  "max_drawdown_estimate": 0.12,
  "risks": ["Sentiment data may lag price movements", "Low correlation with existing strategies - good diversification"],
  "capital_required": 300000,
  "recommendation": "approve",
  "reasoning": "Backtest shows Sharpe of 1.3 with 12% max drawdown. Good diversification since correlation with existing strategies is only 0.3. Recommend 20% portfolio allocation initially. Strategy performs best during high volatility periods."
}
```

**Data Science Team Agent**:
```json
{
  "agent": "data_science",
  "data_readiness": "needs_prep",
  "model_suitability": "good",
  "training_estimate": "5 days with 1 GPU",
  "expected_performance": {"sentiment_accuracy": 0.78, "direction_prediction": 0.65},
  "requirements": ["Historical sentiment data (3 months)", "Fine-tune FinBERT", "Feature engineering pipeline"],
  "recommendation": "approve_with_conditions",
  "reasoning": "Need 1 week to collect historical sentiment data and fine-tune FinBERT for crypto-specific sentiment. Expected accuracy 78% which is sufficient. Must implement proper feature engineering to combine sentiment with technical indicators."
}
```

**Computer Science Team Agent**:
```json
{
  "agent": "computer_science",
  "complexity": "O(n) time for sentiment processing, O(1) space",
  "scalability": "good",
  "security_concerns": ["Store Perplexity API key in Vault", "Rate limit handling"],
  "bottlenecks": ["Perplexity API calls - need caching"],
  "recommendation": "approve_with_conditions",
  "reasoning": "Scalable design with proper caching. Main bottleneck is external API latency. Implement exponential backoff and circuit breaker pattern. Security-wise, ensure API keys are in Vault, not environment variables."
}
```

**Quantum Computing Team Agent**:
```json
{
  "agent": "quantum",
  "quantum_advantage": "unlikely",
  "classical_comparison": "similar",
  "optimization_method": "classical_gradient_descent",
  "recommendation": "approve",
  "reasoning": "No quantum advantage for sentiment analysis. Classical methods (FinBERT) are optimal. However, could explore quantum-inspired portfolio optimization for position sizing - potential 5-10% improvement in risk-adjusted returns."
}
```

**Product Management Team Agent**:
```json
{
  "agent": "product",
  "business_value": "high",
  "strategic_fit": "excellent",
  "priority": "P1",
  "success_metrics": ["Sharpe ratio > 1.2", "Max drawdown < 15%", "Correlation with existing < 0.5"],
  "recommendation": "approve",
  "reasoning": "Aligns with Q4 roadmap goal of diversifying strategies. User research shows demand for sentiment-based trading. Competitive advantage since few algo traders use AI sentiment effectively. High business value."
}
```

### Consensus Calculation

```python
weights = decision_weights["new_trading_strategy"]
# {
#   "finance": 0.35,
#   "data_science": 0.25,
#   "development": 0.15,
#   "computer_science": 0.10,
#   "quantum": 0.10,
#   "product": 0.05
# }

votes = {
    "development": 0.5,      # approve_with_conditions
    "finance": 1.0,          # approve
    "data_science": 0.5,     # approve_with_conditions
    "computer_science": 0.5, # approve_with_conditions
    "quantum": 1.0,          # approve
    "product": 1.0           # approve
}

total_score = (
    0.35 * 1.0 +   # finance
    0.25 * 0.5 +   # data_science
    0.15 * 0.5 +   # development
    0.10 * 0.5 +   # computer_science
    0.10 * 1.0 +   # quantum
    0.05 * 1.0     # product
) = 0.625

# 0.625 >= 0.6 → APPROVED
```

### Final Decision

```json
{
  "decision": "APPROVED_WITH_CONDITIONS",
  "consensus_score": 0.625,
  "conditions": [
    "Implement Perplexity MCP server with caching (1 week)",
    "Collect 3 months historical sentiment data for fine-tuning",
    "Start with 15% portfolio allocation, scale to 20% after 1 month",
    "Implement circuit breaker for API failures",
    "Store API keys in HashiCorp Vault"
  ],
  "action_plan": {
    "week_1": [
      "Data Science: Collect historical sentiment data",
      "Development: Implement Perplexity MCP server",
      "Development: Set up Redis caching layer"
    ],
    "week_2": [
      "Data Science: Fine-tune FinBERT model",
      "Development: Integrate sentiment pipeline with trading engine",
      "Development: Implement circuit breaker and error handling"
    ],
    "week_3": [
      "Data Science: Feature engineering and backtesting",
      "Development: Integration tests and deployment",
      "Finance: Risk assessment and position sizing"
    ],
    "week_4": [
      "Paper trading with 15% allocation",
      "Monitor performance and sentiment accuracy",
      "Iterate based on results"
    ]
  },
  "risks": [
    "Perplexity API rate limits (mitigation: caching)",
    "Sentiment latency (mitigation: Redis cache with 5-min TTL)",
    "Model overfitting (mitigation: walk-forward validation)"
  ],
  "timeline": "3 weeks development + 1 week paper trading = 4 weeks total",
  "success_criteria": [
    "Sharpe ratio >= 1.2",
    "Max drawdown <= 15%",
    "Sentiment accuracy >= 75%",
    "API latency P95 < 500ms"
  ]
}
```

---

## Best Practices

### 1. Always Use Parallel Agents for Major Decisions
- New trading strategies
- Infrastructure changes
- ML model architecture decisions
- Risk management policy changes

### 2. Use Opus for Planning, Sonnet for Execution
- Opus for agent orchestration (complex reasoning)
- Sonnet for implementation (cost-effective)

### 3. Document All Decisions
- Save agent responses to `docs/architecture/decisions/`
- Use Architecture Decision Records (ADR) format

### 4. Continuous Improvement
- Track decision outcomes
- Adjust agent weights based on accuracy
- Refine agent prompts over time

### 5. Human Override
- Always allow human to override agent decisions
- Escalate when consensus < 0.3 (strong disagreement)

---

**Last Updated**: 2025-10-11
**Maintained By**: AI/ML Team & Product Management
