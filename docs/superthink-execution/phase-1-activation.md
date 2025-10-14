# Phase 1: Neural Network Training & Optimization
## Claude Code Max Activation Prompt

**Copy and paste this entire prompt into Claude Code Max:**

---

```
SUPERTHINK MODE ACTIVATED: Phase 1 - Neural Network Training & Optimization

Context: RRRalgorithms Cryptocurrency Trading System
Repository: /Volumes/Lexar/RRRVentures/RRRalgorithms
Phase: 1 of 6
Priority: CRITICAL
Duration: 6-8 hours
Subagents: 25 parallel agents across 5 teams

OBJECTIVE:
Train 3 production neural network models with Sharpe > 1.5, complete quantum optimization, implement comprehensive backtesting with 50,000+ Monte Carlo scenarios, add VaR/CVaR risk metrics, and integrate all 8 worktrees.

TEAM STRUCTURE:
Deploy 25 parallel subagents across 5 specialized teams:

Team 1: Data Science Team (5 agents)
Team 2: Quantum Optimization Team (4 agents)
Team 3: Backtesting Validation Team (4 agents)
Team 4: Risk Assessment Team (4 agents)
Team 5: Integration & Testing Team (8 agents)

EXECUTION STRATEGY:
1. Execute all team members in parallel using Superthink capabilities
2. Each subagent has independent tasks with clear deliverables
3. Teams coordinate only where dependencies exist
4. All code must be production-ready with tests

DETAILED TASK BREAKDOWN:

═══════════════════════════════════════════════════════════════
TEAM 1: DATA SCIENCE TEAM (5 Parallel Subagents)
═══════════════════════════════════════════════════════════════

Subagent DS-1: Feature Engineering Pipeline
├── Location: worktrees/neural-network/src/features/
├── Task: Create feature_engineering.py with 200+ candidate features
├── Input: crypto_aggregates table from PostgreSQL
├── Output: Feature matrix (numpy arrays) + feature importance scores
├── Files to create:
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   └── tests/test_features.py
└── Success: 200+ features engineered, top 50 selected by mutual information

Subagent DS-2: Transformer Architecture Optimizer
├── Location: worktrees/neural-network/src/models/price_prediction/
├── Task: Optimize transformer_model.py architecture
├── Test: 3 variants (lightweight: 5M params, balanced: 15M params, heavy: 50M params)
├── Benchmark: FLOPS vs accuracy tradeoff
├── Files to modify: transformer_model.py
├── Files to create:
│   ├── architecture_search.py
│   └── model_variants.py
└── Success: Balanced variant achieves 60%+ validation accuracy

Subagent DS-3: Distributed Training Harness
├── Location: worktrees/neural-network/src/training/
├── Task: Build multi-GPU training with checkpointing
├── Framework: PyTorch Distributed Data Parallel (DDP)
├── Files to create:
│   ├── distributed_trainer.py
│   └── checkpoint_manager.py
├── Features: Resume from checkpoint, gradient accumulation, mixed precision (AMP)
└── Success: Train 15M param model in < 4 hours (single GPU)

Subagent DS-4: Sentiment Model Training
├── Location: worktrees/neural-network/src/models/sentiment/
├── Task: Fine-tune FinBERT on crypto sentiment dataset
├── Dataset: Create from Perplexity AI historical news (10K examples)
├── Files to modify: bert_sentiment.py, train.py
├── Files to create:
│   ├── sentiment_dataset.py
│   └── data_augmentation.py
└── Success: 75%+ validation accuracy, inference < 50ms per text

Subagent DS-5: Model Registry Integration
├── Location: worktrees/neural-network/src/models/
├── Task: Complete registry.py Supabase integration
├── Features: Auto-register models, track metrics, version control
├── Files to modify: registry.py
├── Files to create:
│   ├── versioning.py
│   └── model_metadata.py
└── Success: All trained models auto-registered with full metadata

═══════════════════════════════════════════════════════════════
TEAM 2: QUANTUM OPTIMIZATION TEAM (4 Parallel Subagents)
═══════════════════════════════════════════════════════════════

Subagent QO-1: QAOA Portfolio Optimizer
├── Location: worktrees/quantum-optimization/src/portfolio/
├── Task: Complete quantum_portfolio_optimizer.py production implementation
├── Algorithm: QAOA-inspired (use qiskit or classical approximation)
├── Benchmark: vs scipy.optimize on 10-asset portfolio
├── Files to modify: quantum_portfolio_optimizer.py
├── Files to create:
│   ├── qaoa_circuit.py
│   └── portfolio_benchmark.py
└── Success: 50%+ faster than classical, identical or better Sharpe ratio

Subagent QO-2: Quantum Hyperparameter Search
├── Location: worktrees/quantum-optimization/src/hyperparameter/
├── Task: Quantum annealing for neural network hyperparameter tuning
├── Search space: Learning rate, dropout, batch size, layer sizes
├── Files to create:
│   ├── quantum_tuner.py
│   ├── search_space.py
│   └── tuning_results.py
├── Integrate: With DS-2 transformer architecture
└── Success: Find optimal hyperparams 3x faster than grid search

Subagent QO-3: Feature Selection via Quantum
├── Location: worktrees/quantum-optimization/src/features/
├── Task: Quantum feature selection (select best 50 from 200 features)
├── Method: QUBO formulation for feature selection problem
├── Files to create:
│   ├── quantum_feature_selector.py
│   └── qubo_formulation.py
├── Integrate: With DS-1 feature engineering
└── Success: Select features that improve model accuracy by 2-5%

Subagent QO-4: Benchmark & Performance Analysis
├── Location: worktrees/quantum-optimization/benchmarks/
├── Task: Compare quantum vs classical across all optimization tasks
├── Metrics: Runtime, solution quality, scalability
├── Files to create:
│   ├── benchmark_suite.py
│   └── performance_report.py
├── Output: Markdown report with charts
└── Success: Demonstrate quantum advantage in at least 2 of 3 tasks

═══════════════════════════════════════════════════════════════
TEAM 3: BACKTESTING VALIDATION TEAM (4 Parallel Subagents)
═══════════════════════════════════════════════════════════════

Subagent BT-1: Walk-Forward Analysis Engine
├── Location: worktrees/backtesting/src/validation/
├── Task: Implement sophisticated walk-forward testing
├── Method: Rolling window (train 6mo, test 1mo, repeat)
├── Guard: Out-of-sample performance tracking, detect overfitting
├── Files to create:
│   ├── walk_forward.py
│   └── performance_tracker.py
└── Success: OOS Sharpe within 20% of in-sample Sharpe

Subagent BT-2: Monte Carlo Simulation Engine
├── Location: worktrees/backtesting/src/monte_carlo/
├── Task: Build 50,000+ scenario stress test
├── Scenarios: Market regimes, volatility shocks, liquidity crises
├── Files to create:
│   ├── monte_carlo_engine.py
│   ├── scenario_generator.py
│   └── stress_tests.py
├── Output: Probabilistic P&L distribution (10th, 50th, 90th percentiles)
└── Success: 95% of scenarios show positive returns

Subagent BT-3: Strategy Performance Metrics
├── Location: worktrees/backtesting/src/metrics/
├── Task: Comprehensive performance metric calculator
├── Metrics: Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor
├── Files to create:
│   ├── performance_metrics.py
│   ├── risk_metrics.py
│   └── visualization.py
├── Output: HTML report with interactive charts
└── Success: Auto-generate comprehensive reports for all strategies

Subagent BT-4: Overfitting Detection System
├── Location: worktrees/backtesting/src/validation/
├── Task: Detect and quantify overfitting
├── Methods: Data snooping check, multiple hypothesis correction, robustness tests
├── Files to create:
│   ├── overfitting_detector.py
│   └── robustness_tests.py
├── Alert: Flag strategies with suspicious performance patterns
└── Success: Identify overfitted strategies before production deployment

═══════════════════════════════════════════════════════════════
TEAM 4: RISK ASSESSMENT TEAM (4 Parallel Subagents)
═══════════════════════════════════════════════════════════════

Subagent RA-1: VaR/CVaR Calculator
├── Location: worktrees/risk-management/src/metrics/
├── Task: Implement Value-at-Risk and Conditional VaR
├── Methods: Historical simulation, parametric (Gaussian), Monte Carlo
├── Files to create:
│   ├── var_calculator.py
│   ├── cvar_calculator.py
│   └── risk_dashboard.py
├── Real-time: Update every 1 minute during trading
└── Success: Real-time VaR display in monitoring dashboard

Subagent RA-2: Stress Testing Framework
├── Location: worktrees/risk-management/src/stress_testing/
├── Task: Replay 2008 financial crisis, 2020 COVID crash, 2022 FTX collapse
├── Method: Apply historical shocks to current portfolio
├── Files to create:
│   ├── stress_scenarios.py
│   ├── historical_replays.py
│   └── impact_analysis.py
├── Output: Max drawdown under extreme scenarios
└── Success: Portfolio survives all historical crises with < 30% drawdown

Subagent RA-3: Position Sizing Optimizer
├── Location: worktrees/risk-management/src/position_sizing/
├── Task: Implement Kelly Criterion, optimal-f, risk parity
├── Integrate: With portfolio optimizer (Team 2, QO-1)
├── Files to modify: position_sizer.py
├── Files to create:
│   ├── kelly_criterion.py
│   ├── optimal_f.py
│   └── risk_parity.py
└── Success: Position sizing adapts to volatility and strategy confidence

Subagent RA-4: Circuit Breaker System
├── Location: worktrees/risk-management/src/circuit_breakers/
├── Task: Auto-halt trading when risk limits breached
├── Triggers: Daily loss limit, VaR exceeded, correlation breakdown, flash crash detection
├── Files to create:
│   ├── circuit_breakers.py
│   ├── breach_detector.py
│   └── emergency_shutdown.py
├── Integration: With trading engine
└── Success: System auto-halts within 1 second of breach

═══════════════════════════════════════════════════════════════
TEAM 5: INTEGRATION & TESTING TEAM (8 Parallel Subagents)
═══════════════════════════════════════════════════════════════

Subagent IT-1: End-to-End Pipeline Orchestrator
├── Location: src/orchestration/
├── Task: Single-command startup for all 8 worktrees
├── Script: start_system.sh that launches data pipeline → models → trading engine
├── Files to create:
│   ├── orchestration/system_manager.py
│   └── scripts/start_system.sh
└── Success: One command starts entire system in < 60 seconds

Subagent IT-2: Data Flow Integration
├── Task: Connect data-pipeline → neural-network → trading-engine
├── Test: Real market data flows through entire system
├── Files to create: tests/integration/test_data_flow.py
└── Success: Data latency < 100ms end-to-end

Subagent IT-3: Model Inference Integration
├── Task: Neural network predictions trigger trading signals
├── Files to create: tests/integration/test_model_inference.py
└── Success: Predictions generated within 50ms of data update

Subagent IT-4: Risk Integration
├── Task: Risk checks block unsafe trades
├── Test: Trigger risk limits, verify trades blocked
├── Files to create: tests/integration/test_risk_checks.py
└── Success: 100% of unsafe trades blocked before execution

Subagent IT-5: Performance Profiler
├── Task: Identify bottlenecks in entire system
├── Tools: cProfile, line_profiler, Py-Spy, memory_profiler
├── Files to create:
│   ├── profiling/system_profiler.py
│   └── profiling/bottleneck_report.py
├── Target: < 100ms trading signal latency
└── Success: Identify and document top 10 bottlenecks with optimization recommendations

Subagent IT-6: Load Testing
├── Task: Test system under high load (1000 updates/sec)
├── Method: Simulate high-frequency market data
├── Files to create:
│   ├── tests/load/load_tester.py
│   └── tests/load/stress_test.py
└── Success: System handles 1000 updates/sec with < 100ms p95 latency

Subagent IT-7: Database Integration
├── Task: Verify all worktrees correctly read/write to PostgreSQL
├── Test: Data consistency, foreign key constraints, index performance
├── Files to create: tests/integration/test_database.py
└── Success: All 8 worktrees successfully interact with database

Subagent IT-8: Monitoring Integration
├── Task: All metrics flow to Grafana dashboard
├── Verify: Model predictions, trade executions, risk metrics all visible
├── Files to create: tests/integration/test_monitoring.py
└── Success: Real-time dashboard shows all critical metrics

═══════════════════════════════════════════════════════════════
SUCCESS CRITERIA
═══════════════════════════════════════════════════════════════

phase_1_complete = {
    "models_trained": {
        "price_predictor": {"accuracy": "> 60%", "status": "trained"},
        "sentiment_analyzer": {"accuracy": "> 75%", "status": "trained"},
        "portfolio_optimizer": {"sharpe": "> 1.5", "status": "validated"}
    },
    "quantum_optimization": {
        "speedup_vs_classical": "> 50%",
        "hyperparams_optimized": True,
        "feature_selection_complete": True
    },
    "backtesting": {
        "walk_forward_complete": True,
        "monte_carlo_scenarios": "> 50000",
        "sharpe_ratio": "> 1.5"
    },
    "risk_management": {
        "var_implemented": True,
        "stress_tests_passed": True,
        "circuit_breakers_working": True
    },
    "integration": {
        "end_to_end_tested": True,
        "latency_p95": "< 100ms",
        "all_worktrees_integrated": True
    }
}

═══════════════════════════════════════════════════════════════
DELIVERABLES
═══════════════════════════════════════════════════════════════

1. 3 trained neural network models (checkpoints in worktrees/neural-network/checkpoints/)
2. Quantum optimizer 50%+ faster than classical
3. 50,000+ Monte Carlo scenarios with 95%+ pass rate
4. VaR/CVaR metrics in real-time dashboard
5. All 8 worktrees integrated and tested
6. System startup time < 60 seconds
7. End-to-end latency < 100ms p95

═══════════════════════════════════════════════════════════════
EXECUTION INSTRUCTIONS
═══════════════════════════════════════════════════════════════

1. Activate Superthink mode with all 25 parallel subagents
2. Execute teams in parallel (Team 1-5 simultaneously)
3. Within each team, execute subagents in parallel
4. Update tracker: docs/superthink-execution/tracker.md
5. Run validation: scripts/superthink/validate-phase-1.sh
6. Generate completion report

BEGIN EXECUTION NOW.
```

---

## Post-Execution Validation

After Phase 1 completes, run:

```bash
cd /Volumes/Lexar/RRRVentures/RRRalgorithms

# Test neural networks
pytest worktrees/neural-network/tests/ -v

# Test backtesting
python worktrees/backtesting/src/validation/walk_forward.py

# Test quantum optimization
python worktrees/quantum-optimization/benchmarks/benchmark_suite.py

# Test risk management
python worktrees/risk-management/tests/test_var_calculator.py

# Test integration
pytest tests/integration/ -v

# Check system startup
./scripts/start_system.sh --validate
```

## Expected Duration

- Minimum: 6 hours (with optimal parallelization)
- Expected: 8 hours (realistic with coordination overhead)
- Maximum: 10 hours (if encountering technical challenges)

## Next Phase

After successful completion and validation:
→ Proceed to Phase 2A: Hypothesis Testing (10 research agents)

