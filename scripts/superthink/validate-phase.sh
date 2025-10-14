#!/bin/bash

# Superthink Phase Validation Script
# Usage: ./validate-phase.sh <phase_number>

set -e

PHASE=$1
BASE_DIR="/Volumes/Lexar/RRRVentures/RRRalgorithms"

if [ -z "$PHASE" ]; then
    echo "Usage: ./validate-phase.sh <phase_number>"
    echo "Example: ./validate-phase.sh 1"
    exit 1
fi

echo "════════════════════════════════════════════════════════════════"
echo "  VALIDATING PHASE $PHASE"
echo "════════════════════════════════════════════════════════════════"
echo ""

cd $BASE_DIR

case $PHASE in
    1)
        echo "Validating Phase 1: Neural Network Training & Optimization"
        echo ""
        
        # Check if neural network models exist
        echo "📦 Checking neural network models..."
        if [ -d "worktrees/neural-network/checkpoints" ]; then
            model_count=$(find worktrees/neural-network/checkpoints -name "*.pt" 2>/dev/null | wc -l)
            echo "  Found $model_count model checkpoint(s)"
        else
            echo "  ⚠️  No checkpoints directory found"
        fi
        
        # Check quantum optimization
        echo ""
        echo "⚛️  Checking quantum optimization..."
        if [ -f "worktrees/quantum-optimization/src/portfolio/quantum_portfolio_optimizer.py" ]; then
            echo "  ✅ Quantum portfolio optimizer found"
        else
            echo "  ⚠️  Quantum portfolio optimizer not found"
        fi
        
        # Check backtesting
        echo ""
        echo "📊 Checking backtesting..."
        if [ -f "worktrees/backtesting/src/monte_carlo/monte_carlo_engine.py" ]; then
            echo "  ✅ Monte Carlo engine found"
        else
            echo "  ⚠️  Monte Carlo engine not found"
        fi
        
        # Check risk management
        echo ""
        echo "🛡️  Checking risk management..."
        if [ -f "worktrees/risk-management/src/metrics/var_calculator.py" ]; then
            echo "  ✅ VaR calculator found"
        else
            echo "  ⚠️  VaR calculator not found"
        fi
        
        # Run tests if available
        echo ""
        echo "🧪 Running tests..."
        if [ -d "worktrees/neural-network/tests" ]; then
            echo "  Running neural network tests..."
            python3 -m pytest worktrees/neural-network/tests/ -v --tb=short 2>/dev/null || echo "  Some tests failed or pytest not available"
        fi
        ;;
        
    2a)
        echo "Validating Phase 2A: Hypothesis Testing"
        echo ""
        
        # Check hypothesis files
        echo "📚 Checking hypothesis documentation..."
        hypothesis_count=$(find research/hypotheses -name "0*.md" 2>/dev/null | wc -l)
        echo "  Found $hypothesis_count hypothesis documents"
        
        # Check testing framework
        echo ""
        echo "🔬 Checking testing framework..."
        if [ -f "research/testing/hypothesis_tester.py" ]; then
            echo "  ✅ Hypothesis tester found"
        else
            echo "  ⚠️  Hypothesis tester not found"
        fi
        
        # Check backtest results
        echo ""
        echo "📈 Checking backtest results..."
        if [ -d "research/data" ]; then
            data_count=$(find research/data -name "*.csv" -o -name "*.parquet" 2>/dev/null | wc -l)
            echo "  Found $data_count data files"
        fi
        ;;
        
    2b)
        echo "Validating Phase 2B: Strategy Implementation"
        echo ""
        
        # Check strategy implementations
        echo "🎯 Checking strategy implementations..."
        if [ -d "src/strategies" ]; then
            strategy_count=$(find src/strategies -name "*.py" 2>/dev/null | wc -l)
            echo "  Found $strategy_count strategy files"
        else
            echo "  ⚠️  Strategies directory not found"
        fi
        ;;
        
    3)
        echo "Validating Phase 3: API Integration"
        echo ""
        
        # Check Polygon WebSocket
        echo "📡 Checking Polygon WebSocket..."
        if [ -f "worktrees/data-pipeline/src/data_pipeline/polygon/websocket_client.py" ]; then
            echo "  ✅ Polygon WebSocket client found"
        else
            echo "  ⚠️  Polygon WebSocket client not found"
        fi
        
        # Check TradingView
        echo ""
        echo "📊 Checking TradingView integration..."
        if [ -f "worktrees/api-integration/src/tradingview/webhook_server.py" ]; then
            echo "  ✅ TradingView webhook server found"
        else
            echo "  ⚠️  TradingView webhook server not found"
        fi
        
        # Check Perplexity
        echo ""
        echo "🤖 Checking Perplexity sentiment..."
        if [ -f "worktrees/data-pipeline/src/data_pipeline/sentiment/perplexity_client.py" ]; then
            echo "  ✅ Perplexity client found"
        else
            echo "  ⚠️  Perplexity client not found"
        fi
        
        # Check Coinbase
        echo ""
        echo "💱 Checking Coinbase connector..."
        if [ -f "worktrees/api-integration/src/exchanges/coinbase/coinbase_rest.py" ]; then
            echo "  ✅ Coinbase REST client found"
        else
            echo "  ⚠️  Coinbase REST client not found"
        fi
        ;;
        
    4)
        echo "Validating Phase 4: Multi-Agent System"
        echo ""
        
        # Check specialist agents
        echo "🤖 Checking specialist agents..."
        if [ -d "src/agents/specialists" ]; then
            agent_count=$(find src/agents/specialists -name "*_agent.py" 2>/dev/null | wc -l)
            echo "  Found $agent_count specialist agents"
        else
            echo "  ⚠️  Specialists directory not found"
        fi
        
        # Check master coordinator
        echo ""
        echo "🎯 Checking master coordinator..."
        if [ -f "src/agents/framework/master_coordinator.py" ]; then
            echo "  ✅ Master coordinator found"
        else
            echo "  ⚠️  Master coordinator not found"
        fi
        
        # Check learning system
        echo ""
        echo "🧠 Checking learning system..."
        if [ -f "src/agents/framework/agent_learning.py" ]; then
            echo "  ✅ Agent learning system found"
        else
            echo "  ⚠️  Agent learning system not found"
        fi
        ;;
        
    5)
        echo "Validating Phase 5: Production Deployment"
        echo ""
        
        # Check Docker optimization
        echo "🐳 Checking Docker..."
        if [ -f "deployment/docker-compose.yml" ]; then
            echo "  ✅ Docker Compose found"
        fi
        
        # Check Kubernetes
        echo ""
        echo "☸️  Checking Kubernetes..."
        if [ -d "deployment/kubernetes" ]; then
            manifest_count=$(find deployment/kubernetes -name "*.yaml" 2>/dev/null | wc -l)
            echo "  Found $manifest_count Kubernetes manifests"
        else
            echo "  ⚠️  Kubernetes directory not found"
        fi
        
        # Check Helm
        echo ""
        echo "⎈  Checking Helm..."
        if [ -f "deployment/helm/Chart.yaml" ]; then
            echo "  ✅ Helm chart found"
        else
            echo "  ⚠️  Helm chart not found"
        fi
        
        # Check monitoring
        echo ""
        echo "📊 Checking monitoring..."
        if [ -d "monitoring/grafana/dashboards" ]; then
            dashboard_count=$(find monitoring/grafana/dashboards -name "*.json" 2>/dev/null | wc -l)
            echo "  Found $dashboard_count Grafana dashboards"
        fi
        ;;
        
    *)
        echo "❌ Invalid phase number: $PHASE"
        echo "Valid phases: 1, 2a, 2b, 3, 4, 5"
        exit 1
        ;;
esac

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  VALIDATION COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Review the results above and address any ⚠️  warnings."
echo ""
echo "Update tracker:"
echo "  vim docs/superthink-execution/tracker.md"
echo ""

