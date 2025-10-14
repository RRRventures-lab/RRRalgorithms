#!/bin/bash

# Superthink Phase 1 Activation Helper
# This script prepares the environment and displays the activation prompt

set -e

echo "════════════════════════════════════════════════════════════════"
echo "  SUPERTHINK PHASE 1: Neural Network Training & Optimization"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Agents: 25 parallel subagents across 5 teams"
echo "Duration: 6-8 hours"
echo "Priority: CRITICAL"
echo ""

# Navigate to neural network worktree
cd /Volumes/Lexar/RRRVentures/RRRalgorithms/worktrees/neural-network

# Check environment
echo "📋 Pre-flight Checks:"
echo ""

# Check if worktree exists
if [ -d "/Volumes/Lexar/RRRVentures/RRRalgorithms/worktrees/neural-network" ]; then
    echo "✅ Neural network worktree found"
else
    echo "❌ Neural network worktree not found"
    exit 1
fi

# Check if Python is available
if command -v python3 &> /dev/null; then
    echo "✅ Python 3 installed: $(python3 --version)"
else
    echo "❌ Python 3 not found"
    exit 1
fi

# Check if PyTorch is available
if python3 -c "import torch" 2>/dev/null; then
    echo "✅ PyTorch installed"
else
    echo "⚠️  PyTorch not found (will need to install)"
fi

# Check if database is accessible
if pg_isready -h localhost 2>/dev/null; then
    echo "✅ PostgreSQL is running"
else
    echo "⚠️  PostgreSQL not running (some features may not work)"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  READY TO LAUNCH"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo ""
echo "1. Open Claude Code Max in Cursor"
echo "2. Copy the activation prompt:"
echo "   cat /Volumes/Lexar/RRRVentures/RRRalgorithms/docs/superthink-execution/phase-1-activation.md"
echo ""
echo "3. Paste into Claude Code Max and execute"
echo ""
echo "4. Monitor progress in tracker:"
echo "   tail -f /Volumes/Lexar/RRRVentures/RRRalgorithms/docs/superthink-execution/tracker.md"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# Offer to display the activation prompt
read -p "Display activation prompt now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    cat /Volumes/Lexar/RRRVentures/RRRalgorithms/docs/superthink-execution/phase-1-activation.md
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "Copy everything above (from 'SUPERTHINK MODE ACTIVATED' to 'BEGIN EXECUTION NOW')"
    echo "════════════════════════════════════════════════════════════════"
fi

