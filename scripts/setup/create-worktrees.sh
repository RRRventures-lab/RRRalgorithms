#!/bin/bash
# Create all worktrees for parallel development

set -e  # Exit on error

echo "🌳 Creating worktrees for parallel development..."

# Define worktrees
declare -a WORKTREES=(
    "neural-network:Neural Network - ML models, training, inference"
    "data-pipeline:Data Pipeline - Ingestion, processing, storage"
    "trading-engine:Trading Engine - Order execution, position management"
    "risk-management:Risk Management - Portfolio risk, position sizing"
    "backtesting:Backtesting - Historical testing, optimization"
    "api-integration:API Integration - TradingView, Polygon, Perplexity"
    "quantum-optimization:Quantum Optimization - Quantum algorithms"
    "monitoring:Monitoring - Observability, alerting, dashboards"
)

# Create worktrees directory
mkdir -p worktrees

# Create each worktree
for worktree_info in "${WORKTREES[@]}"; do
    IFS=':' read -r name description <<< "$worktree_info"
    branch="feature/${name}"
    path="worktrees/${name}"

    echo ""
    echo "📁 Creating worktree: ${name}"
    echo "   Branch: ${branch}"
    echo "   Path: ${path}"
    echo "   Purpose: ${description}"

    # Create worktree with new branch
    git worktree add "${path}" -b "${branch}"

    # Create a README in the worktree
    cat > "${path}/README.md" <<EOF
# ${description}

## Worktree: ${name}

**Branch**: \`${branch}\`
**Path**: \`${path}\`

## Purpose

${description}

## Getting Started

\`\`\`bash
cd ${path}

# Install dependencies (if applicable)
# pip install -r requirements.txt  # For Python
# npm install  # For Node.js

# Run tests
# pytest tests/  # For Python
# npm test  # For Node.js
\`\`\`

## Documentation

See main documentation at \`../../docs/worktrees/WORKTREE_ARCHITECTURE.md\`

For specific details about this component, see:
- Architecture: \`../../docs/architecture/${name}/\`
- API Specs: \`../../docs/api-specs/${name}/\`

## Development Workflow

1. Make changes in this worktree
2. Run tests: \`pytest tests/\` or \`npm test\`
3. Commit: \`git add . && git commit -m "Description"\`
4. Push: \`git push origin ${branch}\`
5. Create PR when ready for integration

## Terminal Session

Keep this terminal open for development:

\`\`\`bash
cd ${path}
# Your development commands here
\`\`\`

## Status

- [ ] Initial setup
- [ ] Dependencies installed
- [ ] Tests passing
- [ ] Integration with main repo

---
**Created**: $(date +%Y-%m-%d)
**Last Updated**: $(date +%Y-%m-%d)
EOF

    echo "   ✅ Worktree created successfully"
done

echo ""
echo "🎉 All worktrees created successfully!"
echo ""
echo "📋 Worktree Summary:"
git worktree list
echo ""
echo "💡 Next Steps:"
echo "   1. Open separate terminals for each worktree"
echo "   2. Start developing in parallel"
echo "   3. Use 'git worktree list' to see all worktrees"
echo "   4. Use 'git worktree remove <path>' to remove a worktree when done"
echo ""
echo "🚀 Example: Open neural-network worktree"
echo "   cd worktrees/neural-network"
echo ""
