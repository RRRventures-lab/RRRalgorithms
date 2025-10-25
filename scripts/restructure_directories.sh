#!/bin/bash
# RRRalgorithms Directory Restructuring Script
# Date: 2025-10-25
# Purpose: Remove duplicate directories and organize structure

set -e  # Exit on error

echo "=== RRRalgorithms Directory Restructuring ==="
echo "Date: $(date)"
echo ""

# Create backup tag first
echo "Creating backup tag..."
git tag backup/pre-restructure-$(date +%Y%m%d-%H%M%S)
echo "✓ Backup tag created"
echo ""

# Phase 1: Remove duplicate directories (keeping src/services/* versions as they're more complete)
echo "Phase 1: Removing duplicate directories..."

# Remove backtesting duplicate (keep services/backtesting)
if [ -d "src/backtesting" ]; then
    echo "  Removing src/backtesting/ (duplicate of services/backtesting/)"
    rm -rf src/backtesting
fi

# Remove data_pipeline duplicates (keep the one with most files)
if [ -d "src/data_pipeline_original" ]; then
    echo "  Removing src/data_pipeline_original/"
    rm -rf src/data_pipeline_original
fi

# Remove monitoring (keep monitoring_original which has more files)
if [ -d "src/monitoring" ]; then
    echo "  Removing src/monitoring/ (smaller version)"
    rm -rf src/monitoring
fi

# Remove neural-network duplicate (keep services/neural_network)
if [ -d "src/neural-network" ]; then
    echo "  Removing src/neural-network/ (duplicate of services/neural_network/)"
    rm -rf src/neural-network
fi

# Remove quantum duplicate (keep services/quantum_optimization)
if [ -d "src/quantum" ]; then
    echo "  Removing src/quantum/ (duplicate of services/quantum_optimization/)"
    rm -rf src/quantum
fi

# Remove trading duplicates (keep services versions)
if [ -d "src/trading" ]; then
    echo "  Removing src/trading/ (duplicates in services/)"
    rm -rf src/trading
fi

echo "✓ Phase 1 complete: Duplicates removed"
echo ""

# Phase 2: Organize documentation
echo "Phase 2: Organizing documentation..."

# Create docs structure
mkdir -p docs/reports/{implementation,performance,audits,downloads}
mkdir -p docs/architecture
mkdir -p docs/guides

# Move report files
echo "  Moving report files to docs/reports/..."
mv *_REPORT.md docs/reports/ 2>/dev/null || true
mv *_DOWNLOAD_REPORT.md docs/reports/downloads/ 2>/dev/null || true
mv AUDIT_REPORT.md docs/reports/audits/ 2>/dev/null || true

# Move implementation/status files
mv *_COMPLETE.md docs/reports/implementation/ 2>/dev/null || true
mv *_IMPLEMENTATION*.md docs/reports/implementation/ 2>/dev/null || true
mv *_STATUS*.md docs/reports/implementation/ 2>/dev/null || true
mv IMPLEMENTATION_STATUS.md docs/reports/implementation/ 2>/dev/null || true

# Move architecture docs
mv ARCHITECTURE*.md docs/architecture/ 2>/dev/null || true

# Move summaries
mv *_SUMMARY*.md docs/reports/ 2>/dev/null || true

# Move quick start guides
mv QUICK_START*.md docs/guides/ 2>/dev/null || true
mv DIVISION_1_DELIVERABLES.md docs/reports/ 2>/dev/null || true
mv HONEST_SUMMARY.md docs/reports/ 2>/dev/null || true
mv NEXT_PHASE_ARCHITECTURE_IMPROVEMENTS.md docs/architecture/ 2>/dev/null || true

echo "✓ Phase 2 complete: Documentation organized"
echo ""

# Phase 3: Clean up data-pipeline name confusion
echo "Phase 3: Cleaning up naming..."

# Keep src/data-pipeline (with hyphen) or src/data_pipeline (with underscore)?
# Let's standardize to underscore for Python
if [ -d "src/data-pipeline" ]; then
    if [ -d "src/data_pipeline" ]; then
        echo "  Both data-pipeline and data_pipeline exist, keeping data_pipeline"
        rm -rf src/data-pipeline
    else
        echo "  Renaming src/data-pipeline to src/data_pipeline"
        mv src/data-pipeline src/data_pipeline
    fi
fi

echo "✓ Phase 3 complete: Naming standardized"
echo ""

# Phase 4: Flatten nested directories (if any exist)
echo "Phase 4: Flattening nested directories..."

# Check for nested patterns and flatten them
# This is complex and risky, so we'll skip for now
echo "  Skipping automatic flattening (requires careful analysis)"
echo "✓ Phase 4 complete (skipped)"
echo ""

# Summary
echo "=== Restructuring Complete ==="
echo ""
echo "Changes made:"
echo "  - Removed duplicate directories in src/"
echo "  - Organized documentation in docs/"
echo "  - Standardized naming conventions"
echo ""
echo "Remaining work:"
echo "  - Review changes carefully"
echo "  - Update import statements if needed"
echo "  - Test critical functionality"
echo "  - Commit changes"
echo ""
echo "To rollback: git reset --hard && git clean -fd"
echo ""
