#!/bin/bash
# =============================================================================
# RRRalgorithms - Start Local Development Server
# Starts the trading system in local development mode
# =============================================================================

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Determine project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  RRRalgorithms - Local Development${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "$PROJECT_ROOT/venv" ]; then
    echo -e "${YELLOW}⚠${NC}  Virtual environment not found!"
    echo ""
    echo "Run setup first:"
    echo "  ./scripts/setup/setup-local.sh"
    echo ""
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}✓${NC} Activating virtual environment..."
source "$PROJECT_ROOT/venv/bin/activate"

# Change to project root
cd "$PROJECT_ROOT"

# Check if database exists
if [ ! -f "$PROJECT_ROOT/data/local.db" ]; then
    echo -e "${YELLOW}⚠${NC}  Database not found, initializing..."
    python3 scripts/setup/init-local-db.py
fi

# Set environment to local
export ENVIRONMENT=local

echo -e "${GREEN}✓${NC} Starting trading system..."
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start the system
python -m src.main "$@"

