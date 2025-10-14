#!/bin/bash
# =============================================================================
# RRRalgorithms - Run Individual Service
# Run a specific service independently
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Determine project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Check arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <service_name>"
    echo ""
    echo "Available services:"
    echo "  data_pipeline      - Market data ingestion"
    echo "  trading_engine     - Order execution"
    echo "  risk_management    - Risk monitoring"
    echo "  monitor            - Live monitoring dashboard"
    echo ""
    echo "Example:"
    echo "  $0 monitor"
    exit 1
fi

SERVICE=$1

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Starting: $SERVICE${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Activate virtual environment
if [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Change to project root
cd "$PROJECT_ROOT"

# Set environment
export ENVIRONMENT=local

# Run the service
echo -e "${GREEN}✓${NC} Running $SERVICE..."
echo ""

python -m src.main --service "$SERVICE"

