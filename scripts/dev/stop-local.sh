#!/bin/bash
# =============================================================================
# RRRalgorithms - Stop Local Development
# Gracefully stop all Python processes related to RRRalgorithms
# =============================================================================

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🛑 Stopping RRRalgorithms services..."
echo ""

# Find and kill Python processes running src.main
PIDS=$(pgrep -f "python.*src.main" || true)

if [ -z "$PIDS" ]; then
    echo -e "${YELLOW}ℹ${NC}  No running services found"
else
    echo "Found processes: $PIDS"
    echo "Sending SIGTERM..."
    echo "$PIDS" | xargs kill -TERM
    
    # Wait a moment
    sleep 2
    
    # Check if still running
    REMAINING=$(pgrep -f "python.*src.main" || true)
    if [ ! -z "$REMAINING" ]; then
        echo "Some processes still running, sending SIGKILL..."
        echo "$REMAINING" | xargs kill -KILL
    fi
    
    echo -e "${GREEN}✓${NC} Services stopped"
fi

echo ""
echo "Done."

