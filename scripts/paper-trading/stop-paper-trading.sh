#!/bin/bash
#
# Stop Paper Trading System
#
# Gracefully stops all paper trading services
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== RRR Trading - Paper Trading Shutdown ===${NC}"
echo ""

cd "$(dirname "$0")/../.."

# Check if system is running
if ! docker-compose ps | grep -q "Up"; then
    echo -e "${YELLOW}Paper trading system is not currently running${NC}"
    exit 0
fi

# Confirm with user
read -p "Stop paper trading system? (yes/no): " -r
echo ""

if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Export final metrics before shutdown
echo -e "${YELLOW}Exporting final metrics...${NC}"
if curl -s http://localhost:8002/metrics > /dev/null 2>&1; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    mkdir -p results/paper-trading
    curl -s http://localhost:8002/metrics > "results/paper-trading/final_metrics_${TIMESTAMP}.txt"
    echo -e "${GREEN}✓ Metrics exported to results/paper-trading/final_metrics_${TIMESTAMP}.txt${NC}"
fi

# Stop services gracefully
echo -e "${YELLOW}Stopping services gracefully...${NC}"
docker-compose -f docker-compose.yml -f docker-compose.paper-trading.yml stop

# Wait a moment
sleep 2

# Remove containers
echo -e "${YELLOW}Removing containers...${NC}"
docker-compose -f docker-compose.yml -f docker-compose.paper-trading.yml down

# Optional: Clean volumes
read -p "Remove data volumes? This will delete all paper trading data (yes/no): " -r
echo ""

if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo -e "${YELLOW}Removing volumes...${NC}"
    docker-compose -f docker-compose.yml -f docker-compose.paper-trading.yml down -v
    echo -e "${GREEN}✓ Volumes removed${NC}"
fi

echo ""
echo -e "${GREEN}=== Paper Trading System Stopped ===${NC}"
echo ""
echo "Summary:"
echo "  - All services stopped"
echo "  - Final metrics exported to results/paper-trading/"
echo ""
echo "To start again:"
echo "  ./scripts/paper-trading/start-paper-trading.sh"
