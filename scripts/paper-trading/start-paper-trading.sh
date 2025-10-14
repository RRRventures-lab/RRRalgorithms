#!/bin/bash
#
# Start Paper Trading System
#
# Launches all services in paper trading mode with safety checks
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== RRR Trading - Paper Trading Startup ===${NC}"
echo ""

# Change to project root
cd "$(dirname "$0")/../.."

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Docker is not running${NC}"
    echo "Please start Docker Desktop and try again"
    exit 1
fi

# Check required files exist
REQUIRED_FILES=(
    "docker-compose.yml"
    "docker-compose.paper-trading.yml"
    "config/api-keys/.env"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}ERROR: Required file not found: $file${NC}"
        exit 1
    fi
done

# Check environment variables
echo -e "${YELLOW}Checking environment variables...${NC}"
if ! grep -q "SUPABASE_URL" config/api-keys/.env; then
    echo -e "${RED}ERROR: SUPABASE_URL not set in config/api-keys/.env${NC}"
    exit 1
fi

if ! grep -q "COINBASE_API_KEY" config/api-keys/.env; then
    echo -e "${YELLOW}WARNING: COINBASE_API_KEY not set (using mock data)${NC}"
fi

# Verify paper trading mode
echo -e "${YELLOW}Verifying paper trading configuration...${NC}"
if grep -q "PAPER_TRADING_MODE=false" docker-compose.paper-trading.yml; then
    echo -e "${RED}ERROR: Paper trading mode is disabled in docker-compose.paper-trading.yml${NC}"
    echo "Please set PAPER_TRADING_MODE=true before proceeding"
    exit 1
fi

# Confirm with user
echo ""
echo -e "${YELLOW}⚠️  IMPORTANT: Paper Trading Mode ⚠️${NC}"
echo "This will start the trading system in PAPER TRADING mode."
echo "No real orders will be placed on exchanges."
echo "Real API keys are NOT required for paper trading."
echo ""
read -p "Continue? (yes/no): " -r
echo ""

if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Pull latest images
echo -e "${GREEN}Pulling latest Docker images...${NC}"
docker-compose pull

# Stop any existing containers
echo -e "${YELLOW}Stopping existing containers...${NC}"
docker-compose -f docker-compose.yml -f docker-compose.paper-trading.yml down --remove-orphans

# Build images if needed
echo -e "${GREEN}Building Docker images (if needed)...${NC}"
docker-compose -f docker-compose.yml -f docker-compose.paper-trading.yml build

# Start services
echo -e "${GREEN}Starting paper trading services...${NC}"
docker-compose -f docker-compose.yml -f docker-compose.paper-trading.yml up -d

# Wait for services to be healthy
echo -e "${YELLOW}Waiting for services to be healthy...${NC}"
sleep 10

# Check service health
echo ""
echo -e "${GREEN}Checking service health...${NC}"
docker-compose -f docker-compose.yml -f docker-compose.paper-trading.yml ps

# Verify paper trading mode is enabled
echo ""
echo -e "${YELLOW}Verifying paper trading mode...${NC}"
if docker-compose logs trading-engine 2>&1 | grep -q "PAPER_TRADING_MODE.*enabled"; then
    echo -e "${GREEN}✓ Paper trading mode confirmed ENABLED${NC}"
else
    echo -e "${RED}✗ WARNING: Could not confirm paper trading mode${NC}"
    echo "Please check logs manually: docker-compose logs trading-engine"
fi

# Display access information
echo ""
echo -e "${GREEN}=== Paper Trading System Started ===${NC}"
echo ""
echo "Services:"
echo "  - Neural Network:     http://localhost:8000"
echo "  - Data Pipeline:      http://localhost:8001"
echo "  - Trading Engine:     http://localhost:8002"
echo "  - Risk Management:    http://localhost:8003"
echo "  - Grafana Dashboard:  http://localhost:3000 (admin/admin)"
echo "  - Prometheus:         http://localhost:9090"
echo ""
echo "Commands:"
echo "  - Monitor logs:       ./scripts/paper-trading/monitor-paper-trading.sh"
echo "  - Stop trading:       ./scripts/paper-trading/stop-paper-trading.sh"
echo "  - View metrics:       open http://localhost:3000"
echo ""
echo -e "${GREEN}Paper trading is now active!${NC}"
echo -e "${YELLOW}Monitor the system at: http://localhost:3000${NC}"
