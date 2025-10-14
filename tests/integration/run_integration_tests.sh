#!/bin/bash
# Integration Test Runner for RRRalgorithms

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}RRRalgorithms Integration Test Suite${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""

# Check if in correct directory
if [ ! -f "config/api-keys/.env" ]; then
    echo -e "${RED}Error: Must run from project root directory${NC}"
    echo -e "${YELLOW}Current directory: $(pwd)${NC}"
    exit 1
fi

# Load environment variables
echo -e "${YELLOW}Loading environment variables...${NC}"
set -a
source config/api-keys/.env
set +a
echo -e "${GREEN}✓ Environment loaded${NC}"
echo ""

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python ${PYTHON_VERSION}${NC}"
echo ""

# Install dependencies if needed
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

echo -e "${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate

echo -e "${YELLOW}Installing test dependencies...${NC}"
pip install -q -r tests/integration/requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Verify Supabase connection
echo -e "${YELLOW}Verifying Supabase connection...${NC}"
if [ -z "$SUPABASE_URL" ]; then
    echo -e "${RED}Error: SUPABASE_URL not set${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Supabase configured${NC}"
echo ""

# Run test suites
echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}Running Integration Tests${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""

# Test 1: MCP Connections
echo -e "${YELLOW}[1/3] Testing MCP Connections...${NC}"
pytest tests/integration/test_mcp_connections.py -v --tb=short
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ MCP Connection Tests PASSED${NC}"
else
    echo -e "${RED}❌ MCP Connection Tests FAILED${NC}"
    exit 1
fi
echo ""

# Test 2: Real-time Subscriptions
echo -e "${YELLOW}[2/3] Testing Real-time Subscriptions...${NC}"
pytest tests/integration/test_realtime_subscriptions.py -v --tb=short
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Real-time Subscription Tests PASSED${NC}"
else
    echo -e "${RED}❌ Real-time Subscription Tests FAILED${NC}"
    exit 1
fi
echo ""

# Test 3: End-to-End Pipeline
echo -e "${YELLOW}[3/3] Testing End-to-End Pipeline...${NC}"
pytest tests/integration/test_end_to_end_pipeline.py -v --tb=short
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ End-to-End Pipeline Tests PASSED${NC}"
else
    echo -e "${RED}❌ End-to-End Pipeline Tests FAILED${NC}"
    exit 1
fi
echo ""

# Summary
echo -e "${BLUE}=============================================================================${NC}"
echo -e "${GREEN}ALL INTEGRATION TESTS PASSED ✅${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""
echo -e "${GREEN}System is ready for paper trading deployment!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Start data pipeline: cd worktrees/data-pipeline && python src/main.py"
echo "  2. Start neural network: cd worktrees/neural-network && python src/main.py"
echo "  3. Start trading engine: cd worktrees/trading-engine && python src/main.py"
echo "  4. Start monitoring: cd worktrees/monitoring && streamlit run src/dashboard/app.py"
echo ""
