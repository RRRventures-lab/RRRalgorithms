#!/bin/bash
# Universal Launch Script for RRRalgorithms Trading System
# Works on M3 MacBook or Mac Mini M4
# Automatically detects Lexar drive and runs from there
#
# Usage:
#   ./scripts/launch.sh              # Start full system
#   ./scripts/launch.sh --dashboard  # Dashboard only
#   ./scripts/launch.sh --trading    # Trading only

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}RRRalgorithms - Universal Launcher${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Detect project root (should be on Lexar)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo -e "${GREEN}Project root: ${PROJECT_ROOT}${NC}"

# Verify we're on Lexar drive
if [[ ! "$PROJECT_ROOT" =~ "/Volumes/Lexar" ]]; then
    echo -e "${YELLOW}⚠️  Warning: Project not on Lexar drive${NC}"
    echo -e "${YELLOW}Current location: ${PROJECT_ROOT}${NC}"
    echo -e "${YELLOW}Expected: /Volumes/Lexar/...${NC}"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Detect machine type
if [[ $(sysctl -n machdep.cpu.brand_string) =~ "M3" ]]; then
    MACHINE="MacBook M3"
elif [[ $(sysctl -n machdep.cpu.brand_string) =~ "M4" ]]; then
    MACHINE="Mac Mini M4"
else
    MACHINE="Mac ($(sysctl -n machdep.cpu.brand_string))"
fi

echo -e "${GREEN}Machine: ${MACHINE}${NC}"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found${NC}"
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements-local.txt
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${GREEN}✅ Virtual environment found${NC}"
fi

# Activate virtual environment
source venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"
echo ""

# Export environment variables
export LEXAR_ROOT="$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export ENVIRONMENT=lexar

# Create necessary directories
mkdir -p data/database data/historical data/cache
mkdir -p logs/trading logs/system logs/archive
mkdir -p models/checkpoints models/production

echo -e "${GREEN}✅ Directories verified${NC}"
echo ""

# Parse command line arguments
MODE="full"
if [ "$1" == "--dashboard" ]; then
    MODE="dashboard"
elif [ "$1" == "--trading" ]; then
    MODE="trading"
fi

# Launch based on mode
echo -e "${BLUE}Starting RRRalgorithms in ${MODE} mode...${NC}"
echo ""

case $MODE in
    "full")
        echo -e "${BLUE}Starting dashboard in background...${NC}"
        nohup streamlit run src/dashboards/mobile_dashboard.py \
            --server.address 0.0.0.0 \
            --server.port 8501 \
            --server.headless true \
            > logs/system/dashboard.log 2>&1 &
        DASHBOARD_PID=$!
        echo -e "${GREEN}✅ Dashboard started (PID: $DASHBOARD_PID)${NC}"
        echo -e "${GREEN}   Access: http://localhost:8501${NC}"
        echo ""
        
        echo -e "${BLUE}Starting trading system...${NC}"
        python -m src.main
        ;;
        
    "dashboard")
        echo -e "${BLUE}Starting dashboard only...${NC}"
        streamlit run src/dashboards/mobile_dashboard.py \
            --server.address 0.0.0.0 \
            --server.port 8501
        ;;
        
    "trading")
        echo -e "${BLUE}Starting trading system only...${NC}"
        python -m src.main
        ;;
esac

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}System stopped gracefully${NC}"
echo -e "${GREEN}========================================${NC}"

