#!/bin/bash
#
# Monitor Paper Trading System
#
# Displays real-time status and metrics
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

cd "$(dirname "$0")/../.."

# Check if system is running
if ! docker-compose ps | grep -q "Up"; then
    echo -e "${RED}ERROR: Paper trading system is not running${NC}"
    echo "Start it with: ./scripts/paper-trading/start-paper-trading.sh"
    exit 1
fi

# Function to display service status
show_service_status() {
    echo -e "${GREEN}=== Service Status ===${NC}"
    docker-compose -f docker-compose.yml -f docker-compose.paper-trading.yml ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
    echo ""
}

# Function to display paper trading metrics
show_metrics() {
    echo -e "${GREEN}=== Paper Trading Metrics ===${NC}"

    # Try to get metrics from trading engine
    if curl -s http://localhost:8002/metrics > /dev/null 2>&1; then
        METRICS=$(curl -s http://localhost:8002/metrics)

        # Parse key metrics
        TOTAL_ORDERS=$(echo "$METRICS" | grep "trading_orders_total" | head -n 1 | awk '{print $2}')
        DAILY_PNL=$(echo "$METRICS" | grep "trading_daily_pnl" | head -n 1 | awk '{print $2}')
        WIN_RATE=$(echo "$METRICS" | grep "trading_win_rate" | head -n 1 | awk '{print $2}')

        echo "Total Orders:      ${TOTAL_ORDERS:-N/A}"
        echo "Daily P&L:         \$${DAILY_PNL:-N/A}"
        echo "Win Rate:          ${WIN_RATE:-N/A}%"
    else
        echo -e "${YELLOW}Unable to fetch metrics (service may be starting)${NC}"
    fi
    echo ""
}

# Function to display recent logs
show_recent_logs() {
    echo -e "${GREEN}=== Recent Activity (last 20 lines) ===${NC}"
    docker-compose -f docker-compose.yml -f docker-compose.paper-trading.yml logs --tail=20
    echo ""
}

# Function to check for errors
check_errors() {
    echo -e "${GREEN}=== Error Check ===${NC}"

    ERROR_COUNT=$(docker-compose logs 2>&1 | grep -ci "error\|exception\|failed" || true)

    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo -e "${RED}Found $ERROR_COUNT error/exception messages in logs${NC}"
        echo "Use 'docker-compose logs [service-name]' to investigate"
    else
        echo -e "${GREEN}No errors detected${NC}"
    fi
    echo ""
}

# Function to verify paper trading mode
verify_paper_trading() {
    echo -e "${GREEN}=== Paper Trading Mode Verification ===${NC}"

    if docker-compose logs trading-engine 2>&1 | grep -q "PAPER_TRADING_MODE.*enabled"; then
        echo -e "${GREEN}✓ Paper trading mode is ENABLED${NC}"
    else
        echo -e "${RED}✗ WARNING: Paper trading mode status unclear${NC}"
        echo "Check logs: docker-compose logs trading-engine | grep PAPER"
    fi
    echo ""
}

# Main menu
show_menu() {
    clear
    echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║   RRR Trading - Paper Trading Monitor     ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
    echo ""

    show_service_status
    show_metrics
    verify_paper_trading

    echo -e "${YELLOW}Options:${NC}"
    echo "  1) Show recent logs"
    echo "  2) Check for errors"
    echo "  3) Tail all logs (live)"
    echo "  4) Tail specific service"
    echo "  5) Open Grafana dashboard"
    echo "  6) Refresh"
    echo "  q) Quit"
    echo ""
    read -p "Select option: " -n 1 -r
    echo ""

    case $REPLY in
        1)
            show_recent_logs
            read -p "Press Enter to continue..."
            ;;
        2)
            check_errors
            read -p "Press Enter to continue..."
            ;;
        3)
            echo -e "${YELLOW}Tailing logs (Ctrl+C to stop)...${NC}"
            docker-compose -f docker-compose.yml -f docker-compose.paper-trading.yml logs -f
            ;;
        4)
            echo "Services: neural-network, data-pipeline, trading-engine, risk-management, monitoring"
            read -p "Enter service name: " SERVICE
            docker-compose logs -f "$SERVICE"
            ;;
        5)
            echo "Opening Grafana dashboard..."
            if command -v open &> /dev/null; then
                open http://localhost:3000
            elif command -v xdg-open &> /dev/null; then
                xdg-open http://localhost:3000
            else
                echo "Navigate to: http://localhost:3000"
            fi
            read -p "Press Enter to continue..."
            ;;
        6)
            ;;
        q|Q)
            echo "Exiting monitor..."
            exit 0
            ;;
        *)
            echo "Invalid option"
            sleep 1
            ;;
    esac
}

# Run interactive menu in loop
while true; do
    show_menu
done
