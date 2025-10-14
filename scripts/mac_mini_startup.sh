#!/bin/bash
# Mac Mini Startup Script
# ========================
# Auto-starts trading system when Mac Mini boots
# Waits for Lexar drive to mount, then starts everything
#
# This script is called by LaunchAgent on Mac Mini boot

set -e

# Configuration
LEXAR_MOUNT="/Volumes/Lexar"
PROJECT_PATH="/Volumes/Lexar/RRRVentures/RRRalgorithms"
MAX_WAIT_SECONDS=120
LOG_FILE="/tmp/rrr_startup.log"

# Redirect all output to log
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================================"
echo "RRRalgorithms - Mac Mini Startup"
echo "========================================================================"
echo "Time: $(date)"
echo ""

# Function to check if Lexar is mounted
check_lexar_mounted() {
    if [ -d "$LEXAR_MOUNT" ]; then
        return 0
    else
        return 1
    fi
}

# Wait for Lexar drive to mount
echo "Waiting for Lexar drive to mount..."
WAIT_COUNT=0
while ! check_lexar_mounted; do
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
    
    if [ $WAIT_COUNT -gt $MAX_WAIT_SECONDS ]; then
        echo "❌ ERROR: Lexar drive not mounted after ${MAX_WAIT_SECONDS}s"
        echo "   Please check drive connection"
        exit 1
    fi
    
    echo "   Waiting... (${WAIT_COUNT}s)"
done

echo "✅ Lexar drive mounted at: $LEXAR_MOUNT"
echo ""

# Verify project directory exists
if [ ! -d "$PROJECT_PATH" ]; then
    echo "❌ ERROR: Project directory not found: $PROJECT_PATH"
    exit 1
fi

echo "✅ Project directory found"
echo ""

# Change to project directory
cd "$PROJECT_PATH"

# Wait a bit more for filesystem to be fully ready
sleep 5

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found"
    echo "   Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements-local.txt
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment found"
fi

# Activate virtual environment
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Set environment variables
export LEXAR_ROOT="$PROJECT_PATH"
export PYTHONPATH="$PROJECT_PATH:$PYTHONPATH"
export ENVIRONMENT=lexar

# Create necessary directories
mkdir -p data/database data/historical data/cache
mkdir -p logs/trading logs/system logs/archive
mkdir -p models/checkpoints models/production

echo "✅ Directories verified"
echo ""

# Start dashboard in background
echo "Starting Streamlit dashboard..."
nohup streamlit run src/dashboards/mobile_dashboard.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    > logs/system/dashboard.log 2>&1 &

DASHBOARD_PID=$!
echo "✅ Dashboard started (PID: $DASHBOARD_PID)"
echo "   Access: http://mac-mini-hostname:8501"
echo "   Log: logs/system/dashboard.log"
echo ""

# Wait for dashboard to start
sleep 5

# Start health monitor in background
echo "Starting health monitor..."
nohup python -m src.monitoring.health_monitor \
    > logs/system/health.log 2>&1 &

HEALTH_PID=$!
echo "✅ Health monitor started (PID: $HEALTH_PID)"
echo "   Log: logs/system/health.log"
echo ""

# Start Telegram bot (if configured)
if grep -q "TELEGRAM_BOT_TOKEN" config/.env.lexar 2>/dev/null; then
    echo "Starting Telegram bot..."
    nohup python -m src.monitoring.telegram_alerts \
        > logs/system/telegram.log 2>&1 &
    
    TELEGRAM_PID=$!
    echo "✅ Telegram bot started (PID: $TELEGRAM_PID)"
    echo "   Log: logs/system/telegram.log"
    echo ""
else
    echo "⚠️  Telegram not configured (skipping)"
    echo "   Configure in config/.env.lexar to enable alerts"
    echo ""
fi

# Start trading system (foreground)
echo "Starting trading system..."
echo "========================================================================"
echo ""

# This runs in foreground - if it exits, everything restarts via LaunchAgent
python -m src.main

# If we get here, system stopped
echo ""
echo "========================================================================"
echo "Trading system stopped"
echo "Time: $(date)"
echo "========================================================================"

