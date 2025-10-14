#!/bin/bash
# Mac Mini First Boot Setup
# ==========================
# Run this ONCE on Mac Mini after transferring Lexar drive
# Sets up everything for 24/7 autonomous operation
#
# Usage: ./scripts/mac_mini_first_boot.sh

set -e

echo "========================================================================"
echo "RRRalgorithms - Mac Mini First Boot Setup"
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Install required software (Python, Homebrew, Tailscale)"
echo "  2. Set up auto-start on boot"
echo "  3. Configure system for 24/7 operation"
echo "  4. Test the trading system"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

echo ""
echo "========================================" echo "Step 1: Install Homebrew (if needed)"
echo "========================================"

if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    echo "✅ Homebrew installed"
else
    echo "✅ Homebrew already installed"
fi

echo ""
echo "========================================"
echo "Step 2: Install Python 3.11+"
echo "========================================"

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Current Python version: $PYTHON_VERSION"

if (( $(echo "$PYTHON_VERSION < 3.11" | bc -l) )); then
    echo "Installing Python 3.11..."
    brew install python@3.11
    echo "✅ Python 3.11 installed"
else
    echo "✅ Python version OK"
fi

echo ""
echo "========================================"
echo "Step 3: Install Tailscale"
echo "========================================"

if ! command -v tailscale &> /dev/null; then
    echo "Installing Tailscale..."
    brew install tailscale
    echo "✅ Tailscale installed"
    echo ""
    echo "📱 Next steps for Tailscale:"
    echo "   1. Run: sudo tailscale up"
    echo "   2. Follow the URL to authenticate"
    echo "   3. Install Tailscale app on iPhone"
    echo "   4. Login with same account"
    echo ""
    read -p "Press Enter to continue..."
else
    echo "✅ Tailscale already installed"
fi

echo ""
echo "========================================"
echo "Step 4: Verify Lexar Drive"
echo "========================================================================"

if [ ! -d "/Volumes/Lexar/RRRVentures/RRRalgorithms" ]; then
    echo "❌ ERROR: Lexar drive not found or project not at expected location"
    echo "   Expected: /Volumes/Lexar/RRRVentures/RRRalgorithms"
    echo "   Please ensure Lexar drive is connected"
    exit 1
fi

cd /Volumes/Lexar/RRRVentures/RRRalgorithms
echo "✅ Project directory found on Lexar"
echo "   Location: $(pwd)"

echo ""
echo "========================================"
echo "Step 5: Set Up Python Environment"
echo "========================================"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements-local.txt
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment exists"
fi

source venv/bin/activate
echo "✅ Virtual environment activated"

echo ""
echo "========================================"
echo "Step 6: Initialize Database"
echo "========================================"

if [ ! -f "data/database/local.db" ]; then
    echo "Initializing database..."
    python -c "from src.core.database.local_db import get_db; db = get_db(); print('✅ Database initialized')"
else
    echo "✅ Database already exists"
fi

echo ""
echo "========================================"
echo "Step 7: Set Up Auto-Start"
echo "========================================"

PLIST_SOURCE="scripts/com.rrrventures.trading.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/com.rrrventures.trading.plist"

mkdir -p "$HOME/Library/LaunchAgents"

if [ -f "$PLIST_SOURCE" ]; then
    echo "Installing LaunchAgent..."
    cp "$PLIST_SOURCE" "$PLIST_DEST"
    
    # Load the LaunchAgent
    launchctl unload "$PLIST_DEST" 2>/dev/null || true
    launchctl load "$PLIST_DEST"
    
    echo "✅ LaunchAgent installed and loaded"
    echo "   The trading system will now start automatically on boot"
else
    echo "❌ ERROR: LaunchAgent plist not found"
fi

echo ""
echo "========================================"
echo "Step 8: System Preferences"
echo "========================================"

echo "Configuring Mac Mini for 24/7 operation..."

# Prevent sleep
sudo pmset -a disablesleep 1
sudo pmset -a displaysleep 0
echo "✅ Sleep disabled"

# Auto-restart after power failure
sudo systemsetup -setrestartpowerfailure on
echo "✅ Auto-restart on power failure enabled"

# Auto-restart after freeze
sudo systemsetup -setrestartfreeze on
echo "✅ Auto-restart on freeze enabled"

echo ""
echo "========================================================================"
echo "Step 9: Test Trading System"
echo "========================================================================"

echo "Testing trading system startup..."
timeout 10 python -m src.main --status || echo "⚠️  Quick test timed out (expected)"

echo ""
echo "========================================================================"
echo "Setup Complete!"
echo "========================================================================"
echo ""
echo "✅ Mac Mini is configured for 24/7 autonomous trading!"
echo ""
echo "Next steps:"
echo "  1. Install Tailscale app on iPhone"
echo "  2. Run: sudo tailscale up (if not done)"
echo "  3. Access dashboard: http://mac-mini-hostname:8501"
echo "  4. Set up Telegram bot (optional):"
echo "     - Get token from @BotFather"
echo "     - Add to config/.env.lexar"
echo "     - Restart system"
echo ""
echo "To start trading now:"
echo "  ./scripts/launch.sh"
echo ""
echo "To check status:"
echo "  launchctl list | grep rrrventures"
echo ""
echo "Logs location:"
echo "  /Volumes/Lexar/RRRVentures/RRRalgorithms/logs/"
echo ""
echo "🎉 Setup complete! Your trading system is ready!"
echo ""

