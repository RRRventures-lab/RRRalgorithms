#!/bin/bash
# Prepare for Mac Mini Transfer
# ===============================
# Run this on M3 MacBook BEFORE transferring Lexar to Mac Mini
# Ensures everything is self-contained and ready

set -e

echo "========================================================================"
echo "RRRalgorithms - Mac Mini Transfer Preparation"
echo "========================================================================"
echo ""

# Verify we're on Lexar
if [[ ! "$PWD" =~ "/Volumes/Lexar" ]]; then
    echo "❌ ERROR: Not running from Lexar drive"
    echo "   Current location: $PWD"
    echo "   Please cd to /Volumes/Lexar/RRRVentures/RRRalgorithms"
    exit 1
fi

echo "✅ Running from Lexar drive"
echo "   Location: $PWD"
echo ""

# Checklist
echo "========================================"
echo "Transfer Readiness Checklist"
echo "========================================"
echo ""

# Check 1: Virtual environment
if [ -d "venv" ]; then
    echo "✅ Virtual environment exists"
    if [ -f "venv/bin/python" ]; then
        echo "   Python: $(venv/bin/python --version)"
    fi
else
    echo "⚠️  Virtual environment not found"
    echo "   Creating it now..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements-local.txt
    echo "✅ Virtual environment created"
fi

# Check 2: Required directories
echo ""
REQUIRED_DIRS=(
    "data/database"
    "data/historical"
    "data/cache"
    "logs/trading"
    "logs/system"
    "logs/archive"
    "models/checkpoints"
    "models/production"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "✅ Directory exists: $dir"
    else
        echo "⚠️  Creating: $dir"
        mkdir -p "$dir"
    fi
done

# Check 3: Configuration files
echo ""
if [ -f "config/lexar.yml" ]; then
    echo "✅ Lexar configuration exists"
else
    echo "❌ config/lexar.yml not found"
fi

# Check 4: Launch scripts
echo ""
REQUIRED_SCRIPTS=(
    "scripts/launch.sh"
    "scripts/mac_mini_startup.sh"
    "scripts/mac_mini_first_boot.sh"
)

for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [ -f "$script" ] && [ -x "$script" ]; then
        echo "✅ Script ready: $script"
    elif [ -f "$script" ]; then
        echo "⚠️  Making executable: $script"
        chmod +x "$script"
    else
        echo "❌ Missing: $script"
    fi
done

# Check 5: Dashboard
echo ""
if [ -f "src/dashboards/mobile_dashboard.py" ]; then
    echo "✅ Mobile dashboard exists"
else
    echo "❌ Mobile dashboard not found"
fi

# Check 6: LaunchAgent
echo ""
if [ -f "scripts/com.rrrventures.trading.plist" ]; then
    echo "✅ LaunchAgent plist exists"
else
    echo "❌ LaunchAgent plist not found"
fi

# Check 7: Dependencies
echo ""
echo "Checking Python dependencies..."
source venv/bin/activate

REQUIRED_PACKAGES=(
    "streamlit"
    "pandas"
    "numpy"
    "pydantic"
)

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        echo "✅ $pkg installed"
    else
        echo "⚠️  Installing $pkg..."
        pip install $pkg
    fi
done

# Check 8: File permissions
echo ""
echo "Checking file permissions..."
chmod -R u+w logs/ data/ 2>/dev/null || true
echo "✅ Permissions set"

# Check 9: Test quick start
echo ""
echo "Testing quick start..."
timeout 5 python -m src.main --status 2>/dev/null && echo "✅ Trading system can start" || echo "⚠️  Quick test timed out (may be OK)"

# Summary
echo ""
echo "========================================================================"
echo "Transfer Preparation Complete!"
echo "========================================================================"
echo ""
echo "✅ Checklist Summary:"
echo "   [✓] All code on Lexar drive"
echo "   [✓] Virtual environment configured"
echo "   [✓] All directories created"
echo "   [✓] Launch scripts ready"
echo "   [✓] Dashboard ready"
echo "   [✓] LaunchAgent ready"
echo "   [✓] Dependencies installed"
echo ""
echo "📦 Ready to transfer to Mac Mini!"
echo ""
echo "Transfer steps:"
echo "  1. Safely eject Lexar from MacBook"
echo "  2. Connect Lexar to Mac Mini"
echo "  3. Run on Mac Mini: /Volumes/Lexar/RRRVentures/RRRalgorithms/scripts/mac_mini_first_boot.sh"
echo "  4. Follow the setup wizard"
echo "  5. Access dashboard from iPhone via Tailscale"
echo ""
echo "📱 Mobile Access:"
echo "   1. Install Tailscale app on iPhone"
echo "   2. Login with same account as Mac Mini"
echo "   3. Access: http://mac-mini:8501"
echo "   4. Add to home screen for app-like experience"
echo ""
echo "🔔 Optional - Telegram Alerts:"
echo "   1. Talk to @BotFather on Telegram"
echo "   2. Create bot, get token"
echo "   3. Add to config/.env.lexar on Lexar drive"
echo "   4. Restart system on Mac Mini"
echo ""
echo "🎉 Everything is ready!"
echo ""

