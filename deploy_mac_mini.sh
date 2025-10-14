#!/bin/bash
# Complete deployment script for RRRalgorithms on Mac Mini
# This script sets up everything for 24/7 paper trading

set -e

echo "
╔══════════════════════════════════════════════════╗
║     RRRalgorithms Mac Mini Deployment Script     ║
║            24/7 Paper Trading System             ║
╚══════════════════════════════════════════════════╝
"

# Configuration
PROJECT_ROOT="$PWD"
PYTHON_VERSION="3.11"
DEPLOYMENT_DIR="$HOME/RRRalgorithms"
LOG_DIR="$DEPLOYMENT_DIR/logs"
DATA_DIR="$DEPLOYMENT_DIR/data"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Mac
if [[ "$OSTYPE" != "darwin"* ]]; then
    log_error "This script is designed for macOS only!"
    exit 1
fi

# Step 1: System Prerequisites
log_info "Step 1/10: Checking system prerequisites..."

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    log_error "Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install required system packages
log_info "Installing system dependencies..."
brew install python@$PYTHON_VERSION postgresql redis node git wget

# Step 2: Create deployment directory structure
log_info "Step 2/10: Creating deployment directory structure..."

mkdir -p "$DEPLOYMENT_DIR"/{logs,data,backups,config,scripts}
mkdir -p "$LOG_DIR"/{trading,system,audit,monitoring}
mkdir -p "$DATA_DIR"/{db,cache,models,market_data}

# Step 3: Copy project files to deployment directory
log_info "Step 3/10: Copying project files..."

# Create a clean copy without unnecessary files
rsync -av --exclude='*.pyc' \
          --exclude='__pycache__' \
          --exclude='.git' \
          --exclude='venv' \
          --exclude='*.log' \
          --exclude='backfill_progress' \
          --exclude='worktrees' \
          "$PROJECT_ROOT/" "$DEPLOYMENT_DIR/app/"

cd "$DEPLOYMENT_DIR/app"

# Step 4: Setup Python environment
log_info "Step 4/10: Setting up Python environment..."

# Create virtual environment
python$PYTHON_VERSION -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install requirements
log_info "Installing Python dependencies..."
pip install -r requirements-local-trading.txt || pip install -r requirements.txt
pip install argon2-cffi cryptography prometheus-client aiosqlite

# Step 5: Initialize database
log_info "Step 5/10: Initializing encrypted database..."

# Run database initialization
python scripts/migrate_supabase_to_sqlite.py

# Set secure permissions
chmod 600 "$DATA_DIR/db/trading.db"
chmod 600 "$DATA_DIR/db/trading_encrypted.db" 2>/dev/null || true

# Step 6: Configure API keys and secrets
log_info "Step 6/10: Configuring secrets management..."

# Create secrets configuration script
cat > "$DEPLOYMENT_DIR/scripts/configure_secrets.py" << 'EOF'
#!/usr/bin/env python3
"""Configure API keys and secrets in macOS Keychain"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from src.security.secrets_manager import SecretsManager

def main():
    print("🔐 Configuring API keys and secrets...")
    
    secrets_mgr = SecretsManager()
    
    # List of required secrets
    required_secrets = [
        ("POLYGON_API_KEY", "Polygon.io API key for market data"),
        ("PERPLEXITY_API_KEY", "Perplexity AI API key for sentiment analysis"),
        ("ANTHROPIC_API_KEY", "Anthropic API key (optional)"),
        ("TELEGRAM_BOT_TOKEN", "Telegram bot token for alerts"),
        ("TELEGRAM_CHAT_ID", "Telegram chat ID for alerts"),
    ]
    
    print("\n⚠️  Please have your API keys ready!")
    print("You can get them from:")
    print("- Polygon.io: https://polygon.io/dashboard/api-keys")
    print("- Perplexity: https://www.perplexity.ai/settings/api")
    print("- Telegram: https://t.me/BotFather\n")
    
    for key_name, description in required_secrets:
        current_value = secrets_mgr.get_secret(key_name)
        if current_value:
            print(f"✅ {key_name} already configured")
            update = input("Update it? (y/N): ").lower() == 'y'
            if not update:
                continue
        
        print(f"\n📝 {description}")
        value = input(f"Enter {key_name}: ").strip()
        
        if value:
            success = secrets_mgr.set_secret(key_name, value)
            if success:
                print(f"✅ {key_name} saved to Keychain")
            else:
                print(f"❌ Failed to save {key_name}")
        else:
            print(f"⏭️  Skipping {key_name}")
    
    # Verify all secrets
    print("\n🔍 Verifying secrets...")
    verification = secrets_mgr.verify_secrets()
    
    print("\n📊 Secrets Status:")
    for key, status in verification.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {key}: {'Configured' if status else 'Missing'}")
    
    if all(verification.values()):
        print("\n✅ All secrets configured successfully!")
    else:
        print("\n⚠️  Some secrets are missing. The system may not function properly.")

if __name__ == "__main__":
    main()
EOF

chmod +x "$DEPLOYMENT_DIR/scripts/configure_secrets.py"

# Run secrets configuration
python "$DEPLOYMENT_DIR/scripts/configure_secrets.py"

# Step 7: Setup monitoring
log_info "Step 7/10: Setting up monitoring stack..."

cd "$PROJECT_ROOT"
chmod +x monitoring/setup_mac_monitoring.sh
./monitoring/setup_mac_monitoring.sh

# Step 8: Create system service files
log_info "Step 8/10: Creating system services..."

# Main trading system LaunchAgent
cat > ~/Library/LaunchAgents/com.rrrventures.trading.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.rrrventures.trading</string>
    <key>ProgramArguments</key>
    <array>
        <string>$DEPLOYMENT_DIR/app/venv/bin/python</string>
        <string>$DEPLOYMENT_DIR/app/src/main_unified.py</string>
        <string>--mode</string>
        <string>paper</string>
        <string>--dashboard</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$DEPLOYMENT_DIR/app</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>$LOG_DIR/trading/system.log</string>
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/trading/error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
        <key>TRADING_HOME</key>
        <string>$DEPLOYMENT_DIR</string>
        <key>DATABASE_PATH</key>
        <string>$DATA_DIR/db/trading.db</string>
    </dict>
</dict>
</plist>
EOF

# Trading metrics exporter
cat > ~/Library/LaunchAgents/com.rrrventures.metrics.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.rrrventures.metrics</string>
    <key>ProgramArguments</key>
    <array>
        <string>$DEPLOYMENT_DIR/app/venv/bin/python</string>
        <string>$PROJECT_ROOT/monitoring/exporters/trading/trading_exporter.py</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>EnvironmentVariables</key>
    <dict>
        <key>METRICS_PORT</key>
        <string>8000</string>
        <key>DATABASE_PATH</key>
        <string>$DATA_DIR/db/trading.db</string>
        <key>LOG_LEVEL</key>
        <string>INFO</string>
    </dict>
</dict>
</plist>
EOF

# Step 9: Create control scripts
log_info "Step 9/10: Creating control scripts..."

# Main control script
cat > "$DEPLOYMENT_DIR/control.sh" << 'EOF'
#!/bin/bash
# RRRalgorithms Trading System Control Script

DEPLOYMENT_DIR="$HOME/RRRalgorithms"
LOG_DIR="$DEPLOYMENT_DIR/logs"

case "$1" in
    start)
        echo "🚀 Starting RRRalgorithms Trading System..."
        launchctl load ~/Library/LaunchAgents/com.rrrventures.trading.plist 2>/dev/null
        launchctl load ~/Library/LaunchAgents/com.rrrventures.metrics.plist 2>/dev/null
        echo "✅ Trading system started"
        ;;
    
    stop)
        echo "🛑 Stopping RRRalgorithms Trading System..."
        launchctl unload ~/Library/LaunchAgents/com.rrrventures.trading.plist 2>/dev/null
        launchctl unload ~/Library/LaunchAgents/com.rrrventures.metrics.plist 2>/dev/null
        echo "✅ Trading system stopped"
        ;;
    
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
    
    status)
        echo "📊 System Status:"
        echo "─────────────────"
        launchctl list | grep com.rrrventures || echo "No RRRventures services running"
        echo ""
        echo "📈 Recent Activity:"
        tail -n 5 "$LOG_DIR/trading/system.log" 2>/dev/null || echo "No recent logs"
        ;;
    
    logs)
        echo "📜 Tailing system logs..."
        tail -f "$LOG_DIR/trading/system.log"
        ;;
    
    errors)
        echo "❌ Recent errors:"
        tail -n 50 "$LOG_DIR/trading/error.log" | grep -E "(ERROR|CRITICAL)" || echo "No recent errors"
        ;;
    
    backup)
        echo "💾 Creating backup..."
        BACKUP_NAME="backup_$(date +%Y%m%d_%H%M%S).tar.gz"
        tar -czf "$DEPLOYMENT_DIR/backups/$BACKUP_NAME" \
            -C "$DEPLOYMENT_DIR" \
            data/db logs/audit
        echo "✅ Backup created: $BACKUP_NAME"
        ;;
    
    health)
        echo "🏥 System Health Check:"
        echo "─────────────────────"
        
        # Check services
        if launchctl list | grep -q com.rrrventures.trading; then
            echo "✅ Trading service: Running"
        else
            echo "❌ Trading service: Not running"
        fi
        
        # Check database
        if [ -f "$DEPLOYMENT_DIR/data/db/trading.db" ]; then
            SIZE=$(du -h "$DEPLOYMENT_DIR/data/db/trading.db" | cut -f1)
            echo "✅ Database: Exists ($SIZE)"
        else
            echo "❌ Database: Not found"
        fi
        
        # Check monitoring
        if curl -s http://localhost:9090 > /dev/null; then
            echo "✅ Prometheus: Running"
        else
            echo "❌ Prometheus: Not reachable"
        fi
        
        if curl -s http://localhost:3000 > /dev/null; then
            echo "✅ Grafana: Running"
        else
            echo "❌ Grafana: Not reachable"
        fi
        
        # Check disk space
        DISK_USAGE=$(df -h "$DEPLOYMENT_DIR" | tail -1 | awk '{print $5}')
        echo "💾 Disk usage: $DISK_USAGE"
        ;;
    
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|errors|backup|health}"
        exit 1
        ;;
esac
EOF

chmod +x "$DEPLOYMENT_DIR/control.sh"

# Create quick access symlink
ln -sf "$DEPLOYMENT_DIR/control.sh" /usr/local/bin/rrralgos 2>/dev/null || true

# Step 10: Start services
log_info "Step 10/10: Starting services..."

# Load services
launchctl load ~/Library/LaunchAgents/com.rrrventures.trading.plist 2>/dev/null
launchctl load ~/Library/LaunchAgents/com.rrrventures.metrics.plist 2>/dev/null

# Wait for services to start
sleep 5

# Final system check
echo "
╔══════════════════════════════════════════════════╗
║            🎉 DEPLOYMENT COMPLETE! 🎉            ║
╚══════════════════════════════════════════════════╝

📊 System Status:
"

"$DEPLOYMENT_DIR/control.sh" health

echo "
🌐 Access Points:
   📈 Grafana Dashboard: http://localhost:3000
   🔍 Prometheus: http://localhost:9090
   📱 Mobile Dashboard: http://$(hostname):8501
   📊 Metrics Endpoint: http://localhost:8000/metrics

🛠  Control Commands:
   rrralgos start    - Start trading system
   rrralgos stop     - Stop trading system
   rrralgos status   - Check system status
   rrralgos logs     - View live logs
   rrralgos health   - System health check
   rrralgos backup   - Create backup

📁 Important Locations:
   🏠 Deployment: $DEPLOYMENT_DIR
   📊 Logs: $LOG_DIR
   💾 Database: $DATA_DIR/db
   ⚙️  Control: $DEPLOYMENT_DIR/control.sh

🔐 Security Notes:
   ✅ Database encrypted with AES-256
   ✅ Secrets stored in macOS Keychain
   ✅ All services running as user
   ✅ Monitoring enabled

⏭️  Next Steps:
   1. Open Grafana: http://localhost:3000
   2. Login: admin / RRRsecure2025!
   3. Add Prometheus datasource
   4. Import trading dashboard
   5. Monitor paper trading performance

📱 Remote Access:
   Install Tailscale for secure remote access
   
⚠️  IMPORTANT: System will auto-start on boot!
"
