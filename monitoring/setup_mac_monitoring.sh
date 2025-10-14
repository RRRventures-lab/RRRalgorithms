#!/bin/bash
# Setup monitoring stack natively on macOS for RRRalgorithms

set -e

echo "🚀 Setting up native macOS monitoring stack..."

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Please install it first:"
    echo "/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

# Install monitoring tools via Homebrew
echo "📦 Installing monitoring tools..."

# Install Prometheus
if ! brew list prometheus &> /dev/null; then
    echo "Installing Prometheus..."
    brew install prometheus
fi

# Install Grafana
if ! brew list grafana &> /dev/null; then
    echo "Installing Grafana..."
    brew install grafana
fi

# Install Alertmanager
if ! brew list alertmanager &> /dev/null; then
    echo "Installing Alertmanager..."
    brew install alertmanager
fi

# Create directories
echo "📁 Creating monitoring directories..."
mkdir -p ~/Library/RRRalgorithms/monitoring/{prometheus,grafana,alertmanager}
mkdir -p ~/Library/RRRalgorithms/monitoring/data/{prometheus,grafana}
mkdir -p ~/Library/RRRalgorithms/monitoring/logs

# Copy configurations
echo "📋 Copying configurations..."
MONITORING_DIR="$PWD/monitoring"
CONFIG_DIR="$HOME/Library/RRRalgorithms/monitoring"

# Prometheus config
cp "$MONITORING_DIR/prometheus/prometheus.yml" "$CONFIG_DIR/prometheus/"
cp -r "$MONITORING_DIR/prometheus/rules" "$CONFIG_DIR/prometheus/"

# Grafana dashboards
mkdir -p "$CONFIG_DIR/grafana/dashboards"
cp "$MONITORING_DIR/grafana/dashboards/"*.json "$CONFIG_DIR/grafana/dashboards/"

# Create Grafana provisioning config
cat > "$CONFIG_DIR/grafana/provisioning.yaml" << 'EOF'
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /Users/$USER/Library/RRRalgorithms/monitoring/grafana/dashboards
EOF

# Create Alertmanager config
cat > "$CONFIG_DIR/alertmanager/alertmanager.yml" << 'EOF'
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'telegram-notifications'

receivers:
- name: 'telegram-notifications'
  webhook_configs:
  - url: 'http://localhost:8089/alerts'
    send_resolved: true
EOF

# Create LaunchAgents for automatic startup
echo "🚀 Creating LaunchAgents..."

# Prometheus LaunchAgent
cat > ~/Library/LaunchAgents/com.rrrventures.prometheus.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.rrrventures.prometheus</string>
    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/bin/prometheus</string>
        <string>--config.file=$CONFIG_DIR/prometheus/prometheus.yml</string>
        <string>--storage.tsdb.path=$CONFIG_DIR/data/prometheus</string>
        <string>--web.enable-lifecycle</string>
        <string>--storage.tsdb.retention.time=30d</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$CONFIG_DIR/logs/prometheus.log</string>
    <key>StandardErrorPath</key>
    <string>$CONFIG_DIR/logs/prometheus.error.log</string>
</dict>
</plist>
EOF

# Grafana LaunchAgent
cat > ~/Library/LaunchAgents/com.rrrventures.grafana.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.rrrventures.grafana</string>
    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/bin/grafana-server</string>
        <string>--homepath=/opt/homebrew/share/grafana</string>
        <string>--config=/opt/homebrew/etc/grafana/grafana.ini</string>
        <string>--packaging=brew</string>
        <string>cfg:default.paths.logs=$CONFIG_DIR/logs</string>
        <string>cfg:default.paths.data=$CONFIG_DIR/data/grafana</string>
        <string>cfg:default.paths.plugins=/opt/homebrew/var/lib/grafana/plugins</string>
        <string>cfg:default.paths.provisioning=$CONFIG_DIR/grafana</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$CONFIG_DIR/logs/grafana.log</string>
    <key>StandardErrorPath</key>
    <string>$CONFIG_DIR/logs/grafana.error.log</string>
</dict>
</plist>
EOF

# Alertmanager LaunchAgent
cat > ~/Library/LaunchAgents/com.rrrventures.alertmanager.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.rrrventures.alertmanager</string>
    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/bin/alertmanager</string>
        <string>--config.file=$CONFIG_DIR/alertmanager/alertmanager.yml</string>
        <string>--storage.path=$CONFIG_DIR/data/alertmanager</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$CONFIG_DIR/logs/alertmanager.log</string>
    <key>StandardErrorPath</key>
    <string>$CONFIG_DIR/logs/alertmanager.error.log</string>
</dict>
</plist>
EOF

# Load LaunchAgents
echo "🔄 Loading LaunchAgents..."
launchctl load ~/Library/LaunchAgents/com.rrrventures.prometheus.plist
launchctl load ~/Library/LaunchAgents/com.rrrventures.grafana.plist
launchctl load ~/Library/LaunchAgents/com.rrrventures.alertmanager.plist

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 5

# Check if services are running
echo "✅ Checking service status..."
if launchctl list | grep -q com.rrrventures.prometheus; then
    echo "✅ Prometheus is running at http://localhost:9090"
else
    echo "❌ Prometheus failed to start"
fi

if launchctl list | grep -q com.rrrventures.grafana; then
    echo "✅ Grafana is running at http://localhost:3000"
    echo "   Default login: admin / RRRsecure2025!"
else
    echo "❌ Grafana failed to start"
fi

if launchctl list | grep -q com.rrrventures.alertmanager; then
    echo "✅ Alertmanager is running at http://localhost:9093"
else
    echo "❌ Alertmanager failed to start"
fi

# Create monitoring control script
cat > ~/Library/RRRalgorithms/monitoring/control.sh << 'EOF'
#!/bin/bash
# Control script for RRRalgorithms monitoring

case "$1" in
    start)
        echo "Starting monitoring services..."
        launchctl load ~/Library/LaunchAgents/com.rrrventures.prometheus.plist 2>/dev/null
        launchctl load ~/Library/LaunchAgents/com.rrrventures.grafana.plist 2>/dev/null
        launchctl load ~/Library/LaunchAgents/com.rrrventures.alertmanager.plist 2>/dev/null
        ;;
    stop)
        echo "Stopping monitoring services..."
        launchctl unload ~/Library/LaunchAgents/com.rrrventures.prometheus.plist 2>/dev/null
        launchctl unload ~/Library/LaunchAgents/com.rrrventures.grafana.plist 2>/dev/null
        launchctl unload ~/Library/LaunchAgents/com.rrrventures.alertmanager.plist 2>/dev/null
        ;;
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
    status)
        echo "Monitoring service status:"
        launchctl list | grep com.rrrventures
        ;;
    logs)
        tail -f ~/Library/RRRalgorithms/monitoring/logs/*.log
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
EOF

chmod +x ~/Library/RRRalgorithms/monitoring/control.sh

echo "
✅ Monitoring stack setup complete!

📊 Access URLs:
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/RRRsecure2025!)
   - Alertmanager: http://localhost:9093

🛠  Control commands:
   ~/Library/RRRalgorithms/monitoring/control.sh start|stop|restart|status|logs

📁 Configuration files:
   ~/Library/RRRalgorithms/monitoring/

🚀 To add the Trading Exporter:
   1. cd monitoring/exporters/trading
   2. pip install prometheus-client aiosqlite
   3. python trading_exporter.py

📱 Next: Configure Grafana data source:
   1. Open http://localhost:3000
   2. Add Prometheus data source: http://localhost:9090
   3. Import dashboard from: ~/Library/RRRalgorithms/monitoring/grafana/dashboards/
"
