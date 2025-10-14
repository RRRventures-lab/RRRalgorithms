# 🎮 Command Center Setup Guide

## Quick Start (5 minutes)

### 1. Install Frontend Dependencies
```bash
cd /Volumes/Lexar/RRRVentures/RRRalgorithms/frontend/command-center
npm install
```

### 2. Configure Environment
```bash
# Create environment file
cat > .env.local << EOF
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
NEXT_PUBLIC_POLYGON_API_KEY=your_polygon_key_here
EOF
```

### 3. Start the Command Center
```bash
# Development mode
npm run dev

# Open in browser
open http://localhost:3000
```

---

## 🖥️ What You'll See

### Main Dashboard
- **Market Overview**: Live BTC, ETH, SOL prices with sparklines
- **Portfolio Summary**: Total value, P&L, performance metrics
- **Control Panel**: Start/stop trading, risk controls
- **System Health**: Connection status, API latency
- **Alerts Panel**: Recent notifications
- **Recent Trades**: Trade history

### Trading Tab
- Full-screen trading charts
- Order book depth
- ML model predictions
- Technical indicators

### Portfolio Tab
- Detailed position breakdown
- Performance analytics
- Historical equity curve
- Risk metrics

### Risk Tab
- Risk exposure monitoring
- Drawdown tracking
- Position sizing calculator
- Stop-loss manager

---

## 🔌 Connect to Backend

### Start Python Backend First
```bash
# Terminal 1: Start trading system
cd /Volumes/Lexar/RRRVentures/RRRalgorithms
python scripts/start_live_feed.py
```

### Start WebSocket Server
```bash
# Terminal 2: Start WebSocket server
python src/api/websocket_server.py
```

### Start Command Center
```bash
# Terminal 3: Start frontend
cd frontend/command-center
npm run dev
```

---

## 📱 Access from Mobile

### Local Network (Same WiFi)
1. Find your Mac's IP address:
   ```bash
   ipconfig getifaddr en0
   # Example: 192.168.1.100
   ```

2. On your iPhone/iPad:
   - Open Safari
   - Navigate to: `http://192.168.1.100:3000`

### Remote Access (Tailscale)
1. Install Tailscale on Mac and iPhone
2. Connect both devices
3. Access: `http://your-mac-name:3000`

---

## 🎨 Customization

### Change Color Theme
Edit `frontend/command-center/src/app/page.tsx`:

```typescript
const darkTheme = createTheme({
  palette: {
    primary: {
      main: '#00ff88',  // Green
      // Change to: '#00aaff' for blue
      // Or: '#ff6b6b' for red
    },
  },
});
```

### Rearrange Widgets
Edit `frontend/command-center/src/components/CommandCenter.tsx`:

Move components around in the grid layout to customize your dashboard.

---

## 🚀 Production Deployment

### Build for Production
```bash
cd frontend/command-center
npm run build
npm run start
```

### Deploy to Vercel (Free Hosting)
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

### Docker Deployment
```bash
# Build Docker image
docker build -t rrr-command-center .

# Run container
docker run -d -p 3000:3000 \
  --name command-center \
  --restart unless-stopped \
  rrr-command-center
```

---

## 🎯 Features Overview

### Real-Time Updates
- ✅ Live price feeds via WebSocket
- ✅ Portfolio updates every second
- ✅ Trade notifications
- ✅ System alerts

### Trading Controls
- ✅ Start/Pause/Stop buttons
- ✅ Emergency stop (closes all positions)
- ✅ Risk level adjustment
- ✅ Strategy selection

### Monitoring
- ✅ System health indicators
- ✅ API connection status
- ✅ Performance metrics
- ✅ Error logging

### Risk Management
- ✅ Position size limits
- ✅ Max drawdown alerts
- ✅ Daily loss limits
- ✅ Risk/reward calculator

---

## 🔧 Troubleshooting

### "Cannot connect to WebSocket"
- Ensure backend is running
- Check NEXT_PUBLIC_WS_URL in .env.local
- Verify no firewall blocking

### "No market data showing"
- Confirm Polygon API key is set
- Check backend logs for errors
- Verify internet connection

### "Page not loading"
- Clear browser cache
- Check console for errors (F12)
- Restart npm run dev

---

## 🎬 Demo Mode

To see the dashboard with mock data (no backend needed):

```bash
# Set demo mode
echo "NEXT_PUBLIC_DEMO_MODE=true" >> .env.local

# Restart
npm run dev
```

---

## 📸 Screenshots

### Dashboard View
- Market overview with live prices
- Portfolio performance metrics
- Quick trading controls
- System health monitoring

### Trading View
- Professional trading charts
- Technical indicators
- Order placement
- ML predictions

### Mobile View
- Responsive design
- Touch-optimized controls
- Swipeable panels
- Full functionality

---

## 🆘 Need Help?

1. Check the console for errors (F12 in browser)
2. Review backend logs
3. Ensure all services are running
4. Check network connectivity

---

**Your professional trading command center is ready! 🚀**

Access it at: http://localhost:3000