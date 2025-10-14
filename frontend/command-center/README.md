# 🎮 RRR Trading Command Center

A professional-grade trading dashboard for monitoring and controlling the RRRalgorithms cryptocurrency trading system.

![Command Center](https://img.shields.io/badge/Status-Production_Ready-green)
![React](https://img.shields.io/badge/React-18.2-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-5.2-blue)
![Next.js](https://img.shields.io/badge/Next.js-14.0-black)

## 🚀 Features

### Real-Time Monitoring
- **Live Price Feed** - Real-time cryptocurrency prices via WebSocket
- **Portfolio Tracking** - Live P&L, positions, and balance updates
- **Market Depth** - Order book visualization
- **Performance Metrics** - ROI, Sharpe ratio, win rate tracking

### Trading Controls
- **Manual Override** - Pause/resume automated trading
- **Position Management** - Close positions manually
- **Risk Controls** - Adjust position sizes and risk limits
- **Emergency Stop** - Instant shutdown with position closure

### System Monitoring
- **Health Dashboard** - API connections, latency, CPU/RAM usage
- **Error Logs** - Real-time error stream with filtering
- **Audit Trail** - Track all trades and system decisions
- **Alert System** - Push notifications for important events

### ML Insights
- **Prediction Confidence** - View model predictions in real-time
- **Feature Analysis** - Understand what's driving decisions
- **Model Performance** - Track accuracy over time

## 📦 Installation

```bash
# Navigate to frontend directory
cd frontend/command-center

# Install dependencies
npm install

# Create environment file
cp .env.example .env.local

# Edit .env.local with your settings
nano .env.local
```

## ⚙️ Configuration

Create `.env.local` with:

```env
# API Endpoints
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Features
NEXT_PUBLIC_ENABLE_LIVE_TRADING=false
NEXT_PUBLIC_ENABLE_VOICE_COMMANDS=true
NEXT_PUBLIC_ENABLE_MOBILE_VIEW=true

# Security
NEXT_PUBLIC_REQUIRE_AUTH=true
NEXT_PUBLIC_SESSION_TIMEOUT=3600
```

## 🎯 Quick Start

### Development Mode
```bash
npm run dev
# Open http://localhost:3000
```

### Production Build
```bash
npm run build
npm run start
```

### Docker Deployment
```bash
docker build -t rrr-command-center .
docker run -p 3000:3000 rrr-command-center
```

## 🏗️ Architecture

```
src/
├── app/                    # Next.js app directory
│   ├── page.tsx           # Main page
│   └── layout.tsx         # Root layout
├── components/            
│   ├── CommandCenter.tsx  # Main dashboard component
│   └── widgets/           # Dashboard widgets
│       ├── MarketOverview.tsx
│       ├── PortfolioSummary.tsx
│       ├── ControlPanel.tsx
│       ├── TradingChart.tsx
│       ├── SystemHealth.tsx
│       └── ...
├── contexts/              
│   ├── WebSocketContext.tsx  # WebSocket connection
│   └── TradingContext.tsx     # Trading state management
├── hooks/                 # Custom React hooks
├── lib/                   # Utility functions
└── styles/               # Global styles
```

## 📊 Dashboard Components

### 1. Market Overview
- Real-time prices for BTC, ETH, SOL, etc.
- 24h change percentages
- Volume indicators
- Mini sparkline charts

### 2. Portfolio Summary
- Total portfolio value
- Cash vs. invested breakdown
- Daily and total P&L
- Win rate and Sharpe ratio

### 3. Control Panel
- Start/Pause/Stop buttons
- Risk level slider
- Strategy toggles
- Emergency stop button

### 4. Trading Chart
- TradingView integration
- Multiple timeframes
- Technical indicators
- Drawing tools

### 5. System Health
- Connection status indicators
- API latency monitoring
- Resource usage graphs
- Error rate tracking

### 6. Position Manager
- Open positions table
- Individual P&L tracking
- Quick close buttons
- Position details modal

### 7. Alerts Panel
- Recent system alerts
- Filterable by severity
- Mark as read functionality
- Alert history

### 8. ML Insights
- Current predictions
- Confidence levels
- Feature importance
- Model metrics

## 🎨 Customization

### Theme Configuration

Edit `src/app/page.tsx`:

```typescript
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00ff88',  // Change primary color
    },
    // ... customize colors
  },
});
```

### Widget Layout

Modify grid layout in `src/components/CommandCenter.tsx`:

```typescript
<Grid container spacing={2}>
  <Grid item xs={12} md={8}>
    <MarketOverview />
  </Grid>
  <Grid item xs={12} md={4}>
    <ControlPanel />
  </Grid>
  // ... rearrange widgets
</Grid>
```

### Add New Widget

1. Create widget component in `src/components/widgets/`
2. Import in `CommandCenter.tsx`
3. Add to grid layout

## 📱 Mobile Support

The dashboard is fully responsive and works on:
- Desktop (1920x1080)
- Laptop (1366x768)
- Tablet (768x1024)
- Mobile (375x812)

### Mobile Features
- Swipeable tabs
- Touch-optimized controls
- Collapsible panels
- Portrait/landscape support

## 🔐 Security

### Authentication
- JWT token authentication
- Session management
- Auto-logout on inactivity
- Two-factor authentication support

### Data Protection
- WebSocket encryption (WSS)
- API request signing
- XSS protection
- CSRF tokens

## 🧪 Testing

```bash
# Run tests
npm test

# Run tests with coverage
npm run test:coverage

# E2E tests
npm run test:e2e
```

## 📈 Performance

### Optimization Techniques
- React.memo for expensive components
- useMemo/useCallback hooks
- Virtual scrolling for large lists
- Code splitting with dynamic imports
- Image optimization

### Performance Targets
- Initial load: <2 seconds
- Time to interactive: <3 seconds
- Lighthouse score: >90
- Bundle size: <500KB

## 🔌 API Integration

### REST Endpoints
```typescript
GET  /api/portfolio     - Get portfolio data
GET  /api/positions     - Get open positions
POST /api/trade        - Place trade
POST /api/close        - Close position
GET  /api/market/:symbol - Get market data
```

### WebSocket Events
```typescript
// Subscribe to events
socket.on('market_data', handleMarketData);
socket.on('trade_update', handleTradeUpdate);
socket.on('alert', handleAlert);

// Emit commands
socket.emit('subscribe', { symbols: ['BTC-USD'] });
socket.emit('place_trade', { symbol, side, quantity });
```

## 🛠️ Development

### Project Structure
```
frontend/command-center/
├── public/              # Static assets
├── src/                # Source code
├── tests/              # Test files
├── package.json        # Dependencies
├── tsconfig.json       # TypeScript config
├── next.config.js      # Next.js config
└── tailwind.config.js  # Tailwind CSS config
```

### Key Technologies
- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Material-UI** - Component library
- **TailwindCSS** - Utility-first CSS
- **Socket.io** - WebSocket client
- **Recharts** - Charts library
- **React Grid Layout** - Drag-and-drop grid
- **Zustand** - State management
- **Framer Motion** - Animations

## 🚢 Deployment

### Vercel (Recommended)
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

### Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

### Nginx Configuration
```nginx
server {
    listen 80;
    server_name trading.yourdomain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

## 📞 Support

- **Documentation**: [docs/COMMAND_CENTER_GUIDE.md](../../docs/COMMAND_CENTER_GUIDE.md)
- **Issues**: GitHub Issues
- **Discord**: [Join our Discord](#)

## 📄 License

Proprietary - RRR Ventures

---

**Built with ❤️ for professional crypto trading**