# Transparency Dashboard - Quick Start Guide

**Get the RRRalgorithms transparency dashboard up and running in 30 minutes**

---

## Overview

This quick start guide will help you set up the complete transparency dashboard:
- **Backend**: FastAPI with Socket.IO for real-time updates
- **Frontend**: Next.js 14 with TypeScript and Tailwind CSS
- **Database**: PostgreSQL (Supabase) with new transparency tables
- **Real-time**: WebSocket connections for live updates

**Estimated Time**: 30-60 minutes

---

## Prerequisites

```bash
# Required software
- Python 3.11+
- Node.js 20+
- PostgreSQL (or Supabase account)
- Redis (for pub/sub)
- Git
```

---

## Step 1: Database Setup (5 minutes)

### Option A: Using Supabase (Recommended)

```bash
# 1. Go to https://supabase.com and create a new project
# 2. Copy your connection string
# 3. Run the schema migration

# From project root
psql "YOUR_SUPABASE_CONNECTION_STRING" < docs/database/transparency_schema.sql

# Or using Supabase web interface:
# - Go to SQL Editor
# - Paste contents of transparency_schema.sql
# - Run query
```

### Option B: Local PostgreSQL

```bash
# 1. Create database
createdb rrralgorithms

# 2. Run migration
psql rrralgorithms < docs/database/transparency_schema.sql

# 3. Verify tables created
psql rrralgorithms -c "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
```

---

## Step 2: Backend Setup (10 minutes)

### Install Dependencies

```bash
# Create new directory for backend
mkdir -p src/api
cd src/api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn python-socketio aioredis psycopg2-binary pydantic python-dotenv slowapi
```

### Create Configuration

```bash
# Create .env file
cat > .env << EOF
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/rrralgorithms
# Or for Supabase:
# DATABASE_URL=postgresql://postgres:password@db.project.supabase.co:5432/postgres

# Redis
REDIS_URL=redis://localhost:6379

# API
API_HOST=0.0.0.0
API_PORT=8000

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:3001

# Security
SECRET_KEY=your-secret-key-here-change-in-production
EOF
```

### Copy Reference Implementation

```bash
# Copy the FastAPI structure from docs
# (Adjust paths based on your actual structure)

# See docs/api/fastapi_structure.py for reference
# Implement the following files:
# - src/api/main.py
# - src/api/websocket.py
# - src/api/routes/trades.py
# - src/api/routes/performance.py
# - src/api/routes/portfolio.py
# - src/api/models/trade.py
# - src/api/database/connection.py
```

### Start Backend Server

```bash
# From src/api directory
uvicorn main:sio_asgi_app --host 0.0.0.0 --port 8000 --reload

# You should see:
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Application startup complete.
```

### Test Backend

```bash
# Test health endpoint
curl http://localhost:8000/health

# Should return: {"status": "healthy"}

# Test API docs
# Open http://localhost:8000/docs in browser
```

---

## Step 3: Frontend Setup (10 minutes)

### Create Next.js App

```bash
# From project root
cd src

# Create Next.js app
npx create-next-app@latest frontend --typescript --tailwind --app --src-dir

# Choose:
# ✔ Would you like to use ESLint? Yes
# ✔ Would you like to use `src/` directory? Yes
# ✔ Would you like to use App Router? Yes
# ✔ Would you like to customize the default import alias? No
```

### Install Dependencies

```bash
cd frontend

# Install core dependencies
npm install @reduxjs/toolkit react-redux @tanstack/react-query socket.io-client axios

# Install UI libraries
npm install lightweight-charts recharts framer-motion lucide-react

# Install utilities
npm install clsx tailwind-merge date-fns zod

# Install shadcn/ui
npx shadcn-ui@latest init
# Choose defaults, dark mode: class

# Add shadcn components
npx shadcn-ui@latest add button card dialog dropdown-menu input select tabs toast
```

### Configure Environment

```bash
# Create .env.local
cat > .env.local << EOF
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
NEXT_PUBLIC_APP_NAME=RRRalgorithms
EOF
```

### Copy Reference Components

```bash
# See docs/frontend/nextjs_structure.md for complete structure

# Create directory structure
mkdir -p src/app/dashboard
mkdir -p src/components/Dashboard
mkdir -p src/lib
mkdir -p src/hooks
mkdir -p src/types
mkdir -p src/store

# Copy/create files from the reference
# Key files to create:
# - src/lib/api.ts
# - src/lib/websocket.tsx
# - src/lib/formatters.ts
# - src/hooks/useSocket.ts
# - src/hooks/usePortfolio.ts
# - src/components/Dashboard/DashboardLayout.tsx
# - src/components/Dashboard/PortfolioOverview.tsx
# - src/app/dashboard/page.tsx
```

### Start Frontend

```bash
# From frontend directory
npm run dev

# You should see:
# ▲ Next.js 14.2.0
# - Local:        http://localhost:3000
```

---

## Step 4: Connect Trading System (5 minutes)

### Add Event Publisher to Trading System

```python
# src/trading_engine/event_publisher.py

import redis.asyncio as redis
import json
from typing import Dict, Any


class EventPublisher:
    """Publish trading events to Redis for dashboard"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None

    async def connect(self):
        """Connect to Redis"""
        self.redis_client = await redis.from_url(self.redis_url)

    async def publish_trade(self, trade: Dict[str, Any]):
        """Publish new trade event"""
        await self.redis_client.publish(
            'trades',
            json.dumps({
                'event_type': 'new_trade',
                **trade
            })
        )

    async def publish_ai_decision(self, decision: Dict[str, Any]):
        """Publish AI decision event"""
        await self.redis_client.publish(
            'ai_decisions',
            json.dumps({
                'event_type': 'new_prediction',
                **decision
            })
        )

    async def publish_performance_update(self, metrics: Dict[str, Any]):
        """Publish performance update"""
        await self.redis_client.publish(
            'performance',
            json.dumps({
                'event_type': 'metrics_update',
                **metrics
            })
        )
```

### Integrate into Trading Engine

```python
# In your trading engine main file

from .event_publisher import EventPublisher

class TradingEngine:
    def __init__(self):
        # ... existing code ...
        self.event_publisher = EventPublisher()

    async def execute_trade(self, trade):
        # ... execute trade logic ...

        # Publish to dashboard
        await self.event_publisher.publish_trade({
            'id': trade.id,
            'timestamp': trade.timestamp.isoformat(),
            'symbol': trade.symbol,
            'side': trade.side,
            'quantity': float(trade.quantity),
            'price': float(trade.price),
            'status': trade.status,
            'strategy': trade.strategy,
            'ai_confidence': trade.ai_confidence
        })

    async def on_ai_prediction(self, prediction):
        # ... handle prediction ...

        # Publish to dashboard
        await self.event_publisher.publish_ai_decision({
            'timestamp': prediction.timestamp.isoformat(),
            'symbol': prediction.symbol,
            'model': prediction.model_name,
            'prediction': prediction.prediction,
            'features': prediction.features,
            'reasoning': prediction.reasoning
        })
```

---

## Step 5: Test End-to-End (5 minutes)

### 1. Verify Backend is Running

```bash
# Terminal 1
curl http://localhost:8000/health

# Should return: {"status": "healthy"}
```

### 2. Verify Frontend is Running

```bash
# Terminal 2
# Open http://localhost:3000/dashboard in browser
# You should see the dashboard layout
```

### 3. Test WebSocket Connection

```javascript
// In browser console (http://localhost:3000/dashboard)
// Check if WebSocket is connected
console.log('Socket connected:', window.io?.connected);
```

### 4. Simulate Trade Event

```bash
# In Python terminal or script
import redis
import json

r = redis.Redis(host='localhost', port=6379)

# Publish test trade
r.publish('trades', json.dumps({
    'event_type': 'new_trade',
    'id': 'test-123',
    'timestamp': '2025-10-25T10:30:00Z',
    'symbol': 'BTC-USD',
    'side': 'buy',
    'quantity': 0.5,
    'price': 50000,
    'status': 'filled',
    'strategy': 'test',
    'ai_confidence': 0.85
}))

# Check browser - you should see the trade appear!
```

---

## Troubleshooting

### Backend Issues

**Problem**: Cannot connect to database
```bash
# Check connection string
psql "YOUR_DATABASE_URL"

# Verify tables exist
psql "YOUR_DATABASE_URL" -c "\dt"
```

**Problem**: Redis connection error
```bash
# Check Redis is running
redis-cli ping
# Should return: PONG

# Start Redis if not running
redis-server
```

### Frontend Issues

**Problem**: WebSocket not connecting
```bash
# Check CORS settings in backend
# Verify CORS_ORIGINS includes http://localhost:3000

# Check WebSocket URL in .env.local
echo $NEXT_PUBLIC_WS_URL
```

**Problem**: API calls failing
```bash
# Check API URL
echo $NEXT_PUBLIC_API_URL

# Test API directly
curl http://localhost:8000/api/v1/portfolio
```

### Trading System Integration

**Problem**: Events not appearing in dashboard
```bash
# Check Redis connection
redis-cli
> SUBSCRIBE trades
# Should show subscription

# In another terminal, publish test event
redis-cli PUBLISH trades '{"test": "data"}'
# Should appear in subscriber
```

---

## Next Steps

### 1. Add More Components

```bash
# Implement remaining pages
# - Live Feed: /app/live-feed/page.tsx
# - Performance: /app/performance/page.tsx
# - AI Insights: /app/ai-insights/page.tsx
# - Backtests: /app/backtests/page.tsx
```

### 2. Add Authentication

```bash
# Install NextAuth.js
npm install next-auth

# Set up authentication
# See: https://next-auth.js.org/getting-started/example
```

### 3. Deploy to Production

```bash
# Backend: Deploy to Railway/Fly.io
# Frontend: Deploy to Vercel

# See deployment section in main design doc
```

### 4. Add Monitoring

```bash
# Install Sentry
npm install @sentry/nextjs

# Configure error tracking
# See: https://docs.sentry.io/platforms/javascript/guides/nextjs/
```

---

## Development Workflow

### Running in Development

```bash
# Terminal 1: Backend
cd src/api
source venv/bin/activate
uvicorn main:sio_asgi_app --reload

# Terminal 2: Frontend
cd src/frontend
npm run dev

# Terminal 3: Redis
redis-server

# Terminal 4: Trading System
cd /home/user/RRRalgorithms
python src/main_unified.py --mode paper
```

### Making Changes

```bash
# Backend changes
# 1. Edit files in src/api/
# 2. Server auto-reloads (--reload flag)
# 3. Test at http://localhost:8000/docs

# Frontend changes
# 1. Edit files in src/frontend/
# 2. Next.js hot-reloads automatically
# 3. View at http://localhost:3000/dashboard
```

---

## Production Checklist

Before deploying to production:

- [ ] Change SECRET_KEY in .env
- [ ] Use production database (not local)
- [ ] Enable SSL/TLS for all connections
- [ ] Set up rate limiting
- [ ] Configure CORS properly
- [ ] Add authentication
- [ ] Set up monitoring (Sentry, etc.)
- [ ] Configure CDN for static assets
- [ ] Enable compression (gzip)
- [ ] Set up backup strategy
- [ ] Load test WebSocket connections
- [ ] Review security best practices
- [ ] Set up CI/CD pipeline
- [ ] Configure logging
- [ ] Set up alerts

---

## Resources

### Documentation
- Main Design: `/docs/TRANSPARENCY_DASHBOARD_DESIGN.md`
- FastAPI Reference: `/docs/api/fastapi_structure.py`
- Frontend Structure: `/docs/frontend/nextjs_structure.md`
- Database Schema: `/docs/database/transparency_schema.sql`

### External Resources
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Next.js Docs](https://nextjs.org/docs)
- [Socket.IO Docs](https://socket.io/docs/v4/)
- [shadcn/ui](https://ui.shadcn.com/)
- [TailwindCSS](https://tailwindcss.com/)

### Support
- Issues: GitHub Issues
- Docs: `/docs` directory
- Examples: `/examples` directory

---

## Success Criteria

You'll know the dashboard is working when:

1. ✅ Backend health check returns 200
2. ✅ Frontend loads without errors
3. ✅ WebSocket connects (check browser console)
4. ✅ Test trade appears in real-time
5. ✅ Portfolio data displays correctly
6. ✅ Charts render properly
7. ✅ No console errors

---

**Congratulations!** Your transparency dashboard is now running. Start trading and watch the real-time updates flow in!

---

**Quick Start Version**: 1.0.0
**Last Updated**: 2025-10-25
**Status**: Ready for Development
