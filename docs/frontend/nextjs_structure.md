# Next.js Frontend Structure for Transparency Dashboard

**Complete component architecture and implementation guide**

---

## Directory Structure

```
src/frontend/
├── app/                          # Next.js 14 App Router
│   ├── layout.tsx               # Root layout
│   ├── page.tsx                 # Home page (redirects to /dashboard)
│   ├── dashboard/
│   │   └── page.tsx            # Main dashboard
│   ├── live-feed/
│   │   └── page.tsx            # Live trading feed
│   ├── performance/
│   │   └── page.tsx            # Performance analytics
│   ├── ai-insights/
│   │   └── page.tsx            # AI decision insights
│   ├── backtests/
│   │   ├── page.tsx            # Backtest list
│   │   └── [id]/
│   │       └── page.tsx        # Backtest details
│   └── api/                     # API routes (optional)
│       └── auth/
│           └── [...nextauth]/
│               └── route.ts
│
├── components/                   # Reusable components
│   ├── Dashboard/
│   │   ├── DashboardLayout.tsx
│   │   ├── PortfolioOverview.tsx
│   │   ├── LiveTradeFeed.tsx
│   │   ├── PortfolioAllocation.tsx
│   │   ├── EquityCurve.tsx
│   │   ├── TopPerformers.tsx
│   │   └── RiskMetrics.tsx
│   │
│   ├── LiveFeed/
│   │   ├── LiveFeedLayout.tsx
│   │   ├── TradeCard.tsx
│   │   ├── AISignalCard.tsx
│   │   ├── PositionClosedCard.tsx
│   │   ├── FeedFilters.tsx
│   │   └── FeedExport.tsx
│   │
│   ├── Performance/
│   │   ├── PerformanceLayout.tsx
│   │   ├── KeyMetrics.tsx
│   │   ├── EquityCurve.tsx
│   │   ├── DrawdownAnalysis.tsx
│   │   ├── ReturnsDistribution.tsx
│   │   ├── RiskAdjustedMetrics.tsx
│   │   ├── TradingStatistics.tsx
│   │   └── MonthlyPerformance.tsx
│   │
│   ├── AIInsights/
│   │   ├── AIInsightsLayout.tsx
│   │   ├── ModelPerformance.tsx
│   │   ├── PredictionCard.tsx
│   │   ├── FeatureImportance.tsx
│   │   ├── ConfidenceCalibration.tsx
│   │   ├── AccuracyByTimeframe.tsx
│   │   └── Explainability.tsx
│   │
│   ├── Backtests/
│   │   ├── BacktestLayout.tsx
│   │   ├── StrategyComparison.tsx
│   │   ├── BacktestDetails.tsx
│   │   ├── BacktestEquityCurve.tsx
│   │   ├── MonthlyReturns.tsx
│   │   ├── TradeDistribution.tsx
│   │   └── BacktestExport.tsx
│   │
│   ├── Charts/
│   │   ├── CandlestickChart.tsx
│   │   ├── LineChart.tsx
│   │   ├── PieChart.tsx
│   │   ├── Heatmap.tsx
│   │   └── Histogram.tsx
│   │
│   └── UI/                       # shadcn/ui components
│       ├── button.tsx
│       ├── card.tsx
│       ├── dialog.tsx
│       ├── dropdown-menu.tsx
│       ├── input.tsx
│       ├── select.tsx
│       ├── tabs.tsx
│       └── toast.tsx
│
├── lib/                          # Utilities
│   ├── api.ts                   # API client
│   ├── websocket.ts             # WebSocket client
│   ├── formatters.ts            # Number/date formatters
│   ├── constants.ts             # Constants
│   └── utils.ts                 # Helper functions
│
├── hooks/                        # Custom React hooks
│   ├── useSocket.ts             # WebSocket hook
│   ├── usePortfolio.ts          # Portfolio data hook
│   ├── useTrades.ts             # Trades data hook
│   ├── usePerformance.ts        # Performance data hook
│   └── useAIInsights.ts         # AI insights hook
│
├── store/                        # Redux store
│   ├── store.ts                 # Configure store
│   ├── slices/
│   │   ├── portfolioSlice.ts
│   │   ├── tradesSlice.ts
│   │   ├── performanceSlice.ts
│   │   └── aiInsightsSlice.ts
│   └── api/
│       └── apiSlice.ts          # RTK Query API
│
├── types/                        # TypeScript types
│   ├── trade.ts
│   ├── portfolio.ts
│   ├── performance.ts
│   ├── aiDecision.ts
│   └── backtest.ts
│
└── styles/
    ├── globals.css              # Global styles
    └── theme.css                # Theme variables
```

---

## Sample Implementation

### 1. Root Layout (app/layout.tsx)

```typescript
import { Inter } from 'next/font/google';
import { Providers } from '@/components/Providers';
import { Toaster } from '@/components/UI/toast';
import '@/styles/globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata = {
  title: 'RRRalgorithms - Trading Transparency Dashboard',
  description: 'Real-time algorithmic trading with complete transparency',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>
        <Providers>
          {children}
          <Toaster />
        </Providers>
      </body>
    </html>
  );
}
```

### 2. Providers Component

```typescript
// components/Providers.tsx
'use client';

import { Provider } from 'react-redux';
import { store } from '@/store/store';
import { SocketProvider } from '@/lib/websocket';

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <Provider store={store}>
      <SocketProvider>
        {children}
      </SocketProvider>
    </Provider>
  );
}
```

### 3. Main Dashboard Page

```typescript
// app/dashboard/page.tsx
import { DashboardLayout } from '@/components/Dashboard/DashboardLayout';
import { PortfolioOverview } from '@/components/Dashboard/PortfolioOverview';
import { LiveTradeFeed } from '@/components/Dashboard/LiveTradeFeed';
import { PortfolioAllocation } from '@/components/Dashboard/PortfolioAllocation';
import { EquityCurve } from '@/components/Dashboard/EquityCurve';
import { TopPerformers } from '@/components/Dashboard/TopPerformers';
import { RiskMetrics } from '@/components/Dashboard/RiskMetrics';

export default function DashboardPage() {
  return (
    <DashboardLayout>
      {/* Portfolio Overview - Full Width */}
      <div className="col-span-full">
        <PortfolioOverview />
      </div>

      {/* Live Feed - 2/3 width */}
      <div className="lg:col-span-2">
        <LiveTradeFeed />
      </div>

      {/* Right Sidebar - 1/3 width */}
      <div className="lg:col-span-1 space-y-4">
        <PortfolioAllocation />
        <RiskMetrics />
      </div>

      {/* Equity Curve - 2/3 width */}
      <div className="lg:col-span-2">
        <EquityCurve period="30d" />
      </div>

      {/* Top Performers - 1/3 width */}
      <div className="lg:col-span-1">
        <TopPerformers />
      </div>
    </DashboardLayout>
  );
}
```

### 4. Dashboard Layout Component

```typescript
// components/Dashboard/DashboardLayout.tsx
'use client';

import { ReactNode } from 'react';
import { Sidebar } from './Sidebar';
import { Header } from './Header';

interface DashboardLayoutProps {
  children: ReactNode;
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
  return (
    <div className="flex h-screen bg-gray-900 text-white">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <Header />

        {/* Content */}
        <main className="flex-1 overflow-y-auto p-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
```

### 5. Portfolio Overview Component

```typescript
// components/Dashboard/PortfolioOverview.tsx
'use client';

import { usePortfolio } from '@/hooks/usePortfolio';
import { Card } from '@/components/UI/card';
import { formatCurrency, formatPercentage } from '@/lib/formatters';
import { TrendingUp, TrendingDown } from 'lucide-react';

export function PortfolioOverview() {
  const { data: portfolio, isLoading } = usePortfolio();

  if (isLoading) {
    return <PortfolioOverviewSkeleton />;
  }

  if (!portfolio) {
    return null;
  }

  const metrics = [
    {
      label: 'Total Value',
      value: formatCurrency(portfolio.total_value),
      change: formatCurrency(portfolio.daily_pnl),
      changePercent: formatPercentage(portfolio.daily_return),
      positive: portfolio.daily_pnl >= 0,
    },
    {
      label: 'Daily P&L',
      value: formatCurrency(portfolio.daily_pnl),
      positive: portfolio.daily_pnl >= 0,
    },
    {
      label: 'Daily Return',
      value: formatPercentage(portfolio.daily_return),
      positive: portfolio.daily_return >= 0,
    },
    {
      label: 'Sharpe Ratio',
      value: portfolio.sharpe_ratio?.toFixed(2) || '-',
      positive: (portfolio.sharpe_ratio || 0) > 1,
    },
  ];

  return (
    <Card className="p-6 bg-gray-800 border-gray-700">
      <h2 className="text-xl font-bold mb-4">Portfolio Overview</h2>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {metrics.map((metric, index) => (
          <div key={index} className="flex flex-col">
            <span className="text-sm text-gray-400">{metric.label}</span>
            <div className="flex items-center gap-2 mt-1">
              <span className="text-2xl font-bold">{metric.value}</span>
              {metric.positive !== undefined && (
                metric.positive ? (
                  <TrendingUp className="text-green-500" size={20} />
                ) : (
                  <TrendingDown className="text-red-500" size={20} />
                )
              )}
            </div>
            {metric.change && (
              <span className={`text-sm mt-1 ${
                metric.positive ? 'text-green-500' : 'text-red-500'
              }`}>
                {metric.change} ({metric.changePercent})
              </span>
            )}
          </div>
        ))}
      </div>

      {/* Status Indicator */}
      <div className="mt-4 flex items-center gap-2">
        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
        <span className="text-sm text-gray-400">
          System Status: Trading Active
        </span>
      </div>
    </Card>
  );
}

function PortfolioOverviewSkeleton() {
  return (
    <Card className="p-6 bg-gray-800 border-gray-700">
      <div className="animate-pulse">
        <div className="h-6 bg-gray-700 rounded w-48 mb-4" />
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="space-y-2">
              <div className="h-4 bg-gray-700 rounded w-24" />
              <div className="h-8 bg-gray-700 rounded w-32" />
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}
```

### 6. WebSocket Hook

```typescript
// hooks/useSocket.ts
'use client';

import { useEffect, useState } from 'react';
import { useSocket as useSocketContext } from '@/lib/websocket';

export function useSocketEvent<T>(
  event: string,
  handler: (data: T) => void
) {
  const socket = useSocketContext();
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    if (!socket) return;

    // Listen for connection events
    socket.on('connect', () => setIsConnected(true));
    socket.on('disconnect', () => setIsConnected(false));

    // Listen for specific event
    socket.on(event, handler);

    return () => {
      socket.off(event, handler);
      socket.off('connect');
      socket.off('disconnect');
    };
  }, [socket, event, handler]);

  return { socket, isConnected };
}
```

### 7. Portfolio Hook with Real-time Updates

```typescript
// hooks/usePortfolio.ts
'use client';

import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useSocketEvent } from './useSocket';
import { api } from '@/lib/api';
import { Portfolio } from '@/types/portfolio';

export function usePortfolio() {
  const queryClient = useQueryClient();

  // Fetch initial data
  const query = useQuery({
    queryKey: ['portfolio'],
    queryFn: async () => {
      const response = await api.get<Portfolio>('/api/v1/portfolio');
      return response.data;
    },
    refetchInterval: 5000, // Refetch every 5 seconds as fallback
  });

  // Listen for real-time updates
  useSocketEvent<Portfolio>('performance:metrics_update', (data) => {
    // Update the query cache with new data
    queryClient.setQueryData(['portfolio'], (old: Portfolio | undefined) => {
      if (!old) return data;
      return { ...old, ...data };
    });
  });

  return query;
}
```

### 8. API Client

```typescript
// lib/api.ts
import axios from 'axios';

export const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for auth
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);
```

### 9. WebSocket Context Provider

```typescript
// lib/websocket.tsx
'use client';

import { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { io, Socket } from 'socket.io-client';

const SocketContext = createContext<Socket | null>(null);

export function SocketProvider({ children }: { children: ReactNode }) {
  const [socket, setSocket] = useState<Socket | null>(null);

  useEffect(() => {
    const socketInstance = io(
      process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:8000',
      {
        transports: ['websocket'],
        autoConnect: true,
      }
    );

    socketInstance.on('connect', () => {
      console.log('WebSocket connected');

      // Subscribe to all channels
      socketInstance.emit('subscribe', {
        channels: ['trades', 'performance', 'ai_decisions', 'positions'],
      });
    });

    socketInstance.on('disconnect', () => {
      console.log('WebSocket disconnected');
    });

    socketInstance.on('error', (error) => {
      console.error('WebSocket error:', error);
    });

    setSocket(socketInstance);

    return () => {
      socketInstance.disconnect();
    };
  }, []);

  return (
    <SocketContext.Provider value={socket}>
      {children}
    </SocketContext.Provider>
  );
}

export function useSocket() {
  const context = useContext(SocketContext);
  if (context === undefined) {
    throw new Error('useSocket must be used within SocketProvider');
  }
  return context;
}
```

### 10. Formatters

```typescript
// lib/formatters.ts

export function formatCurrency(
  value: number,
  options: Intl.NumberFormatOptions = {}
): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
    ...options,
  }).format(value);
}

export function formatPercentage(
  value: number,
  decimals: number = 2
): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(decimals)}%`;
}

export function formatNumber(
  value: number,
  decimals: number = 2
): string {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
}

export function formatDate(
  date: Date | string,
  options: Intl.DateTimeFormatOptions = {}
): string {
  const dateObj = typeof date === 'string' ? new Date(date) : date;

  return new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    ...options,
  }).format(dateObj);
}

export function formatTime(
  date: Date | string,
  options: Intl.DateTimeFormatOptions = {}
): string {
  const dateObj = typeof date === 'string' ? new Date(date) : date;

  return new Intl.DateTimeFormat('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    ...options,
  }).format(dateObj);
}

export function formatRelativeTime(date: Date | string): string {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  const now = new Date();
  const diffMs = now.getTime() - dateObj.getTime();
  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffSecs / 60);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffSecs < 60) return `${diffSecs}s ago`;
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;

  return formatDate(dateObj);
}
```

### 11. TypeScript Types

```typescript
// types/portfolio.ts
export interface Portfolio {
  total_value: number;
  cash: number;
  positions_value: number;
  daily_pnl: number;
  total_pnl: number;
  daily_return: number;
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  positions: Position[];
}

export interface Position {
  symbol: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  allocation_pct: number;
}

// types/trade.ts
export interface Trade {
  id: string;
  timestamp: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  order_type: 'market' | 'limit';
  status: 'filled' | 'pending' | 'cancelled';
  pnl?: number;
  strategy: string;
  ai_confidence?: number;
  ai_reasoning?: string;
  risk_analysis?: RiskAnalysis;
}

export interface RiskAnalysis {
  position_size_pct: number;
  stop_loss: number;
  take_profit: number;
  risk_reward_ratio: number;
}

// types/performance.ts
export interface PerformanceMetrics {
  timestamp: string;
  portfolio_value: number;
  daily_pnl: number;
  total_return: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
}

export interface EquityCurvePoint {
  timestamp: string;
  value: number;
}

// types/aiDecision.ts
export interface AIDecision {
  id: string;
  timestamp: string;
  symbol: string;
  model_name: string;
  prediction: {
    direction: 'up' | 'down' | 'neutral';
    confidence: number;
    price_target: number;
    time_horizon: string;
  };
  features: Record<string, number>;
  reasoning: string;
  outcome?: 'profitable' | 'loss' | 'pending';
  actual_return?: number;
}
```

---

## Package.json

```json
{
  "name": "rrralgorithms-dashboard",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "type-check": "tsc --noEmit"
  },
  "dependencies": {
    "next": "14.2.0",
    "react": "18.3.0",
    "react-dom": "18.3.0",
    "@reduxjs/toolkit": "^2.2.0",
    "react-redux": "^9.1.0",
    "@tanstack/react-query": "^5.28.0",
    "socket.io-client": "^4.7.0",
    "axios": "^1.6.0",
    "lightweight-charts": "^4.1.0",
    "recharts": "^2.12.0",
    "framer-motion": "^11.0.0",
    "lucide-react": "^0.363.0",
    "tailwindcss": "^3.4.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.2.0",
    "date-fns": "^3.3.0",
    "zod": "^3.22.0"
  },
  "devDependencies": {
    "@types/node": "^20.11.0",
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "typescript": "^5.3.0",
    "eslint": "^8.56.0",
    "eslint-config-next": "14.2.0"
  }
}
```

---

## Environment Variables

```env
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
NEXT_PUBLIC_APP_NAME=RRRalgorithms
```

---

## Tailwind Config

```javascript
// tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ['class'],
  content: [
    './app/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        border: 'hsl(var(--border))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
        },
        success: {
          DEFAULT: 'hsl(142, 76%, 36%)',
          foreground: 'hsl(142, 76%, 96%)',
        },
        danger: {
          DEFAULT: 'hsl(0, 84%, 60%)',
          foreground: 'hsl(0, 84%, 96%)',
        },
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [require('tailwindcss-animate')],
}
```

---

## Getting Started

```bash
# Create Next.js app
npx create-next-app@latest rrralgorithms-dashboard --typescript --tailwind --app

# Install dependencies
npm install

# Install additional packages
npm install @reduxjs/toolkit react-redux
npm install @tanstack/react-query
npm install socket.io-client
npm install axios
npm install lightweight-charts recharts
npm install framer-motion
npm install lucide-react
npm install clsx tailwind-merge
npm install date-fns
npm install zod

# Install shadcn/ui
npx shadcn-ui@latest init
npx shadcn-ui@latest add button card dialog dropdown-menu input select tabs toast

# Run development server
npm run dev
```

---

## Deployment

### Vercel (Recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Production
vercel --prod
```

### Docker

```dockerfile
# Dockerfile
FROM node:20-alpine AS base

# Install dependencies
FROM base AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci

# Build
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# Production
FROM base AS runner
WORKDIR /app

ENV NODE_ENV production

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000

CMD ["node", "server.js"]
```

```bash
# Build and run
docker build -t rrralgorithms-dashboard .
docker run -p 3000:3000 rrralgorithms-dashboard
```

---

## Best Practices

### 1. Code Organization
- Keep components small and focused
- Use custom hooks for reusable logic
- Separate business logic from UI

### 2. Performance
- Use React.memo for expensive components
- Implement virtual scrolling for large lists
- Optimize bundle size with dynamic imports

### 3. Type Safety
- Define all types in `/types`
- Use Zod for runtime validation
- Enable strict TypeScript

### 4. Error Handling
- Implement error boundaries
- Show user-friendly error messages
- Log errors to monitoring service

### 5. Testing
- Unit test components
- Integration test API calls
- E2E test critical flows

---

This structure provides a solid foundation for building the transparency dashboard with Next.js 14, TypeScript, and modern React patterns.
