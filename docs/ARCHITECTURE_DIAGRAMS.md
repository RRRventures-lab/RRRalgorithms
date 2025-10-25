# RRRalgorithms Transparency Dashboard - Architecture Diagrams

**Visual architecture and component diagrams**

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            USER DEVICES                                      │
│                                                                               │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐│
│  │   Desktop    │   │   Tablet     │   │   Mobile     │   │   API Users  ││
│  │   Browser    │   │   Browser    │   │   Browser    │   │   (cURL)     ││
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘│
│         │                  │                  │                  │          │
└─────────┼──────────────────┼──────────────────┼──────────────────┼──────────┘
          │                  │                  │                  │
          └──────────────────┼──────────────────┼──────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLOUDFLARE CDN                                       │
│  • DDoS Protection                                                           │
│  • SSL/TLS Termination                                                       │
│  • Static Asset Caching                                                      │
│  • Geographic Distribution                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
│     VERCEL        │  │   RAILWAY/FLY.IO  │  │     SUPABASE      │
│   (Frontend)      │  │    (Backend)      │  │   (Database)      │
│                   │  │                   │  │                   │
│  Next.js 14       │◄─┤  FastAPI          │◄─┤  PostgreSQL       │
│  React 18         │  │  Socket.IO        │  │  Real-time DB     │
│  TypeScript       │  │  Python 3.11      │  │  Auth & Storage   │
│  Tailwind CSS     │  │  Uvicorn          │  │  Row-level        │
│                   │  │                   │  │  Security         │
│  • SSR/SSG        │  │  • REST API       │  │                   │
│  • Edge Runtime   │  │  • WebSocket      │  │  • Indexes        │
│  • Auto-scaling   │  │  • Rate Limiting  │  │  • Backups        │
│  • CDN            │  │  • Redis Pub/Sub  │  │  • Replication    │
└───────────────────┘  └────────┬──────────┘  └───────────────────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
                    ▼           ▼           ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │  Redis   │ │  Sentry  │ │ Trading  │
              │  Cache   │ │  Errors  │ │  System  │
              │          │ │          │ │ (Python) │
              │• Pub/Sub │ │• Logging │ │          │
              │• Session │ │• Alerts  │ │• Orders  │
              │• Queue   │ │• APM     │ │• AI      │
              └──────────┘ └──────────┘ └──────────┘
```

---

## Data Flow Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         TRADING SYSTEM                                    │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐    │
│  │ Trading    │   │ Risk       │   │ AI Model   │   │ Portfolio  │    │
│  │ Engine     │   │ Manager    │   │ Inference  │   │ Manager    │    │
│  └─────┬──────┘   └─────┬──────┘   └─────┬──────┘   └─────┬──────┘    │
│        │                │                │                │             │
│        │                │                │                │             │
│        └────────────────┴────────────────┴────────────────┘             │
│                              │                                           │
│                              ▼                                           │
│                    ┌──────────────────┐                                 │
│                    │ Event Publisher  │                                 │
│                    │ (Redis Client)   │                                 │
│                    └─────────┬────────┘                                 │
│                              │                                           │
└──────────────────────────────┼──────────────────────────────────────────┘
                               │
                               │ Publish Events
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         REDIS PUB/SUB                                     │
│  Channels:                                                                │
│  • trades            (new orders, fills, cancellations)                  │
│  • ai_decisions      (predictions, features, reasoning)                  │
│  • performance       (portfolio value, P&L, metrics)                     │
│  • positions         (open positions, updates, closures)                 │
└──────────────────────────────────────────────────────────────────────────┘
                               │
                               │ Subscribe
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      FASTAPI BACKEND                                      │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                  Socket.IO Server                                │    │
│  │                                                                   │    │
│  │  • Receives events from Redis                                   │    │
│  │  • Broadcasts to connected WebSocket clients                    │    │
│  │  • Manages client subscriptions (rooms)                         │    │
│  │  • Handles reconnections                                        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                               │                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     REST API                                     │    │
│  │                                                                   │    │
│  │  /api/v1/portfolio     - Portfolio state & metrics              │    │
│  │  /api/v1/trades        - Trade history & details                │    │
│  │  /api/v1/performance   - Performance analytics                  │    │
│  │  /api/v1/ai            - AI predictions & insights              │    │
│  │  /api/v1/backtests     - Backtest results                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                               │                                           │
│                               │ Persist Data                              │
│                               │                                           │
└───────────────────────────────┼──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      POSTGRESQL DATABASE                                  │
│                                                                           │
│  Core Tables:                    Transparency Tables:                    │
│  • crypto_aggregates             • ai_decisions                          │
│  • crypto_trades                 • trade_feed                            │
│  • crypto_quotes                 • performance_snapshots                 │
│  • market_sentiment              • strategy_performance                  │
│                                  • backtest_results                       │
│                                  • feature_importance                     │
└──────────────────────────────────────────────────────────────────────────┘
                                │
                                │ Query Data
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       NEXT.JS FRONTEND                                    │
│                                                                           │
│  ┌──────────────────────────┐    ┌──────────────────────────┐           │
│  │   Initial Page Load      │    │   Real-time Updates      │           │
│  │                          │    │                          │           │
│  │  1. Server-side render   │    │  1. WebSocket connect    │           │
│  │  2. Fetch initial data   │    │  2. Subscribe channels   │           │
│  │  3. Hydrate React app    │    │  3. Receive events       │           │
│  │  4. Connect WebSocket    │    │  4. Update UI instantly  │           │
│  └──────────────────────────┘    └──────────────────────────┘           │
│                                                                           │
│  Pages:                                                                   │
│  • /dashboard         - Main command center                              │
│  • /live-feed         - Detailed trading feed                            │
│  • /performance       - Analytics & metrics                              │
│  • /ai-insights       - AI decision transparency                         │
│  • /backtests         - Backtest results                                 │
└──────────────────────────────────────────────────────────────────────────┘
                                │
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                           END USER                                        │
│  • Sees real-time updates (<200ms latency)                              │
│  • Full transparency into AI decisions                                   │
│  • Complete trading history                                              │
│  • Performance analytics                                                 │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Component Tree

```
App (Root)
│
├── Providers
│   ├── Redux Provider
│   │   └── Store (state management)
│   └── Socket Provider
│       └── WebSocket connection
│
├── Layout
│   ├── Sidebar
│   │   ├── Logo
│   │   ├── Navigation Menu
│   │   │   ├── Dashboard Link
│   │   │   ├── Live Feed Link
│   │   │   ├── Performance Link
│   │   │   ├── AI Insights Link
│   │   │   └── Backtests Link
│   │   └── User Profile
│   │
│   └── Header
│       ├── Search Bar
│       ├── Notifications
│       └── Settings Dropdown
│
└── Routes
    │
    ├── /dashboard
    │   ├── Portfolio Overview
    │   │   ├── Total Value Card
    │   │   ├── Daily P&L Card
    │   │   ├── Daily Return Card
    │   │   └── Sharpe Ratio Card
    │   │
    │   ├── Live Trade Feed
    │   │   ├── Trade Card (x N)
    │   │   │   ├── Trade Header
    │   │   │   ├── Order Details
    │   │   │   ├── AI Decision
    │   │   │   └── Risk Management
    │   │   └── Load More Button
    │   │
    │   ├── Portfolio Allocation
    │   │   ├── Pie Chart
    │   │   └── Asset Breakdown
    │   │
    │   ├── Equity Curve
    │   │   ├── Chart (Lightweight Charts)
    │   │   └── Time Period Selector
    │   │
    │   ├── Top Performers
    │   │   └── Performer List Item (x 5)
    │   │
    │   └── Risk Metrics
    │       ├── Max Drawdown
    │       ├── Risk Score
    │       ├── Position Limit
    │       └── Daily Loss Limit
    │
    ├── /live-feed
    │   ├── Feed Filters
    │   │   ├── All Button
    │   │   ├── Trades Button
    │   │   └── Signals Button
    │   │
    │   ├── Export Button
    │   │
    │   └── Feed Items (infinite scroll)
    │       ├── Trade Card
    │       ├── AI Signal Card
    │       └── Position Closed Card
    │
    ├── /performance
    │   ├── Key Metrics Grid
    │   │   ├── Total Return
    │   │   ├── Sharpe Ratio
    │   │   ├── Max Drawdown
    │   │   ├── Win Rate
    │   │   └── Total Trades
    │   │
    │   ├── Equity Curve Chart
    │   │   ├── Main Chart
    │   │   ├── Time Period Selector
    │   │   └── Chart Controls
    │   │
    │   ├── Drawdown Analysis
    │   │   ├── Current Drawdown
    │   │   ├── Max Drawdown
    │   │   └── Recovery Stats
    │   │
    │   ├── Returns Distribution
    │   │   ├── Histogram Chart
    │   │   └── Statistics
    │   │
    │   ├── Risk-Adjusted Metrics
    │   │   ├── Sharpe Ratio
    │   │   ├── Sortino Ratio
    │   │   ├── Calmar Ratio
    │   │   └── More Metrics
    │   │
    │   ├── Trading Statistics
    │   │   ├── Win/Loss Stats
    │   │   ├── Avg Win/Loss
    │   │   └── Profit Factor
    │   │
    │   └── Monthly Performance
    │       └── Heatmap Chart
    │
    ├── /ai-insights
    │   ├── Model Performance
    │   │   ├── Prediction Accuracy
    │   │   ├── Avg Confidence
    │   │   ├── Hit Rate
    │   │   └── Avg Error
    │   │
    │   ├── Recent Predictions
    │   │   ├── Prediction Card (x N)
    │   │   │   ├── Prediction Details
    │   │   │   ├── Key Features
    │   │   │   ├── Model Reasoning
    │   │   │   └── Outcome Status
    │   │   └── Load More
    │   │
    │   ├── Feature Importance
    │   │   └── Bar Chart
    │   │
    │   ├── Confidence Calibration
    │   │   └── Calibration Plot
    │   │
    │   └── Accuracy by Timeframe
    │       └── Horizontal Bar Chart
    │
    └── /backtests
        ├── Strategy Comparison Table
        │   └── Strategy Row (x N)
        │
        ├── Backtest Details
        │   ├── Overview Stats
        │   ├── Equity Curve
        │   ├── Performance Metrics
        │   ├── Trading Statistics
        │   └── Action Buttons
        │
        ├── Monthly Returns
        │   └── Returns Grid
        │
        └── Trade Distribution
            └── Distribution Chart
```

---

## Database Schema Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CORE TABLES (Existing)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  crypto_aggregates          crypto_trades          crypto_quotes        │
│  ┌───────────────┐         ┌──────────────┐       ┌──────────────┐     │
│  │ id            │         │ id           │       │ id           │     │
│  │ ticker        │         │ ticker       │       │ ticker       │     │
│  │ event_time    │         │ event_time   │       │ event_time   │     │
│  │ open          │         │ price        │       │ bid_price    │     │
│  │ high          │         │ size         │       │ ask_price    │     │
│  │ low           │         │ exchange_id  │       │ bid_size     │     │
│  │ close         │         │ trade_id     │       │ ask_size     │     │
│  │ volume        │         │ created_at   │       │ created_at   │     │
│  │ vwap          │         └──────────────┘       └──────────────┘     │
│  │ created_at    │                                                      │
│  └───────────────┘                                                      │
│                                                                          │
│  market_sentiment                                                       │
│  ┌───────────────┐                                                      │
│  │ id            │                                                      │
│  │ asset         │                                                      │
│  │ source        │                                                      │
│  │ sentiment_label│                                                     │
│  │ sentiment_score│                                                     │
│  │ confidence    │                                                      │
│  │ event_time    │                                                      │
│  │ created_at    │                                                      │
│  └───────────────┘                                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      TRANSPARENCY TABLES (New)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ai_decisions                trade_feed                                 │
│  ┌────────────────────┐      ┌────────────────────┐                    │
│  │ id                 │      │ id                 │                    │
│  │ timestamp          │      │ timestamp          │                    │
│  │ symbol             │      │ event_type         │                    │
│  │ model_name         │      │ symbol             │                    │
│  │ prediction (JSONB) │      │ data (JSONB)       │                    │
│  │ features (JSONB)   │      │ visibility         │                    │
│  │ reasoning          │      │ user_id            │                    │
│  │ outcome            │      │ strategy_name      │                    │
│  │ actual_return      │      │ created_at         │                    │
│  │ confidence_score   │      └────────────────────┘                    │
│  │ created_at         │               │                                 │
│  └────────────┬───────┘               │                                 │
│               │                       │                                 │
│               │                       │                                 │
│               ▼                       ▼                                 │
│  trade_attribution                                                      │
│  ┌────────────────────┐                                                 │
│  │ id                 │                                                 │
│  │ trade_id           │───────┐                                        │
│  │ ai_decision_id     │◄──────┘                                        │
│  │ prediction_accuracy│                                                 │
│  │ followed_prediction│                                                 │
│  │ override_reason    │                                                 │
│  │ created_at         │                                                 │
│  └────────────────────┘                                                 │
│                                                                          │
│  performance_snapshots          strategy_performance                    │
│  ┌────────────────────┐         ┌────────────────────┐                 │
│  │ id                 │         │ id                 │                 │
│  │ timestamp          │         │ strategy_name      │                 │
│  │ portfolio_value    │         │ timeframe          │                 │
│  │ cash               │         │ total_trades       │                 │
│  │ positions_value    │         │ winning_trades     │                 │
│  │ daily_return       │         │ losing_trades      │                 │
│  │ total_return       │         │ win_rate           │                 │
│  │ sharpe_ratio       │         │ total_return       │                 │
│  │ max_drawdown       │         │ sharpe_ratio       │                 │
│  │ win_rate           │         │ max_drawdown       │                 │
│  │ metrics (JSONB)    │         │ metrics (JSONB)    │                 │
│  │ created_at         │         │ updated_at         │                 │
│  └────────────────────┘         └────────────────────┘                 │
│                                                                          │
│  backtest_results              feature_importance                       │
│  ┌────────────────────┐         ┌────────────────────┐                 │
│  │ id                 │         │ id                 │                 │
│  │ strategy_name      │         │ model_name         │                 │
│  │ backtest_id        │         │ feature_name       │                 │
│  │ start_date         │         │ importance_score   │                 │
│  │ end_date           │         │ period_start       │                 │
│  │ initial_capital    │         │ period_end         │                 │
│  │ final_capital      │         │ sample_size        │                 │
│  │ total_return       │         │ metadata (JSONB)   │                 │
│  │ sharpe_ratio       │         │ created_at         │                 │
│  │ max_drawdown       │         └────────────────────┘                 │
│  │ win_rate           │                                                 │
│  │ equity_curve (JSONB)│                                                │
│  │ trades (JSONB)     │                                                 │
│  │ metrics (JSONB)    │                                                 │
│  │ created_at         │                                                 │
│  └────────────────────┘                                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

Relationships:
─────────────  One-to-many
◄──────────►  Many-to-one
```

---

## WebSocket Event Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        EVENT LIFECYCLE                                    │
└──────────────────────────────────────────────────────────────────────────┘

1. TRADE EXECUTION
──────────────────

Trading System                Redis              FastAPI             Frontend
     │                         │                   │                    │
     │  1. Execute Trade       │                   │                    │
     ├────────────────────────▶│                   │                    │
     │  publish('trades', {    │                   │                    │
     │    event: 'new_trade',  │                   │                    │
     │    data: {...}          │                   │                    │
     │  })                     │                   │                    │
     │                         │                   │                    │
     │                         │  2. Subscribe     │                    │
     │                         │◄──────────────────┤                    │
     │                         │  PSUBSCRIBE trades│                    │
     │                         │                   │                    │
     │                         │  3. Broadcast     │                    │
     │                         ├──────────────────▶│                    │
     │                         │  Event data       │                    │
     │                         │                   │                    │
     │                         │                   │  4. Emit WebSocket │
     │                         │                   ├───────────────────▶│
     │                         │                   │  'trades:new_trade'│
     │                         │                   │                    │
     │                         │                   │                    │  5. Update UI
     │                         │                   │                    ├─────────────▶
     │                         │                   │                    │  React state
     │                         │                   │                    │  Animation
     │                         │                   │                    │  Notification
     │                         │                   │                    │
     ▼                         ▼                   ▼                    ▼

Time: <200ms from trade execution to UI update


2. AI PREDICTION
────────────────

AI Model                      Redis              FastAPI             Frontend
     │                         │                   │                    │
     │  1. Make Prediction     │                   │                    │
     ├────────────────────────▶│                   │                    │
     │  publish('ai_decisions',│                   │                    │
     │    {prediction: {...}}  │                   │                    │
     │  })                     │                   │                    │
     │                         │                   │                    │
     │                         │  2. Broadcast     │                    │
     │                         ├──────────────────▶│                    │
     │                         │                   │                    │
     │                         │                   │  3. Emit           │
     │                         │                   ├───────────────────▶│
     │                         │                   │  'ai_decisions:    │
     │                         │                   │   new_prediction'  │
     │                         │                   │                    │
     │                         │                   │                    │  4. Display
     │                         │                   │                    ├─────────────▶
     │                         │                   │                    │  Prediction
     │                         │                   │                    │  Card
     │                         │                   │                    │
     ▼                         ▼                   ▼                    ▼


3. PERFORMANCE UPDATE
─────────────────────

Portfolio Manager             Redis              FastAPI             Frontend
     │                         │                   │                    │
     │  1. Calculate Metrics   │                   │                    │
     │     (every 1 second)    │                   │                    │
     ├────────────────────────▶│                   │                    │
     │  publish('performance', │                   │                    │
     │    {metrics: {...}}     │                   │                    │
     │  })                     │                   │                    │
     │                         │                   │                    │
     │                         │  2. Throttle      │                    │
     │                         │     (1/sec max)   │                    │
     │                         ├──────────────────▶│                    │
     │                         │                   │                    │
     │                         │                   │  3. Emit           │
     │                         │                   ├───────────────────▶│
     │                         │                   │  'performance:     │
     │                         │                   │   metrics_update'  │
     │                         │                   │                    │
     │                         │                   │                    │  4. Update
     │                         │                   │                    ├─────────────▶
     │                         │                   │                    │  All metrics
     │                         │                   │                    │  Equity curve
     │                         │                   │                    │
     ▼                         ▼                   ▼                    ▼
```

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PRODUCTION DEPLOYMENT                             │
└─────────────────────────────────────────────────────────────────────────────┘

                                ┌──────────────┐
                                │   INTERNET   │
                                └──────┬───────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │  CLOUDFLARE CDN │
                              │  • DDoS Protection
                              │  • SSL/TLS
                              │  • Caching
                              │  • Rate Limiting
                              └────────┬────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
     │     VERCEL      │     │  RAILWAY/FLY.IO │     │   MONITORING    │
     │   (Frontend)    │     │    (Backend)    │     │                 │
     ├─────────────────┤     ├─────────────────┤     ├─────────────────┤
     │                 │     │                 │     │   Sentry        │
     │ • Edge Network  │     │ • Auto-scaling  │     │   - Errors      │
     │ • CDN           │◄────┤ • Health checks │     │   - Performance │
     │ • Auto-deploy   │ API │ • Logs          │     │   - Alerts      │
     │ • Rollback      │     │ • Metrics       │     │                 │
     └─────────────────┘     └────────┬────────┘     │   Uptime Robot  │
                                      │               │   - Uptime      │
                      ┌───────────────┼───────┐      │   - Latency     │
                      │               │       │      │                 │
                      ▼               ▼       ▼      │   Plausible     │
              ┌──────────────┐ ┌──────────┐ ┌────────┐ - Analytics  │
              │   SUPABASE   │ │  REDIS   │ │ SECRETS│ - Privacy    │
              │  (Database)  │ │  (Upstash)│ │ (ENV)  │              │
              ├──────────────┤ ├──────────┤ └────────┘ - GDPR      │
              │              │ │          │             └──────────────┘
              │• PostgreSQL  │ │• Pub/Sub │
              │• Backups     │ │• Cache   │
              │• Replication │ │• Session │
              │• Monitoring  │ │• Queue   │
              └──────────────┘ └──────────┘

Geographic Distribution:
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                           │
│  US East          US West         Europe          Asia                   │
│  ┌──────┐        ┌──────┐        ┌──────┐        ┌──────┐              │
│  │ CDN  │        │ CDN  │        │ CDN  │        │ CDN  │              │
│  │ Node │        │ Node │        │ Node │        │ Node │              │
│  └──────┘        └──────┘        └──────┘        └──────┘              │
│     │                │                │                │                  │
│     └────────────────┼────────────────┼────────────────┘                  │
│                      │                │                                   │
│                      ▼                ▼                                   │
│              Primary Region      Backup Region                           │
│              (US East)           (EU West)                               │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Security Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SECURITY LAYERS                                      │
└─────────────────────────────────────────────────────────────────────────────┘

Layer 1: Network Security
┌──────────────────────────────────────────────────────────────┐
│  Cloudflare                                                   │
│  • DDoS Protection (up to 100 Tbps)                          │
│  • WAF (Web Application Firewall)                            │
│  • Bot Management                                            │
│  • IP Reputation Filtering                                   │
└──────────────────────────────────────────────────────────────┘

Layer 2: Transport Security
┌──────────────────────────────────────────────────────────────┐
│  SSL/TLS                                                      │
│  • TLS 1.3                                                   │
│  • HSTS (Strict-Transport-Security)                         │
│  • Certificate Pinning                                       │
└──────────────────────────────────────────────────────────────┘

Layer 3: Application Security
┌──────────────────────────────────────────────────────────────┐
│  Backend (FastAPI)                                            │
│  • Rate Limiting (100 req/min per IP)                        │
│  • Input Validation (Pydantic)                               │
│  • SQL Injection Prevention (Parameterized queries)          │
│  • XSS Prevention (Content-Security-Policy)                  │
│  • CSRF Protection                                           │
│  • CORS Configuration                                        │
└──────────────────────────────────────────────────────────────┘

Layer 4: Authentication & Authorization
┌──────────────────────────────────────────────────────────────┐
│  JWT Authentication                                           │
│  • Access Tokens (15 min expiry)                             │
│  • Refresh Tokens (7 day expiry)                             │
│  • HttpOnly Cookies (not accessible by JS)                   │
│  • Token Rotation                                            │
│  • Role-based Access Control (RBAC)                          │
└──────────────────────────────────────────────────────────────┘

Layer 5: Data Security
┌──────────────────────────────────────────────────────────────┐
│  Database (Supabase)                                          │
│  • Encryption at Rest (AES-256)                              │
│  • Encryption in Transit (TLS)                               │
│  • Row-level Security (RLS)                                  │
│  • Automated Backups (daily)                                 │
│  • Point-in-time Recovery                                    │
└──────────────────────────────────────────────────────────────┘

Layer 6: Secrets Management
┌──────────────────────────────────────────────────────────────┐
│  Environment Variables                                        │
│  • Vercel Secret Management                                  │
│  • Railway Secret Management                                 │
│  • No secrets in code/git                                    │
│  • Separate secrets per environment                          │
└──────────────────────────────────────────────────────────────┘

Layer 7: Monitoring & Alerting
┌──────────────────────────────────────────────────────────────┐
│  Security Monitoring                                          │
│  • Failed login attempts                                     │
│  • Unusual API activity                                      │
│  • Unauthorized access attempts                              │
│  • Data exfiltration detection                               │
│  • Real-time alerts (Slack/Email)                            │
└──────────────────────────────────────────────────────────────┘
```

---

This comprehensive set of diagrams provides a complete visual understanding of the RRRalgorithms transparency dashboard architecture, data flow, components, and deployment strategy.
