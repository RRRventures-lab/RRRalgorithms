# RRRalgorithms Trading System - Data Model & Architecture Analysis

**Analysis Date**: 2025-10-25  
**System Version**: 2.0.0  
**Status**: Production Ready (Local Architecture)

---

## Executive Summary

The RRRalgorithms system is a comprehensive cryptocurrency algorithmic trading platform with:
- **Multiple data sources**: Polygon.io, TradingView, on-chain data
- **Real-time market data processing**: OHLCV, trades, quotes, order book metrics
- **AI/ML prediction models**: Neural networks with feature engineering capabilities
- **Automated trading engine**: Complete order lifecycle management with risk controls
- **Backtesting framework**: Historical strategy validation
- **Database architecture**: SQLite (local) / PostgreSQL/Supabase (cloud)
- **API exposure**: FastAPI-based transparency dashboard

---

## 1. DATA MODELS & SCHEMAS

### 1.1 Core Market Data Models

**OHLCV Data (Candlesticks)**
```python
# From: src/data_pipeline/polygon/models.py
from pydantic import BaseModel
from decimal import Decimal
from datetime import datetime

class Aggregate(BaseModel):
    """OHLCV aggregate bar (candlestick) data"""
    ticker: str                    # e.g., "X:BTCUSD"
    timestamp: int                 # Unix timestamp (ms)
    open: Decimal                  # Open price
    high: Decimal                  # High price
    low: Decimal                   # Low price
    close: Decimal                 # Close price
    volume: Decimal                # Trading volume
    vwap: Optional[Decimal]        # Volume weighted average price
    trade_count: Optional[int]     # Number of trades
```

**Individual Trades**
```python
class Trade(BaseModel):
    ticker: str                    # e.g., "X:BTCUSD"
    timestamp: int                 # Unix timestamp (ns)
    price: Decimal                 # Trade price
    size: Decimal                  # Trade size
    exchange: Optional[int]        # Exchange ID
    conditions: Optional[List[int]] # Trade conditions
```

**Bid/Ask Quotes**
```python
class Quote(BaseModel):
    ticker: str                    # Ticker symbol
    timestamp: int                 # Unix timestamp (ns)
    bid_price: Decimal             # Bid price
    bid_size: Decimal              # Bid size
    ask_price: Decimal             # Ask price
    ask_size: Decimal              # Ask size
    exchange: Optional[int]        # Exchange ID
    
    @property
    def spread(self) -> Decimal:
        return self.ask_price - self.bid_price
    
    @property
    def mid_price(self) -> Decimal:
        return (self.bid_price + self.ask_price) / 2
```

### 1.2 Trading Models

**Order Model**
```python
# src/services/trading_engine/oms/order_manager.py
order = {
    "order_id": str,                    # UUID
    "exchange_id": str,                 # Exchange identifier
    "symbol": str,                      # Trading pair (e.g., "BTC-USD")
    "side": str,                        # "buy" or "sell"
    "order_type": str,                  # "market", "limit", "stop_loss", "take_profit"
    "quantity": float,                  # Amount to trade
    "price": Optional[float],           # Limit price
    "stop_price": Optional[float],      # Stop price
    "time_in_force": str,              # "gtc", "ioc", "fok", "day"
    "status": str,                      # "pending", "open", "filled", "cancelled"
    "filled_quantity": float,           # Amount filled
    "average_fill_price": Optional[float],
    "strategy_id": Optional[str],       # Associated strategy
    "signal_id": Optional[str],         # Associated signal
    "metadata": Dict,                   # Additional context
    "created_at": str,                  # ISO timestamp
    "filled_at": Optional[str],         # ISO timestamp
}
```

**Position Model**
```python
# src/services/trading_engine/positions/position_manager.py
position = {
    "position_id": str,                 # UUID
    "symbol": str,                      # Trading pair
    "side": str,                        # "long" or "short"
    "quantity": float,                  # Position size
    "entry_price": float,               # Entry price
    "current_price": float,             # Current market price
    "unrealized_pnl": float,            # Unrealized P&L
    "realized_pnl": float,              # Realized P&L
    "total_pnl": float,                 # Total P&L
    "status": str,                      # "open" or "closed"
    "strategy_id": Optional[str],       # Strategy that opened it
    "open_order_id": str,               # Order that opened it
    "close_order_id": Optional[str],    # Order that closed it
    "metadata": Dict,                   # Additional metadata
    "opened_at": str,                   # ISO timestamp
    "closed_at": Optional[str],         # ISO timestamp
    "updated_at": str,                  # ISO timestamp
}
```

**Portfolio Model**
```python
# src/services/trading_engine/portfolio/portfolio_manager.py
portfolio = {
    "total_value": float,               # Total portfolio value
    "cash": float,                      # Cash balance
    "equity": float,                    # Invested equity value
    "total_pnl": float,                 # Total profit/loss
    "daily_pnl": float,                 # Daily profit/loss
    "total_return_pct": float,          # Total return percentage
    "daily_return_pct": float,          # Daily return percentage
}
```

### 1.3 Sentiment & Alternative Data Models

**Market Sentiment**
```python
sentiment_data = {
    "asset": str,                       # Asset symbol (e.g., "BTC")
    "source": str,                      # Data source ("perplexity", "twitter", etc.)
    "sentiment_label": str,             # "bullish", "neutral", "bearish"
    "sentiment_score": float,           # Range: -1.0 to 1.0
    "confidence": float,                # Range: 0.0 to 1.0
    "text": Optional[str],              # Source text
    "bullish_count": int,               # Number of bullish signals
    "bearish_count": int,               # Number of bearish signals
    "sources_count": int,               # Number of sources analyzed
    "timestamp": float,                 # Unix timestamp
}
```

---

## 2. DATABASE ARCHITECTURE

### 2.1 Current Database Configuration

**Location**: `/home/user/RRRalgorithms/src/database/`

**Dual Backend Support**:
- **SQLite** (Local development): `data/db/trading.db`
- **PostgreSQL/Supabase** (Cloud production): Configured via environment variables

**Database Client Factory** (`client_factory.py`):
```python
from src.database import get_db, get_database_client

# Usage
db = get_db()  # Returns configured DatabaseClient (SQLite or Supabase)
```

### 2.2 Database Schema Overview

**Schema Location**: 
- `/home/user/RRRalgorithms/src/database/schema.sql` (SQLite)
- `/home/user/RRRalgorithms/config/database/schema.sql` (PostgreSQL)
- `/home/user/RRRalgorithms/config/supabase/schema.sql` (Supabase)

**Major Table Groups**:

| Category | Tables | Purpose |
|----------|--------|---------|
| **Core** | `symbols` | Master list of trading instruments |
| **Market Data** | `market_data`, `trades_data`, `quotes` | OHLCV, trades, bid/ask quotes |
| **Trading** | `orders`, `trades`, `positions` | Order management, execution, positions |
| **Portfolio** | `portfolio_snapshots` | Portfolio state snapshots over time |
| **Risk** | `risk_limits`, `risk_events` | Risk thresholds and alerts |
| **ML** | `ml_models`, `ml_predictions` | Model registry and predictions |
| **Sentiment** | `market_sentiment` | Alternative sentiment data |
| **Backtesting** | `backtest_runs`, `backtest_trades` | Historical test results |
| **System** | `system_events`, `performance_metrics`, `audit_log` | Monitoring and logging |

### 2.3 Key Database Tables

**Market Data Table Structure**:
```sql
CREATE TABLE market_data (
    id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    timestamp INTEGER NOT NULL,      -- Unix timestamp
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    vwap REAL,
    trade_count INTEGER,
    created_at INTEGER,
    FOREIGN KEY (symbol) REFERENCES symbols(symbol)
);
-- Indexes: (symbol, timestamp), (created_at)
```

**Positions Table**:
```sql
CREATE TABLE positions (
    id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL UNIQUE,
    quantity REAL NOT NULL,
    average_price REAL NOT NULL,
    current_price REAL,
    unrealized_pnl REAL,
    realized_pnl REAL DEFAULT 0,
    opened_at INTEGER NOT NULL,
    updated_at INTEGER,
    FOREIGN KEY (symbol) REFERENCES symbols(symbol)
);
```

**Orders Table**:
```sql
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,             -- 'buy' or 'sell'
    order_type TEXT NOT NULL,       -- 'market', 'limit', 'stop', 'stop_limit'
    quantity REAL NOT NULL,
    price REAL,                     -- For limit orders
    stop_price REAL,                -- For stop orders
    status TEXT NOT NULL,           -- 'pending', 'open', 'filled', 'cancelled'
    filled_quantity REAL,
    average_fill_price REAL,
    timestamp INTEGER NOT NULL,
    filled_at INTEGER,
    cancelled_at INTEGER,
    strategy TEXT,
    created_at INTEGER,
    FOREIGN KEY (symbol) REFERENCES symbols(symbol)
);
```

**ML Models & Predictions**:
```sql
CREATE TABLE ml_models (
    id INTEGER PRIMARY KEY,
    model_name TEXT NOT NULL UNIQUE,
    model_type TEXT NOT NULL,       -- 'lstm', 'transformer', 'cnn', etc.
    version TEXT NOT NULL,
    file_path TEXT NOT NULL,        -- Path to model file/checkpoint
    metrics TEXT,                   -- JSON: {accuracy, loss, etc.}
    hyperparameters TEXT,           -- JSON: model config
    trained_at INTEGER NOT NULL,
    active INTEGER DEFAULT 0,
    created_at INTEGER
);

CREATE TABLE ml_predictions (
    id INTEGER PRIMARY KEY,
    model_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    prediction_type TEXT NOT NULL,  -- 'price', 'direction', 'volatility'
    prediction_value REAL NOT NULL,
    confidence REAL,
    features TEXT,                  -- JSON: input features used
    created_at INTEGER,
    FOREIGN KEY (model_id) REFERENCES ml_models(id)
);
```

---

## 3. MOCK DATA GENERATORS

### 3.1 MockDataSource (`src/data_pipeline/mock_data_source.py`)

**Location**: `/home/user/RRRalgorithms/src/data_pipeline/mock_data_source.py`

**Purpose**: Generate realistic OHLCV data for local development and testing without requiring external APIs.

**Key Classes**:

```python
@dataclass
class MockDataSource:
    """Generate realistic OHLCV data without requiring external APIs"""
    
    symbols: List[str] = field(default_factory=lambda: ["BTC-USD", "ETH-USD"])
    base_prices: Dict[str, float] = field(default_factory=lambda: DEFAULT_BASE_PRICES.copy())
    volatility: float = 0.02                 # 2% volatility
    update_interval: float = 1.0             # 1 second between updates
    trend_strength: float = 0.0001
    
    def get_latest_data(self, symbol: Optional[str] = None) -> Dict:
        """Return the most recent OHLCV snapshot"""
        # Returns: {symbol: {open, high, low, close, volume, timestamp}}
    
    def get_historical_data(self, symbol: str, periods: int = 100, 
                           interval_seconds: int = 60) -> pd.DataFrame:
        """Build a synthetic OHLCV history suitable for backtests"""
    
    def stream(self, callback: Callable, interval: Optional[float] = None) -> None:
        """Continuously emit market data until callback raises KeyboardInterrupt"""
```

**Default Symbols & Prices**:
```python
DEFAULT_BASE_PRICES = {
    "BTC-USD": 50_000.0,
    "ETH-USD": 3_000.0,
    "SOL-USD": 100.0,
    "MATIC-USD": 0.80,
    "AVAX-USD": 35.0,
}

DEFAULT_BASE_VOLUMES = {
    "BTC-USD": 1_000_000,
    "ETH-USD": 500_000,
    "SOL-USD": 100_000,
    "MATIC-USD": 50_000,
    "AVAX-USD": 75_000,
}
```

### 3.2 MockOrderBook (`src/data_pipeline/mock_data_source.py`)

```python
class MockOrderBook:
    """Generate synthetic order book snapshots"""
    
    def get_orderbook(self, levels: int = 10) -> Dict:
        """Return order book with bids and asks"""
        # Returns: {symbol, timestamp, bids, asks}
```

### 3.3 MockSentimentData (`src/data_pipeline/mock_data_source.py`)

```python
class MockSentimentData:
    """Simple sentiment generator using mean-reverting random walk"""
    
    def get_sentiment(self, symbol: str) -> Dict:
        """Return sentiment data"""
        # Returns: {symbol, timestamp, sentiment, confidence, sources_count, etc.}
```

---

## 4. EXISTING DATABASE & ML INFRASTRUCTURE

### 4.1 Database Configuration

**Settings File**: `/home/user/RRRalgorithms/src/core/settings.py`

```python
class Settings(BaseSettings):
    # Database Configuration
    database_url: Optional[str]                 # PostgreSQL URL
    supabase_url: Optional[str]                 # Supabase project URL
    supabase_service_key: Optional[str]         # Service key
    supabase_key: Optional[str]                 # Anon key
    
    database_pool_min: int = 5                  # Connection pool minimum
    database_pool_max: int = 20                 # Connection pool maximum
    database_timeout: int = 30                  # Timeout in seconds
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 50
```

**Database Pool Manager**: `/home/user/RRRalgorithms/src/core/database.py`

```python
class DatabasePool:
    """Thread-safe database connection pool manager"""
    
    def get_postgres_connection(self):
        """Get PostgreSQL connection from pool"""
    
    def get_supabase_client(self) -> Client:
        """Get Supabase client"""
    
    def get_pool_stats(self) -> Dict:
        """Get connection pool statistics"""
```

### 4.2 API Database Operations

**Supabase Client**: `/home/user/RRRalgorithms/src/data_pipeline/supabase_client.py`

Provides high-level methods for:
- **Market Data**: Insert/query OHLCV, trades, quotes
- **Sentiment**: Insert/query market sentiment
- **Trading**: Insert/update orders and positions
- **ML Models**: Register models and store predictions
- **Real-time**: WebSocket subscriptions for trades, orders, signals

### 4.3 ML Model Infrastructure

**Current Status**: No trained production models found. Framework is in place but models need to be trained and registered.

**ML Service**: `/home/user/RRRalgorithms/src/microservices/ml_service.py`

```python
class MLService:
    """Microservice for machine learning predictions"""
    
    predictor: ProductionPredictor           # Reference to model (not yet implemented)
    redis_cache: RedisCache                  # Prediction caching
    memory_cache: MemoryCache                # Local caching
    
    metrics = {
        'total_predictions': 0,
        'successful_predictions': 0,
        'cache_hits': 0,
        'cache_misses': 0,
        'avg_inference_time': 0.0,
    }
```

**Neural Network Components**:
- **Features**: `/home/user/RRRalgorithms/src/services/neural_network/features/technical_indicators.py`
  - 25+ technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
  - All vectorized with NumPy for efficiency
  
- **Data Loading**: `/home/user/RRRalgorithms/src/services/neural_network/utils/data_loader.py`
  - Loads crypto OHLCV data from CSV
  - Preprocessing and standardization

### 4.4 Trading Engine Integration

**Trading Engine**: `/home/user/RRRalgorithms/src/services/trading_engine/main.py`

Main components:
1. **Exchange Interface** (`exchanges/`):
   - `PaperExchange`: Simulated trading
   - `CoinbaseExchange`: Live trading
   - `ExchangeInterface`: Abstract base

2. **Order Management** (`oms/order_manager.py`):
   - Order creation, modification, cancellation
   - Status tracking
   - Database persistence

3. **Position Management** (`positions/position_manager.py`):
   - Open position tracking
   - P&L calculations
   - Position lifecycle

4. **Portfolio Management** (`portfolio/portfolio_manager.py`):
   - Aggregate portfolio metrics
   - Daily snapshots
   - Risk analytics

5. **Strategy Execution** (`executor/strategy_executor.py`):
   - Signal processing
   - Order execution
   - Risk validation

---

## 5. API INTEGRATION & ENDPOINTS

### 5.1 Transparency Dashboard API

**Location**: `/home/user/RRRalgorithms/src/api/main.py`

**Framework**: FastAPI

**Key Endpoints**:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/api/portfolio` | GET | Portfolio overview |
| `/api/portfolio/positions` | GET | Open positions |
| `/api/trades` | GET | Trade history (paginated) |
| `/api/performance` | GET | Performance metrics |
| `/api/performance/equity-curve` | GET | Equity curve data |
| `/api/ai/decisions` | GET | AI predictions (paginated) |
| `/api/ai/models` | GET | Active model info |
| `/api/backtests` | GET | Backtest results |
| `/api/backtests/{id}` | GET | Detailed backtest results |
| `/api/stats` | GET | System-wide statistics |

**Status**: Endpoints are currently returning mock data. Database connections need to be implemented.

### 5.2 WebSocket Server

**Location**: `/home/user/RRRalgorithms/src/api/websocket_server.py`

Real-time streaming for:
- Market data updates
- Trade executions
- Order updates
- Portfolio changes

---

## 6. SYSTEM CONFIGURATION

### 6.1 Configuration Files

**Main Settings**: `/home/user/RRRalgorithms/src/core/settings.py`
- Pydantic BaseSettings with environment variable validation
- All config validated at startup

**Config Loader**: `/home/user/RRRalgorithms/src/core/config.py`
- Legacy configuration loader
- Project root discovery
- Environment file loading

### 6.2 Environment Variables Required

```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/trading_db    # PostgreSQL
SUPABASE_URL=https://xxx.supabase.co                             # Supabase URL
SUPABASE_KEY=...                                                  # Service key
DATABASE_PATH=...                                                 # Default DB path
DATABASE_TYPE=sqlite                                              # DB type

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=50

# API Keys
POLYGON_API_KEY=...
PERPLEXITY_API_KEY=...
ANTHROPIC_API_KEY=...
COINBASE_API_KEY=...
COINBASE_API_SECRET=...

# Trading Configuration
PAPER_TRADING=true
LIVE_TRADING=false
INITIAL_CAPITAL=100000.0

# Risk Management
MAX_POSITION_SIZE_PCT=0.20
MAX_DAILY_LOSS_PCT=0.05

# ML Configuration
MODEL_CACHE_SIZE=10
MODEL_INFERENCE_BATCH_SIZE=32
CUDA_VISIBLE_DEVICES=0
```

---

## 7. DATA FLOW & INTEGRATION POINTS

### 7.1 Data Ingestion Pipeline

```
Polygon.io (WebSocket)
    ↓
Data Processors (normalization, quality check)
    ↓
SQLite/Supabase Storage
    ↓
Feature Engineering (technical indicators)
    ↓
ML Prediction Pipeline
    ↓
Trading Engine (signals)
    ↓
Order Execution
    ↓
Position Tracking
    ↓
Portfolio Snapshots & API
```

### 7.2 Real-time Data Flow

```
Market Data Source
    ↓
MockDataSource (for testing) or Real API
    ↓
Async Trading Engine (async_trading_engine.py)
    ↓
Database Batch Insert
    ↓
WebSocket Broadcasting
    ↓
Dashboard & Mobile UI
```

---

## 8. WHAT NEEDS TO BE CONNECTED TO DATABASE

### 8.1 Mock Data → Real Database

**Currently Using**: `MockDataSource` class generates synthetic OHLCV data

**To Replace With**:
1. Real Polygon.io WebSocket data
2. Real TradingView webhooks
3. On-chain data from Etherscan/blockchain APIs
4. Direct database insertion of market data

### 8.2 API Endpoints → Database Queries

**Current Status**: All API endpoints return hardcoded mock data

**Lines with TODO comments**:
- Line 18: `database_connected = False` (needs real connection)
- Line 28: `# TODO: Initialize database connection`
- Line 107: `# TODO: Connect to actual database` (Portfolio endpoint)
- Line 130: `# TODO: Connect to actual database` (Positions endpoint)
- Line 182: `# TODO: Connect to actual database` (Trades endpoint)
- Line 234: `# TODO: Connect to actual database and calculate real metrics`
- Line 271: `# TODO: Connect to actual database`
- Line 315: `# TODO: Connect to actual database`
- Line 374: `# TODO: Connect to actual database`
- Line 425: `# TODO: Connect to actual database`
- Line 470: `# TODO: Connect to actual database`
- Line 513: `# TODO: Connect to actual database`

**Required Queries**:
```python
# Portfolio
SELECT SUM(quantity * current_price) FROM positions WHERE status='open'

# Positions
SELECT * FROM positions WHERE status='open' ORDER BY updated_at DESC

# Trades
SELECT * FROM trades ORDER BY timestamp DESC LIMIT ? OFFSET ?

# Performance Metrics
SELECT AVG(total_pnl), MAX(total_pnl), MIN(total_pnl), 
       COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 END) / COUNT(*) as win_rate

# AI Decisions
SELECT * FROM ml_predictions WHERE symbol=? ORDER BY timestamp DESC LIMIT ?

# Models
SELECT * FROM ml_models WHERE active=1

# Backtests
SELECT * FROM backtest_runs ORDER BY created_at DESC LIMIT ?

# System Stats
SELECT COUNT(*) as trades_count, SUM(volume) as volume_sum
```

### 8.3 ML Models → Database Registration

**Missing**: Model training and registration pipeline

**What's Needed**:
1. Train neural network models (Transformer, LSTM, CNN)
2. Save model files/checkpoints
3. Register in `ml_models` table
4. Log hyperparameters and metrics
5. Create prediction insertion pipeline
6. Cache predictions in Redis for API response

---

## 9. SUMMARY OF FILES & LOCATIONS

### Data Models
- `/home/user/RRRalgorithms/src/data_pipeline/polygon/models.py` - Pydantic models

### Database
- `/home/user/RRRalgorithms/src/database/` - Database client abstraction
- `/home/user/RRRalgorithms/src/database/sqlite_client.py` - SQLite implementation
- `/home/user/RRRalgorithms/src/database/client_factory.py` - Factory pattern
- `/home/user/RRRalgorithms/src/database/schema.sql` - SQLite schema
- `/home/user/RRRalgorithms/config/database/schema.sql` - PostgreSQL schema
- `/home/user/RRRalgorithms/config/supabase/schema.sql` - Supabase schema

### Mock Data
- `/home/user/RRRalgorithms/src/data_pipeline/mock_data_source.py` - Mock generators

### Configuration
- `/home/user/RRRalgorithms/src/core/settings.py` - Pydantic settings
- `/home/user/RRRalgorithms/src/core/config.py` - Config loader

### API
- `/home/user/RRRalgorithms/src/api/main.py` - FastAPI app with TODO endpoints

### Trading Engine
- `/home/user/RRRalgorithms/src/services/trading_engine/main.py` - Main engine
- `/home/user/RRRalgorithms/src/services/trading_engine/oms/order_manager.py` - Orders
- `/home/user/RRRalgorithms/src/services/trading_engine/positions/position_manager.py` - Positions
- `/home/user/RRRalgorithms/src/services/trading_engine/portfolio/portfolio_manager.py` - Portfolio

### ML Infrastructure
- `/home/user/RRRalgorithms/src/microservices/ml_service.py` - ML service
- `/home/user/RRRalgorithms/src/services/neural_network/features/technical_indicators.py` - Features
- `/home/user/RRRalgorithms/src/services/neural_network/utils/data_loader.py` - Data loading

### Database Client Libraries
- `/home/user/RRRalgorithms/src/data_pipeline/supabase_client.py` - Supabase wrapper
- `/home/user/RRRalgorithms/src/core/database.py` - Database pool manager

---

## 10. NEXT STEPS FOR IMPLEMENTATION

1. **Implement database connection in API**: Replace mock data with real database queries
2. **Train and register ML models**: Create model registry and prediction pipeline
3. **Integrate real market data**: Connect Polygon.io and other data sources
4. **Cache layer**: Implement Redis caching for API responses
5. **Real-time subscriptions**: Enable WebSocket connections for live updates
6. **Audit logging**: Implement comprehensive audit trail
7. **Error handling**: Add transaction rollback and error recovery
8. **Testing**: Unit tests for all data access layers

