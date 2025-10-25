"""
API Request/Response Models
Pydantic models for input validation and response serialization
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


class PeriodEnum(str, Enum):
    """Valid time periods"""
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    SEVEN_DAYS = "7d"
    THIRTY_DAYS = "30d"
    ALL = "all"


class IntervalEnum(str, Enum):
    """Valid data intervals"""
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"


class TradesQuery(BaseModel):
    """Query parameters for trades endpoint"""
    limit: int = Field(default=50, ge=1, le=500, description="Maximum number of trades to return")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")
    symbol: Optional[str] = Field(None, max_length=20, description="Filter by symbol")

    @validator('symbol')
    def validate_symbol(cls, v):
        if v:
            # Remove whitespace and convert to uppercase
            v = v.strip().upper()
            # Basic validation: should be alphanumeric with optional dash
            if not v.replace('-', '').isalnum():
                raise ValueError('Symbol must be alphanumeric')
        return v


class PerformanceQuery(BaseModel):
    """Query parameters for performance endpoint"""
    period: PeriodEnum = Field(default=PeriodEnum.ONE_DAY, description="Time period")


class EquityCurveQuery(BaseModel):
    """Query parameters for equity curve endpoint"""
    period: PeriodEnum = Field(default=PeriodEnum.SEVEN_DAYS, description="Time period")
    interval: IntervalEnum = Field(default=IntervalEnum.ONE_HOUR, description="Data interval")


class AIDecisionsQuery(BaseModel):
    """Query parameters for AI decisions endpoint"""
    limit: int = Field(default=50, ge=1, le=200, description="Maximum number of decisions")
    model: Optional[str] = Field(None, max_length=100, description="Filter by model name")


class BacktestsQuery(BaseModel):
    """Query parameters for backtests endpoint"""
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of backtests")


# Response Models

class PortfolioPosition(BaseModel):
    """Portfolio position model"""
    id: str
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    opened_at: str


class PortfolioSummary(BaseModel):
    """Portfolio summary response"""
    total_equity: float
    cash_balance: float
    invested: float
    total_pnl: float
    total_pnl_percent: float
    day_pnl: float
    day_pnl_percent: float
    positions_count: int
    open_orders: int
    timestamp: str


class Trade(BaseModel):
    """Trade model"""
    id: str
    timestamp: str
    symbol: str
    event_type: str
    side: Optional[str] = None
    size: Optional[float] = None
    price: Optional[float] = None
    pnl: Optional[float] = None


class TradesResponse(BaseModel):
    """Trades list response"""
    trades: List[Trade]
    total_count: int
    limit: int
    offset: int
    timestamp: str


class PerformanceMetrics(BaseModel):
    """Performance metrics response"""
    period: str
    total_return: float
    total_return_percent: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_percent: float
    win_rate: float
    profit_factor: float
    total_trades: int
    timestamp: str


class EquityPoint(BaseModel):
    """Equity curve data point"""
    timestamp: str
    equity: float


class EquityCurveResponse(BaseModel):
    """Equity curve response"""
    period: str
    interval: str
    data: List[EquityPoint]
    initial_equity: float
    current_equity: float
    timestamp: str


class AIDecision(BaseModel):
    """AI decision model"""
    id: str
    timestamp: str
    model_name: str
    symbol: str
    prediction: dict
    reasoning: str
    outcome: Optional[str] = None
    actual_return: Optional[float] = None
    features: dict


class AIDecisionsResponse(BaseModel):
    """AI decisions list response"""
    decisions: List[AIDecision]
    total_count: int
    limit: int
    timestamp: str


class AIModel(BaseModel):
    """AI model info"""
    name: str
    type: str
    status: str
    accuracy: float
    predictions_today: int
    avg_confidence: float
    win_rate: float


class AIModelsResponse(BaseModel):
    """AI models list response"""
    models: List[AIModel]
    total_count: int
    timestamp: str


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    database: str
    components: dict


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    message: str
    timestamp: str
    request_id: Optional[str] = None


# Login/Auth Models

class LoginRequest(BaseModel):
    """Login request"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)


class TokenResponse(BaseModel):
    """Token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshTokenRequest(BaseModel):
    """Refresh token request"""
    refresh_token: str
