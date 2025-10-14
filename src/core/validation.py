"""
Input Validation Framework
===========================

Pydantic-based validation for all system inputs.
Prevents invalid data from entering the system.

Usage:
    from src.core.validation import TradeRequest

    trade = TradeRequest(
        symbol="BTC-USD",
        side="buy",
        quantity=1.0,
        price=50000.0
    )
"""

from datetime import datetime
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, root_validator, validator

from src.core.constants import (
    ValidationConstants,
    OrderSide,
    OrderType,
    OrderStatus,
    SignalDirection,
)


# =============================================================================
# Base Models
# =============================================================================

class StrictBaseModel(BaseModel):
    """Base model with strict validation"""
    
    class Config:
        # Strict validation
        validate_assignment = True
        str_strip_whitespace = True
        str_min_length = 1
        # Allow using enums
        use_enum_values = True
        # Validate on attribute assignment
        validate_default = True


# =============================================================================
# Market Data Validation
# =============================================================================

class OHLCVData(StrictBaseModel):
    """OHLCV market data validation"""
    
    open: float = Field(
        ...,
        gt=ValidationConstants.MIN_PRICE,
        lt=ValidationConstants.MAX_PRICE,
        description="Opening price"
    )
    high: float = Field(
        ...,
        gt=ValidationConstants.MIN_PRICE,
        lt=ValidationConstants.MAX_PRICE,
        description="High price"
    )
    low: float = Field(
        ...,
        gt=ValidationConstants.MIN_PRICE,
        lt=ValidationConstants.MAX_PRICE,
        description="Low price"
    )
    close: float = Field(
        ...,
        gt=ValidationConstants.MIN_PRICE,
        lt=ValidationConstants.MAX_PRICE,
        description="Closing price"
    )
    volume: float = Field(
        ...,
        ge=0,
        description="Trading volume"
    )
    
    @validator('high')
    def high_must_be_highest(cls, v, values):
        """Validate high is the highest price"""
        if 'low' in values and v < values['low']:
            raise ValueError('High must be >= low')
        if 'open' in values and v < values['open']:
            raise ValueError('High must be >= open')
        if 'close' in values and v < values['close']:
            raise ValueError('High must be >= close')
        return v
    
    @validator('low')
    def low_must_be_lowest(cls, v, values):
        """Validate low is the lowest price"""
        if 'open' in values and v > values['open']:
            raise ValueError('Low must be <= open')
        if 'close' in values and v > values['close']:
            raise ValueError('Low must be <= close')
        return v


class MarketDataInput(StrictBaseModel):
    """Market data input validation"""
    
    symbol: str = Field(
        ...,
        min_length=ValidationConstants.MIN_SYMBOL_LENGTH,
        max_length=ValidationConstants.MAX_SYMBOL_LENGTH,
        description="Trading symbol"
    )
    timestamp: float = Field(
        ...,
        gt=0,
        description="Unix timestamp"
    )
    ohlcv: OHLCVData
    
    @validator('symbol')
    def symbol_must_be_valid(cls, v):
        """Validate symbol format"""
        if not re.match(ValidationConstants.VALID_SYMBOL_PATTERN, v):
            raise ValueError(f"Invalid symbol format: {v}")
        return v.upper()
    
    @validator('timestamp')
    def timestamp_must_be_reasonable(cls, v):
        """Validate timestamp is not too far in past or future"""
        now = datetime.now().timestamp()
        # Allow up to 1 day in the past, 1 hour in the future
        if v < now - 86400:
            raise ValueError("Timestamp is too far in the past")
        if v > now + 3600:
            raise ValueError("Timestamp is in the future")
        return v


# =============================================================================
# Trading Request Validation
# =============================================================================

class TradeRequest(StrictBaseModel):
    """Trade execution request validation"""
    
    symbol: str = Field(
        ...,
        min_length=ValidationConstants.MIN_SYMBOL_LENGTH,
        max_length=ValidationConstants.MAX_SYMBOL_LENGTH
    )
    side: str = Field(
        ...,
        description="Order side: buy or sell"
    )
    order_type: str = Field(
        ...,
        description="Order type: market, limit, stop, stop_limit"
    )
    quantity: float = Field(
        ...,
        gt=ValidationConstants.MIN_QUANTITY,
        lt=ValidationConstants.MAX_QUANTITY,
        description="Order quantity"
    )
    price: Optional[float] = Field(
        None,
        gt=ValidationConstants.MIN_PRICE,
        lt=ValidationConstants.MAX_PRICE,
        description="Order price (required for limit orders)"
    )
    stop_price: Optional[float] = Field(
        None,
        gt=ValidationConstants.MIN_PRICE,
        lt=ValidationConstants.MAX_PRICE,
        description="Stop price (required for stop orders)"
    )
    timestamp: float = Field(
        default_factory=lambda: datetime.now().timestamp(),
        description="Order timestamp"
    )
    strategy: Optional[str] = Field(
        None,
        max_length=ValidationConstants.MAX_STRATEGY_NAME_LENGTH,
        description="Strategy name"
    )
    notes: Optional[str] = Field(
        None,
        max_length=ValidationConstants.MAX_NOTES_LENGTH,
        description="Additional notes"
    )
    
    @validator('symbol')
    def validate_symbol(cls, v):
        """Validate symbol format"""
        if not re.match(ValidationConstants.VALID_SYMBOL_PATTERN, v):
            raise ValueError(f"Invalid symbol format: {v}")
        return v.upper()
    
    @validator('side')
    def validate_side(cls, v):
        """Validate order side"""
        valid_sides = [e.value for e in OrderSide]
        if v.lower() not in valid_sides:
            raise ValueError(f"Side must be one of: {valid_sides}")
        return v.lower()
    
    @validator('order_type')
    def validate_order_type(cls, v):
        """Validate order type"""
        valid_types = [e.value for e in OrderType]
        if v.lower() not in valid_types:
            raise ValueError(f"Order type must be one of: {valid_types}")
        return v.lower()
    
    @root_validator(skip_on_failure=True)
    def validate_price_for_limit_orders(cls, values):
        """Validate price is provided for limit orders"""
        order_type = values.get('order_type')
        price = values.get('price')

        if order_type in ['limit', 'stop_limit'] and price is None:
            raise ValueError(f"{order_type} orders require a price")
        
        return values
    
    @root_validator(skip_on_failure=True)
    def validate_stop_price_for_stop_orders(cls, values):
        """Validate stop_price is provided for stop orders"""
        order_type = values.get('order_type')
        stop_price = values.get('stop_price')
        
        if order_type in ['stop', 'stop_limit'] and stop_price is None:
            raise ValueError(f"{order_type} orders require a stop_price")
        
        return values


class TradeUpdateRequest(StrictBaseModel):
    """Trade update request validation"""
    
    status: Optional[str] = Field(None)
    executed_quantity: Optional[float] = Field(None, ge=0)
    executed_price: Optional[float] = Field(None, gt=0)
    commission: Optional[float] = Field(None, ge=0)
    pnl: Optional[float] = None
    notes: Optional[str] = Field(None, max_length=ValidationConstants.MAX_NOTES_LENGTH)
    
    @validator('status')
    def validate_status(cls, v):
        """Validate order status"""
        if v is not None:
            valid_statuses = [e.value for e in OrderStatus]
            if v.lower() not in valid_statuses:
                raise ValueError(f"Status must be one of: {valid_statuses}")
        return v.lower() if v else None
    
    @root_validator(skip_on_failure=True)
    def at_least_one_field(cls, values):
        """Ensure at least one field is being updated"""
        if all(v is None for v in values.values()):
            raise ValueError("At least one field must be provided for update")
        return values


# =============================================================================
# Position Validation
# =============================================================================

class PositionRequest(StrictBaseModel):
    """Position management request"""
    
    symbol: str = Field(
        ...,
        min_length=ValidationConstants.MIN_SYMBOL_LENGTH,
        max_length=ValidationConstants.MAX_SYMBOL_LENGTH
    )
    quantity: float = Field(
        ...,
        description="Position quantity (positive for long, negative for short)"
    )
    average_price: float = Field(
        ...,
        gt=ValidationConstants.MIN_PRICE,
        lt=ValidationConstants.MAX_PRICE,
        description="Average entry price"
    )
    current_price: Optional[float] = Field(
        None,
        gt=ValidationConstants.MIN_PRICE,
        lt=ValidationConstants.MAX_PRICE,
        description="Current market price"
    )
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not re.match(ValidationConstants.VALID_SYMBOL_PATTERN, v):
            raise ValueError(f"Invalid symbol format: {v}")
        return v.upper()


# =============================================================================
# ML Prediction Validation
# =============================================================================

class PredictionRequest(StrictBaseModel):
    """ML prediction request"""
    
    symbol: str = Field(
        ...,
        min_length=ValidationConstants.MIN_SYMBOL_LENGTH,
        max_length=ValidationConstants.MAX_SYMBOL_LENGTH
    )
    current_price: float = Field(
        ...,
        gt=ValidationConstants.MIN_PRICE,
        lt=ValidationConstants.MAX_PRICE
    )
    features: Optional[Dict[str, float]] = Field(
        default_factory=dict,
        description="Additional features for prediction"
    )
    horizon: Optional[int] = Field(
        1,
        gt=0,
        lt=1440,  # Max 1 day (1440 minutes)
        description="Prediction horizon in minutes"
    )
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not re.match(ValidationConstants.VALID_SYMBOL_PATTERN, v):
            raise ValueError(f"Invalid symbol format: {v}")
        return v.upper()


class PredictionOutput(StrictBaseModel):
    """ML prediction output validation"""
    
    symbol: str
    timestamp: float
    predicted_price: float = Field(..., gt=0)
    predicted_change: float
    direction: str
    confidence: float = Field(..., ge=0, le=1)
    horizon_minutes: Optional[int] = None
    model_type: Optional[str] = None
    
    @validator('direction')
    def validate_direction(cls, v):
        """Validate signal direction"""
        valid_directions = [e.value for e in SignalDirection]
        if v.lower() not in valid_directions:
            raise ValueError(f"Direction must be one of: {valid_directions}")
        return v.lower()


# =============================================================================
# Portfolio Metrics Validation
# =============================================================================

class PortfolioMetricsInput(StrictBaseModel):
    """Portfolio metrics validation"""
    
    timestamp: float = Field(..., gt=0)
    total_value: float = Field(..., gt=0)
    cash: float = Field(..., ge=0)
    positions_value: float = Field(..., ge=0)
    daily_pnl: float
    total_pnl: float
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = Field(None, ge=0, le=1)
    win_rate: Optional[float] = Field(None, ge=0, le=1)
    
    @root_validator(skip_on_failure=True)
    def validate_totals(cls, values):
        """Validate cash + positions = total"""
        total = values.get('total_value')
        cash = values.get('cash')
        positions = values.get('positions_value')
        
        if total and cash is not None and positions is not None:
            expected_total = cash + positions
            if abs(total - expected_total) > 0.01:  # Allow 1 cent rounding error
                raise ValueError(
                    f"Total value ({total}) doesn't match cash ({cash}) + "
                    f"positions ({positions})"
                )
        
        return values


# =============================================================================
# Configuration Validation
# =============================================================================

class TradingConfigInput(StrictBaseModel):
    """Trading configuration validation"""
    
    max_position_size: float = Field(
        ...,
        gt=0,
        le=1,
        description="Maximum position size as fraction of portfolio"
    )
    max_daily_loss: float = Field(
        ...,
        gt=0,
        le=1,
        description="Maximum daily loss as fraction of portfolio"
    )
    initial_capital: float = Field(
        ...,
        gt=0,
        description="Initial capital in dollars"
    )
    update_interval: float = Field(
        ...,
        gt=0,
        lt=3600,
        description="Update interval in seconds"
    )
    symbols: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="Trading symbols"
    )
    
    @validator('symbols', each_item=True)
    def validate_symbols(cls, v):
        """Validate each symbol"""
        if not re.match(ValidationConstants.VALID_SYMBOL_PATTERN, v):
            raise ValueError(f"Invalid symbol format: {v}")
        return v.upper()


# =============================================================================
# Helper Functions
# =============================================================================

def validate_trade_request(data: Dict[str, Any]) -> TradeRequest:
    """
    Validate trade request data.
    
    Args:
        data: Trade request data
        
    Returns:
        Validated TradeRequest
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        return TradeRequest(**data)
    except Exception as e:
        from src.core.exceptions import InputValidationError
        raise InputValidationError(
            message=f"Trade request validation failed: {str(e)}",
            details={'errors': str(e)}
        )


def validate_market_data(symbol: str, timestamp: float, ohlcv: Dict[str, float]) -> MarketDataInput:
    """
    Validate market data.
    
    Args:
        symbol: Trading symbol
        timestamp: Unix timestamp
        ohlcv: OHLCV data
        
    Returns:
        Validated MarketDataInput
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        return MarketDataInput(
            symbol=symbol,
            timestamp=timestamp,
            ohlcv=OHLCVData(**ohlcv)
        )
    except Exception as e:
        from src.core.exceptions import DataValidationError
        raise DataValidationError(
            message=f"Market data validation failed: {str(e)}",
            field='ohlcv',
            value=ohlcv
        )


def validate_prediction_request(data: Dict[str, Any]) -> PredictionRequest:
    """
    Validate prediction request.
    
    Args:
        data: Prediction request data
        
    Returns:
        Validated PredictionRequest
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        return PredictionRequest(**data)
    except Exception as e:
        from src.core.exceptions import InputValidationError
        raise InputValidationError(
            message=f"Prediction request validation failed: {str(e)}",
            details={'errors': str(e)}
        )


__all__ = [
    'StrictBaseModel',
    'OHLCVData',
    'MarketDataInput',
    'TradeRequest',
    'TradeUpdateRequest',
    'PositionRequest',
    'PredictionRequest',
    'PredictionOutput',
    'PortfolioMetricsInput',
    'TradingConfigInput',
    'validate_trade_request',
    'validate_market_data',
    'validate_prediction_request',
]
