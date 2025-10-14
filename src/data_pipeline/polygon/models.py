from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field
from typing import Optional, List

"""
Pydantic models for Polygon.io API responses.
"""



class Aggregate(BaseModel):
    """OHLCV aggregate bar (candlestick) data."""

    ticker: Optional[str] = Field(None, description="Ticker symbol (e.g., X:BTCUSD)")
    timestamp: int = Field(..., alias="t", description="Unix timestamp (ms)")
    open: Decimal = Field(..., alias="o", description="Open price")
    high: Decimal = Field(..., alias="h", description="High price")
    low: Decimal = Field(..., alias="l", description="Low price")
    close: Decimal = Field(..., alias="c", description="Close price")
    volume: Decimal = Field(..., alias="v", description="Trading volume")
    vwap: Optional[Decimal] = Field(None, alias="vw", description="Volume weighted average price")
    trade_count: Optional[int] = Field(None, alias="n", description="Number of trades")

    class Config:
        populate_by_name = True

    @property
    def datetime(self) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(self.timestamp / 1000)


class Trade(BaseModel):
    """Individual trade execution."""

    ticker: str = Field(..., alias="sym", description="Ticker symbol")
    timestamp: int = Field(..., alias="t", description="Unix timestamp (ns)")
    price: Decimal = Field(..., alias="p", description="Trade price")
    size: Decimal = Field(..., alias="s", description="Trade size")
    exchange: Optional[int] = Field(None, alias="x", description="Exchange ID")
    conditions: Optional[List[int]] = Field(None, alias="c", description="Trade conditions")
    id: Optional[str] = Field(None, alias="i", description="Trade ID")

    class Config:
        populate_by_name = True

    @property
    def datetime(self) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(self.timestamp / 1_000_000_000)


class Quote(BaseModel):
    """Bid/Ask quote."""

    ticker: str = Field(..., alias="sym", description="Ticker symbol")
    timestamp: int = Field(..., alias="t", description="Unix timestamp (ns)")
    bid_price: Decimal = Field(..., alias="bp", description="Bid price")
    bid_size: Decimal = Field(..., alias="bs", description="Bid size")
    ask_price: Decimal = Field(..., alias="ap", description="Ask price")
    ask_size: Decimal = Field(..., alias="as", description="Ask size")
    exchange: Optional[int] = Field(None, alias="x", description="Exchange ID")

    class Config:
        populate_by_name = True

    @property
    def datetime(self) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(self.timestamp / 1_000_000_000)

    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        return self.ask_price - self.bid_price

    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price."""
        return (self.bid_price + self.ask_price) / 2


class TickerDetails(BaseModel):
    """Ticker/symbol details and metadata."""

    ticker: str = Field(..., description="Ticker symbol")
    name: str = Field(..., description="Asset name")
    market: str = Field(..., description="Market type (crypto, stocks, etc.)")
    locale: str = Field(..., description="Locale (us, global)")
    primary_exchange: Optional[str] = Field(None, description="Primary exchange")
    type: str = Field(..., description="Asset type")
    active: bool = Field(..., description="Whether ticker is active")
    currency_name: Optional[str] = Field(None, description="Currency name (for crypto)")
    base_currency_symbol: Optional[str] = Field(None, description="Base currency")
    base_currency_name: Optional[str] = Field(None, description="Base currency name")

    class Config:
        populate_by_name = True


class MarketStatus(BaseModel):
    """Market status (open/closed)."""

    market: str = Field(..., description="Market name")
    server_time: datetime = Field(..., alias="serverTime")
    exchanges: dict = Field(..., description="Exchange statuses")
    currencies: dict = Field(..., description="Currency market statuses")

    class Config:
        populate_by_name = True


class AggregatesResponse(BaseModel):
    """Response wrapper for aggregates endpoint."""

    ticker: str
    query_count: int = Field(..., alias="queryCount")
    results_count: int = Field(..., alias="resultsCount")
    adjusted: bool
    results: List[Aggregate]
    status: str
    request_id: Optional[str] = Field(None, alias="request_id")
    count: Optional[int] = None

    class Config:
        populate_by_name = True


class TradesResponse(BaseModel):
    """Response wrapper for trades endpoint."""

    results: List[Trade]
    status: str
    request_id: str = Field(..., alias="request_id")
    count: int

    class Config:
        populate_by_name = True


class LastTradeResponse(BaseModel):
    """Response for last trade endpoint."""

    status: str
    ticker: str = Field(..., alias="symbol")
    last: Trade
    request_id: Optional[str] = Field(None, alias="request_id")

    class Config:
        populate_by_name = True


class LastQuoteResponse(BaseModel):
    """Response for last quote endpoint."""

    status: str
    ticker: str = Field(..., alias="symbol")
    last: Quote
    request_id: Optional[str] = Field(None, alias="request_id")

    class Config:
        populate_by_name = True
