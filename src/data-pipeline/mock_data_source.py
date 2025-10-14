from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional
import math
import numpy as np
import pandas as pd
import random
import time


"""
Mock data source for local development.
Generates realistic cryptocurrency market data without requiring API keys.
"""



class MockDataSource:
    """
    Generate realistic mock market data for testing.
    Simulates OHLCV data with realistic price movements.
    """
    
    def __init__(
        self,
        symbols: List[str] = None,
        base_prices: Optional[Dict[str, float]] = None,
        volatility: float = 0.02,
        update_interval: float = 1.0,
        trend_strength: float = 0.0001
    ):
        """
        Initialize mock data source.
        
        Args:
            symbols: List of trading pairs (e.g., ['BTC-USD', 'ETH-USD'])
            base_prices: Starting prices for each symbol
            volatility: Price volatility (0.02 = 2% standard deviation)
            update_interval: Seconds between updates
            trend_strength: Trend component strength
        """
        self.symbols = symbols or ['BTC-USD', 'ETH-USD']
        self.volatility = volatility
        self.update_interval = update_interval
        self.trend_strength = trend_strength
        
        # Set base prices
        default_prices = {
            'BTC-USD': 50000.0,
            'ETH-USD': 3000.0,
            'SOL-USD': 100.0,
            'MATIC-USD': 0.80,
            'AVAX-USD': 35.0,
        }
        
        self.base_prices = base_prices or default_prices
        self.current_prices = {s: self.base_prices.get(s, 100.0) for s in self.symbols}
        
        # Track state for each symbol
        self.trends = {s: 0.0 for s in self.symbols}
        self.last_update = {s: time.time() for s in self.symbols}
        
        # Volume parameters
        self.base_volumes = {
            'BTC-USD': 1000000,
            'ETH-USD': 500000,
            'SOL-USD': 100000,
            'MATIC-USD': 50000,
            'AVAX-USD': 75000,
        }
    
    def _generate_price_movement(self, symbol: str) -> float:
        """Generate realistic price movement."""
        # Geometric Brownian Motion with trend
        dt = self.update_interval / 86400  # Convert to days
        
        # Random walk component
        random_component = random.gauss(0, self.volatility * math.sqrt(dt))
        
        # Trend component (mean-reverting with drift)
        self.trends[symbol] = 0.95 * self.trends[symbol] + random.gauss(0, 0.0001)
        trend_component = self.trends[symbol] * dt
        
        # Combine components
        total_change = random_component + trend_component
        
        return total_change
    
    def _generate_ohlcv(self, symbol: str) -> Dict[str, float]:
        """Generate OHLCV bar for a symbol."""
        # Update price
        price_change = self._generate_price_movement(symbol)
        new_price = self.current_prices[symbol] * (1 + price_change)
        
        # Ensure price doesn't go negative
        new_price = max(new_price, 0.01)
        
        # Generate OHLC with realistic micro-structure
        open_price = self.current_prices[symbol]
        close_price = new_price
        
        # High and low with some randomness
        high_variation = abs(random.gauss(0, self.volatility * 0.5))
        low_variation = abs(random.gauss(0, self.volatility * 0.5))
        
        high_price = max(open_price, close_price) * (1 + high_variation)
        low_price = min(open_price, close_price) * (1 - low_variation)
        
        # Ensure OHLC consistency
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Generate volume with some randomness
        base_volume = self.base_volumes.get(symbol, 100000)
        volume = base_volume * random.uniform(0.7, 1.3)
        
        # Update current price
        self.current_prices[symbol] = close_price
        self.last_update[symbol] = time.time()
        
        return {
            'symbol': symbol,
            'timestamp': time.time(),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': round(volume, 2)
        }
    
    @lru_cache(maxsize=128)
    
    def get_latest_data(self, symbol: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Get latest market data for symbol(s).
        
        Args:
            symbol: Specific symbol or None for all symbols
        
        Returns:
            Dictionary of symbol -> OHLCV data
        """
        if symbol:
            return {symbol: self._generate_ohlcv(symbol)}
        else:
            return {s: self._generate_ohlcv(s) for s in self.symbols}
    
    @lru_cache(maxsize=128)
    
    def get_historical_data(
        self,
        symbol: str,
        periods: int = 100,
        interval_seconds: int = 60
    ) -> pd.DataFrame:
        """
        Generate historical data for backtesting.
        
        Args:
            symbol: Trading symbol
            periods: Number of periods to generate
            interval_seconds: Seconds per bar
        
        Returns:
            DataFrame with OHLCV data
        """
        # Reset to base price for consistency
        self.current_prices[symbol] = self.base_prices.get(symbol, 100.0)
        
        data = []
        current_time = time.time() - (periods * interval_seconds)
        
        for i in range(periods):
            ohlcv = self._generate_ohlcv(symbol)
            ohlcv['timestamp'] = current_time + (i * interval_seconds)
            data.append(ohlcv)
        
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        return df
    
    def stream(self, callback, interval: Optional[float] = None):
        """
        Stream real-time data updates.
        
        Args:
            callback: Function to call with each update
            interval: Update interval (uses self.update_interval if None)
        """
        interval = interval or self.update_interval
        
        try:
            while True:
                data = self.get_latest_data()
                callback(data)
                time.sleep(interval)
        except KeyboardInterrupt:
            pass


class MockOrderBook:
    """Generate realistic order book data."""
    
    def __init__(self, symbol: str, mid_price: float, spread_pct: float = 0.001):
        """
        Initialize mock order book.
        
        Args:
            symbol: Trading symbol
            mid_price: Current mid price
            spread_pct: Bid-ask spread as percentage
        """
        self.symbol = symbol
        self.mid_price = mid_price
        self.spread_pct = spread_pct
    
    @lru_cache(maxsize=128)
    
    def get_orderbook(self, levels: int = 10) -> Dict[str, List[List[float]]]:
        """
        Generate order book with specified depth.
        
        Args:
            levels: Number of price levels
        
        Returns:
            Dictionary with 'bids' and 'asks' lists
        """
        spread = self.mid_price * self.spread_pct
        best_bid = self.mid_price - spread / 2
        best_ask = self.mid_price + spread / 2
        
        bids = []
        asks = []
        
        for i in range(levels):
            # Bids (decreasing prices)
            bid_price = best_bid * (1 - i * 0.0005)
            bid_size = random.uniform(0.1, 2.0) * (1 + i * 0.5)
            bids.append([bid_price, bid_size])
            
            # Asks (increasing prices)
            ask_price = best_ask * (1 + i * 0.0005)
            ask_size = random.uniform(0.1, 2.0) * (1 + i * 0.5)
            asks.append([ask_price, ask_size])
        
        return {
            'symbol': self.symbol,
            'timestamp': time.time(),
            'bids': bids,
            'asks': asks
        }


class MockSentimentData:
    """Generate mock sentiment data for testing."""
    
    def __init__(self):
        self.base_sentiment = 0.5  # Neutral
        self.sentiment_drift = 0.0
    
    @lru_cache(maxsize=128)
    
    def get_sentiment(self, symbol: str) -> Dict[str, float]:
        """
        Generate sentiment score.
        
        Returns:
            Dictionary with sentiment metrics
        """
        # Mean-reverting random walk
        self.sentiment_drift = 0.9 * self.sentiment_drift + random.gauss(0, 0.05)
        sentiment = np.clip(self.base_sentiment + self.sentiment_drift, 0, 1)
        
        return {
            'symbol': symbol,
            'timestamp': time.time(),
            'sentiment': sentiment,
            'confidence': random.uniform(0.6, 0.9),
            'sources_count': random.randint(10, 100),
            'bullish_count': int(sentiment * 100),
            'bearish_count': int((1 - sentiment) * 100)
        }


if __name__ == "__main__":
    # Test the mock data source
    print("🧪 Testing Mock Data Source\n")
    
    # Create mock data source
    mock = MockDataSource(symbols=['BTC-USD', 'ETH-USD'])
    
    # Get latest data
    print("📊 Latest Market Data:")
    data = mock.get_latest_data()
    for symbol, ohlcv in data.items():
        print(f"\n{symbol}:")
        print(f"  Open:   ${ohlcv['open']:,.2f}")
        print(f"  High:   ${ohlcv['high']:,.2f}")
        print(f"  Low:    ${ohlcv['low']:,.2f}")
        print(f"  Close:  ${ohlcv['close']:,.2f}")
        print(f"  Volume: {ohlcv['volume']:,.2f}")
    
    # Generate historical data
    print("\n📈 Historical Data:")
    hist = mock.get_historical_data('BTC-USD', periods=10)
    print(hist[['datetime', 'open', 'high', 'low', 'close', 'volume']].to_string(index=False))
    
    # Test order book
    print("\n📖 Order Book:")
    orderbook = MockOrderBook('BTC-USD', 50000)
    book = orderbook.get_orderbook(levels=5)
    print(f"\nTop 5 Bids:")
    for price, size in book['bids'][:5]:
        print(f"  ${price:,.2f} × {size:.4f}")
    print(f"\nTop 5 Asks:")
    for price, size in book['asks'][:5]:
        print(f"  ${price:,.2f} × {size:.4f}")
    
    # Test sentiment
    print("\n💭 Sentiment Data:")
    sentiment = MockSentimentData()
    sent_data = sentiment.get_sentiment('BTC-USD')
    print(f"  Sentiment: {sent_data['sentiment']:.2%}")
    print(f"  Confidence: {sent_data['confidence']:.2%}")
    print(f"  Bullish: {sent_data['bullish_count']} | Bearish: {sent_data['bearish_count']}")

