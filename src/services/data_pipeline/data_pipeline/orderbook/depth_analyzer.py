from .binance_orderbook_client import OrderBookSnapshot
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional
import logging
import psycopg2

"""
Order Book Depth Analyzer

Analyzes order book imbalances and generates trading signals.
Implements Hypothesis 002: Order Book Imbalance Predicts Short-Term Returns.
"""



logger = logging.getLogger(__name__)


@dataclass
class ImbalanceSignal:
    """Trading signal generated from order book imbalance"""
    timestamp: datetime
    symbol: str
    signal: str  # 'LONG', 'SHORT', 'NEUTRAL'
    confidence: float  # 0.0 to 1.0
    bid_ask_ratio: float
    depth_imbalance: float
    reasoning: str
    
    # Signal persistence
    imbalance_duration_seconds: int = 0
    consecutive_imbalances: int = 1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'signal': self.signal,
            'confidence': self.confidence,
            'bid_ask_ratio': self.bid_ask_ratio,
            'depth_imbalance': self.depth_imbalance,
            'reasoning': self.reasoning,
            'imbalance_duration_seconds': self.imbalance_duration_seconds,
            'consecutive_imbalances': self.consecutive_imbalances
        }


class DepthAnalyzer:
    """
    Analyze order book depth and generate trading signals.
    
    Key signal: When bid/ask ratio > 2:1 (or < 0.5:1), price likely moves
    in the direction of the imbalance within 5-15 minutes.
    """
    
    def __init__(
        self,
        bullish_threshold: float = 2.0,  # Bid/ask ratio for long signal
        bearish_threshold: float = 0.5,  # Bid/ask ratio for short signal
        min_persistence_seconds: int = 60,  # Imbalance must persist this long
        db_connection_string: Optional[str] = None
    ):
        """
        Initialize depth analyzer.
        
        Args:
            bullish_threshold: Bid/ask ratio threshold for bullish signal
            bearish_threshold: Bid/ask ratio threshold for bearish signal
            min_persistence_seconds: Minimum imbalance duration for signal
            db_connection_string: PostgreSQL connection string (optional)
        """
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold
        self.min_persistence_seconds = min_persistence_seconds
        self.db_connection_string = db_connection_string
        
        # Track imbalance history for persistence detection
        self.imbalance_history: Dict[str, List[OrderBookSnapshot]] = {}
    
    def analyze_snapshot(self, snapshot: OrderBookSnapshot) -> ImbalanceSignal:
        """
        Analyze a single order book snapshot and generate signal.
        
        Args:
            snapshot: Current order book state
            
        Returns:
            ImbalanceSignal with trading recommendation
        """
        symbol = snapshot.symbol
        bid_ask_ratio = float(snapshot.bid_ask_ratio)
        depth_imbalance = float(snapshot.depth_imbalance)
        
        # Track snapshot in history
        if symbol not in self.imbalance_history:
            self.imbalance_history[symbol] = []
        self.imbalance_history[symbol].append(snapshot)
        
        # Keep only recent history (last 10 minutes)
        cutoff_time = datetime.utcnow() - timedelta(minutes=10)
        self.imbalance_history[symbol] = [
            s for s in self.imbalance_history[symbol]
            if s.timestamp >= cutoff_time
        ]
        
        # Check for persistent imbalance
        persistence_info = self._check_persistence(symbol, bid_ask_ratio)
        
        # Generate signal
        signal_type = 'NEUTRAL'
        confidence = 0.0
        reasoning = "No significant imbalance"
        
        if bid_ask_ratio > self.bullish_threshold:
            # Bullish imbalance (more bids than asks)
            signal_type = 'LONG'
            
            # Confidence based on:
            # 1. Magnitude of imbalance
            # 2. Persistence duration
            magnitude_factor = min((bid_ask_ratio - self.bullish_threshold) / 2.0, 0.5)
            persistence_factor = min(persistence_info['duration_seconds'] / 300.0, 0.3)
            consecutive_factor = min(persistence_info['consecutive'] / 10.0, 0.2)
            
            confidence = magnitude_factor + persistence_factor + consecutive_factor
            confidence = min(confidence, 0.75)  # Cap at 75%
            
            reasoning = (
                f"Bullish order book imbalance: {bid_ask_ratio:.2f}:1 bid/ask ratio "
                f"(threshold: {self.bullish_threshold}:1). "
                f"Persisted for {persistence_info['duration_seconds']}s "
                f"({persistence_info['consecutive']} consecutive snapshots)."
            )
            
        elif bid_ask_ratio < self.bearish_threshold:
            # Bearish imbalance (more asks than bids)
            signal_type = 'SHORT'
            
            # Confidence calculation (inverse of bullish)
            magnitude_factor = min((self.bearish_threshold - bid_ask_ratio) / 0.5, 0.5)
            persistence_factor = min(persistence_info['duration_seconds'] / 300.0, 0.3)
            consecutive_factor = min(persistence_info['consecutive'] / 10.0, 0.2)
            
            confidence = magnitude_factor + persistence_factor + consecutive_factor
            confidence = min(confidence, 0.75)
            
            reasoning = (
                f"Bearish order book imbalance: {bid_ask_ratio:.2f}:1 bid/ask ratio "
                f"(threshold: {self.bearish_threshold}:1). "
                f"Persisted for {persistence_info['duration_seconds']}s "
                f"({persistence_info['consecutive']} consecutive snapshots)."
            )
        
        return ImbalanceSignal(
            timestamp=snapshot.timestamp,
            symbol=symbol,
            signal=signal_type,
            confidence=confidence,
            bid_ask_ratio=bid_ask_ratio,
            depth_imbalance=depth_imbalance,
            reasoning=reasoning,
            imbalance_duration_seconds=persistence_info['duration_seconds'],
            consecutive_imbalances=persistence_info['consecutive']
        )
    
    def _check_persistence(self, symbol: str, current_ratio: float) -> Dict:
        """
        Check if imbalance has persisted over time.
        
        Returns:
            Dict with duration_seconds and consecutive count
        """
        if symbol not in self.imbalance_history or not self.imbalance_history[symbol]:
            return {'duration_seconds': 0, 'consecutive': 0}
        
        history = self.imbalance_history[symbol]
        
        # Count consecutive imbalances in same direction
        consecutive = 0
        for snapshot in reversed(history):
            ratio = float(snapshot.bid_ask_ratio)
            
            # Check if imbalance is in same direction
            same_direction = (
                (current_ratio > self.bullish_threshold and ratio > self.bullish_threshold) or
                (current_ratio < self.bearish_threshold and ratio < self.bearish_threshold)
            )
            
            if same_direction:
                consecutive += 1
            else:
                break
        
        # Calculate duration
        if consecutive > 0:
            first_snapshot = history[-consecutive]
            duration = (history[-1].timestamp - first_snapshot.timestamp).total_seconds()
        else:
            duration = 0
        
        return {
            'duration_seconds': int(duration),
            'consecutive': consecutive
        }
    
    def store_metrics(self, snapshot: OrderBookSnapshot):
        """
        Store order book metrics to database.
        
        Args:
            snapshot: Order book snapshot to store
        """
        if not self.db_connection_string:
            logger.warning("No database connection string provided, skipping storage")
            return
        
        try:
            conn = psycopg2.connect(self.db_connection_string)
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO order_book_metrics (
                    timestamp, asset, exchange,
                    bid_ask_ratio, depth_imbalance,
                    mid_price, spread_bps,
                    bid_depth_1pct, ask_depth_1pct
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (timestamp, asset, exchange) DO NOTHING
            """, (
                snapshot.timestamp,
                snapshot.symbol,
                'binance',
                float(snapshot.bid_ask_ratio),
                float(snapshot.depth_imbalance),
                float(snapshot.mid_price),
                float(snapshot.spread_bps),
                float(snapshot.bid_depth_1pct),
                float(snapshot.ask_depth_1pct)
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing order book metrics: {e}")
    
    def load_historical_metrics(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict]:
        """
        Load historical order book metrics from database.
        
        Args:
            symbol: Trading symbol
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            List of metric dictionaries
        """
        if not self.db_connection_string:
            raise ValueError("Database connection string required for historical data")
        
        try:
            conn = psycopg2.connect(self.db_connection_string)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("""
                SELECT *
                FROM order_book_metrics
                WHERE asset = %s
                  AND timestamp >= %s
                  AND timestamp <= %s
                ORDER BY timestamp ASC
            """, (symbol, start_time, end_time))
            
            metrics = cur.fetchall()
            
            cur.close()
            conn.close()
            
            return [dict(row) for row in metrics]
            
        except Exception as e:
            logger.error(f"Error loading historical metrics: {e}")
            return []
    
    def backtest_strategy(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        holding_period_minutes: int = 15
    ) -> Dict:
        """
        Backtest the order book imbalance strategy on historical data.
        
        Args:
            symbol: Trading symbol
            start_time: Start of backtest period
            end_time: End of backtest period
            holding_period_minutes: How long to hold positions
            
        Returns:
            Dict with backtest results (win rate, avg return, etc.)
        """
        # Load historical metrics
        metrics = self.load_historical_metrics(symbol, start_time, end_time)
        
        if not metrics:
            return {'error': 'No historical data available'}
        
        logger.info(f"Backtesting on {len(metrics)} historical snapshots")
        
        # TODO: Implement full backtest logic
        # This is a placeholder for the actual backtesting implementation
        # which would need price data to calculate actual returns
        
        trades = []
        for i, metric in enumerate(metrics):
            # Generate signal
            # (This is simplified - in reality we'd use OrderBookSnapshot objects)
            bid_ask_ratio = metric['bid_ask_ratio']
            
            if bid_ask_ratio > self.bullish_threshold:
                signal = 'LONG'
            elif bid_ask_ratio < self.bearish_threshold:
                signal = 'SHORT'
            else:
                continue
            
            # Would need to:
            # 1. Look ahead {holding_period_minutes} to get exit price
            # 2. Calculate return
            # 3. Track trade
            
            trades.append({
                'entry_time': metric['timestamp'],
                'signal': signal,
                'entry_price': metric['mid_price'],
                # 'exit_price': ...,  # Need price data
                # 'return': ...
            })
        
        return {
            'total_signals': len(trades),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'note': 'Full backtest requires price data integration'
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("Order Book Depth Analyzer Demo")
    print("="*80)
    print()
    
    # Create analyzer
    analyzer = DepthAnalyzer(
        bullish_threshold=2.0,
        bearish_threshold=0.5,
        min_persistence_seconds=60
    )
    
    # Simulate some order book snapshots
    from decimal import Decimal
    
    # Snapshot 1: Bullish imbalance
    snapshot1 = OrderBookSnapshot(
        symbol='BTCUSDT',
        timestamp=datetime.utcnow(),
        bids=[(Decimal('67000'), Decimal('10')), (Decimal('66990'), Decimal('5'))],
        asks=[(Decimal('67010'), Decimal('3')), (Decimal('67020'), Decimal('2'))]
    )
    
    signal1 = analyzer.analyze_snapshot(snapshot1)
    print(f"Signal 1: {signal1.signal}")
    print(f"  Confidence: {signal1.confidence:.2%}")
    print(f"  Bid/Ask Ratio: {signal1.bid_ask_ratio:.2f}:1")
    print(f"  Reasoning: {signal1.reasoning}")
    print()
    
    # Snapshot 2: Bearish imbalance
    snapshot2 = OrderBookSnapshot(
        symbol='BTCUSDT',
        timestamp=datetime.utcnow(),
        bids=[(Decimal('67000'), Decimal('2')), (Decimal('66990'), Decimal('1'))],
        asks=[(Decimal('67010'), Decimal('10')), (Decimal('67020'), Decimal('8'))]
    )
    
    signal2 = analyzer.analyze_snapshot(snapshot2)
    print(f"Signal 2: {signal2.signal}")
    print(f"  Confidence: {signal2.confidence:.2%}")
    print(f"  Bid/Ask Ratio: {signal2.bid_ask_ratio:.2f}:1")
    print(f"  Reasoning: {signal2.reasoning}")
    print()
    
    print("✅ Depth analyzer demo complete")


