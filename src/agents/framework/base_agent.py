from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache
from typing import List, Dict, Optional, Any


"""
Base Agent - Foundation for all specialized trading agents.

Every agent analyzes market state and produces a signal with confidence level.
Agents are specialized and focus on their domain of expertise.
"""



class SignalDirection(Enum):
    """Trading signal direction"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


@dataclass
class MarketState:
    """
    Complete market state snapshot passed to agents.
    
    Agents extract relevant information for their analysis.
    """
    timestamp: datetime
    asset: str  # 'BTC-USD', 'ETH-USD', etc.
    
    # Price data
    price: float
    volume_24h: float
    price_change_24h: float
    
    # Order book (if available)
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_depth: Optional[float] = None  # Total bids within 1% of mid
    ask_depth: Optional[float] = None  # Total asks within 1% of mid
    
    # On-chain data (if available)
    whale_flow_24h: Optional[float] = None  # Net whale flow to exchanges
    exchange_balance: Optional[float] = None  # Total on exchanges
    
    # Sentiment (if available)
    news_sentiment: Optional[float] = None  # -1 to +1
    social_sentiment: Optional[float] = None
    
    # Technical indicators
    rsi_14: Optional[float] = None
    macd: Optional[Dict[str, float]] = None
    
    # Market regime
    volatility: Optional[float] = None  # Annualized
    trend_strength: Optional[float] = None  # -1 to +1
    
    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentSignal:
    """
    Signal produced by an agent.
    
    Each agent expresses its view with direction, confidence, and reasoning.
    """
    agent_id: str
    agent_type: str  # 'technical', 'onchain', 'microstructure', etc.
    timestamp: datetime
    asset: str
    
    # Core signal
    direction: SignalDirection
    confidence: float  # 0.0 to 1.0
    
    # Supporting information
    reasoning: str  # Human-readable explanation
    evidence: List[str] = field(default_factory=list)  # Key facts supporting signal
    
    # Position sizing recommendation (optional)
    suggested_position_size: Optional[float] = None  # Fraction of portfolio (0-1)
    
    # Risk assessment
    risk_level: str = "MEDIUM"  # LOW, MEDIUM, HIGH
    stop_loss: Optional[float] = None  # Suggested stop loss price
    take_profit: Optional[float] = None  # Suggested take profit price
    
    # Agent-specific data
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/storage"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'timestamp': self.timestamp.isoformat(),
            'asset': self.asset,
            'direction': self.direction.value,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'evidence': self.evidence,
            'suggested_position_size': self.suggested_position_size,
            'risk_level': self.risk_level,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }


class BaseAgent(ABC):
    """
    Abstract base class for all trading agents.
    
    Agents are specialized experts that analyze specific aspects of the market.
    They run independently and their signals are aggregated by the coordinator.
    """
    
    def __init__(self, agent_id: str, agent_type: str):
        """
        Initialize agent.
        
        Args:
            agent_id: Unique identifier (e.g., 'technical_001')
            agent_type: Agent category (e.g., 'technical', 'onchain')
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.enabled = True
        
        # Performance tracking
        self.signals_generated = 0
        self.last_signal_time: Optional[datetime] = None
    
    @abstractmethod
    def analyze(self, market_state: MarketState) -> AgentSignal:
        """
        Analyze market state and produce trading signal.
        
        This is the core method each agent must implement.
        
        Args:
            market_state: Complete market data snapshot
            
        Returns:
            AgentSignal with direction, confidence, and reasoning
        """
        pass
    
    def can_analyze(self, market_state: MarketState) -> bool:
        """
        Check if agent has sufficient data to analyze.
        
        Agents should return False if required data is missing.
        This prevents agents from producing signals without proper information.
        
        Args:
            market_state: Market data to check
            
        Returns:
            True if agent can analyze, False otherwise
        """
        # Default: can analyze if enabled
        return self.enabled
    
    @lru_cache(maxsize=128)
    
    def get_confidence_adjustment(self, market_state: MarketState) -> float:
        """
        Calculate confidence adjustment based on market conditions.
        
        Agents may reduce confidence in certain market regimes:
        - High volatility may reduce confidence for some strategies
        - Low liquidity may reduce confidence for execution
        - Market regime mismatches may reduce confidence
        
        Args:
            market_state: Current market conditions
            
        Returns:
            Multiplier for confidence (0.5-1.0 typically)
        """
        return 1.0  # Default: no adjustment
    
    @lru_cache(maxsize=128)
    
    def get_status(self) -> Dict:
        """Get agent status and statistics"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'enabled': self.enabled,
            'signals_generated': self.signals_generated,
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None
        }
    
    def enable(self):
        """Enable agent"""
        self.enabled = True
    
    def disable(self):
        """Disable agent"""
        self.enabled = False
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, type={self.agent_type})"


class SimpleThresholdAgent(BaseAgent):
    """
    Example implementation: Simple threshold-based agent.
    
    This demonstrates how to subclass BaseAgent.
    """
    
    def __init__(
        self,
        agent_id: str = "threshold_001",
        threshold_high: float = 70.0,
        threshold_low: float = 30.0
    ):
        super().__init__(agent_id, agent_type="technical")
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
    
    def analyze(self, market_state: MarketState) -> AgentSignal:
        """Simple RSI-based signal"""
        
        # Check if we have RSI data
        if market_state.rsi_14 is None:
            return AgentSignal(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                timestamp=market_state.timestamp,
                asset=market_state.asset,
                direction=SignalDirection.NEUTRAL,
                confidence=0.0,
                reasoning="Insufficient data (RSI not available)"
            )
        
        rsi = market_state.rsi_14
        
        # Generate signal based on RSI thresholds
        if rsi > self.threshold_high:
            direction = SignalDirection.SHORT
            confidence = min((rsi - self.threshold_high) / 30.0, 1.0)
            reasoning = f"RSI overbought: {rsi:.1f} > {self.threshold_high}"
            evidence = [f"RSI = {rsi:.1f}", "Overbought condition"]
            
        elif rsi < self.threshold_low:
            direction = SignalDirection.LONG
            confidence = min((self.threshold_low - rsi) / 30.0, 1.0)
            reasoning = f"RSI oversold: {rsi:.1f} < {self.threshold_low}"
            evidence = [f"RSI = {rsi:.1f}", "Oversold condition"]
            
        else:
            direction = SignalDirection.NEUTRAL
            confidence = 0.0
            reasoning = f"RSI neutral: {rsi:.1f}"
            evidence = [f"RSI = {rsi:.1f}"]
        
        self.signals_generated += 1
        self.last_signal_time = market_state.timestamp
        
        return AgentSignal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            timestamp=market_state.timestamp,
            asset=market_state.asset,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            risk_level="MEDIUM"
        )


# Example usage
if __name__ == "__main__":
    # Create test market state
    market_state = MarketState(
        timestamp=datetime.utcnow(),
        asset="BTC-USD",
        price=67000.0,
        volume_24h=25000000000.0,
        price_change_24h=-2.3,
        rsi_14=75.5,  # Overbought
        volatility=0.65
    )
    
    # Create agent
    agent = SimpleThresholdAgent()
    
    print("="*80)
    print("Base Agent Demo")
    print("="*80)
    print()
    
    print(f"Agent: {agent}")
    print(f"Status: {agent.get_status()}")
    print()
    
    # Generate signal
    print("Analyzing market state...")
    signal = agent.analyze(market_state)
    
    print()
    print(f"Signal: {signal.direction.value}")
    print(f"Confidence: {signal.confidence:.2%}")
    print(f"Reasoning: {signal.reasoning}")
    print(f"Evidence: {', '.join(signal.evidence)}")
    print()
    
    print("✅ Base agent demo complete")

