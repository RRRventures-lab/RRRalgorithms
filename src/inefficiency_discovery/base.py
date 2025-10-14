from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd


"""
Base classes for inefficiency detection system
"""



class InefficiencyType(Enum):
    """Types of market inefficiencies"""
    LATENCY_ARBITRAGE = "latency_arbitrage"
    FUNDING_RATE = "funding_rate"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    SENTIMENT_DIVERGENCE = "sentiment_divergence"
    SEASONALITY = "seasonality"
    ORDER_FLOW_TOXICITY = "order_flow_toxicity"
    TRIANGULAR_ARBITRAGE = "triangular_arbitrage"
    SPREAD_ANOMALY = "spread_anomaly"
    LIQUIDITY_IMBALANCE = "liquidity_imbalance"


class SignificanceLevel(Enum):
    """Statistical significance levels"""
    HIGH = "high"  # p < 0.01
    MEDIUM = "medium"  # p < 0.05
    LOW = "low"  # p < 0.10
    INSIGNIFICANT = "insignificant"  # p >= 0.10


@dataclass
class InefficiencySignal:
    """
    Represents a detected market inefficiency
    """
    signal_id: str
    timestamp: datetime
    inefficiency_type: InefficiencyType
    
    # Asset information
    symbols: List[str]
    exchange: Optional[str] = None
    
    # Signal characteristics
    confidence: float = 0.0  # 0-1 confidence score
    expected_return: float = 0.0  # Expected return %
    expected_duration: Optional[int] = None  # Expected duration in seconds
    
    # Statistical metrics
    p_value: float = 1.0
    z_score: float = 0.0
    sharpe_ratio: Optional[float] = None
    
    # Trade parameters
    direction: str = "neutral"  # 'long', 'short', 'neutral', 'pair'
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: Optional[float] = None
    
    # Context
    market_regime: Optional[str] = None
    volatility_regime: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_significant(self, alpha: float = 0.01) -> bool:
        """Check if signal is statistically significant"""
        return self.p_value < alpha and self.confidence > 0.7
    
    @lru_cache(maxsize=128)
    
    def get_significance_level(self) -> SignificanceLevel:
        """Get significance level based on p-value"""
        if self.p_value < 0.01:
            return SignificanceLevel.HIGH
        elif self.p_value < 0.05:
            return SignificanceLevel.MEDIUM
        elif self.p_value < 0.10:
            return SignificanceLevel.LOW
        else:
            return SignificanceLevel.INSIGNIFICANT
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'signal_id': self.signal_id,
            'timestamp': self.timestamp.isoformat(),
            'inefficiency_type': self.inefficiency_type.value,
            'symbols': self.symbols,
            'exchange': self.exchange,
            'confidence': self.confidence,
            'expected_return': self.expected_return,
            'expected_duration': self.expected_duration,
            'p_value': self.p_value,
            'z_score': self.z_score,
            'sharpe_ratio': self.sharpe_ratio,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'position_size': self.position_size,
            'market_regime': self.market_regime,
            'volatility_regime': self.volatility_regime,
            'description': self.description,
            'metadata': self.metadata
        }


@dataclass
class BacktestResult:
    """Results from backtesting an inefficiency"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    
    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Risk metrics
    volatility: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR 95%
    
    # Statistical tests
    t_statistic: float
    p_value: float
    
    # Additional metrics
    calmar_ratio: Optional[float] = None
    omega_ratio: Optional[float] = None
    
    def is_viable(self) -> bool:
        """Check if strategy is viable for live trading"""
        return (
            self.sharpe_ratio > 2.0 and
            self.win_rate > 0.55 and
            self.profit_factor > 1.5 and
            self.p_value < 0.01 and
            self.max_drawdown > -0.20  # Less than 20% drawdown
        )


class BaseInefficiencyDetector(ABC):
    """
    Abstract base class for all inefficiency detectors
    """
    
    def __init__(self, name: str):
        self.name = name
        self.signals_generated: List[InefficiencySignal] = []
        self.is_running = False
        
    @abstractmethod
    async def detect(self, data: pd.DataFrame) -> List[InefficiencySignal]:
        """
        Main detection method - must be implemented by subclasses
        
        Args:
            data: Market data to analyze
            
        Returns:
            List of detected inefficiency signals
        """
        pass
    
    @abstractmethod
    def calculate_statistics(self, signal: InefficiencySignal) -> InefficiencySignal:
        """
        Calculate statistical metrics for a signal
        
        Args:
            signal: Signal to analyze
            
        Returns:
            Signal with updated statistics
        """
        pass
    
    def validate_signal(self, signal: InefficiencySignal) -> bool:
        """
        Validate a signal before returning
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if signal is valid
        """
        # Basic validation
        if signal.confidence < 0.5:
            return False
        
        if abs(signal.expected_return) < 0.001:  # Less than 0.1%
            return False
        
        if signal.p_value > 0.10:
            return False
        
        return True
    
    def filter_signals(self, signals: List[InefficiencySignal]) -> List[InefficiencySignal]:
        """Filter signals based on quality criteria"""
        return [s for s in signals if self.validate_signal(s)]
    
    async def start(self):
        """Start the detector"""
        self.is_running = True
        
    async def stop(self):
        """Stop the detector"""
        self.is_running = False
    
    @lru_cache(maxsize=128)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics"""
        return {
            'name': self.name,
            'total_signals': len(self.signals_generated),
            'significant_signals': len([s for s in self.signals_generated if s.is_significant()]),
            'avg_confidence': np.mean([s.confidence for s in self.signals_generated]) if self.signals_generated else 0,
            'avg_expected_return': np.mean([s.expected_return for s in self.signals_generated]) if self.signals_generated else 0
        }


class BaseDataCollector(ABC):
    """
    Abstract base class for data collectors
    """
    
    def __init__(self, name: str):
        self.name = name
        self.is_running = False
        self.data_buffer: List[Dict[str, Any]] = []
        
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to data source"""
        pass
    
    @abstractmethod
    async def subscribe(self, symbols: List[str]) -> bool:
        """Subscribe to data for symbols"""
        pass
    
    @abstractmethod
    async def collect(self) -> pd.DataFrame:
        """Collect and return data"""
        pass
    
    async def start(self):
        """Start collecting data"""
        self.is_running = True
        
    async def stop(self):
        """Stop collecting data"""
        self.is_running = False


class BaseAnalyzer(ABC):
    """
    Abstract base class for data analyzers
    """
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data and return metrics"""
        pass
    
    @abstractmethod
    def calculate_significance(self, results: Dict[str, Any]) -> float:
        """Calculate statistical significance (p-value)"""
        pass

