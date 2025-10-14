from .base_agent import AgentSignal, SignalDirection
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional
import logging

"""
Consensus Builder - Aggregate signals from multiple agents.

Combines signals from specialized agents using various methods:
1. Majority voting
2. Confidence weighting
3. Expertise weighting (domain-specific)
4. Regime-dependent weighting
5. Bayesian aggregation
"""



logger = logging.getLogger(__name__)


class ConsensusMethod(Enum):
    """Method for aggregating agent signals"""
    MAJORITY_VOTE = "majority_vote"  # Simple vote
    CONFIDENCE_WEIGHTED = "confidence_weighted"  # Weight by confidence
    EXPERTISE_WEIGHTED = "expertise_weighted"  # Weight by agent type
    ADAPTIVE = "adaptive"  # Learn optimal weights from performance


@dataclass
class ConsensusResult:
    """
    Aggregated decision from multiple agents.
    """
    timestamp: datetime
    asset: str
    
    # Final decision
    direction: SignalDirection
    confidence: float  # 0.0 to 1.0
    
    # Consensus metrics
    agreement_level: float  # 0.0 to 1.0 (how much agents agree)
    participating_agents: int
    
    # Breakdown
    agent_signals: List[AgentSignal] = field(default_factory=list)
    weights_used: Dict[str, float] = field(default_factory=dict)
    
    # Supporting information
    reasoning: str = ""
    has_conflicts: bool = False
    conflict_description: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'asset': self.asset,
            'direction': self.direction.value,
            'confidence': self.confidence,
            'agreement_level': self.agreement_level,
            'participating_agents': self.participating_agents,
            'reasoning': self.reasoning,
            'has_conflicts': self.has_conflicts,
            'conflict_description': self.conflict_description,
            'weights_used': self.weights_used
        }


class ConsensusBuilder:
    """
    Aggregate signals from multiple agents into single decision.
    
    Handles conflicts, weights agents appropriately, and produces
    final trading decision with confidence level.
    """
    
    def __init__(
        self,
        method: ConsensusMethod = ConsensusMethod.CONFIDENCE_WEIGHTED,
        min_agents: int = 3,
        conflict_threshold: float = 0.3
    ):
        """
        Initialize consensus builder.
        
        Args:
            method: Aggregation method to use
            min_agents: Minimum agents required for consensus
            conflict_threshold: Max disagreement before flagging conflict
        """
        self.method = method
        self.min_agents = min_agents
        self.conflict_threshold = conflict_threshold
        
        # Expertise weights (domain-specific trust)
        self.expertise_weights = {
            'technical': 1.0,
            'onchain': 1.2,  # Unique to crypto, weight higher
            'microstructure': 1.1,
            'sentiment': 0.9,  # Noisier signal
            'arbitrage': 1.5,  # High confidence when found
            'regime': 1.0
        }
        
        # Performance-based weights (learned from historical accuracy)
        self.performance_weights: Dict[str, float] = {}
    
    def aggregate(self, signals: List[AgentSignal]) -> ConsensusResult:
        """
        Aggregate agent signals into consensus decision.
        
        Args:
            signals: List of signals from different agents
            
        Returns:
            ConsensusResult with final decision
        """
        if len(signals) < self.min_agents:
            logger.warning(f"Insufficient agents: {len(signals)} < {self.min_agents}")
            return self._neutral_consensus(signals)
        
        # Filter out neutral signals for voting
        active_signals = [s for s in signals if s.direction != SignalDirection.NEUTRAL]
        
        if not active_signals:
            return self._neutral_consensus(signals)
        
        # Select aggregation method
        if self.method == ConsensusMethod.MAJORITY_VOTE:
            return self._majority_vote(signals, active_signals)
        elif self.method == ConsensusMethod.CONFIDENCE_WEIGHTED:
            return self._confidence_weighted(signals, active_signals)
        elif self.method == ConsensusMethod.EXPERTISE_WEIGHTED:
            return self._expertise_weighted(signals, active_signals)
        elif self.method == ConsensusMethod.ADAPTIVE:
            return self._adaptive_weighted(signals, active_signals)
        else:
            return self._confidence_weighted(signals, active_signals)
    
    def _neutral_consensus(self, signals: List[AgentSignal]) -> ConsensusResult:
        """Return neutral consensus when insufficient data"""
        return ConsensusResult(
            timestamp=datetime.utcnow(),
            asset=signals[0].asset if signals else "UNKNOWN",
            direction=SignalDirection.NEUTRAL,
            confidence=0.0,
            agreement_level=0.0,
            participating_agents=len(signals),
            agent_signals=signals,
            reasoning="Insufficient active signals for consensus"
        )
    
    def _majority_vote(
        self,
        all_signals: List[AgentSignal],
        active_signals: List[AgentSignal]
    ) -> ConsensusResult:
        """Simple majority voting"""
        long_votes = sum(1 for s in active_signals if s.direction == SignalDirection.LONG)
        short_votes = sum(1 for s in active_signals if s.direction == SignalDirection.SHORT)
        
        total_votes = long_votes + short_votes
        
        if long_votes > short_votes:
            direction = SignalDirection.LONG
            confidence = long_votes / len(all_signals)
            agreement = long_votes / total_votes if total_votes > 0 else 0
        elif short_votes > long_votes:
            direction = SignalDirection.SHORT
            confidence = short_votes / len(all_signals)
            agreement = short_votes / total_votes if total_votes > 0 else 0
        else:
            # Tie
            return self._neutral_consensus(all_signals)
        
        # Check for conflicts
        has_conflicts = agreement < (1 - self.conflict_threshold)
        conflict_desc = f"Split vote: {long_votes} LONG vs {short_votes} SHORT" if has_conflicts else ""
        
        return ConsensusResult(
            timestamp=datetime.utcnow(),
            asset=all_signals[0].asset,
            direction=direction,
            confidence=confidence,
            agreement_level=agreement,
            participating_agents=len(all_signals),
            agent_signals=all_signals,
            reasoning=f"Majority vote: {long_votes} LONG, {short_votes} SHORT",
            has_conflicts=has_conflicts,
            conflict_description=conflict_desc
        )
    
    def _confidence_weighted(
        self,
        all_signals: List[AgentSignal],
        active_signals: List[AgentSignal]
    ) -> ConsensusResult:
        """Weight signals by agent confidence"""
        long_weight = sum(s.confidence for s in active_signals if s.direction == SignalDirection.LONG)
        short_weight = sum(s.confidence for s in active_signals if s.direction == SignalDirection.SHORT)
        
        total_weight = long_weight + short_weight
        
        if total_weight == 0:
            return self._neutral_consensus(all_signals)
        
        if long_weight > short_weight:
            direction = SignalDirection.LONG
            confidence = long_weight / (long_weight + short_weight)
            agreement = long_weight / total_weight
        else:
            direction = SignalDirection.SHORT
            confidence = short_weight / (long_weight + short_weight)
            agreement = short_weight / total_weight
        
        # Normalize confidence to [0, 1]
        max_possible = len(active_signals)
        normalized_confidence = max(long_weight, short_weight) / max_possible if max_possible > 0 else 0
        
        # Check conflicts
        has_conflicts = abs(long_weight - short_weight) / total_weight < self.conflict_threshold
        conflict_desc = f"Close confidence split: {long_weight:.2f} vs {short_weight:.2f}" if has_conflicts else ""
        
        return ConsensusResult(
            timestamp=datetime.utcnow(),
            asset=all_signals[0].asset,
            direction=direction,
            confidence=normalized_confidence,
            agreement_level=agreement,
            participating_agents=len(all_signals),
            agent_signals=all_signals,
            reasoning=f"Confidence-weighted: {long_weight:.2f} LONG, {short_weight:.2f} SHORT",
            has_conflicts=has_conflicts,
            conflict_description=conflict_desc
        )
    
    def _expertise_weighted(
        self,
        all_signals: List[AgentSignal],
        active_signals: List[AgentSignal]
    ) -> ConsensusResult:
        """Weight signals by agent expertise (domain-specific trust)"""
        long_weight = sum(
            s.confidence * self.expertise_weights.get(s.agent_type, 1.0)
            for s in active_signals if s.direction == SignalDirection.LONG
        )
        short_weight = sum(
            s.confidence * self.expertise_weights.get(s.agent_type, 1.0)
            for s in active_signals if s.direction == SignalDirection.SHORT
        )
        
        total_weight = long_weight + short_weight
        
        if total_weight == 0:
            return self._neutral_consensus(all_signals)
        
        direction = SignalDirection.LONG if long_weight > short_weight else SignalDirection.SHORT
        agreement = max(long_weight, short_weight) / total_weight
        confidence = agreement  # Confidence = agreement level
        
        # Calculate weights used for transparency
        weights_used = {
            s.agent_id: self.expertise_weights.get(s.agent_type, 1.0)
            for s in all_signals
        }
        
        # Check conflicts
        has_conflicts = abs(long_weight - short_weight) / total_weight < self.conflict_threshold
        
        return ConsensusResult(
            timestamp=datetime.utcnow(),
            asset=all_signals[0].asset,
            direction=direction,
            confidence=confidence,
            agreement_level=agreement,
            participating_agents=len(all_signals),
            agent_signals=all_signals,
            weights_used=weights_used,
            reasoning=f"Expertise-weighted: {long_weight:.2f} LONG, {short_weight:.2f} SHORT",
            has_conflicts=has_conflicts
        )
    
    def _adaptive_weighted(
        self,
        all_signals: List[AgentSignal],
        active_signals: List[AgentSignal]
    ) -> ConsensusResult:
        """
        Weight signals adaptively based on historical performance.
        
        Falls back to expertise weighting if no performance history.
        """
        if not self.performance_weights:
            logger.info("No performance weights available, using expertise weighting")
            return self._expertise_weighted(all_signals, active_signals)
        
        long_weight = sum(
            s.confidence * self.performance_weights.get(s.agent_id, 1.0)
            for s in active_signals if s.direction == SignalDirection.LONG
        )
        short_weight = sum(
            s.confidence * self.performance_weights.get(s.agent_id, 1.0)
            for s in active_signals if s.direction == SignalDirection.SHORT
        )
        
        total_weight = long_weight + short_weight
        
        if total_weight == 0:
            return self._neutral_consensus(all_signals)
        
        direction = SignalDirection.LONG if long_weight > short_weight else SignalDirection.SHORT
        confidence = max(long_weight, short_weight) / total_weight
        agreement = confidence
        
        weights_used = {
            s.agent_id: self.performance_weights.get(s.agent_id, 1.0)
            for s in all_signals
        }
        
        return ConsensusResult(
            timestamp=datetime.utcnow(),
            asset=all_signals[0].asset,
            direction=direction,
            confidence=confidence,
            agreement_level=agreement,
            participating_agents=len(all_signals),
            agent_signals=all_signals,
            weights_used=weights_used,
            reasoning=f"Adaptive-weighted: {long_weight:.2f} LONG, {short_weight:.2f} SHORT"
        )
    
    def update_performance_weights(self, agent_id: str, new_weight: float):
        """
        Update performance-based weight for an agent.
        
        Called by agent performance tracker after evaluating accuracy.
        
        Args:
            agent_id: Agent identifier
            new_weight: New performance weight (typically 0.5-2.0)
        """
        self.performance_weights[agent_id] = max(0.1, min(new_weight, 3.0))  # Clamp to [0.1, 3.0]
        logger.info(f"Updated {agent_id} weight to {new_weight:.2f}")


# Example usage
if __name__ == "__main__":
    from .base_agent import AgentSignal, SignalDirection, MarketState
    
    print("="*80)
    print("Consensus Builder Demo")
    print("="*80)
    print()
    
    # Create mock signals from different agents
    timestamp = datetime.utcnow()
    signals = [
        AgentSignal(
            agent_id="technical_001",
            agent_type="technical",
            timestamp=timestamp,
            asset="BTC-USD",
            direction=SignalDirection.LONG,
            confidence=0.7,
            reasoning="RSI oversold"
        ),
        AgentSignal(
            agent_id="onchain_001",
            agent_type="onchain",
            timestamp=timestamp,
            asset="BTC-USD",
            direction=SignalDirection.LONG,
            confidence=0.8,
            reasoning="Whale outflows from exchanges"
        ),
        AgentSignal(
            agent_id="microstructure_001",
            agent_type="microstructure",
            timestamp=timestamp,
            asset="BTC-USD",
            direction=SignalDirection.SHORT,
            confidence=0.6,
            reasoning="Order book imbalance bearish"
        ),
        AgentSignal(
            agent_id="sentiment_001",
            agent_type="sentiment",
            timestamp=timestamp,
            asset="BTC-USD",
            direction=SignalDirection.NEUTRAL,
            confidence=0.0,
            reasoning="Mixed sentiment"
        )
    ]
    
    # Test different consensus methods
    methods = [
        ConsensusMethod.MAJORITY_VOTE,
        ConsensusMethod.CONFIDENCE_WEIGHTED,
        ConsensusMethod.EXPERTISE_WEIGHTED
    ]
    
    for method in methods:
        print(f"\n{method.value.upper()}:")
        print("-" * 40)
        
        builder = ConsensusBuilder(method=method)
        result = builder.aggregate(signals)
        
        print(f"Direction: {result.direction.value}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Agreement: {result.agreement_level:.2%}")
        print(f"Agents: {result.participating_agents}")
        print(f"Reasoning: {result.reasoning}")
        if result.has_conflicts:
            print(f"⚠️  Conflicts detected: {result.conflict_description}")
    
    print("\n" + "="*80)
    print("✅ Consensus builder demo complete")

