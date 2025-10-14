from .framework.base_agent import BaseAgent, AgentSignal, MarketState
from .framework.consensus_builder import ConsensusBuilder
from .framework.coordinator import MasterCoordinator

"""
Multi-Agent Trading System

Hierarchical agent architecture for emergent pattern discovery:
- Market Analysis Agents: Technical, On-Chain, Microstructure, Sentiment
- Strategy Selection Agents: Trend, Mean-Reversion, Arbitrage
- Risk Assessment Agents: Portfolio, Execution, Market Risk
- Master Coordinator: Aggregates signals and makes final decisions
"""


__all__ = [
    'BaseAgent',
    'AgentSignal',
    'MarketState',
    'MasterCoordinator',
    'ConsensusBuilder',
]

