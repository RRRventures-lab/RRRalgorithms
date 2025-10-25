"""
Integration Layer for Neural Network System

Connects neural network models with:
- Data pipeline (Polygon.io, Perplexity)
- Database (Supabase)
- Trading engine
- Transparency dashboard
"""

from .data_integration import DataPipelineConnector, RealtimeDataStream
from .database_integration import PredictionStorage, ModelRegistry
from .trading_integration import TradingSignalGenerator

__all__ = [
    'DataPipelineConnector',
    'RealtimeDataStream',
    'PredictionStorage',
    'ModelRegistry',
    'TradingSignalGenerator',
]
