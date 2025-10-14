from ..base import InefficiencySignal, InefficiencyType, BaseInefficiencyDetector
from ..collectors import EnhancedPolygonCollector, OrderFlowAnalyzer
from ..collectors.perplexity_sentiment import PerplexitySentimentAnalyzer
from ..detectors import (
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional, Any
import asyncio
import logging
import numpy as np
import pandas as pd


"""
Master Orchestrator for Parallel Multi-Agent Inefficiency Discovery

Coordinates multiple detector agents running in parallel to discover novel market inefficiencies.
"""


    LatencyArbitrageDetector,
    FundingRateArbitrageDetector,
    CorrelationAnomalyDetector,
    SentimentDivergenceDetector,
    SeasonalityDetector,
    OrderFlowToxicityDetector
)

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorStats:
    """Statistics for the orchestrator"""
    start_time: datetime
    total_cycles: int = 0
    total_signals: int = 0
    significant_signals: int = 0
    signals_by_type: Dict[str, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    avg_expected_return: float = 0.0
    detectors_active: int = 0


class MasterOrchestrator:
    """
    Coordinates parallel inefficiency detection across multiple agents
    
    Architecture:
    - 6 specialized detector agents
    - 1 data collection layer
    - 1 validation/backtesting layer
    - Parallel execution with asyncio
    
    Flow:
    1. Data collectors gather multi-source data
    2. Detectors run in parallel analyzing different patterns
    3. Signals are aggregated and deduplicated
    4. Top signals are validated via backtesting
    5. Viable signals are forwarded to trading engine
    """
    
    def __init__(self, 
                 polygon_api_key: Optional[str] = None,
                 perplexity_api_key: Optional[str] = None,
                 enable_sentiment: bool = True):
        """
        Initialize orchestrator with data sources and detectors
        
        Args:
            polygon_api_key: Polygon.io API key
            perplexity_api_key: Perplexity AI API key
            enable_sentiment: Whether to enable sentiment analysis
        """
        self.is_running = False
        self.stats = OrchestratorStats(start_time=datetime.now())
        
        # Data collectors
        logger.info("Initializing data collectors...")
        self.polygon_collector = EnhancedPolygonCollector(api_key=polygon_api_key)
        self.order_flow_analyzer = OrderFlowAnalyzer()
        
        if enable_sentiment:
            self.sentiment_analyzer = PerplexitySentimentAnalyzer(api_key=perplexity_api_key)
        else:
            self.sentiment_analyzer = None
        
        # Initialize detectors
        logger.info("Initializing detector agents...")
        self.detectors: Dict[str, BaseInefficiencyDetector] = {
            'latency_arbitrage': LatencyArbitrageDetector(),
            'funding_rate': FundingRateArbitrageDetector(),
            'correlation_anomaly': CorrelationAnomalyDetector(),
            'sentiment_divergence': SentimentDivergenceDetector() if enable_sentiment else None,
            'seasonality': SeasonalityDetector(),
            'order_flow_toxicity': OrderFlowToxicityDetector()
        }
        
        # Remove None detectors
        self.detectors = {k: v for k, v in self.detectors.items() if v is not None}
        
        self.stats.detectors_active = len(self.detectors)
        
        # Signal aggregation
        self.all_signals: List[InefficiencySignal] = []
        self.recent_signals: List[InefficiencySignal] = []
        
        # Configuration
        self.cycle_interval_seconds = 60  # Run detection every 60 seconds
        self.max_signals_per_cycle = 50
        self.min_confidence_threshold = 0.6
        
        logger.info(f"✅ Orchestrator initialized with {len(self.detectors)} active detectors")
    
    async def start(self, symbols: List[str]):
        """
        Start the orchestrator
        
        Args:
            symbols: List of symbols to monitor
        """
        logger.info(f"🚀 Starting Master Orchestrator for {len(symbols)} symbols...")
        
        self.is_running = True
        
        # Start data collectors
        await self._start_collectors(symbols)
        
        # Main orchestration loop
        while self.is_running:
            try:
                await self._run_detection_cycle(symbols)
                await asyncio.sleep(self.cycle_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in orchestration cycle: {e}")
                await asyncio.sleep(5)
        
        # Cleanup
        await self._stop_collectors()
        
        logger.info("✅ Master Orchestrator stopped")
    
    async def _start_collectors(self, symbols: List[str]):
        """Start data collectors"""
        logger.info("Starting data collectors...")
        
        # Start Polygon collector
        connected = await self.polygon_collector.connect()
        if connected:
            await self.polygon_collector.subscribe(symbols, channels=['XT', 'XQ'])
            logger.info(f"✅ Polygon collector subscribed to {len(symbols)} symbols")
        else:
            logger.error("Failed to connect to Polygon.io")
    
    async def _stop_collectors(self):
        """Stop data collectors"""
        await self.polygon_collector.stop()
        
        if self.sentiment_analyzer:
            await self.sentiment_analyzer.close()
    
    async def _run_detection_cycle(self, symbols: List[str]):
        """
        Run one detection cycle
        
        Process:
        1. Collect recent data
        2. Run all detectors in parallel
        3. Aggregate and deduplicate signals
        4. Filter by quality
        5. Store results
        """
        cycle_start = datetime.now()
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 Detection Cycle {self.stats.total_cycles + 1} - {cycle_start.strftime('%H:%M:%S')}")
        logger.info(f"{'='*60}")
        
        # Step 1: Collect data from all sources
        data = await self._collect_data(symbols)
        
        if data.empty:
            logger.warning("No data collected, skipping cycle")
            return
        
        logger.info(f"📊 Collected {len(data)} data points")
        
        # Step 2: Run detectors in parallel
        detector_tasks = []
        for detector_name, detector in self.detectors.items():
            task = self._run_detector(detector_name, detector, data)
            detector_tasks.append(task)
        
        # Wait for all detectors
        detector_results = await asyncio.gather(*detector_tasks, return_exceptions=True)
        
        # Step 3: Aggregate signals
        cycle_signals = []
        for detector_name, result in zip(self.detectors.keys(), detector_results):
            if isinstance(result, list):
                cycle_signals.extend(result)
                logger.info(f"  {detector_name}: {len(result)} signals")
            elif isinstance(result, Exception):
                logger.error(f"  {detector_name}: Error - {result}")
        
        # Step 4: Filter and deduplicate
        filtered_signals = self._filter_signals(cycle_signals)
        
        logger.info(f"\n📈 Cycle Results:")
        logger.info(f"  Total signals: {len(cycle_signals)}")
        logger.info(f"  After filtering: {len(filtered_signals)}")
        logger.info(f"  Avg confidence: {np.mean([s.confidence for s in filtered_signals]):.2%}" if filtered_signals else "  No signals")
        logger.info(f"  Avg expected return: {np.mean([s.expected_return for s in filtered_signals]):.2%}" if filtered_signals else "")
        
        # Step 5: Store results
        self.recent_signals = filtered_signals
        self.all_signals.extend(filtered_signals)
        
        # Update stats
        self.stats.total_cycles += 1
        self.stats.total_signals += len(cycle_signals)
        self.stats.significant_signals += len(filtered_signals)
        
        for signal in filtered_signals:
            signal_type = signal.inefficiency_type.value
            self.stats.signals_by_type[signal_type] = self.stats.signals_by_type.get(signal_type, 0) + 1
        
        if filtered_signals:
            self.stats.avg_confidence = np.mean([s.confidence for s in filtered_signals])
            self.stats.avg_expected_return = np.mean([s.expected_return for s in filtered_signals])
        
        # Print top signals
        if filtered_signals:
            logger.info(f"\n🎯 Top Signals:")
            top_signals = sorted(filtered_signals, key=lambda s: s.confidence * s.expected_return, reverse=True)[:5]
            
            for i, signal in enumerate(top_signals, 1):
                logger.info(f"\n  {i}. {signal.inefficiency_type.value.upper()}")
                logger.info(f"     Symbols: {', '.join(signal.symbols)}")
                logger.info(f"     Confidence: {signal.confidence:.1%}")
                logger.info(f"     Expected Return: {signal.expected_return:.2%}")
                logger.info(f"     Direction: {signal.direction}")
                logger.info(f"     P-value: {signal.p_value:.4f}")
                logger.info(f"     Description: {signal.description[:100]}...")
        
        cycle_duration = (datetime.now() - cycle_start).total_seconds()
        logger.info(f"\n⏱️  Cycle completed in {cycle_duration:.1f}s")
    
    async def _collect_data(self, symbols: List[str]) -> pd.DataFrame:
        """
        Collect data from all sources
        
        Returns:
            Unified DataFrame with all data
        """
        data_frames = []
        
        # Polygon tick data
        polygon_data = await self.polygon_collector.collect()
        if not polygon_data.empty:
            data_frames.append(polygon_data)
        
        # Combine all data sources
        if data_frames:
            combined_data = pd.concat(data_frames, ignore_index=True)
            return combined_data
        
        return pd.DataFrame()
    
    async def _run_detector(self, detector_name: str, 
                           detector: BaseInefficiencyDetector,
                           data: pd.DataFrame) -> List[InefficiencySignal]:
        """
        Run a single detector
        
        Args:
            detector_name: Name of detector
            detector: Detector instance
            data: Input data
            
        Returns:
            List of signals
        """
        try:
            signals = await detector.detect(data)
            return signals
        except Exception as e:
            logger.error(f"Error in {detector_name}: {e}")
            return []
    
    def _filter_signals(self, signals: List[InefficiencySignal]) -> List[InefficiencySignal]:
        """
        Filter and deduplicate signals
        
        Criteria:
        - Confidence > threshold
        - P-value < 0.05
        - Expected return > 0
        - Remove duplicates (same symbol + type)
        """
        if not signals:
            return []
        
        # Filter by quality
        filtered = []
        for signal in signals:
            if (signal.confidence >= self.min_confidence_threshold and
                signal.p_value < 0.05 and
                signal.expected_return > 0):
                filtered.append(signal)
        
        # Deduplicate
        # Keep highest confidence signal for each (symbols, type) combination
        unique_signals = {}
        for signal in filtered:
            key = (tuple(sorted(signal.symbols)), signal.inefficiency_type)
            
            if key not in unique_signals:
                unique_signals[key] = signal
            else:
                # Keep higher confidence
                if signal.confidence > unique_signals[key].confidence:
                    unique_signals[key] = signal
        
        deduplicated = list(unique_signals.values())
        
        # Sort by quality score (confidence * expected_return)
        deduplicated.sort(key=lambda s: s.confidence * s.expected_return, reverse=True)
        
        # Limit number of signals
        return deduplicated[:self.max_signals_per_cycle]
    
    @lru_cache(maxsize=128)
    
    def get_recent_signals(self, top_n: int = 10) -> List[InefficiencySignal]:
        """Get most recent signals"""
        return self.recent_signals[:top_n]
    
    @lru_cache(maxsize=128)
    
    def get_all_signals(self) -> List[InefficiencySignal]:
        """Get all discovered signals"""
        return self.all_signals
    
    @lru_cache(maxsize=128)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        uptime = (datetime.now() - self.stats.start_time).total_seconds()
        
        return {
            'uptime_seconds': uptime,
            'total_cycles': self.stats.total_cycles,
            'total_signals': self.stats.total_signals,
            'significant_signals': self.stats.significant_signals,
            'signals_by_type': self.stats.signals_by_type,
            'avg_confidence': self.stats.avg_confidence,
            'avg_expected_return': self.stats.avg_expected_return,
            'detectors_active': self.stats.detectors_active,
            'cycle_interval': self.cycle_interval_seconds,
            'signals_per_hour': (self.stats.total_signals / uptime * 3600) if uptime > 0 else 0
        }
    
    @lru_cache(maxsize=128)
    
    def get_detector_statistics(self) -> Dict[str, Dict]:
        """Get statistics for each detector"""
        stats = {}
        
        for detector_name, detector in self.detectors.items():
            stats[detector_name] = detector.get_statistics()
        
        return stats
    
    async def stop(self):
        """Stop the orchestrator"""
        logger.info("Stopping orchestrator...")
        self.is_running = False


# Example usage
async def main():
    """Example: Run orchestrator for BTC and ETH"""
    
    orchestrator = MasterOrchestrator()
    
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    
    try:
        await orchestrator.start(symbols)
    except KeyboardInterrupt:
        await orchestrator.stop()
        
        # Print final statistics
        stats = orchestrator.get_statistics()
        print(f"\n{'='*60}")
        print("📊 Final Statistics")
        print(f"{'='*60}")
        print(f"Total Cycles: {stats['total_cycles']}")
        print(f"Total Signals: {stats['total_signals']}")
        print(f"Significant Signals: {stats['significant_signals']}")
        print(f"Uptime: {stats['uptime_seconds']:.0f}s")
        print(f"\nSignals by Type:")
        for signal_type, count in stats['signals_by_type'].items():
            print(f"  {signal_type}: {count}")


if __name__ == "__main__":
    asyncio.run(main())

