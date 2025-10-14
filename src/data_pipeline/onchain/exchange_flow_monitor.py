from .blockchain_client import BlockchainClient
from .etherscan_client import EtherscanClient
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from typing import Dict, List, Optional
import logging


"""
Exchange Flow Monitor - Track net inflow/outflow from exchanges.

Monitors the balance changes on known exchange addresses to detect
accumulation (outflows, bullish) vs distribution (inflows, bearish) patterns.
"""



logger = logging.getLogger(__name__)


@dataclass
class FlowMetric:
    """Exchange flow metric for a specific asset and timeframe"""
    asset: str  # 'BTC' or 'ETH'
    timestamp: datetime
    exchange_name: str
    inflow: Decimal  # Amount deposited to exchange
    outflow: Decimal  # Amount withdrawn from exchange
    net_flow: Decimal  # outflow - inflow (positive = accumulation)
    balance_change: Decimal  # Current balance change
    
    def is_bullish(self) -> bool:
        """Positive net flow (withdrawals > deposits) is bullish"""
        return self.net_flow > 0
    
    def signal_strength(self) -> str:
        """Classify flow magnitude"""
        abs_flow = abs(float(self.net_flow))
        if self.asset == 'ETH':
            if abs_flow > 50000:
                return "CRITICAL"
            elif abs_flow > 20000:
                return "STRONG"
            elif abs_flow > 5000:
                return "MEDIUM"
        else:  # BTC
            if abs_flow > 1000:
                return "CRITICAL"
            elif abs_flow > 500:
                return "STRONG"
            elif abs_flow > 100:
                return "MEDIUM"
        return "WEAK"


class ExchangeFlowMonitor:
    """
    Monitor exchange inflow/outflow to detect accumulation/distribution.
    
    Key signals:
    - High inflow = potential selling pressure (bearish)
    - High outflow = accumulation, holders moving to cold storage (bullish)
    """
    
    # Major exchange addresses for Ethereum
    ETH_EXCHANGE_ADDRESSES = {
        "Binance": [
            "0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE",
            "0xD551234Ae421e3BCBA99A0Da6d736074f22192FF",
        ],
        "Coinbase": [
            "0x71660c4005BA85c37ccec55d0C4493E66Fe775d3",
            "0x503828976D22510aad0201ac7EC88293211D23Da",
        ],
        "Kraken": [
            "0x2910543Af39abA0Cd09dBb2D50200b3E800A63D2",
        ],
    }
    
    # Major exchange addresses for Bitcoin
    BTC_EXCHANGE_ADDRESSES = {
        "Binance": [
            "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo",  # Example
        ],
        "Coinbase": [
            "3FupZp77ySr7jwoLYEJ9mwzJpvoNBXsBnE",  # Example
        ],
    }
    
    def __init__(self):
        """Initialize exchange flow monitor"""
        self.eth_client = EtherscanClient()
        self.btc_client = BlockchainClient()
        self.flow_history: List[FlowMetric] = []
    
    @lru_cache(maxsize=128)
    
    def get_eth_exchange_balance(self, exchange_name: str) -> Decimal:
        """Get total ETH balance across all addresses for an exchange"""
        addresses = self.ETH_EXCHANGE_ADDRESSES.get(exchange_name, [])
        total_balance = Decimal(0)
        
        for addr in addresses:
            try:
                balance = self.eth_client.get_eth_balance(addr)
                total_balance += balance
            except Exception as e:
                logger.error(f"Error fetching balance for {addr}: {e}")
        
        return total_balance
    
    def calculate_eth_flow(
        self,
        exchange_name: str,
        lookback_hours: int = 24
    ) -> FlowMetric:
        """
        Calculate ETH flow for an exchange over time period.
        
        Args:
            exchange_name: Name of exchange (must be in ETH_EXCHANGE_ADDRESSES)
            lookback_hours: Time window to analyze
            
        Returns:
            FlowMetric with inflow/outflow data
        """
        addresses = self.ETH_EXCHANGE_ADDRESSES.get(exchange_name, [])
        if not addresses:
            raise ValueError(f"Unknown exchange: {exchange_name}")
        
        # Calculate cutoff timestamp
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        cutoff_timestamp = int(cutoff_time.timestamp())
        start_block = self.eth_client.get_block_number_by_timestamp(cutoff_timestamp)
        
        total_inflow = Decimal(0)
        total_outflow = Decimal(0)
        
        # Aggregate flows across all exchange addresses
        for addr in addresses:
            try:
                txs = self.eth_client.get_normal_transactions(
                    address=addr,
                    start_block=start_block,
                    sort="asc"
                )
                
                for tx in txs:
                    value_wei = int(tx.get('value', 0))
                    value_eth = Decimal(value_wei) / Decimal(10**18)
                    
                    # Check if this is inflow or outflow
                    to_addr = tx.get('to', '').lower()
                    from_addr = tx.get('from', '').lower()
                    
                    if to_addr == addr.lower():
                        # Inflow (deposit to exchange)
                        total_inflow += value_eth
                    elif from_addr == addr.lower():
                        # Outflow (withdrawal from exchange)
                        total_outflow += value_eth
                        
            except Exception as e:
                logger.error(f"Error processing {addr}: {e}")
                continue
        
        net_flow = total_outflow - total_inflow
        
        return FlowMetric(
            asset='ETH',
            timestamp=datetime.utcnow(),
            exchange_name=exchange_name,
            inflow=total_inflow,
            outflow=total_outflow,
            net_flow=net_flow,
            balance_change=net_flow  # Simplified
        )
    
    @lru_cache(maxsize=128)
    
    def get_aggregate_eth_flow(self, lookback_hours: int = 24) -> Dict:
        """Get aggregate ETH flow across all major exchanges"""
        flows = []
        
        for exchange_name in self.ETH_EXCHANGE_ADDRESSES.keys():
            try:
                flow = self.calculate_eth_flow(exchange_name, lookback_hours)
                flows.append(flow)
            except Exception as e:
                logger.error(f"Error calculating flow for {exchange_name}: {e}")
        
        # Aggregate
        total_inflow = sum(f.inflow for f in flows)
        total_outflow = sum(f.outflow for f in flows)
        net_flow = total_outflow - total_inflow
        
        return {
            'asset': 'ETH',
            'lookback_hours': lookback_hours,
            'total_inflow': float(total_inflow),
            'total_outflow': float(total_outflow),
            'net_flow': float(net_flow),
            'is_bullish': net_flow > 0,
            'exchanges': [
                {
                    'name': f.exchange_name,
                    'inflow': float(f.inflow),
                    'outflow': float(f.outflow),
                    'net_flow': float(f.net_flow)
                }
                for f in flows
            ]
        }
    
    def generate_signal(self, lookback_hours: int = 24) -> Dict:
        """
        Generate trading signal based on exchange flow.
        
        Returns:
            Signal dict with direction, confidence, reasoning
        """
        flow = self.get_aggregate_eth_flow(lookback_hours)
        
        # Thresholds for ETH
        STRONG_INFLOW = 50000  # >50K ETH inflow = strong bearish
        STRONG_OUTFLOW = 50000  # >50K ETH outflow = strong bullish
        MEDIUM_THRESHOLD = 20000
        
        net_flow = flow['net_flow']
        
        if net_flow < -STRONG_INFLOW:
            return {
                'signal': 'SHORT',
                'confidence': 0.70,
                'reasoning': f"Strong ETH inflow to exchanges: {abs(net_flow):.0f} ETH in {lookback_hours}h (selling pressure)",
                'data': flow
            }
        elif net_flow < -MEDIUM_THRESHOLD:
            return {
                'signal': 'SHORT',
                'confidence': 0.55,
                'reasoning': f"Moderate ETH inflow to exchanges: {abs(net_flow):.0f} ETH in {lookback_hours}h",
                'data': flow
            }
        elif net_flow > STRONG_OUTFLOW:
            return {
                'signal': 'LONG',
                'confidence': 0.65,
                'reasoning': f"Strong ETH outflow from exchanges: {net_flow:.0f} ETH in {lookback_hours}h (accumulation)",
                'data': flow
            }
        elif net_flow > MEDIUM_THRESHOLD:
            return {
                'signal': 'LONG',
                'confidence': 0.55,
                'reasoning': f"Moderate ETH outflow from exchanges: {net_flow:.0f} ETH in {lookback_hours}h",
                'data': flow
            }
        else:
            return {
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'reasoning': "No significant exchange flow detected",
                'data': flow
            }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("Exchange Flow Monitor Demo")
    print("="*80)
    print()
    
    monitor = ExchangeFlowMonitor()
    
    print("📊 Calculating exchange flows...")
    print()
    
    # Get aggregate flow
    flow = monitor.get_aggregate_eth_flow(lookback_hours=24)
    
    print("24-Hour Exchange Flow Summary:")
    print(f"  Total Inflow: {flow['total_inflow']:.2f} ETH")
    print(f"  Total Outflow: {flow['total_outflow']:.2f} ETH")
    print(f"  Net Flow: {flow['net_flow']:.2f} ETH")
    print(f"  Direction: {'📈 BULLISH (Accumulation)' if flow['is_bullish'] else '📉 BEARISH (Distribution)'}")
    print()
    
    print("By Exchange:")
    for ex in flow['exchanges']:
        print(f"  {ex['name']}:")
        print(f"    Inflow: {ex['inflow']:.2f} ETH")
        print(f"    Outflow: {ex['outflow']:.2f} ETH")
        print(f"    Net: {ex['net_flow']:.2f} ETH")
    print()
    
    # Generate signal
    signal = monitor.generate_signal()
    print("🎯 Trading Signal:")
    print(f"  Signal: {signal['signal']}")
    print(f"  Confidence: {signal['confidence']:.0%}")
    print(f"  Reasoning: {signal['reasoning']}")
    print()
    
    print("✅ Exchange flow monitor demo complete")

