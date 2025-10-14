from .etherscan_client import EtherscanClient
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Optional, Set
import json
import logging


"""
Whale Tracker - Monitor large cryptocurrency movements.

Tracks large transactions (>$1M) from whale wallets to exchanges,
which historically correlates with price drops within 2-6 hours.

Hypothesis 001: Whale Exchange Deposits Predict Price Drops
"""



logger = logging.getLogger(__name__)


@dataclass
class WhaleTransaction:
    """Represents a large crypto transaction from whale to exchange"""
    tx_hash: str
    timestamp: datetime
    from_address: str
    to_address: str
    value: Decimal  # Amount in native currency (BTC or ETH)
    value_usd: float  # USD value at time of transaction
    asset: str  # 'BTC' or 'ETH' or token symbol
    exchange_name: Optional[str]  # 'Binance', 'Coinbase', etc.
    is_to_exchange: bool  # True if deposit to exchange, False if withdrawal
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        d['value'] = str(self.value)
        return d
    
    def signal_strength(self) -> str:
        """Calculate signal strength based on transaction characteristics"""
        # Larger transfers = stronger signal
        if self.value_usd > 100_000_000:  # >$100M
            return "CRITICAL"
        elif self.value_usd > 50_000_000:  # >$50M
            return "STRONG"
        elif self.value_usd > 10_000_000:  # >$10M
            return "MEDIUM"
        else:  # >$1M
            return "WEAK"


class WhaleTracker:
    """
    Track whale wallet transactions and detect potential market-moving events.
    
    Monitors known whale addresses and exchange deposit addresses to detect
    large transfers that may precede price movements.
    """
    
    # Known exchange deposit addresses (Ethereum)
    # In production, maintain a comprehensive database of exchange addresses
    EXCHANGE_ADDRESSES = {
        # Binance (example addresses - update with current ones)
        "0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE": "Binance",
        "0xD551234Ae421e3BCBA99A0Da6d736074f22192FF": "Binance",
        "0x564286362092D8e7936f0549571a803B203aAceD": "Binance",
        
        # Coinbase (example addresses)
        "0x71660c4005BA85c37ccec55d0C4493E66Fe775d3": "Coinbase",
        "0x503828976D22510aad0201ac7EC88293211D23Da": "Coinbase",
        "0xddfAbCdc4D8FfC6d5beaf154f18B778f892A0740": "Coinbase",
        
        # Kraken (example addresses)  
        "0x2910543Af39abA0Cd09dBb2D50200b3E800A63D2": "Kraken",
        "0x0A869d79a7052C7f1b55a8EbAbbEa3420F0D1E13": "Kraken",
        
        # Bitfinex (example addresses)
        "0x876EabF441B2EE5B5b0554Fd502a8E0600950cFa": "Bitfinex",
        "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb": "Bitfinex",
    }
    
    # Known whale addresses (update with current large holders)
    # These are examples - in production, dynamically identify from on-chain data
    KNOWN_WHALES = {
        "0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8": "Binance Cold Wallet",
        "0xF977814e90dA44bFA03b6295A0616a897441aceC": "Binance Cold Wallet 2",
        "0x8103683202aa8DA10536036EDef04CDd865C225E": "Large Holder 1",
        # Add more whale addresses
    }
    
    def __init__(
        self,
        etherscan_client: Optional[EtherscanClient] = None,
        min_value_usd: float = 1_000_000,  # $1M minimum
        lookback_hours: int = 24
    ):
        """
        Initialize whale tracker.
        
        Args:
            etherscan_client: Etherscan API client (if None, creates new one)
            min_value_usd: Minimum transaction value to track (USD)
            lookback_hours: How far back to look for transactions
        """
        self.etherscan = etherscan_client or EtherscanClient()
        self.min_value_usd = min_value_usd
        self.lookback_hours = lookback_hours
        
        # Tracking state
        self.tracked_transactions: List[WhaleTransaction] = []
        self.last_check_time: Optional[datetime] = None
    
    def is_exchange_address(self, address: str) -> Optional[str]:
        """
        Check if address is a known exchange.
        
        Returns:
            Exchange name if known, None otherwise
        """
        return self.EXCHANGE_ADDRESSES.get(address)
    
    def is_whale_address(self, address: str) -> bool:
        """Check if address is a known whale wallet"""
        return address in self.KNOWN_WHALES
    
    @lru_cache(maxsize=128)
    
    def get_eth_price_usd(self) -> float:
        """
        Get current ETH price in USD.
        
        TODO: Integrate with Polygon.io or Coinbase API for real price
        For now, returns placeholder value
        """
        # In production, fetch from Polygon.io or Coinbase
        return 3000.0  # Placeholder
    
    def scan_whale_address(
        self,
        whale_address: str,
        start_block: Optional[int] = None,
        end_block: Optional[int] = None
    ) -> List[WhaleTransaction]:
        """
        Scan a whale address for large transactions to exchanges.
        
        Args:
            whale_address: Ethereum address to monitor
            start_block: Starting block (if None, uses recent blocks)
            end_block: Ending block (if None, uses latest)
            
        Returns:
            List of whale transactions to exchanges
        """
        logger.info(f"Scanning whale address {whale_address[:10]}...")
        
        # Get recent transactions
        txs = self.etherscan.get_normal_transactions(
            address=whale_address,
            start_block=start_block or 0,
            end_block=end_block or 99999999,
            sort="desc"
        )
        
        eth_price = self.get_eth_price_usd()
        whale_txs = []
        
        for tx in txs:
            # Parse transaction
            to_address = tx.get('to', '').lower()
            from_address = tx.get('from', '').lower()
            value_wei = int(tx.get('value', 0))
            value_eth = Decimal(value_wei) / Decimal(10**18)
            value_usd = float(value_eth) * eth_price
            
            # Check if this is a large transfer to an exchange
            if value_usd >= self.min_value_usd:
                exchange_name = self.is_exchange_address(to_address)
                
                if exchange_name:
                    # Whale depositing to exchange (bearish signal)
                    tx_time = datetime.fromtimestamp(int(tx.get('timeStamp', 0)))
                    
                    whale_tx = WhaleTransaction(
                        tx_hash=tx.get('hash', ''),
                        timestamp=tx_time,
                        from_address=from_address,
                        to_address=to_address,
                        value=value_eth,
                        value_usd=value_usd,
                        asset='ETH',
                        exchange_name=exchange_name,
                        is_to_exchange=True
                    )
                    whale_txs.append(whale_tx)
                    logger.info(
                        f"🐋 Whale transaction detected: {value_usd/1e6:.1f}M USD "
                        f"to {exchange_name} at {tx_time}"
                    )
        
        return whale_txs
    
    def scan_all_whales(
        self,
        lookback_hours: Optional[int] = None
    ) -> List[WhaleTransaction]:
        """
        Scan all known whale addresses for recent large transactions.
        
        Args:
            lookback_hours: How far back to scan (if None, uses instance default)
            
        Returns:
            List of all whale transactions found
        """
        lookback = lookback_hours or self.lookback_hours
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback)
        
        # Convert timestamp to block number
        cutoff_timestamp = int(cutoff_time.timestamp())
        start_block = self.etherscan.get_block_number_by_timestamp(cutoff_timestamp)
        
        logger.info(f"Scanning {len(self.KNOWN_WHALES)} whale addresses...")
        logger.info(f"Lookback: {lookback} hours (block {start_block})")
        
        all_whale_txs = []
        for whale_addr in self.KNOWN_WHALES.keys():
            try:
                whale_txs = self.scan_whale_address(whale_addr, start_block=start_block)
                all_whale_txs.extend(whale_txs)
            except Exception as e:
                logger.error(f"Error scanning {whale_addr}: {e}")
                continue
        
        self.tracked_transactions = all_whale_txs
        self.last_check_time = datetime.utcnow()
        
        return all_whale_txs
    
    @lru_cache(maxsize=128)
    
    def get_aggregate_flow(
        self,
        time_window_hours: int = 24
    ) -> Dict[str, float]:
        """
        Calculate aggregate flow metrics over time window.
        
        Returns:
            Dict with flow metrics:
            - total_to_exchanges: Total USD flowing to exchanges
            - total_from_exchanges: Total USD flowing from exchanges
            - net_flow: Net flow (negative = more deposits = bearish)
            - transaction_count: Number of whale transactions
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        # Filter transactions within time window
        recent_txs = [
            tx for tx in self.tracked_transactions
            if tx.timestamp >= cutoff_time
        ]
        
        total_to_exchanges = sum(
            tx.value_usd for tx in recent_txs if tx.is_to_exchange
        )
        total_from_exchanges = sum(
            tx.value_usd for tx in recent_txs if not tx.is_to_exchange
        )
        
        return {
            'total_to_exchanges': total_to_exchanges,
            'total_from_exchanges': total_from_exchanges,
            'net_flow': total_from_exchanges - total_to_exchanges,  # Positive = withdrawals (bullish)
            'transaction_count': len(recent_txs),
            'time_window_hours': time_window_hours
        }
    
    def generate_signal(self) -> Dict:
        """
        Generate trading signal based on recent whale activity.
        
        Returns:
            Signal dict with:
            - signal: 'SHORT' | 'LONG' | 'NEUTRAL'
            - confidence: 0.0-1.0
            - reasoning: Explanation
            - data: Supporting metrics
        """
        flow = self.get_aggregate_flow(time_window_hours=2)  # 2-hour window
        
        # Threshold: $50M to exchanges triggers short signal
        THRESHOLD_STRONG = 50_000_000
        THRESHOLD_MEDIUM = 20_000_000
        
        if flow['total_to_exchanges'] > THRESHOLD_STRONG:
            return {
                'signal': 'SHORT',
                'confidence': 0.75,
                'reasoning': f"Large whale deposits to exchanges: ${flow['total_to_exchanges']/1e6:.1f}M USD in 2 hours",
                'data': flow
            }
        elif flow['total_to_exchanges'] > THRESHOLD_MEDIUM:
            return {
                'signal': 'SHORT',
                'confidence': 0.55,
                'reasoning': f"Moderate whale deposits to exchanges: ${flow['total_to_exchanges']/1e6:.1f}M USD in 2 hours",
                'data': flow
            }
        elif flow['total_from_exchanges'] > THRESHOLD_MEDIUM:
            return {
                'signal': 'LONG',
                'confidence': 0.60,
                'reasoning': f"Whale withdrawals from exchanges: ${flow['total_from_exchanges']/1e6:.1f}M USD in 2 hours (accumulation)",
                'data': flow
            }
        else:
            return {
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'reasoning': "No significant whale activity detected",
                'data': flow
            }
    
    def save_transactions(self, filepath: Path):
        """Save tracked transactions to JSON file"""
        data = {
            'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
            'transaction_count': len(self.tracked_transactions),
            'transactions': [tx.to_dict() for tx in self.tracked_transactions]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.tracked_transactions)} transactions to {filepath}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("Whale Tracker Demo")
    print("="*80)
    print()
    
    # Initialize tracker
    tracker = WhaleTracker(min_value_usd=1_000_000, lookback_hours=168)  # 1 week
    
    # Scan all known whales
    print("📡 Scanning whale addresses for large transactions...")
    print(f"   Tracking {len(WhaleTracker.KNOWN_WHALES)} whale addresses")
    print(f"   Minimum transaction size: ${tracker.min_value_usd/1e6:.1f}M USD")
    print()
    
    whale_txs = tracker.scan_all_whales()
    
    print(f"\n🐋 Found {len(whale_txs)} whale transactions to exchanges")
    print()
    
    # Show top 5 largest
    if whale_txs:
        print("Top 5 Largest Transactions:")
        sorted_txs = sorted(whale_txs, key=lambda x: x.value_usd, reverse=True)
        for i, tx in enumerate(sorted_txs[:5], 1):
            print(f"{i}. ${tx.value_usd/1e6:.1f}M USD → {tx.exchange_name}")
            print(f"   Time: {tx.timestamp}")
            print(f"   Signal: {tx.signal_strength()}")
            print()
    
    # Calculate aggregate flow
    flow = tracker.get_aggregate_flow(time_window_hours=24)
    print("📊 24-Hour Aggregate Flow:")
    print(f"   To Exchanges: ${flow['total_to_exchanges']/1e6:.1f}M USD")
    print(f"   From Exchanges: ${flow['total_from_exchanges']/1e6:.1f}M USD")
    print(f"   Net Flow: ${flow['net_flow']/1e6:.1f}M USD")
    print(f"   Transactions: {flow['transaction_count']}")
    print()
    
    # Generate trading signal
    signal = tracker.generate_signal()
    print("🎯 Trading Signal:")
    print(f"   Signal: {signal['signal']}")
    print(f"   Confidence: {signal['confidence']:.0%}")
    print(f"   Reasoning: {signal['reasoning']}")
    print()
    
    print("✅ Whale tracker demo complete")
    print()
    print("⚠️  Note: This demo uses example whale/exchange addresses.")
    print("   For production, maintain updated database of current addresses.")

