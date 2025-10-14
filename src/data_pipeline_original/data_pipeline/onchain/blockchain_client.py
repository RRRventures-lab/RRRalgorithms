from datetime import datetime
from decimal import Decimal
from functools import lru_cache
from typing import List, Dict, Optional
import logging
import requests
import time


"""
Blockchain.com API client for Bitcoin blockchain data.

Free tier: No API key required for basic usage
Rate limits: 1 request per 10 seconds on free tier
"""


logger = logging.getLogger(__name__)


class BlockchainClient:
    """Client for Blockchain.com API to fetch Bitcoin transactions and data"""
    
    BASE_URL = "https://blockchain.info"
    
    def __init__(self, rate_limit: float = 10.0):
        """
        Initialize Blockchain.com client.
        
        Args:
            rate_limit: Seconds between requests (default 10 = 1 req/10sec)
        """
        self.rate_limit = rate_limit
        self.last_request_time = 0.0
        self.session = requests.Session()
    
    def _wait_if_needed(self):
        """Rate limiting: wait if we're exceeding request limits"""
        if self.rate_limit > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit:
                time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with rate limiting and error handling"""
        self._wait_if_needed()
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {}
    
    @lru_cache(maxsize=128)
    
    def get_address_info(self, address: str) -> Dict:
        """
        Get address information including balance and transaction count.
        
        Returns:
            Dict with address, balance (satoshis), tx_count, etc.
        """
        endpoint = f"address/{address}"
        params = {'format': 'json'}
        return self._make_request(endpoint, params)
    
    @lru_cache(maxsize=128)
    
    def get_address_transactions(
        self,
        address: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """
        Get transactions for an address.
        
        Args:
            address: Bitcoin address
            limit: Number of transactions to return (max 100)
            offset: Number of transactions to skip
            
        Returns:
            List of transaction dictionaries
        """
        endpoint = f"rawaddr/{address}"
        params = {
            'format': 'json',
            'limit': min(limit, 100),
            'offset': offset
        }
        
        data = self._make_request(endpoint, params)
        return data.get('txs', [])
    
    @lru_cache(maxsize=128)
    
    def get_btc_balance(self, address: str) -> Decimal:
        """
        Get BTC balance for an address.
        
        Returns:
            Balance in BTC (not satoshis)
        """
        info = self.get_address_info(address)
        satoshis = info.get('final_balance', 0)
        return Decimal(satoshis) / Decimal(10**8)
    
    @lru_cache(maxsize=128)
    
    def get_large_transactions(
        self,
        address: str,
        min_value_btc: float = 100.0
    ) -> List[Dict]:
        """
        Get large transactions from an address.
        
        Args:
            address: Bitcoin address to monitor
            min_value_btc: Minimum transaction value in BTC
            
        Returns:
            List of large transactions
        """
        txs = self.get_address_transactions(address, limit=100)
        large_txs = []
        
        for tx in txs:
            # Calculate total output value
            for output in tx.get('out', []):
                value_satoshis = output.get('value', 0)
                value_btc = value_satoshis / 10**8
                
                if value_btc >= min_value_btc:
                    large_txs.append({
                        'hash': tx.get('hash'),
                        'time': datetime.fromtimestamp(tx.get('time', 0)),
                        'value_btc': value_btc,
                        'output_address': output.get('addr')
                    })
        
        return large_txs


# Example usage
if __name__ == "__main__":
    client = BlockchainClient()
    
    print("="*80)
    print("Blockchain.com Client Demo")
    print("="*80)
    print()
    
    # Example Bitcoin whale address (public exchange cold wallet)
    test_address = "1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s"  # Bitfinex cold wallet
    
    print(f"📊 Address Info for {test_address[:20]}...")
    info = client.get_address_info(test_address)
    print(f"  Balance: {info.get('final_balance', 0) / 10**8:.8f} BTC")
    print(f"  Transactions: {info.get('n_tx', 0)}")
    print()
    
    print("✅ Blockchain.com client test complete")

