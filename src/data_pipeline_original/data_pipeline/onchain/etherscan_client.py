from datetime import datetime
from decimal import Decimal
from functools import lru_cache
from typing import List, Dict, Optional
import logging
import os
import requests
import time


"""
Etherscan API client for Ethereum blockchain data.

Free tier: 5 requests/second, 100,000 requests/day
No API key required for basic usage, but recommended for higher limits.
"""


logger = logging.getLogger(__name__)


class EtherscanClient:
    """Client for Etherscan API to fetch Ethereum transactions and account data"""
    
    BASE_URL = "https://api.etherscan.io/api"
    
    def __init__(self, api_key: Optional[str] = None, rate_limit: float = 0.2):
        """
        Initialize Etherscan client.
        
        Args:
            api_key: Etherscan API key (optional, but recommended)
            rate_limit: Seconds between requests (default 0.2 = 5 req/sec)
        """
        self.api_key = api_key or os.getenv("ETHERSCAN_API_KEY", "")
        self.rate_limit = rate_limit
        self.last_request_time = 0.0
        self.session = requests.Session()
        
        if not self.api_key:
            logger.warning("No Etherscan API key provided. Using free tier with strict limits.")
    
    def _wait_if_needed(self):
        """Rate limiting: wait if we're exceeding request limits"""
        if self.rate_limit > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit:
                time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, params: Dict) -> Dict:
        """Make API request with rate limiting and error handling"""
        self._wait_if_needed()
        
        # Add API key if available
        if self.api_key:
            params['apikey'] = self.api_key
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == '0' and 'rate limit' in data.get('message', '').lower():
                logger.warning("Rate limit hit, waiting 1 second...")
                time.sleep(1)
                return self._make_request(params)  # Retry once
            
            if data.get('status') == '0':
                logger.error(f"Etherscan API error: {data.get('message')}")
                return {'status': '0', 'result': []}
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {'status': '0', 'result': []}
    
    @lru_cache(maxsize=128)
    
    def get_normal_transactions(
        self, 
        address: str,
        start_block: int = 0,
        end_block: int = 99999999,
        sort: str = "desc"
    ) -> List[Dict]:
        """
        Get list of normal transactions for an address.
        
        Args:
            address: Ethereum address (0x...)
            start_block: Starting block number
            end_block: Ending block number
            sort: 'asc' or 'desc'
            
        Returns:
            List of transaction dictionaries
        """
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'startblock': start_block,
            'endblock': end_block,
            'sort': sort
        }
        
        data = self._make_request(params)
        return data.get('result', [])
    
    @lru_cache(maxsize=128)
    
    def get_internal_transactions(
        self,
        address: str,
        start_block: int = 0,
        end_block: int = 99999999,
        sort: str = "desc"
    ) -> List[Dict]:
        """Get list of internal transactions (contract transfers) for an address"""
        params = {
            'module': 'account',
            'action': 'txlistinternal',
            'address': address,
            'startblock': start_block,
            'endblock': end_block,
            'sort': sort
        }
        
        data = self._make_request(params)
        return data.get('result', [])
    
    @lru_cache(maxsize=128)
    
    def get_erc20_transfers(
        self,
        address: Optional[str] = None,
        contract_address: Optional[str] = None,
        start_block: int = 0,
        end_block: int = 99999999,
        sort: str = "desc"
    ) -> List[Dict]:
        """
        Get ERC-20 token transfer events.
        
        Args:
            address: Filter by address (sender or receiver)
            contract_address: Filter by token contract
            start_block: Starting block number
            end_block: Ending block number
            sort: 'asc' or 'desc'
        """
        params = {
            'module': 'account',
            'action': 'tokentx',
            'startblock': start_block,
            'endblock': end_block,
            'sort': sort
        }
        
        if address:
            params['address'] = address
        if contract_address:
            params['contractaddress'] = contract_address
        
        data = self._make_request(params)
        return data.get('result', [])
    
    @lru_cache(maxsize=128)
    
    def get_eth_balance(self, address: str) -> Decimal:
        """
        Get ETH balance for an address.
        
        Returns:
            Balance in ETH (not wei)
        """
        params = {
            'module': 'account',
            'action': 'balance',
            'address': address,
            'tag': 'latest'
        }
        
        data = self._make_request(params)
        if data.get('status') == '1':
            wei_balance = int(data.get('result', 0))
            return Decimal(wei_balance) / Decimal(10**18)
        return Decimal(0)
    
    @lru_cache(maxsize=128)
    
    def get_block_number_by_timestamp(self, timestamp: int, closest: str = "before") -> int:
        """
        Get block number by Unix timestamp.
        
        Args:
            timestamp: Unix timestamp
            closest: 'before' or 'after'
            
        Returns:
            Block number
        """
        params = {
            'module': 'block',
            'action': 'getblocknobytime',
            'timestamp': timestamp,
            'closest': closest
        }
        
        data = self._make_request(params)
        if data.get('status') == '1':
            return int(data.get('result', 0))
        return 0
    
    @lru_cache(maxsize=128)
    
    def get_large_transactions(
        self,
        min_value_eth: float = 100.0,
        start_block: Optional[int] = None,
        end_block: Optional[int] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Get large ETH transactions (whale transactions).
        
        Note: This is a convenience method that's limited. For production,
        you'd need to either:
        1. Monitor known whale addresses
        2. Use a paid service like Glassnode
        3. Run your own Ethereum node
        
        Args:
            min_value_eth: Minimum transaction value in ETH
            start_block: Starting block (if None, uses recent blocks)
            end_block: Ending block
            limit: Max number of transactions to return
            
        Returns:
            List of large transactions
        """
        # For free tier, we need to know specific addresses
        # This is a placeholder - you'd need to maintain a list of known whale addresses
        logger.warning(
            "get_large_transactions requires known whale addresses. "
            "Consider using Glassnode or maintaining a whale address list."
        )
        return []
    
    @lru_cache(maxsize=128)
    
    def get_gas_oracle(self) -> Dict:
        """
        Get current gas price recommendations.
        
        Returns:
            Dict with SafeGasPrice, ProposeGasPrice, FastGasPrice in Gwei
        """
        params = {
            'module': 'gastracker',
            'action': 'gasoracle'
        }
        
        data = self._make_request(params)
        if data.get('status') == '1':
            result = data.get('result', {})
            return {
                'safe': float(result.get('SafeGasPrice', 0)),
                'proposed': float(result.get('ProposeGasPrice', 0)),
                'fast': float(result.get('FastGasPrice', 0)),
                'timestamp': datetime.utcnow().isoformat()
            }
        return {'safe': 0, 'proposed': 0, 'fast': 0}


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Initialize client
    client = EtherscanClient()
    
    print("="*80)
    print("Etherscan Client Demo")
    print("="*80)
    print()
    
    # Test 1: Get gas prices
    print("📊 Current Gas Prices:")
    gas_prices = client.get_gas_oracle()
    print(f"  Safe: {gas_prices['safe']} Gwei")
    print(f"  Proposed: {gas_prices['proposed']} Gwei")
    print(f"  Fast: {gas_prices['fast']} Gwei")
    print()
    
    # Test 2: Get balance for a known address (Vitalik's public address)
    vitalik_address = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    print(f"💰 ETH Balance for {vitalik_address[:10]}...:")
    balance = client.get_eth_balance(vitalik_address)
    print(f"  Balance: {balance:.4f} ETH")
    print()
    
    # Test 3: Get recent transactions
    print(f"📝 Recent Transactions for {vitalik_address[:10]}...:")
    txs = client.get_normal_transactions(vitalik_address, sort="desc")
    if txs:
        for tx in txs[:3]:  # Show first 3
            value_eth = int(tx.get('value', 0)) / 10**18
            print(f"  - Hash: {tx.get('hash', '')[:16]}...")
            print(f"    Value: {value_eth:.4f} ETH")
            print(f"    From: {tx.get('from', '')[:16]}...")
            print(f"    To: {tx.get('to', '')[:16]}...")
            print()
    else:
        print("  No transactions found")
    
    print("✅ Etherscan client test complete")

