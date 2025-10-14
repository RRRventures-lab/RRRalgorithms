# DeFi Integration - Technical Implementation Guide

**Audience**: Engineering Team (Solidity, Python, DevOps)
**Date**: October 11, 2025
**Version**: 1.0

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Smart Contract Specifications](#smart-contract-specifications)
3. [Off-Chain Bot Implementation](#off-chain-bot-implementation)
4. [Infrastructure Setup](#infrastructure-setup)
5. [Testing Strategy](#testing-strategy)
6. [Deployment Checklist](#deployment-checklist)

---

## 1. Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         RRRalgorithms DeFi System               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐         ┌──────────────────┐               │
│  │  Ethereum Node │◄────────┤ Flashbots Protect│               │
│  │   (Geth/Erigon)│         │       RPC         │               │
│  └────────┬───────┘         └──────────────────┘               │
│           │                                                      │
│           │ Web3.py                                             │
│           ▼                                                      │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              Off-Chain Trading Bot (Python)          │      │
│  ├──────────────────────────────────────────────────────┤      │
│  │ • Arbitrage Scanner                                  │      │
│  │ • LP Position Manager                                │      │
│  │ • Yield Optimizer                                    │      │
│  │ • Gas Price Monitor                                  │      │
│  │ • Risk Monitor                                       │      │
│  └─────┬────────────────────────────────────────────────┘      │
│        │                                                         │
│        │ Smart Contract Calls                                   │
│        ▼                                                         │
│  ┌──────────────────────────────────────────────────────┐      │
│  │          Smart Contracts (Solidity)                  │      │
│  ├──────────────────────────────────────────────────────┤      │
│  │ • TradingVault (capital management)                  │      │
│  │ • StrategyRouter (strategy selection)                │      │
│  │ • UniswapV4Integration (DEX swaps, LP)               │      │
│  │ • CurveIntegration (stablecoin swaps)                │      │
│  │ • FlashLoanArbitrage (Aave flash loans)              │      │
│  │ • MultiSigWallet (governance)                        │      │
│  └─────┬────────────────────────────────────────────────┘      │
│        │                                                         │
│        │ DEX Interactions                                       │
│        ▼                                                         │
│  ┌──────────────────────────────────────────────────────┐      │
│  │               DeFi Protocols                         │      │
│  ├──────────────────────────────────────────────────────┤      │
│  │ • Uniswap V4 (spot, LP)                              │      │
│  │ • Curve (stablecoin swaps)                           │      │
│  │ • Aave (flash loans, lending)                        │      │
│  │ • 1inch (DEX aggregator)                             │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Market Data Ingestion**: Polygon.io WebSocket → Data Pipeline → TimescaleDB
2. **Opportunity Detection**: Python bot queries DEX prices via The Graph subgraphs
3. **Signal Generation**: Neural network predicts profitable arbitrage/LP opportunities
4. **Execution**: Bot calls smart contract → Flashbots Protect RPC → Ethereum mainnet
5. **Monitoring**: Prometheus scrapes metrics → Grafana dashboards → Alert Manager

---

## 2. Smart Contract Specifications

### 2.1 TradingVault.sol

**Purpose**: Securely hold capital and route to approved strategies.

**Key Functions**:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

contract TradingVault is Initializable, AccessControlUpgradeable {
    using SafeERC20 for IERC20;

    bytes32 public constant STRATEGIST_ROLE = keccak256("STRATEGIST_ROLE");
    bytes32 public constant GUARDIAN_ROLE = keccak256("GUARDIAN_ROLE");

    // Approved strategies
    mapping(address => bool) public strategies;

    // Capital allocation per strategy
    mapping(address => uint256) public allocations;

    // Total value locked
    uint256 public totalValueLocked;

    // Events
    event StrategyApproved(address indexed strategy, uint256 allocation);
    event StrategyRevoked(address indexed strategy);
    event CapitalDeployed(address indexed strategy, uint256 amount);
    event ProfitRealized(address indexed strategy, uint256 profit);
    event EmergencyWithdrawal(address indexed token, uint256 amount);

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    function initialize(address admin) public initializer {
        __AccessControl_init();
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(GUARDIAN_ROLE, admin);
    }

    /**
     * @notice Approve a new strategy
     * @param strategy Address of strategy contract
     * @param allocation Maximum capital allocation
     */
    function approveStrategy(
        address strategy,
        uint256 allocation
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(strategy != address(0), "Invalid strategy");
        strategies[strategy] = true;
        allocations[strategy] = allocation;
        emit StrategyApproved(strategy, allocation);
    }

    /**
     * @notice Execute a strategy
     * @param strategy Strategy contract address
     * @param data Encoded function call
     * @return profit Amount of profit generated
     */
    function executeStrategy(
        address strategy,
        bytes calldata data
    ) external onlyRole(STRATEGIST_ROLE) returns (uint256 profit) {
        require(strategies[strategy], "Strategy not approved");

        // Record TVL before
        uint256 balanceBefore = address(this).balance;

        // Execute strategy
        (bool success, bytes memory result) = strategy.call(data);
        require(success, "Strategy execution failed");

        // Calculate profit
        uint256 balanceAfter = address(this).balance;
        profit = balanceAfter > balanceBefore ? balanceAfter - balanceBefore : 0;

        emit ProfitRealized(strategy, profit);
        return profit;
    }

    /**
     * @notice Emergency withdrawal (guardian only)
     * @param token Token to withdraw (address(0) for ETH)
     */
    function emergencyWithdraw(
        address token
    ) external onlyRole(GUARDIAN_ROLE) {
        if (token == address(0)) {
            // Withdraw ETH
            uint256 balance = address(this).balance;
            payable(msg.sender).transfer(balance);
            emit EmergencyWithdrawal(token, balance);
        } else {
            // Withdraw ERC20
            IERC20 tokenContract = IERC20(token);
            uint256 balance = tokenContract.balanceOf(address(this));
            tokenContract.safeTransfer(msg.sender, balance);
            emit EmergencyWithdrawal(token, balance);
        }
    }

    // Allow contract to receive ETH
    receive() external payable {}
}
```

### 2.2 UniswapV4Integration.sol

**Purpose**: Execute swaps and manage liquidity positions on Uniswap V4.

**Key Functions**:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@uniswap/v4-core/contracts/interfaces/IPoolManager.sol";
import "@uniswap/v4-core/contracts/libraries/PoolKey.sol";

contract UniswapV4Integration {
    IPoolManager public poolManager;

    constructor(address _poolManager) {
        poolManager = IPoolManager(_poolManager);
    }

    /**
     * @notice Execute a swap on Uniswap V4
     * @param key Pool key (tokenA, tokenB, fee, tickSpacing, hooks)
     * @param zeroForOne Swap direction (token0 -> token1)
     * @param amountSpecified Amount to swap (negative = exactInput)
     * @param sqrtPriceLimitX96 Price limit (0 = no limit)
     */
    function swap(
        PoolKey memory key,
        bool zeroForOne,
        int256 amountSpecified,
        uint160 sqrtPriceLimitX96
    ) external returns (int256 amount0, int256 amount1) {
        IPoolManager.SwapParams memory params = IPoolManager.SwapParams({
            zeroForOne: zeroForOne,
            amountSpecified: amountSpecified,
            sqrtPriceLimitX96: sqrtPriceLimitX96
        });

        (amount0, amount1) = poolManager.swap(key, params, "");
        return (amount0, amount1);
    }

    /**
     * @notice Add liquidity to a concentrated position
     * @param key Pool key
     * @param tickLower Lower tick of position
     * @param tickUpper Upper tick of position
     * @param liquidity Amount of liquidity to add
     */
    function addLiquidity(
        PoolKey memory key,
        int24 tickLower,
        int24 tickUpper,
        uint128 liquidity
    ) external returns (uint256 amount0, uint256 amount1) {
        IPoolManager.ModifyLiquidityParams memory params =
            IPoolManager.ModifyLiquidityParams({
                tickLower: tickLower,
                tickUpper: tickUpper,
                liquidityDelta: int256(uint256(liquidity)),
                salt: bytes32(0)
            });

        (int256 delta0, int256 delta1) = poolManager.modifyLiquidity(key, params, "");
        return (uint256(delta0), uint256(delta1));
    }

    /**
     * @notice Remove liquidity from a position
     */
    function removeLiquidity(
        PoolKey memory key,
        int24 tickLower,
        int24 tickUpper,
        uint128 liquidity
    ) external returns (uint256 amount0, uint256 amount1) {
        IPoolManager.ModifyLiquidityParams memory params =
            IPoolManager.ModifyLiquidityParams({
                tickLower: tickLower,
                tickUpper: tickUpper,
                liquidityDelta: -int256(uint256(liquidity)),
                salt: bytes32(0)
            });

        (int256 delta0, int256 delta1) = poolManager.modifyLiquidity(key, params, "");
        return (uint256(-delta0), uint256(-delta1));
    }
}
```

### 2.3 FlashLoanArbitrage.sol

**Purpose**: Execute risk-free arbitrage using Aave flash loans.

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@aave/core-v3/contracts/flashloan/base/FlashLoanSimpleReceiverBase.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract FlashLoanArbitrage is FlashLoanSimpleReceiverBase {
    address public owner;

    constructor(address _addressProvider)
        FlashLoanSimpleReceiverBase(IPoolAddressesProvider(_addressProvider))
    {
        owner = msg.sender;
    }

    /**
     * @notice Execute flash loan arbitrage
     * @param asset Token to borrow
     * @param amount Amount to borrow
     * @param params Encoded arbitrage path
     */
    function executeArbitrage(
        address asset,
        uint256 amount,
        bytes calldata params
    ) external {
        require(msg.sender == owner, "Not owner");

        // Request flash loan from Aave
        POOL.flashLoanSimple(
            address(this),
            asset,
            amount,
            params,
            0 // referralCode
        );
    }

    /**
     * @notice Callback executed by Aave after flash loan
     * @dev Must repay loan + premium within this function
     */
    function executeOperation(
        address asset,
        uint256 amount,
        uint256 premium,
        address initiator,
        bytes calldata params
    ) external override returns (bool) {
        require(msg.sender == address(POOL), "Not Aave pool");

        // Decode arbitrage path
        (address dexA, address dexB, address intermediateToken) =
            abi.decode(params, (address, address, address));

        // Step 1: Swap on DEX A (e.g., Uniswap)
        // ETH → USDC at DEX A
        uint256 intermediateAmount = _swapOnDex(dexA, asset, intermediateToken, amount);

        // Step 2: Swap on DEX B (e.g., Curve)
        // USDC → ETH at DEX B (higher price)
        uint256 finalAmount = _swapOnDex(dexB, intermediateToken, asset, intermediateAmount);

        // Check profit (must cover flash loan premium)
        uint256 totalDebt = amount + premium;
        require(finalAmount > totalDebt, "Arbitrage not profitable");

        // Approve Aave to pull repayment
        IERC20(asset).approve(address(POOL), totalDebt);

        return true;
    }

    function _swapOnDex(
        address dex,
        address tokenIn,
        address tokenOut,
        uint256 amountIn
    ) internal returns (uint256 amountOut) {
        // Call DEX router to execute swap
        // Implementation depends on DEX (Uniswap, Curve, etc.)
        // Return amount received
    }
}
```

---

## 3. Off-Chain Bot Implementation

### 3.1 Project Structure

```
worktrees/defi-integration/
├── src/
│   ├── blockchain/
│   │   ├── __init__.py
│   │   ├── node_client.py           # Web3 connection manager
│   │   ├── transaction_manager.py   # Nonce, gas, signing
│   │   └── event_listener.py        # Listen to contract events
│   ├── contracts/
│   │   ├── __init__.py
│   │   ├── vault.py                 # TradingVault interface
│   │   ├── uniswap_v4.py            # Uniswap V4 interface
│   │   ├── curve.py                 # Curve interface
│   │   └── aave.py                  # Aave flash loan interface
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── arbitrage_scanner.py     # Detect arbitrage opportunities
│   │   ├── lp_manager.py            # Manage LP positions
│   │   ├── yield_optimizer.py       # Optimize yield allocation
│   │   └── rebalancer.py            # Rebalance portfolio
│   ├── mev/
│   │   ├── __init__.py
│   │   ├── flashbots_client.py      # Flashbots RPC integration
│   │   └── bundle_builder.py        # Construct MEV bundles
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── gas_tracker.py           # Track gas prices
│   │   ├── pool_health.py           # Monitor DEX pool health
│   │   └── risk_monitor.py          # Alert on smart contract risks
│   └── main.py                      # Entry point
├── contracts/                       # Solidity contracts
│   ├── TradingVault.sol
│   ├── UniswapV4Integration.sol
│   └── FlashLoanArbitrage.sol
├── tests/
│   ├── test_vault.py
│   ├── test_arbitrage.py
│   └── test_lp_manager.py
├── requirements.txt
└── README.md
```

### 3.2 Core Bot Implementation

**File**: `src/blockchain/node_client.py`

```python
"""
Ethereum node client with Flashbots integration
"""

from web3 import Web3
from flashbots import flashbot
from eth_account import Account
import os
from typing import Optional

class EthereumClient:
    """Manages connection to Ethereum node and Flashbots"""

    def __init__(self):
        # Primary: Self-hosted Geth node
        self.primary_rpc = os.getenv("GETH_RPC_URL", "http://localhost:8545")

        # Backup: Infura
        self.backup_rpc = os.getenv("INFURA_URL")

        # Flashbots RPC for MEV protection
        self.flashbots_rpc = "https://rpc.flashbots.net"

        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(self.primary_rpc))

        # Load private key from environment
        self.account = Account.from_key(os.getenv("PRIVATE_KEY"))

        # Initialize Flashbots
        flashbot(self.w3, self.account)

        # Verify connection
        if not self.w3.is_connected():
            print("⚠️  Primary RPC failed, switching to backup...")
            self.w3 = Web3(Web3.HTTPProvider(self.backup_rpc))
            assert self.w3.is_connected(), "Cannot connect to Ethereum"

        print(f"✅ Connected to Ethereum (block: {self.w3.eth.block_number})")

    def get_gas_price(self) -> dict:
        """Get current gas prices (EIP-1559)"""
        latest_block = self.w3.eth.get_block('latest')
        base_fee = latest_block['baseFeePerGas']

        # Query pending transactions for priority fee estimates
        priority_fee_percentiles = [25, 50, 75]  # Low, medium, high

        return {
            'base_fee': base_fee,
            'priority_fee_low': self.w3.eth.max_priority_fee // 2,
            'priority_fee_medium': self.w3.eth.max_priority_fee,
            'priority_fee_high': self.w3.eth.max_priority_fee * 2,
        }

    def send_transaction(
        self,
        to: str,
        data: str,
        value: int = 0,
        use_flashbots: bool = True
    ) -> str:
        """Send transaction (optionally via Flashbots)"""

        gas_prices = self.get_gas_price()

        tx_params = {
            'from': self.account.address,
            'to': to,
            'value': value,
            'data': data,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'maxFeePerGas': gas_prices['base_fee'] * 2 + gas_prices['priority_fee_medium'],
            'maxPriorityFeePerGas': gas_prices['priority_fee_medium'],
            'gas': 500000,  # Estimate gas
            'chainId': self.w3.eth.chain_id,
        }

        # Sign transaction
        signed_tx = self.w3.eth.account.sign_transaction(tx_params, self.account.key)

        if use_flashbots:
            # Send via Flashbots (private mempool)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            print(f"📤 Transaction sent via Flashbots: {tx_hash.hex()}")
        else:
            # Send via public mempool
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            print(f"📤 Transaction sent: {tx_hash.hex()}")

        return tx_hash.hex()

    def wait_for_transaction(self, tx_hash: str, timeout: int = 120) -> dict:
        """Wait for transaction to be mined"""
        try:
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
            if receipt['status'] == 1:
                print(f"✅ Transaction confirmed: {tx_hash}")
                return receipt
            else:
                print(f"❌ Transaction reverted: {tx_hash}")
                return receipt
        except Exception as e:
            print(f"⏱️  Transaction timeout: {e}")
            return None
```

**File**: `src/strategies/arbitrage_scanner.py`

```python
"""
Scan for arbitrage opportunities across DEXs
"""

import asyncio
from web3 import Web3
from typing import List, Dict, Optional
import time

class ArbitrageScanner:
    """Detect profitable arbitrage opportunities"""

    def __init__(self, eth_client, min_profit_bps: int = 50):
        """
        Args:
            eth_client: EthereumClient instance
            min_profit_bps: Minimum profit in basis points (50 = 0.5%)
        """
        self.w3 = eth_client.w3
        self.min_profit_bps = min_profit_bps

        # DEX router addresses
        self.dexes = {
            'uniswap_v4': '0x...', # Uniswap V4 PoolManager
            'curve': '0x...',       # Curve router
            'balancer': '0x...',    # Balancer vault
        }

        # Tokens to monitor
        self.tokens = {
            'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
            'USDC': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
            'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',
        }

    async def scan_arbitrage(self) -> List[Dict]:
        """Scan all DEX pairs for arbitrage opportunities"""
        opportunities = []

        # Check all token pairs
        pairs = [
            ('WETH', 'USDC'),
            ('WETH', 'DAI'),
            ('USDC', 'DAI'),
        ]

        for token_a, token_b in pairs:
            # Get prices on all DEXs
            prices = await self._get_prices_across_dexs(token_a, token_b)

            # Find arbitrage opportunity
            arb = self._find_best_arbitrage(prices, token_a, token_b)

            if arb and arb['profit_bps'] >= self.min_profit_bps:
                opportunities.append(arb)
                print(f"🎯 Arbitrage found: {arb['path']} - Profit: {arb['profit_bps']} bps")

        return opportunities

    async def _get_prices_across_dexs(
        self,
        token_a: str,
        token_b: str
    ) -> Dict[str, float]:
        """Get prices for a token pair on all DEXs"""
        prices = {}

        # Query Uniswap V4
        prices['uniswap_v4'] = await self._get_uniswap_price(token_a, token_b)

        # Query Curve
        prices['curve'] = await self._get_curve_price(token_a, token_b)

        # Query Balancer
        prices['balancer'] = await self._get_balancer_price(token_a, token_b)

        return prices

    def _find_best_arbitrage(
        self,
        prices: Dict[str, float],
        token_a: str,
        token_b: str
    ) -> Optional[Dict]:
        """Find most profitable arbitrage path"""

        # Find DEX with lowest price (buy here)
        buy_dex = min(prices, key=prices.get)
        buy_price = prices[buy_dex]

        # Find DEX with highest price (sell here)
        sell_dex = max(prices, key=prices.get)
        sell_price = prices[sell_dex]

        # Calculate profit (in basis points)
        profit_bps = int((sell_price - buy_price) / buy_price * 10000)

        if profit_bps <= 0:
            return None

        return {
            'token_a': token_a,
            'token_b': token_b,
            'buy_dex': buy_dex,
            'sell_dex': sell_dex,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'profit_bps': profit_bps,
            'path': f"{token_a} → {token_b} ({buy_dex} → {sell_dex})",
            'timestamp': time.time(),
        }

    async def _get_uniswap_price(self, token_a: str, token_b: str) -> float:
        """Get Uniswap V4 price via slot0()"""
        # Query pool's sqrtPriceX96
        # Convert to human-readable price
        # Return price
        pass

    async def _get_curve_price(self, token_a: str, token_b: str) -> float:
        """Get Curve price via get_dy()"""
        pass

    async def _get_balancer_price(self, token_a: str, token_b: str) -> float:
        """Get Balancer price via getAmountOut()"""
        pass
```

---

## 4. Infrastructure Setup

### 4.1 Ethereum Full Node (Geth)

**AWS EC2 Instance Specs**:
- Instance Type: `c6a.4xlarge` (16 vCPU, 32 GB RAM)
- Storage: 2 TB NVMe SSD (EBS gp3)
- OS: Ubuntu 22.04 LTS
- Monthly Cost: ~$500

**Geth Installation**:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Geth
sudo add-apt-repository -y ppa:ethereum/ethereum
sudo apt update
sudo apt install -y ethereum

# Create Geth data directory
sudo mkdir -p /mnt/ethereum/geth
sudo chown ubuntu:ubuntu /mnt/ethereum/geth

# Start Geth (snap sync for faster sync)
geth --datadir /mnt/ethereum/geth \
     --http --http.addr 0.0.0.0 --http.port 8545 \
     --http.api eth,net,web3,txpool \
     --ws --ws.addr 0.0.0.0 --ws.port 8546 \
     --ws.api eth,net,web3,txpool \
     --syncmode snap \
     --cache 16384 \
     --maxpeers 50

# Sync time: ~6-12 hours (snap sync)
```

**Monitoring Geth Sync**:

```bash
# Attach to Geth console
geth attach /mnt/ethereum/geth/geth.ipc

# Check sync status
> eth.syncing

# Check current block
> eth.blockNumber

# Check peer count
> net.peerCount
```

### 4.2 Flashbots Setup

**No installation required** - just use Flashbots Protect RPC:

```python
# Add Flashbots RPC to Web3
flashbots_rpc = "https://rpc.flashbots.net"
w3 = Web3(Web3.HTTPProvider(flashbots_rpc))

# All transactions sent via this RPC are automatically protected
```

### 4.3 The Graph Node (for DEX data)

**Install The Graph Node**:

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Clone The Graph repo
git clone https://github.com/graphprotocol/graph-node
cd graph-node/docker

# Configure environment
nano docker-compose.yml
# Set ethereum RPC: http://YOUR_GETH_IP:8545

# Start The Graph node
docker-compose up -d

# Query Uniswap V3 subgraph
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "{ pools(first: 10) { id token0 { symbol } token1 { symbol } } }"}' \
  https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3
```

---

## 5. Testing Strategy

### 5.1 Smart Contract Testing (Foundry)

```bash
# Install Foundry
curl -L https://foundry.paradigm.xyz | bash
foundryup

# Navigate to contracts directory
cd worktrees/defi-integration/contracts

# Run tests
forge test

# Run tests with gas reporting
forge test --gas-report

# Run tests with coverage
forge coverage
```

**Example Test**: `test/TradingVault.t.sol`

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "../src/TradingVault.sol";

contract TradingVaultTest is Test {
    TradingVault public vault;
    address public admin = address(1);
    address public strategist = address(2);

    function setUp() public {
        vault = new TradingVault();
        vault.initialize(admin);

        vm.prank(admin);
        vault.grantRole(vault.STRATEGIST_ROLE(), strategist);
    }

    function testApproveStrategy() public {
        address mockStrategy = address(3);
        uint256 allocation = 1 ether;

        vm.prank(admin);
        vault.approveStrategy(mockStrategy, allocation);

        assertTrue(vault.strategies(mockStrategy));
        assertEq(vault.allocations(mockStrategy), allocation);
    }

    function testExecuteStrategyRevertsIfNotApproved() public {
        address mockStrategy = address(3);
        bytes memory data = "";

        vm.prank(strategist);
        vm.expectRevert("Strategy not approved");
        vault.executeStrategy(mockStrategy, data);
    }

    function testEmergencyWithdraw() public {
        // Fund vault with 10 ETH
        vm.deal(address(vault), 10 ether);

        uint256 balanceBefore = admin.balance;

        // Guardian withdraws
        vm.prank(admin);
        vault.emergencyWithdraw(address(0));

        assertEq(admin.balance, balanceBefore + 10 ether);
        assertEq(address(vault).balance, 0);
    }
}
```

### 5.2 Bot Testing (Pytest)

```bash
# Install dependencies
pip install pytest pytest-asyncio web3 flashbots

# Run tests
cd worktrees/defi-integration
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

**Example Test**: `tests/test_arbitrage_scanner.py`

```python
import pytest
from src.blockchain.node_client import EthereumClient
from src.strategies.arbitrage_scanner import ArbitrageScanner

@pytest.fixture
def eth_client():
    return EthereumClient()

@pytest.fixture
def scanner(eth_client):
    return ArbitrageScanner(eth_client, min_profit_bps=50)

@pytest.mark.asyncio
async def test_scan_arbitrage(scanner):
    """Test arbitrage scanning"""
    opportunities = await scanner.scan_arbitrage()

    assert isinstance(opportunities, list)

    if len(opportunities) > 0:
        opp = opportunities[0]
        assert 'token_a' in opp
        assert 'token_b' in opp
        assert opp['profit_bps'] >= 50

def test_find_best_arbitrage(scanner):
    """Test arbitrage detection logic"""
    prices = {
        'uniswap_v4': 2000.0,  # WETH/USDC
        'curve': 2010.0,       # Higher price
        'balancer': 1995.0,    # Lower price
    }

    arb = scanner._find_best_arbitrage(prices, 'WETH', 'USDC')

    assert arb is not None
    assert arb['buy_dex'] == 'balancer'
    assert arb['sell_dex'] == 'curve'
    assert arb['profit_bps'] == 75  # (2010-1995)/1995 * 10000
```

---

## 6. Deployment Checklist

### Pre-Deployment (Testnet)

- [ ] Deploy Geth full node (synced to latest block)
- [ ] Deploy all smart contracts to Sepolia testnet
- [ ] Run 100+ test transactions (swaps, LP, flash loans)
- [ ] Verify gas costs are within budget (<$50 per arbitrage)
- [ ] Test Flashbots integration (no frontrunning detected)
- [ ] Run bot for 7 days in paper trading mode
- [ ] Internal security audit (Slither, Mythril)

### Pre-Deployment (Mainnet)

- [ ] External audit complete (Certik or Trail of Bits)
- [ ] All critical/high vulnerabilities addressed
- [ ] Multi-sig wallet configured (3-of-5)
- [ ] Timelock controller deployed (24-hour delay)
- [ ] Nexus Mutual insurance purchased ($1M coverage)
- [ ] Emergency pause mechanism tested
- [ ] Monitoring dashboards configured (Grafana)
- [ ] Alert system tested (PagerDuty, Slack)

### Deployment (Mainnet)

- [ ] Deploy TradingVault contract (verify on Etherscan)
- [ ] Deploy StrategyRouter contract
- [ ] Deploy UniswapV4Integration contract
- [ ] Deploy FlashLoanArbitrage contract
- [ ] Transfer ownership to multi-sig wallet
- [ ] Approve initial strategies (Uniswap, Curve)
- [ ] Fund vault with $50K test capital
- [ ] Start bot in live mode (monitor closely for 24 hours)
- [ ] Gradually increase capital to $250K over 4 weeks

### Post-Deployment

- [ ] Monitor 24/7 for first 7 days
- [ ] Review transaction logs daily
- [ ] Track profitability (target: >0.5% per trade)
- [ ] Monitor gas costs (alert if >$50 per trade)
- [ ] Check for smart contract exploits (Forta alerts)
- [ ] Weekly team review meeting
- [ ] Monthly external audit (ongoing)

---

## Appendix: Useful Commands

### Foundry Commands

```bash
# Compile contracts
forge build

# Run specific test
forge test --match-test testApproveStrategy

# Deploy contract
forge create src/TradingVault.sol:TradingVault \
  --private-key $PRIVATE_KEY \
  --rpc-url $RPC_URL

# Verify contract on Etherscan
forge verify-contract \
  --chain-id 1 \
  --compiler-version v0.8.20 \
  $CONTRACT_ADDRESS \
  src/TradingVault.sol:TradingVault \
  $ETHERSCAN_API_KEY
```

### Web3.py Commands

```python
# Get ETH balance
balance = w3.eth.get_balance("0x...")
print(f"Balance: {w3.from_wei(balance, 'ether')} ETH")

# Get ERC20 balance
token = w3.eth.contract(address=token_address, abi=erc20_abi)
balance = token.functions.balanceOf(account).call()

# Estimate gas for transaction
gas = w3.eth.estimate_gas(tx_params)
print(f"Estimated gas: {gas}")

# Get transaction receipt
receipt = w3.eth.get_transaction_receipt(tx_hash)
print(f"Status: {'Success' if receipt['status'] == 1 else 'Failed'}")
```

---

**END OF TECHNICAL GUIDE**

**Next**: Begin Phase 1 implementation (see DEFI_INTEGRATION_STRATEGY.md Section 3.1)
