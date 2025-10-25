"""
Integration tests for multi-exchange routing
"""

import pytest
import asyncio
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from src.services.trading_engine.multi_exchange_router import (
    MultiExchangeRouter,
    RouteOption,
    ExchangeType
)


class TestMultiExchangeRouter:
    """Test multi-exchange routing functionality"""

    @pytest.fixture
    async def router(self):
        """Create router instance"""
        router = MultiExchangeRouter()
        # Mock initialize to avoid real API calls
        with patch.object(router, 'initialize', return_value=asyncio.coroutine(lambda: None)()):
            yield router

    @pytest.mark.asyncio
    async def test_router_initialization(self, router):
        """Test router initializes exchanges"""
        await router.initialize()

        # Check exchanges are loaded
        assert "coinbase" in router.configs
        assert "hyperliquid" in router.configs
        assert "cdp" in router.configs

    @pytest.mark.asyncio
    async def test_best_price_routing(self, router):
        """Test finding best price across exchanges"""
        # Mock exchange responses
        router.exchanges = {
            "coinbase": Mock(
                get_orderbook=AsyncMock(return_value={
                    "bids": [["50000", "1.5"]],
                    "asks": [["50100", "2.0"]]
                })
            ),
            "hyperliquid": Mock(
                client=Mock(
                    get_orderbook=AsyncMock(return_value={
                        "bids": [["49950", "3.0"]],
                        "asks": [["50050", "1.0"]]
                    })
                )
            )
        }

        router.configs = {
            "coinbase": Mock(
                enabled=True,
                fee_taker=Decimal("0.006"),
                fee_maker=Decimal("0.004"),
                type=ExchangeType.CEX
            ),
            "hyperliquid": Mock(
                enabled=True,
                fee_taker=Decimal("0.0005"),
                fee_maker=Decimal("0.0002"),
                type=ExchangeType.PERP_DEX
            )
        }

        # Find best buy price
        best = await router.get_best_price("BTC-USD", "buy", Decimal("0.1"))

        assert best is not None
        assert best.exchange in ["coinbase", "hyperliquid"]
        assert best.price > 0
        assert best.fee > 0

    @pytest.mark.asyncio
    async def test_arbitrage_detection(self, router):
        """Test arbitrage opportunity detection"""
        # Mock price differences
        router.get_best_price = AsyncMock(side_effect=[
            RouteOption(
                exchange="coinbase",
                price=Decimal("50000"),
                liquidity=Decimal("10"),
                fee=Decimal("30"),
                execution_time=100.0,
                score=50000.0
            ),
            RouteOption(
                exchange="hyperliquid",
                price=Decimal("50600"),
                liquidity=Decimal("5"),
                fee=Decimal("25"),
                execution_time=1000.0,
                score=50600.0
            )
        ])

        # Detect arbitrage
        opportunities = await router.detect_arbitrage("BTC-USD", min_profit=Decimal("0.01"))

        # Should find buy low on coinbase, sell high on hyperliquid
        assert len(opportunities) > 0
        assert opportunities[0]["profit_pct"] > Decimal("0.01")

    @pytest.mark.asyncio
    async def test_order_execution(self, router):
        """Test order execution on best exchange"""
        # Mock best route
        router.get_best_price = AsyncMock(return_value=RouteOption(
            exchange="hyperliquid",
            price=Decimal("50000"),
            liquidity=Decimal("10"),
            fee=Decimal("25"),
            execution_time=1000.0,
            score=50000.0
        ))

        # Mock exchange execution
        router.exchanges = {
            "hyperliquid": Mock(
                client=Mock(
                    place_order=AsyncMock(return_value={"order_id": "123"})
                )
            )
        }

        # Execute order
        result = await router.execute_best_route(
            "BTC-USD",
            "buy",
            Decimal("0.1")
        )

        assert result is not None
        assert "order_id" in result

    @pytest.mark.asyncio
    async def test_aggregated_orderbook(self, router):
        """Test orderbook aggregation"""
        # Mock exchange orderbooks
        router.exchanges = {
            "coinbase": Mock(
                get_orderbook=AsyncMock(return_value={
                    "bids": [["50000", "1.0"]],
                    "asks": [["50100", "2.0"]]
                })
            ),
            "hyperliquid": Mock(
                client=Mock(
                    get_orderbook=AsyncMock(return_value={
                        "bids": [["50050", "0.5"]],
                        "asks": [["50150", "1.5"]]
                    })
                )
            )
        }

        router.configs = {
            "coinbase": Mock(enabled=True),
            "hyperliquid": Mock(enabled=True)
        }

        # Get aggregated book
        book = await router.get_aggregated_orderbook("BTC-USD")

        assert "bids" in book
        assert "asks" in book
        assert len(book["bids"]) > 0
        assert len(book["asks"]) > 0


class TestCoinbaseCDP:
    """Test Coinbase CDP SDK integration"""

    @pytest.mark.asyncio
    async def test_create_evm_account(self):
        """Test EVM account creation"""
        from src.services.trading_engine.exchanges.coinbase_cdp import CoinbaseCDPClient

        client = CoinbaseCDPClient()
        account = await client.create_evm_account("test_account")

        assert account.name == "test_account"
        assert account.address.startswith("0x")

    @pytest.mark.asyncio
    async def test_token_swap(self):
        """Test token swap functionality"""
        from src.services.trading_engine.exchanges.coinbase_cdp import CoinbaseCDPClient

        client = CoinbaseCDPClient()
        await client.create_evm_account("trading")

        # Mock swap
        result = await client.swap_tokens(
            account_name="trading",
            from_token="USDC",
            to_token="ETH",
            amount=Decimal("100"),
            network="base"
        )

        assert "hash" in result
        assert result["from_token"] == "USDC"
        assert result["to_token"] == "ETH"


class TestHyperliquid:
    """Test Hyperliquid DEX integration"""

    @pytest.mark.asyncio
    async def test_get_positions(self):
        """Test getting positions"""
        from src.services.trading_engine.exchanges.hyperliquid import HyperliquidClient

        client = HyperliquidClient(testnet=True)

        # Mock API response
        with patch.object(client, '_request', return_value={
            "assetPositions": [{
                "position": {
                    "coin": "BTC",
                    "szi": "0.5",
                    "entryPx": "50000",
                    "unrealizedPnl": "500",
                    "leverage": {"value": 5}
                },
                "markPx": "51000"
            }]
        }):
            positions = await client.get_positions()

            assert len(positions) > 0
            assert positions[0].coin == "BTC"
            assert positions[0].side == "long"

    @pytest.mark.asyncio
    async def test_place_order(self):
        """Test order placement"""
        from src.services.trading_engine.exchanges.hyperliquid import (
            HyperliquidClient,
            OrderSide,
            OrderType
        )

        client = HyperliquidClient(testnet=True)

        # Mock order placement
        with patch.object(client, '_request', return_value={
            "status": {"statuses": [{"resting": {"oid": "123456"}}]}
        }):
            order = await client.place_order(
                coin="BTC",
                side=OrderSide.BUY,
                size=Decimal("0.1"),
                price=Decimal("50000"),
                leverage=5
            )

            assert order.order_id == "123456"
            assert order.coin == "BTC"
            assert order.side == OrderSide.BUY

    @pytest.mark.asyncio
    async def test_get_leaderboard(self):
        """Test leaderboard retrieval"""
        from src.services.trading_engine.exchanges.hyperliquid import HyperliquidClient

        client = HyperliquidClient(testnet=True)

        # Mock leaderboard response
        with patch.object(client, '_request', return_value={
            "leaderboard": [
                {"user": "0x123", "pnl": "10000", "volume": "1000000", "winRate": "0.65"}
            ]
        }):
            leaderboard = await client.get_leaderboard("day")

            assert len(leaderboard) > 0
            assert "user" in leaderboard[0]
            assert "pnl" in leaderboard[0]


def test_exchange_config():
    """Test exchange configuration"""
    from src.services.trading_engine.multi_exchange_router import ExchangeConfig, ExchangeType

    config = ExchangeConfig(
        name="Test Exchange",
        type=ExchangeType.CEX,
        enabled=True,
        fee_maker=Decimal("0.001"),
        fee_taker=Decimal("0.002"),
        min_order_size=Decimal("10"),
        supports_margin=True,
        max_leverage=50
    )

    assert config.name == "Test Exchange"
    assert config.type == ExchangeType.CEX
    assert config.max_leverage == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
