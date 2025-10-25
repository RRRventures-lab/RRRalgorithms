"""
Multi-Exchange Router - Intelligent order routing across exchanges

Supports:
- Coinbase Advanced Trade (CEX)
- Coinbase CDP (DeFi/Onchain)
- Hyperliquid (Decentralized Perpetuals)
- Binance (CEX)

Features:
- Best execution routing
- Liquidity aggregation
- Arbitrage detection
- Fee optimization
"""

import asyncio
from typing import Dict, List, Optional, Any, Literal
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import logging

from .exchanges.coinbase_exchange import CoinbaseExchange
from .exchanges.coinbase_cdp import CDPIntegration
from .exchanges.hyperliquid import HyperliquidIntegration
from .exchanges.binance_exchange import BinanceExchange

logger = logging.getLogger(__name__)


class ExchangeType(Enum):
    """Exchange types"""
    CEX = "centralized"
    DEX = "decentralized"
    DEFI = "defi"
    PERP_DEX = "perpetual_dex"


@dataclass
class ExchangeConfig:
    """Exchange configuration"""
    name: str
    type: ExchangeType
    enabled: bool
    fee_maker: Decimal
    fee_taker: Decimal
    min_order_size: Decimal
    supports_margin: bool
    max_leverage: int


@dataclass
class RouteOption:
    """Routing option"""
    exchange: str
    price: Decimal
    liquidity: Decimal
    fee: Decimal
    execution_time: float  # estimated ms
    score: float  # overall score


class MultiExchangeRouter:
    """
    Intelligent multi-exchange order router

    Features:
    - Smart order routing for best execution
    - Cross-exchange arbitrage detection
    - Liquidity aggregation across venues
    - Fee optimization
    - Network effect analysis
    """

    def __init__(self):
        self.exchanges: Dict[str, Any] = {}
        self.configs: Dict[str, ExchangeConfig] = {}
        self.market_data: Dict[str, Dict] = {}  # Exchange -> Market data

    async def initialize(self):
        """Initialize all exchange integrations"""

        # Coinbase Advanced Trade (CEX)
        try:
            coinbase = CoinbaseExchange()
            self.exchanges["coinbase"] = coinbase
            self.configs["coinbase"] = ExchangeConfig(
                name="Coinbase",
                type=ExchangeType.CEX,
                enabled=True,
                fee_maker=Decimal("0.004"),  # 0.4%
                fee_taker=Decimal("0.006"),  # 0.6%
                min_order_size=Decimal("1"),
                supports_margin=False,
                max_leverage=1
            )
            logger.info("Coinbase Advanced Trade initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Coinbase: {e}")

        # Coinbase CDP (DeFi/Onchain)
        try:
            cdp = CDPIntegration()
            await cdp.initialize()
            self.exchanges["cdp"] = cdp
            self.configs["cdp"] = ExchangeConfig(
                name="Coinbase CDP",
                type=ExchangeType.DEFI,
                enabled=True,
                fee_maker=Decimal("0.003"),  # DeFi swap fees vary
                fee_taker=Decimal("0.003"),
                min_order_size=Decimal("0.01"),
                supports_margin=False,
                max_leverage=1
            )
            logger.info("Coinbase CDP initialized")
        except Exception as e:
            logger.error(f"Failed to initialize CDP: {e}")

        # Hyperliquid (Decentralized Perpetuals)
        try:
            hyperliquid = HyperliquidIntegration(testnet=False)
            await hyperliquid.initialize()
            self.exchanges["hyperliquid"] = hyperliquid
            self.configs["hyperliquid"] = ExchangeConfig(
                name="Hyperliquid",
                type=ExchangeType.PERP_DEX,
                enabled=True,
                fee_maker=Decimal("0.0002"),  # 0.02% maker
                fee_taker=Decimal("0.0005"),  # 0.05% taker
                min_order_size=Decimal("0.001"),
                supports_margin=True,
                max_leverage=50
            )
            logger.info("Hyperliquid initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Hyperliquid: {e}")

        # Binance (CEX) - if needed
        try:
            binance = BinanceExchange()
            self.exchanges["binance"] = binance
            self.configs["binance"] = ExchangeConfig(
                name="Binance",
                type=ExchangeType.CEX,
                enabled=False,  # Disabled by default
                fee_maker=Decimal("0.001"),  # 0.1%
                fee_taker=Decimal("0.001"),
                min_order_size=Decimal("0.0001"),
                supports_margin=True,
                max_leverage=125
            )
            logger.info("Binance available (disabled)")
        except Exception as e:
            logger.warning(f"Binance not initialized: {e}")

        logger.info(
            f"Multi-exchange router initialized with {len(self.exchanges)} exchanges"
        )

    async def get_best_price(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        amount: Decimal
    ) -> Optional[RouteOption]:
        """
        Find best price across all exchanges

        Args:
            symbol: Trading pair (e.g., "BTC-USD", "ETH-USD")
            side: Buy or sell
            amount: Order amount

        Returns:
            Best routing option
        """
        options: List[RouteOption] = []

        # Query all enabled exchanges
        for exchange_name, exchange in self.exchanges.items():
            config = self.configs[exchange_name]

            if not config.enabled:
                continue

            try:
                # Get orderbook or quote
                if exchange_name == "coinbase":
                    # CEX orderbook
                    book = await exchange.get_orderbook(symbol)
                    price, liquidity = self._analyze_orderbook(
                        book, side, amount
                    )

                elif exchange_name == "hyperliquid":
                    # DEX orderbook
                    coin = symbol.split("-")[0]  # BTC-USD -> BTC
                    book = await exchange.client.get_orderbook(coin)
                    price, liquidity = self._analyze_orderbook(
                        book, side, amount
                    )

                elif exchange_name == "cdp":
                    # DeFi price estimate (swap quote)
                    # Simplified - would need actual swap quote
                    price = Decimal("0")
                    liquidity = Decimal("0")
                    continue  # Skip for now

                else:
                    continue

                # Calculate fees
                fee = (
                    amount * price * config.fee_taker
                    if side == "buy"
                    else amount * config.fee_maker
                )

                # Estimate execution time
                exec_time = {
                    ExchangeType.CEX: 100.0,  # 100ms
                    ExchangeType.PERP_DEX: 1000.0,  # 1s (blockchain)
                    ExchangeType.DEFI: 2000.0  # 2s (blockchain + swap)
                }.get(config.type, 500.0)

                # Calculate score (lower is better for buy, higher for sell)
                score = self._calculate_route_score(
                    price=price,
                    fee=fee,
                    liquidity=liquidity,
                    exec_time=exec_time,
                    side=side
                )

                options.append(RouteOption(
                    exchange=exchange_name,
                    price=price,
                    liquidity=liquidity,
                    fee=fee,
                    execution_time=exec_time,
                    score=score
                ))

            except Exception as e:
                logger.error(
                    f"Failed to get price from {exchange_name}: {e}"
                )
                continue

        if not options:
            return None

        # Return best option
        options.sort(
            key=lambda x: x.score,
            reverse=(side == "sell")  # Higher score better for sell
        )

        return options[0]

    def _analyze_orderbook(
        self,
        book: Dict,
        side: str,
        amount: Decimal
    ) -> tuple[Decimal, Decimal]:
        """Analyze orderbook for execution"""
        levels = book.get("asks" if side == "buy" else "bids", [])

        total_volume = Decimal("0")
        weighted_price = Decimal("0")

        for price_str, volume_str in levels:
            price = Decimal(price_str)
            volume = Decimal(volume_str)

            if total_volume >= amount:
                break

            fill_amount = min(volume, amount - total_volume)
            weighted_price += price * fill_amount
            total_volume += fill_amount

        if total_volume == 0:
            return Decimal("0"), Decimal("0")

        avg_price = weighted_price / total_volume
        return avg_price, total_volume

    def _calculate_route_score(
        self,
        price: Decimal,
        fee: Decimal,
        liquidity: Decimal,
        exec_time: float,
        side: str
    ) -> float:
        """
        Calculate routing score

        For buys: lower total cost is better
        For sells: higher proceeds is better
        """
        total_cost = price + fee  # Simplified

        # Factors:
        # - Price (most important)
        # - Liquidity (important for large orders)
        # - Speed (minor factor)

        price_score = float(total_cost)
        liquidity_score = float(liquidity) / 100.0  # Normalize
        speed_score = 1000.0 / exec_time  # Faster is better

        # Weighted score
        score = (
            price_score * 0.7 +
            liquidity_score * 0.2 +
            speed_score * 0.1
        )

        return score

    async def execute_best_route(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        amount: Decimal,
        order_type: str = "market"
    ) -> Optional[Dict[str, Any]]:
        """
        Execute order on best exchange

        Returns:
            Order result
        """
        # Find best route
        best_route = await self.get_best_price(symbol, side, amount)

        if not best_route:
            logger.error("No viable route found")
            return None

        logger.info(
            f"Best route: {best_route.exchange} @ {best_route.price} "
            f"(fee: {best_route.fee}, liquidity: {best_route.liquidity})"
        )

        # Execute on selected exchange
        exchange = self.exchanges[best_route.exchange]

        try:
            if best_route.exchange == "coinbase":
                result = await exchange.place_order(
                    symbol=symbol,
                    side=side,
                    amount=amount,
                    order_type=order_type
                )

            elif best_route.exchange == "hyperliquid":
                from .exchanges.hyperliquid import OrderSide, OrderType
                hl_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
                coin = symbol.split("-")[0]

                result = await exchange.client.place_order(
                    coin=coin,
                    side=hl_side,
                    size=amount,
                    order_type=OrderType.MARKET
                )

            elif best_route.exchange == "cdp":
                # DeFi swap
                from_token, to_token = symbol.split("-")
                if side == "sell":
                    from_token, to_token = to_token, from_token

                result = await exchange.execute_defi_swap(
                    from_token=from_token,
                    to_token=to_token,
                    amount=amount
                )

            else:
                logger.error(f"Unknown exchange: {best_route.exchange}")
                return None

            logger.info(f"Order executed: {result}")
            return result

        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return None

    async def detect_arbitrage(
        self,
        symbol: str,
        min_profit: Decimal = Decimal("0.01")  # 1%
    ) -> List[Dict[str, Any]]:
        """
        Detect arbitrage opportunities across exchanges

        Returns:
            List of arbitrage opportunities
        """
        opportunities = []

        # Get prices from all exchanges
        prices = {}
        for exchange_name in self.exchanges:
            if not self.configs[exchange_name].enabled:
                continue

            try:
                buy_route = await self.get_best_price(symbol, "buy", Decimal("1"))
                sell_route = await self.get_best_price(symbol, "sell", Decimal("1"))

                if buy_route and sell_route:
                    prices[exchange_name] = {
                        "buy": buy_route.price,
                        "sell": sell_route.price
                    }

            except Exception as e:
                logger.error(f"Failed to get prices from {exchange_name}: {e}")

        # Find arbitrage: buy low on one exchange, sell high on another
        for buy_exchange, buy_prices in prices.items():
            for sell_exchange, sell_prices in prices.items():
                if buy_exchange == sell_exchange:
                    continue

                buy_price = buy_prices["buy"]
                sell_price = sell_prices["sell"]

                # Calculate profit
                profit_pct = (sell_price - buy_price) / buy_price

                if profit_pct >= min_profit:
                    opportunities.append({
                        "symbol": symbol,
                        "buy_exchange": buy_exchange,
                        "sell_exchange": sell_exchange,
                        "buy_price": buy_price,
                        "sell_price": sell_price,
                        "profit_pct": profit_pct,
                        "estimated_profit": profit_pct * 100  # For $100
                    })

        return opportunities

    async def get_aggregated_orderbook(
        self,
        symbol: str,
        depth: int = 20
    ) -> Dict[str, List]:
        """
        Aggregate orderbooks from all exchanges

        Returns combined bid/ask ladder
        """
        all_bids = []
        all_asks = []

        for exchange_name, exchange in self.exchanges.items():
            if not self.configs[exchange_name].enabled:
                continue

            try:
                book = None

                if exchange_name == "coinbase":
                    book = await exchange.get_orderbook(symbol)
                elif exchange_name == "hyperliquid":
                    coin = symbol.split("-")[0]
                    book = await exchange.client.get_orderbook(coin)

                if book:
                    all_bids.extend([
                        (Decimal(p), Decimal(v), exchange_name)
                        for p, v in book.get("bids", [])
                    ])
                    all_asks.extend([
                        (Decimal(p), Decimal(v), exchange_name)
                        for p, v in book.get("asks", [])
                    ])

            except Exception as e:
                logger.error(f"Failed to get orderbook from {exchange_name}: {e}")

        # Sort and aggregate
        all_bids.sort(key=lambda x: x[0], reverse=True)  # Highest bids first
        all_asks.sort(key=lambda x: x[0])  # Lowest asks first

        return {
            "bids": all_bids[:depth],
            "asks": all_asks[:depth]
        }


# Example usage
async def main():
    """Example multi-exchange routing"""
    router = MultiExchangeRouter()
    await router.initialize()

    # Find best price
    best = await router.get_best_price("BTC-USD", "buy", Decimal("0.1"))
    print(f"Best price: {best}")

    # Execute order
    result = await router.execute_best_route("BTC-USD", "buy", Decimal("0.01"))
    print(f"Order result: {result}")

    # Detect arbitrage
    arb_ops = await router.detect_arbitrage("ETH-USD")
    print(f"Arbitrage opportunities: {arb_ops}")

    # Aggregated orderbook
    book = await router.get_aggregated_orderbook("BTC-USD")
    print(f"Aggregated orderbook: {book}")


if __name__ == "__main__":
    asyncio.run(main())
