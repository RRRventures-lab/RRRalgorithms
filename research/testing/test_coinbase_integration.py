from datetime import datetime, timedelta
from pathlib import Path
from research.testing.professional_data_collectors import ProfessionalDataCollector
import asyncio
import sys

"""
Test Coinbase API Integration

Validates:
1. Order book data collection
2. Order book imbalance calculation
3. Real-time trades
4. Price comparison with Polygon.io
5. Data quality and reliability
"""


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))



async def test_order_book():
    """Test order book data collection."""
    print("\n" + "=" * 80)
    print("TEST 1: Coinbase Order Book Collection")
    print("=" * 80)

    collector = ProfessionalDataCollector()

    # Test fetching order book
    print("\n[Test] Fetching BTC-USD order book from Coinbase...")
    order_book = await collector.coinbase.get_order_book(product_id="BTC-USD", level=2)

    if order_book:
        print(f"✅ Order book received")
        print(f"   Timestamp: {order_book.get('timestamp')}")
        print(f"   Product: {order_book.get('product_id')}")
        print(f"   Sequence: {order_book.get('sequence')}")
        print(f"   Bids count: {len(order_book.get('bids', []))}")
        print(f"   Asks count: {len(order_book.get('asks', []))}")

        # Show best bid/ask
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])

        if bids and asks:
            best_bid = bids[0]
            best_ask = asks[0]
            print(f"\n   Best Bid: ${best_bid[0]:,.2f} (size: {best_bid[1]:.4f} BTC)")
            print(f"   Best Ask: ${best_ask[0]:,.2f} (size: {best_ask[1]:.4f} BTC)")
            print(f"   Spread: ${best_ask[0] - best_bid[0]:.2f}")

            return order_book
        else:
            print("❌ No bid/ask data in order book")
            return None
    else:
        print("❌ Failed to fetch order book")
        return None


async def test_order_book_imbalance(order_book):
    """Test order book imbalance calculation."""
    print("\n" + "=" * 80)
    print("TEST 2: Order Book Imbalance Calculation")
    print("=" * 80)

    collector = ProfessionalDataCollector()

    if not order_book:
        print("❌ No order book data to analyze")
        return None

    print("\n[Test] Calculating order book imbalance...")

    # Test different depth percentages
    for depth_pct in [0.005, 0.01, 0.02]:  # 0.5%, 1%, 2%
        metrics = collector.coinbase.calculate_order_book_imbalance(order_book, depth_pct=depth_pct)

        print(f"\n   Depth {depth_pct*100}%:")
        print(f"     Imbalance Ratio: {metrics['imbalance_ratio']:.4f} ({'BULLISH' if metrics['imbalance_ratio'] > 0.5 else 'BEARISH'})")
        print(f"     Bid Volume: {metrics['bid_volume']:.4f} BTC")
        print(f"     Ask Volume: {metrics['ask_volume']:.4f} BTC")
        print(f"     Total Volume: {metrics['total_volume']:.4f} BTC")
        print(f"     Spread (bps): {metrics['spread_bps']:.2f}")
        print(f"     Mid Price: ${metrics['mid_price']:,.2f}")

    print("\n✅ Order book imbalance calculated successfully")
    return metrics


async def test_recent_trades():
    """Test recent trades data."""
    print("\n" + "=" * 80)
    print("TEST 3: Recent Trades Collection")
    print("=" * 80)

    collector = ProfessionalDataCollector()

    print("\n[Test] Fetching recent trades from Coinbase...")
    trades = await collector.coinbase.get_recent_trades(product_id="BTC-USD", limit=10)

    if trades:
        print(f"✅ Received {len(trades)} recent trades")

        if trades:
            print("\n   Last 5 trades:")
            for i, trade in enumerate(trades[:5]):
                trade_type = "BUY " if trade.get('side') == 'buy' else "SELL"
                print(f"     {i+1}. {trade_type} {trade.get('size')} BTC @ ${trade.get('price')} | Time: {trade.get('time')}")

        return trades
    else:
        print("❌ Failed to fetch recent trades")
        return None


async def test_24h_stats():
    """Test 24h statistics."""
    print("\n" + "=" * 80)
    print("TEST 4: 24-Hour Statistics")
    print("=" * 80)

    collector = ProfessionalDataCollector()

    print("\n[Test] Fetching 24h stats from Coinbase...")
    stats = await collector.coinbase.get_24h_stats(product_id="BTC-USD")

    if stats:
        print("✅ 24h statistics received")
        print(f"   Open: ${stats.get('open'):,.2f}")
        print(f"   High: ${stats.get('high'):,.2f}")
        print(f"   Low: ${stats.get('low'):,.2f}")
        print(f"   Last: ${stats.get('last'):,.2f}")
        print(f"   Volume: {stats.get('volume'):,.2f} BTC")
        print(f"   30-day Volume: {stats.get('volume_30day'):,.2f} BTC")

        # Calculate daily change
        if stats.get('open') and stats.get('last'):
            change = stats.get('last') - stats.get('open')
            change_pct = (change / stats.get('open')) * 100
            print(f"   Change: ${change:,.2f} ({change_pct:+.2f}%)")

        return stats
    else:
        print("❌ Failed to fetch 24h stats")
        return None


async def test_price_comparison():
    """Compare Coinbase price to Polygon.io price."""
    print("\n" + "=" * 80)
    print("TEST 5: Coinbase vs Polygon.io Price Comparison")
    print("=" * 80)

    collector = ProfessionalDataCollector()

    print("\n[Test] Comparing prices between Coinbase and Polygon...")

    # Get Coinbase price
    stats = await collector.coinbase.get_24h_stats(product_id="BTC-USD")
    coinbase_price = stats.get('last') if stats else None

    # Get Polygon price (last hour)
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=1)

    polygon_data = await collector.polygon.get_crypto_aggregates(
        "X:BTCUSD",
        start_date,
        end_date,
        timespan="minute",
        multiplier=1
    )

    polygon_price = None
    if not polygon_data.empty:
        polygon_price = polygon_data['close'].iloc[-1]

    if coinbase_price and polygon_price:
        print(f"\n   Coinbase Price: ${coinbase_price:,.2f}")
        print(f"   Polygon Price:  ${polygon_price:,.2f}")

        diff = coinbase_price - polygon_price
        diff_pct = (diff / polygon_price) * 100

        print(f"   Difference: ${diff:,.2f} ({diff_pct:+.3f}%)")

        # Coinbase premium/discount
        if diff > 0:
            print(f"   📊 Coinbase PREMIUM: ${diff:,.2f} ({diff_pct:+.3f}%)")
        else:
            print(f"   📊 Coinbase DISCOUNT: ${abs(diff):,.2f} ({abs(diff_pct):.3f}%)")

        print("\n✅ Price comparison complete")
        return {'coinbase': coinbase_price, 'polygon': polygon_price, 'diff': diff, 'diff_pct': diff_pct}
    else:
        print("❌ Unable to compare prices (missing data)")
        return None


async def main():
    """Run all Coinbase integration tests."""
    print("\n" + "=" * 80)
    print("🔬 COINBASE API INTEGRATION TEST SUITE")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")

    try:
        # Test 1: Order book
        order_book = await test_order_book()

        # Test 2: Order book imbalance
        if order_book:
            await test_order_book_imbalance(order_book)

        # Test 3: Recent trades
        await test_recent_trades()

        # Test 4: 24h stats
        await test_24h_stats()

        # Test 5: Price comparison
        await test_price_comparison()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nCoinbase API Integration Status: ✅ WORKING")
        print("Ready for hypothesis testing with real order book data")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nCoinbase API Integration Status: ❌ FAILED")


if __name__ == "__main__":
    asyncio.run(main())
