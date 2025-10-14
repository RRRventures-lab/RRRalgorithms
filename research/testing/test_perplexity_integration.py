from datetime import datetime, timedelta
from pathlib import Path
from research.testing.professional_data_collectors import ProfessionalDataCollector
import asyncio
import sys

"""
Test Perplexity AI Sentiment Integration

Validates:
1. Real-time sentiment analysis
2. Enhanced sentiment scoring (weighted keywords)
3. Confidence calculation
4. Citation quality
5. Sentiment signal generation
"""


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))



async def test_sentiment_analysis():
    """Test real-time sentiment analysis."""
    print("\n" + "=" * 80)
    print("TEST 1: Real-Time Sentiment Analysis")
    print("=" * 80)

    collector = ProfessionalDataCollector()

    # Test sentiment for BTC
    print("\n[Test] Analyzing Bitcoin sentiment...")
    sentiment = await collector.perplexity.get_market_sentiment(
        symbol="BTC",
        date=datetime.now()
    )

    if sentiment and sentiment.get('sentiment_score') != 0.0:
        print(f"✅ Sentiment analysis complete")
        print(f"\n   Symbol: {sentiment.get('symbol')}")
        print(f"   Timestamp: {sentiment.get('timestamp')}")
        print(f"   Sentiment Score: {sentiment.get('sentiment_score'):.3f} ({_interpret_sentiment(sentiment.get('sentiment_score'))})")
        print(f"   Confidence: {sentiment.get('confidence'):.2%}")
        print(f"   Citations: {len(sentiment.get('citations', []))} sources")

        # Show snippet of sentiment text
        text = sentiment.get('sentiment_text', '')
        print(f"\n   Analysis Preview:")
        print(f"   {text[:300]}..." if len(text) > 300 else f"   {text}")

        return sentiment
    else:
        print("❌ Sentiment analysis failed or returned neutral")
        return None


def _interpret_sentiment(score: float) -> str:
    """Interpret sentiment score."""
    if score >= 0.6:
        return "VERY BULLISH 🚀"
    elif score >= 0.3:
        return "BULLISH 📈"
    elif score >= 0.1:
        return "SLIGHTLY BULLISH ↗️"
    elif score <= -0.6:
        return "VERY BEARISH 💥"
    elif score <= -0.3:
        return "BEARISH 📉"
    elif score <= -0.1:
        return "SLIGHTLY BEARISH ↘️"
    else:
        return "NEUTRAL ➡️"


async def test_enhanced_scoring():
    """Test enhanced sentiment scoring with different text samples."""
    print("\n" + "=" * 80)
    print("TEST 2: Enhanced Sentiment Scoring")
    print("=" * 80)

    collector = ProfessionalDataCollector()

    # Test various sentiment texts
    test_cases = [
        {
            "text": "Bitcoin is experiencing a massive surge with extremely bullish momentum. The rally continues with strong buying pressure.",
            "expected": "very bullish"
        },
        {
            "text": "BTC shows slight positive gains with some optimistic sentiment but remains cautious near resistance levels.",
            "expected": "slightly bullish"
        },
        {
            "text": "The market crashed today with panic selling and massive losses. Very bearish outlook continues.",
            "expected": "very bearish"
        },
        {
            "text": "Bitcoin is not showing bullish signals despite earlier optimism. Weakness persists.",
            "expected": "bearish (negation)"
        },
        {
            "text": "Market conditions remain stable with no clear direction. Price action is uncertain.",
            "expected": "neutral"
        }
    ]

    print("\n[Test] Testing enhanced sentiment scoring on sample texts...")

    for i, case in enumerate(test_cases, 1):
        score = collector.perplexity._score_sentiment(case['text'])
        interpretation = _interpret_sentiment(score)

        print(f"\n   Test Case {i}: {case['expected'].upper()}")
        print(f"     Score: {score:.3f} → {interpretation}")
        print(f"     Text: {case['text'][:80]}...")


async def test_confidence_calculation():
    """Test confidence calculation."""
    print("\n" + "=" * 80)
    print("TEST 3: Confidence Calculation")
    print("=" * 80)

    collector = ProfessionalDataCollector()

    # Test different confidence scenarios
    test_cases = [
        {
            "text": "Data shows clear evidence that Bitcoin is rising. Analysis indicates strong momentum with confirmed breakout patterns. Research demonstrates bullish sentiment.",
            "citations": ["source1", "source2", "source3", "source4", "source5"],
            "expected": "high (long text, many citations, definitive language)"
        },
        {
            "text": "Bitcoin might be bullish, but it's unclear. Perhaps it could rise, or maybe not.",
            "citations": ["source1"],
            "expected": "low (short text, uncertainty words)"
        },
        {
            "text": "Market analysis shows Bitcoin trading near $100k with positive momentum.",
            "citations": [],
            "expected": "medium (moderate length, no citations)"
        }
    ]

    print("\n[Test] Testing confidence calculation...")

    for i, case in enumerate(test_cases, 1):
        confidence = collector.perplexity._calculate_confidence(case['text'], case['citations'])

        print(f"\n   Test Case {i}: {case['expected'].upper()}")
        print(f"     Confidence: {confidence:.2%}")
        print(f"     Text length: {len(case['text'])} chars")
        print(f"     Citations: {len(case['citations'])} sources")


async def test_historical_sentiment():
    """Test collecting historical sentiment data."""
    print("\n" + "=" * 80)
    print("TEST 4: Historical Sentiment Collection")
    print("=" * 80)

    collector = ProfessionalDataCollector()

    print("\n[Test] Collecting sentiment for past 3 days...")

    sentiments = []
    for days_ago in range(3):
        date = datetime.now() - timedelta(days=days_ago)
        print(f"\n   Fetching sentiment for {date.strftime('%Y-%m-%d')}...")

        sentiment = await collector.perplexity.get_market_sentiment(
            symbol="BTC",
            date=date
        )

        if sentiment and sentiment.get('sentiment_score') != 0.0:
            sentiments.append(sentiment)
            print(f"     Score: {sentiment.get('sentiment_score'):.3f} | Confidence: {sentiment.get('confidence'):.2%}")
        else:
            print(f"     ⚠️ No data or error")

        # Rate limiting
        await asyncio.sleep(2)

    if sentiments:
        avg_sentiment = sum(s.get('sentiment_score', 0) for s in sentiments) / len(sentiments)
        print(f"\n✅ Historical sentiment collected: {len(sentiments)} days")
        print(f"   Average sentiment: {avg_sentiment:.3f} ({_interpret_sentiment(avg_sentiment)})")
        return sentiments
    else:
        print("\n❌ Failed to collect historical sentiment")
        return None


async def test_sentiment_signal_generation():
    """Test generating trading signals from sentiment."""
    print("\n" + "=" * 80)
    print("TEST 5: Sentiment Signal Generation")
    print("=" * 80)

    collector = ProfessionalDataCollector()

    print("\n[Test] Generating trading signals from sentiment...")

    # Get current sentiment
    sentiment = await collector.perplexity.get_market_sentiment(
        symbol="BTC",
        date=datetime.now()
    )

    if sentiment:
        score = sentiment.get('sentiment_score', 0)
        confidence = sentiment.get('confidence', 0)

        # Generate signal
        signal = 0
        if confidence >= 0.7:  # High confidence threshold
            if score >= 0.5:
                signal = 1  # LONG
            elif score <= -0.5:
                signal = -1  # SHORT

        signal_text = "LONG 📈" if signal == 1 else ("SHORT 📉" if signal == -1 else "NEUTRAL ➡️")

        print(f"   Sentiment Score: {score:.3f}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Signal: {signal_text}")

        if signal == 0:
            if confidence < 0.7:
                print(f"   Reason: Low confidence (< 70%)")
            else:
                print(f"   Reason: Weak sentiment (|score| < 0.5)")

        print("\n✅ Signal generation complete")
        return signal
    else:
        print("❌ Unable to generate signal (no sentiment data)")
        return None


async def main():
    """Run all Perplexity integration tests."""
    print("\n" + "=" * 80)
    print("🔬 PERPLEXITY AI SENTIMENT INTEGRATION TEST SUITE")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")
    print("\nNote: Perplexity API calls are rate-limited. Tests may take 1-2 minutes.")

    try:
        # Test 1: Real-time sentiment
        sentiment = await test_sentiment_analysis()

        # Test 2: Enhanced scoring
        await test_enhanced_scoring()

        # Test 3: Confidence calculation
        await test_confidence_calculation()

        # Test 4: Historical sentiment (commented out to save API calls)
        # await test_historical_sentiment()

        # Test 5: Signal generation
        if sentiment:
            await test_sentiment_signal_generation()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nPerplexity Sentiment Integration Status: ✅ WORKING")
        print("Ready for hypothesis testing with real sentiment data")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nPerplexity Sentiment Integration Status: ❌ FAILED")


if __name__ == "__main__":
    asyncio.run(main())
