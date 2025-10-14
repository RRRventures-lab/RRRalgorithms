from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from news, social media, and market analysis.
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import httpx
import json
import logging
import os
import re


"""
Perplexity AI Sentiment Analyzer for Cryptocurrency Markets
============================================================

This module uses Perplexity AI to analyze cryptocurrency market sentiment

Features:
- Query Perplexity for real-time market sentiment
- Extract sentiment labels (bullish/neutral/bearish)
- Calculate confidence scores
- Store sentiment data in Supabase
- Scheduled execution (every 15 minutes)
- Support for multiple cryptocurrencies

Usage:
    from data_pipeline.perplexity.sentiment_analyzer import PerplexitySentimentAnalyzer
    from src.database import SQLiteClient as DatabaseClient

    supabase = get_db()
    analyzer = PerplexitySentimentAnalyzer(supabase_client=supabase)

    # Analyze sentiment for Bitcoin
    sentiment = await analyzer.analyze_sentiment("BTC")

    # Run scheduled sentiment analysis
    await analyzer.run_scheduled()
"""



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentimentLabel(str, Enum):
    """Sentiment classification labels."""
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"


class SentimentResult(BaseModel):
    """Sentiment analysis result."""
    asset: str
    sentiment_label: SentimentLabel
    sentiment_score: float  # -1.0 (bearish) to 1.0 (bullish)
    confidence: float  # 0.0 to 1.0
    text: str
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = {}
    timestamp: datetime


class PerplexitySentimentAnalyzer:
    """
    Sentiment analyzer using Perplexity AI.

    Queries Perplexity for cryptocurrency market sentiment and stores
    results in Supabase for use by trading algorithms.
    """

    # Perplexity API endpoint
    API_URL = "https://api.perplexity.ai/chat/completions"

    # Default assets to track
    DEFAULT_ASSETS = [
        "BTC",  # Bitcoin
        "ETH",  # Ethereum
        "SOL",  # Solana
        "ADA",  # Cardano
        "DOT",  # Polkadot
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        supabase_client=None,
        assets: Optional[List[str]] = None,
        model: str = "llama-3.1-sonar-large-128k-online",
        update_interval: int = 900,  # 15 minutes
    ):
        """
        Initialize Perplexity sentiment analyzer.

        Args:
            api_key: Perplexity API key (or set PERPLEXITY_API_KEY env var)
            supabase_client: SupabaseClient instance for data storage
            assets: List of crypto assets to analyze (default: major coins)
            model: Perplexity model to use
            update_interval: Update interval in seconds (default: 15 min)
        """
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Perplexity API key required. Set PERPLEXITY_API_KEY env var or pass api_key parameter."
            )

        self.supabase_client = supabase_client
        if not self.supabase_client:
            logger.warning("No Supabase client provided. Data will not be stored.")

        self.assets = assets or self.DEFAULT_ASSETS
        self.model = model
        self.update_interval = update_interval

        self.running = False
        self.analysis_count = 0
        self.error_count = 0

        logger.info(f"Perplexity sentiment analyzer initialized for {len(self.assets)} assets")

    def _build_sentiment_prompt(self, asset: str) -> str:
        """
        Build prompt for sentiment analysis.

        Args:
            asset: Crypto asset symbol (e.g., "BTC")

        Returns:
            Formatted prompt string
        """
        return f"""Analyze the current market sentiment for {asset} (cryptocurrency).

Search for and analyze:
1. Recent news articles (last 24 hours)
2. Market trends and price movements
3. Social media sentiment
4. Expert opinions and analyst reports
5. On-chain metrics and activity

Provide a sentiment analysis with:
- Overall sentiment: BULLISH, NEUTRAL, or BEARISH
- Confidence level: 0-100%
- Sentiment score: -1.0 (most bearish) to +1.0 (most bullish)
- Key reasoning: Brief explanation (2-3 sentences)

Format your response EXACTLY as JSON:
{{
  "sentiment_label": "bullish|neutral|bearish",
  "sentiment_score": <float between -1.0 and 1.0>,
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "Brief explanation of the sentiment"
}}

Current time: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
"""

    async def query_perplexity(self, prompt: str) -> str:
        """
        Query Perplexity API.

        Args:
            prompt: Prompt to send to Perplexity

        Returns:
            Response text from Perplexity
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a cryptocurrency market analyst. Provide accurate, data-driven sentiment analysis in JSON format."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,  # Lower temperature for more consistent analysis
            "max_tokens": 1000,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    self.API_URL,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()

                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return content

            except httpx.HTTPError as e:
                logger.error(f"Perplexity API error: {e}")
                raise
            except Exception as e:
                logger.error(f"Error querying Perplexity: {e}")
                raise

    def _parse_sentiment_response(self, response_text: str, asset: str) -> SentimentResult:
        """
        Parse Perplexity response into structured sentiment data.

        Args:
            response_text: Raw response from Perplexity
            asset: Asset symbol

        Returns:
            SentimentResult object
        """
        try:
            # Try to extract JSON from response
            # Sometimes LLMs wrap JSON in markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    raise ValueError("No JSON found in response")

            # Parse JSON
            data = json.loads(json_text)

            # Normalize sentiment label
            label = data["sentiment_label"].lower()
            if label not in ["bullish", "neutral", "bearish"]:
                logger.warning(f"Invalid sentiment label: {label}, defaulting to neutral")
                label = "neutral"

            # Clamp values to valid ranges
            score = max(-1.0, min(1.0, float(data["sentiment_score"])))
            confidence = max(0.0, min(1.0, float(data["confidence"])))

            return SentimentResult(
                asset=asset,
                sentiment_label=SentimentLabel(label),
                sentiment_score=score,
                confidence=confidence,
                text=response_text,
                reasoning=data.get("reasoning", ""),
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error parsing sentiment response: {e}")
            logger.debug(f"Response text: {response_text}")

            # Return neutral sentiment as fallback
            return SentimentResult(
                asset=asset,
                sentiment_label=SentimentLabel.NEUTRAL,
                sentiment_score=0.0,
                confidence=0.0,
                text=response_text,
                reasoning=f"Error parsing response: {str(e)}",
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )

    async def analyze_sentiment(self, asset: str) -> SentimentResult:
        """
        Analyze sentiment for a specific asset.

        Args:
            asset: Crypto asset symbol (e.g., "BTC")

        Returns:
            SentimentResult with analysis
        """
        try:
            logger.info(f"Analyzing sentiment for {asset}...")

            # Build prompt
            prompt = self._build_sentiment_prompt(asset)

            # Query Perplexity
            response = await self.query_perplexity(prompt)

            # Parse response
            sentiment = self._parse_sentiment_response(response, asset)

            logger.info(
                f"{asset} sentiment: {sentiment.sentiment_label.value} "
                f"(score: {sentiment.sentiment_score:.2f}, "
                f"confidence: {sentiment.confidence:.2f})"
            )

            # Store in Supabase
            if self.supabase_client:
                db_data = {
                    "asset": asset,
                    "source": "perplexity",
                    "sentiment_label": sentiment.sentiment_label.value,
                    "sentiment_score": sentiment.sentiment_score,
                    "confidence": sentiment.confidence,
                    "text": sentiment.text,
                    "metadata": {
                        "reasoning": sentiment.reasoning,
                        "model": self.model,
                        **sentiment.metadata
                    }
                }
                self.supabase_client.insert_market_sentiment(db_data)

            self.analysis_count += 1
            return sentiment

        except Exception as e:
            logger.error(f"Error analyzing sentiment for {asset}: {e}")
            self.error_count += 1
            raise

    async def analyze_all_assets(self) -> List[SentimentResult]:
        """
        Analyze sentiment for all tracked assets.

        Returns:
            List of SentimentResult objects
        """
        logger.info(f"Analyzing sentiment for {len(self.assets)} assets...")

        results = []
        for asset in self.assets:
            try:
                sentiment = await self.analyze_sentiment(asset)
                results.append(sentiment)

                # Small delay to respect rate limits
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Failed to analyze {asset}: {e}")
                continue

        logger.info(f"Completed sentiment analysis for {len(results)}/{len(self.assets)} assets")
        return results

    async def run_scheduled(self):
        """
        Run sentiment analysis on a schedule.

        This method runs indefinitely, analyzing sentiment at regular intervals.
        """
        self.running = True
        logger.info(f"Starting scheduled sentiment analysis (interval: {self.update_interval}s)")

        while self.running:
            try:
                start_time = datetime.now()

                # Analyze all assets
                await self.analyze_all_assets()

                # Calculate time elapsed
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(f"Analysis completed in {elapsed:.1f}s")

                # Wait for next interval
                sleep_time = max(0, self.update_interval - elapsed)
                if sleep_time > 0:
                    logger.info(f"Next analysis in {sleep_time:.0f}s")
                    await asyncio.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                self.running = False
                break

            except Exception as e:
                logger.error(f"Error in scheduled run: {e}")
                self.error_count += 1
                await asyncio.sleep(60)  # Wait 1 minute before retry

        logger.info("Stopped scheduled sentiment analysis")

    async def stop(self):
        """Stop the scheduled analysis."""
        logger.info("Stopping sentiment analyzer...")
        self.running = False

    @lru_cache(maxsize=128)

    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "running": self.running,
            "assets": self.assets,
            "analysis_count": self.analysis_count,
            "error_count": self.error_count,
            "update_interval": self.update_interval,
        }


# =============================================================================
# Example Usage
# =============================================================================

async def main():
    """Example usage of Perplexity sentiment analyzer."""
    from src.database import SQLiteClient as DatabaseClient

    # Initialize Supabase client
    supabase = get_db()

    # Initialize sentiment analyzer
    analyzer = PerplexitySentimentAnalyzer(
        supabase_client=supabase,
        assets=["BTC", "ETH"],  # Track only BTC and ETH for demo
        update_interval=900,  # 15 minutes
    )

    # Option 1: Analyze once
    sentiment = await analyzer.analyze_sentiment("BTC")
    print(f"\nBTC Sentiment: {sentiment.sentiment_label.value}")
    print(f"Score: {sentiment.sentiment_score:.2f}")
    print(f"Confidence: {sentiment.confidence:.2f}")
    print(f"Reasoning: {sentiment.reasoning}")

    # Option 2: Run scheduled analysis (uncomment to use)
    # await analyzer.run_scheduled()


if __name__ == "__main__":
    asyncio.run(main())
