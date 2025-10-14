from dataclasses import dataclass, field
from datetime import datetime, timedelta
from dotenv import load_dotenv
from functools import lru_cache
from typing import Dict, List, Optional, Any
import aiohttp
import asyncio
import json
import logging
import numpy as np
import os


"""
Perplexity AI integration for market sentiment and narrative analysis
"""


load_dotenv('config/api-keys/.env')

logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """Sentiment analysis result"""
    symbol: str
    timestamp: datetime
    score: float  # -1 to +1 (negative to positive)
    confidence: float  # 0 to 1
    magnitude: float  # 0 to 1 (strength of sentiment)
    
    # Source breakdown
    news_sentiment: Optional[float] = None
    social_sentiment: Optional[float] = None
    expert_sentiment: Optional[float] = None
    
    # Metadata
    sources_count: int = 0
    narrative: str = ""
    key_themes: List[str] = field(default_factory=list)
    raw_response: str = ""


@dataclass
class NarrativeShift:
    """Detected narrative shift in market"""
    symbol: str
    timestamp: datetime
    old_narrative: str
    new_narrative: str
    shift_magnitude: float  # How significant the shift is
    confidence: float
    
    # Context
    related_events: List[str] = field(default_factory=list)
    price_impact_estimate: Optional[float] = None  # Expected price impact


class PerplexitySentimentAnalyzer:
    """
    Uses Perplexity AI to analyze market sentiment and detect narrative shifts
    
    Features:
    - Real-time sentiment scoring
    - Narrative shift detection
    - Multi-source sentiment aggregation
    - Sentiment-price correlation analysis
    """
    
    API_URL = "https://api.perplexity.ai/chat/completions"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('PERPLEXITY_API_KEY')
        if not self.api_key:
            raise ValueError("Perplexity API key required")
        
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Cache for recent queries
        self.sentiment_cache: Dict[str, SentimentScore] = {}
        self.cache_ttl_seconds = 300  # 5 minutes
        
        # Historical tracking
        self.narrative_history: Dict[str, List[str]] = {}  # symbol -> list of narratives
        
        # Rate limiting
        self.last_request_time = datetime.now()
        self.min_request_interval = 1.0  # 1 second between requests
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self):
        """Enforce rate limiting"""
        time_since_last = (datetime.now() - self.last_request_time).total_seconds()
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = datetime.now()
    
    async def _query_perplexity(self, prompt: str, model: str = "llama-3.1-sonar-small-128k-online") -> str:
        """
        Query Perplexity API
        
        Args:
            prompt: Query prompt
            model: Model to use
            
        Returns:
            Response text
        """
        await self._ensure_session()
        await self._rate_limit()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a financial analyst specializing in cryptocurrency markets. Provide concise, factual analysis."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": 500
        }
        
        try:
            async with self.session.post(self.API_URL, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    error_text = await response.text()
                    logger.error(f"Perplexity API error {response.status}: {error_text}")
                    return ""
        
        except Exception as e:
            logger.error(f"Error querying Perplexity: {e}")
            return ""
    
    async def get_sentiment_score(self, symbol: str, timeframe: str = '1h') -> Optional[SentimentScore]:
        """
        Get sentiment score for a symbol
        
        Args:
            symbol: Symbol to analyze (e.g., 'BTC', 'ETH')
            timeframe: Time frame ('1h', '4h', '24h', '7d')
            
        Returns:
            SentimentScore object
        """
        # Check cache
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.sentiment_cache:
            cached = self.sentiment_cache[cache_key]
            age = (datetime.now() - cached.timestamp).total_seconds()
            if age < self.cache_ttl_seconds:
                logger.info(f"Using cached sentiment for {symbol} (age: {age:.0f}s)")
                return cached
        
        # Build prompt
        prompt = f"""Analyze the current market sentiment for {symbol} over the last {timeframe}.

Please provide:
1. Overall sentiment: POSITIVE, NEGATIVE, or NEUTRAL
2. Sentiment score from -1 (very negative) to +1 (very positive)
3. Confidence level: LOW, MEDIUM, or HIGH
4. Main narrative/theme in the market
5. Key factors influencing sentiment

Include analysis from:
- Recent news articles
- Social media (Twitter/X, Reddit)
- Market analyst opinions
- On-chain metrics (if available)

Format your response as:
SENTIMENT: [POSITIVE/NEGATIVE/NEUTRAL]
SCORE: [number from -1 to +1]
CONFIDENCE: [LOW/MEDIUM/HIGH]
NARRATIVE: [main market narrative]
FACTORS: [key factors, comma-separated]
"""
        
        response = await self._query_perplexity(prompt)
        
        if not response:
            return None
        
        # Parse response
        sentiment_score = self._parse_sentiment_response(symbol, response)
        
        # Cache result
        self.sentiment_cache[cache_key] = sentiment_score
        
        return sentiment_score
    
    def _parse_sentiment_response(self, symbol: str, response: str) -> SentimentScore:
        """Parse Perplexity response into SentimentScore"""
        
        # Extract values using simple parsing
        sentiment = "NEUTRAL"
        score = 0.0
        confidence_str = "MEDIUM"
        narrative = ""
        factors = []
        
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            if line.startswith('SENTIMENT:'):
                sentiment = line.split(':', 1)[1].strip().upper()
            elif line.startswith('SCORE:'):
                try:
                    score = float(line.split(':', 1)[1].strip())
                    score = max(-1.0, min(1.0, score))  # Clamp to [-1, 1]
                except:
                    pass
            elif line.startswith('CONFIDENCE:'):
                confidence_str = line.split(':', 1)[1].strip().upper()
            elif line.startswith('NARRATIVE:'):
                narrative = line.split(':', 1)[1].strip()
            elif line.startswith('FACTORS:'):
                factors_str = line.split(':', 1)[1].strip()
                factors = [f.strip() for f in factors_str.split(',')]
        
        # Convert confidence to numeric
        confidence_map = {'LOW': 0.3, 'MEDIUM': 0.6, 'HIGH': 0.9}
        confidence = confidence_map.get(confidence_str, 0.5)
        
        # Calculate magnitude (absolute value of score)
        magnitude = abs(score)
        
        return SentimentScore(
            symbol=symbol,
            timestamp=datetime.now(),
            score=score,
            confidence=confidence,
            magnitude=magnitude,
            narrative=narrative,
            key_themes=factors,
            raw_response=response,
            sources_count=self._count_sources(response)
        )
    
    def _count_sources(self, response: str) -> int:
        """Estimate number of sources referenced"""
        # Simple heuristic: count mentions of source types
        source_keywords = ['news', 'twitter', 'reddit', 'analyst', 'report', 'article', 'post']
        count = sum(1 for keyword in source_keywords if keyword.lower() in response.lower())
        return max(count, 1)
    
    async def detect_narrative_shift(self, symbol: str) -> Optional[NarrativeShift]:
        """
        Detect if market narrative has shifted
        
        Args:
            symbol: Symbol to analyze
            
        Returns:
            NarrativeShift if detected, None otherwise
        """
        # Get current sentiment
        current_sentiment = await self.get_sentiment_score(symbol, '24h')
        
        if not current_sentiment:
            return None
        
        current_narrative = current_sentiment.narrative
        
        # Initialize history if needed
        if symbol not in self.narrative_history:
            self.narrative_history[symbol] = []
        
        # Get historical narratives
        history = self.narrative_history[symbol]
        
        if len(history) == 0:
            # First observation
            self.narrative_history[symbol].append(current_narrative)
            return None
        
        # Compare with most recent narrative
        previous_narrative = history[-1]
        
        # Calculate shift magnitude using simple similarity
        shift_magnitude = self._calculate_narrative_similarity(previous_narrative, current_narrative)
        
        # If narratives are significantly different, it's a shift
        if shift_magnitude < 0.5:  # Less than 50% similar
            # Add to history
            self.narrative_history[symbol].append(current_narrative)
            
            # Limit history size
            if len(self.narrative_history[symbol]) > 10:
                self.narrative_history[symbol] = self.narrative_history[symbol][-10:]
            
            return NarrativeShift(
                symbol=symbol,
                timestamp=datetime.now(),
                old_narrative=previous_narrative,
                new_narrative=current_narrative,
                shift_magnitude=1.0 - shift_magnitude,
                confidence=current_sentiment.confidence,
                related_events=current_sentiment.key_themes
            )
        
        return None
    
    def _calculate_narrative_similarity(self, narrative1: str, narrative2: str) -> float:
        """
        Calculate similarity between two narratives
        
        Returns:
            Similarity score 0-1 (1 = identical)
        """
        if not narrative1 or not narrative2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(narrative1.lower().split())
        words2 = set(narrative2.lower().split())
        
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    async def get_multi_asset_sentiment(self, symbols: List[str]) -> Dict[str, SentimentScore]:
        """
        Get sentiment scores for multiple symbols
        
        Args:
            symbols: List of symbols
            
        Returns:
            Dictionary of symbol -> SentimentScore
        """
        tasks = [self.get_sentiment_score(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        sentiment_map = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, SentimentScore):
                sentiment_map[symbol] = result
            else:
                logger.warning(f"Failed to get sentiment for {symbol}: {result}")
        
        return sentiment_map
    
    async def analyze_sentiment_trend(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Analyze sentiment trend over time
        
        Args:
            symbol: Symbol to analyze
            days: Number of days to analyze
            
        Returns:
            Trend analysis results
        """
        # Query for different timeframes
        timeframes = ['24h', '7d']
        
        sentiments = {}
        for tf in timeframes:
            sentiment = await self.get_sentiment_score(symbol, tf)
            if sentiment:
                sentiments[tf] = sentiment
        
        if len(sentiments) < 2:
            return {}
        
        # Calculate trend
        recent_score = sentiments['24h'].score
        historical_score = sentiments['7d'].score
        
        trend = "increasing" if recent_score > historical_score else "decreasing"
        trend_strength = abs(recent_score - historical_score)
        
        return {
            'symbol': symbol,
            'trend': trend,
            'trend_strength': trend_strength,
            'current_score': recent_score,
            'historical_score': historical_score,
            'current_narrative': sentiments['24h'].narrative,
            'momentum': recent_score - historical_score
        }
    
    def calculate_sentiment_divergence(self, symbol: str, price_change: float, sentiment_change: float) -> float:
        """
        Calculate divergence between sentiment and price
        
        Args:
            symbol: Symbol
            price_change: Price change % (e.g., 0.05 for 5% increase)
            sentiment_change: Sentiment change (e.g., 0.2 for +0.2 shift)
            
        Returns:
            Divergence score (positive = bullish divergence, negative = bearish)
        """
        # Normalize both to same scale
        normalized_price = np.sign(price_change) * min(abs(price_change) / 0.1, 1.0)  # Cap at 10%
        normalized_sentiment = np.sign(sentiment_change) * min(abs(sentiment_change), 1.0)
        
        # Divergence is when they move in opposite directions
        divergence = normalized_sentiment - normalized_price
        
        return divergence


# Example usage
async def main():
    """Example usage of Perplexity sentiment analyzer"""
    analyzer = PerplexitySentimentAnalyzer()
    
    try:
        # Get sentiment for Bitcoin
        sentiment = await analyzer.get_sentiment_score('BTC', '24h')
        
        if sentiment:
            print(f"\n=== {sentiment.symbol} Sentiment Analysis ===")
            print(f"Score: {sentiment.score:.2f} (Confidence: {sentiment.confidence:.0%})")
            print(f"Narrative: {sentiment.narrative}")
            print(f"Key Themes: {', '.join(sentiment.key_themes)}")
        
        # Check for narrative shift
        shift = await analyzer.detect_narrative_shift('BTC')
        if shift:
            print(f"\n⚠️  Narrative Shift Detected!")
            print(f"From: {shift.old_narrative}")
            print(f"To: {shift.new_narrative}")
            print(f"Magnitude: {shift.shift_magnitude:.2%}")
        
        # Multi-asset sentiment
        symbols = ['BTC', 'ETH', 'SOL']
        sentiments = await analyzer.get_multi_asset_sentiment(symbols)
        
        print(f"\n=== Multi-Asset Sentiment ===")
        for symbol, sent in sentiments.items():
            print(f"{symbol}: {sent.score:.2f} ({sent.narrative[:50]}...)")
    
    finally:
        await analyzer.close()


if __name__ == "__main__":
    asyncio.run(main())

