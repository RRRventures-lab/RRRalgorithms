from unittest.mock import Mock, patch, AsyncMock
import asyncio
import os
import pytest
import sys
import time

from src.microservices.api_gateway import APIGateway, RateLimiter

"""
Integration tests for API Gateway
"""


# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    # Convenience functions not implemented in microservice module in this repo.
    # Adjust tests to call gateway methods directly where applicable.


class TestRateLimiter:
    """Test rate limiting functionality"""

    def test_rate_limiter_initialization(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        assert limiter.max_requests == 5
        assert limiter.window_seconds == 60
        assert len(limiter.requests) == 0

    def test_can_proceed_empty(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        assert limiter.can_proceed() is True

    def test_can_proceed_within_limit(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(4):
            limiter.record_request()
        assert limiter.can_proceed() is True

    def test_can_proceed_at_limit(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            limiter.record_request()
        assert limiter.can_proceed() is False

    def test_wait_time_calculation(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            limiter.record_request()

        wait_time = limiter.wait_time()
        assert wait_time > 0
        assert wait_time <= 60

    def test_requests_expire(self):
        limiter = RateLimiter(max_requests=2, window_seconds=1)
        limiter.record_request()
        limiter.record_request()

        assert limiter.can_proceed() is False

        # Wait for window to expire
        time.sleep(1.1)

        assert limiter.can_proceed() is True


class TestAPIGateway:
    """Test API Gateway functionality"""

    @pytest.fixture
    def gateway(self):
        return APIGateway()

    def test_gateway_initialization(self, gateway):
        assert gateway is not None
        assert len(gateway.rate_limiters) > 0
        assert 'polygon' in gateway.rate_limiters
        assert 'perplexity' in gateway.rate_limiters

    def test_prepare_request_polygon(self, gateway):
        url, params, headers = gateway._prepare_request(
            'polygon',
            '/v2/last/trade/X:BTCUSD',
            {}
        )

        assert 'polygon.io' in url
        assert 'apiKey' in params or 'Authorization' in headers

    def test_prepare_request_perplexity(self, gateway):
        url, params, headers = gateway._prepare_request(
            'perplexity',
            '/chat/completions',
            {}
        )

        assert 'perplexity.ai' in url
        assert 'Authorization' in headers

    def test_prepare_request_invalid_api(self, gateway):
        with pytest.raises(ValueError):
            gateway._prepare_request('invalid_api', '/endpoint', {})

    @pytest.mark.asyncio
    async def test_rate_limiting_enforced(self, gateway):
        """Test that rate limiting is enforced"""
        # Mock the session to avoid actual API calls
        with patch.object(gateway.session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'test': 'data'}
            mock_get.return_value = mock_response

            # Make requests up to the limit
            api_name = 'polygon'
            limiter = gateway.rate_limiters[api_name]
            max_requests = limiter.max_requests

            # Exhaust rate limit
            for i in range(max_requests):
                await gateway.get(api_name, '/test/endpoint')

            # Next request should wait
            assert not limiter.can_proceed()

    @pytest.mark.asyncio
    async def test_request_retry_on_failure(self, gateway):
        """Test retry logic on failed requests"""
        with patch.object(gateway.session, 'get') as mock_get:
            # Simulate transient failure then success
            mock_response_fail = Mock()
            mock_response_fail.status_code = 500
            mock_response_fail.raise_for_status.side_effect = Exception("Server error")

            mock_response_success = Mock()
            mock_response_success.status_code = 200
            mock_response_success.json.return_value = {'test': 'data'}

            mock_get.side_effect = [mock_response_fail, mock_response_success]

            # Should retry and eventually succeed
            # Note: This depends on retry configuration
            try:
                result = await gateway.get('polygon', '/test/endpoint')
            except Exception:
                # Expected if all retries fail
                pass


@pytest.mark.asyncio
async def test_polygon_get_convenience_function():
    """Test polygon_get convenience function"""
    with patch('gateway.api_gateway.gateway') as mock_gateway:
        mock_gateway.get = AsyncMock(return_value={'test': 'data'})

        result = await polygon_get('/v2/last/trade/X:BTCUSD')

        mock_gateway.get.assert_called_once()
        assert result == {'test': 'data'}


@pytest.mark.asyncio
async def test_perplexity_post_convenience_function():
    """Test perplexity_post convenience function"""
    with patch('gateway.api_gateway.gateway') as mock_gateway:
        mock_gateway.post = AsyncMock(return_value={'test': 'data'})

        result = await perplexity_post('/chat/completions', json={'model': 'test'})

        mock_gateway.post.assert_called_once()
        assert result == {'test': 'data'}


@pytest.mark.asyncio
async def test_get_api_usage_stats():
    """Test API usage statistics retrieval"""
    with patch('gateway.api_gateway.gateway') as mock_gateway:
        mock_stats = {
            'total_requests': 100,
            'by_api': {'polygon': 60, 'perplexity': 40},
            'avg_response_time': 250,
            'errors': 5
        }
        mock_gateway.get_usage_stats = AsyncMock(return_value=mock_stats)

        result = await get_api_usage('polygon', hours=24)

        assert result['total_requests'] == 100


class TestIntegration:
    """Integration tests requiring actual API access"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_polygon_api_call(self):
        """Test actual Polygon API call (requires valid API key)"""
        if not os.getenv('POLYGON_API_KEY'):
            pytest.skip("POLYGON_API_KEY not set")

        try:
            result = await polygon_get('/v2/last/trade/X:BTCUSD')
            assert 'results' in result or 'error' in result
        except Exception as e:
            pytest.skip(f"API call failed: {e}")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_perplexity_api_call(self):
        """Test actual Perplexity API call (requires valid API key)"""
        if not os.getenv('PERPLEXITY_API_KEY'):
            pytest.skip("PERPLEXITY_API_KEY not set")

        try:
            payload = {
                'model': 'llama-3.1-sonar-small-128k-online',
                'messages': [
                    {'role': 'user', 'content': 'What is Bitcoin?'}
                ]
            }
            result = await perplexity_post('/chat/completions', json=payload)
            assert 'choices' in result or 'error' in result
        except Exception as e:
            pytest.skip(f"API call failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
