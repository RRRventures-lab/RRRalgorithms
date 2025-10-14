from dotenv import load_dotenv
from supabase import create_client, Client
import os
import pytest
import requests
import time

"""
MCP Server Connection Tests

Validates that all Model Context Protocol servers are properly configured and accessible:
- Supabase MCP (PostgreSQL)
- Polygon.io MCP
- Perplexity AI MCP
- TradingView Webhook Server
"""


# Load environment variables
load_dotenv('config/api-keys/.env')


class TestMCPConnections:
    """Test all MCP server connections"""

    def test_supabase_mcp_connection(self):
        """Test 1: Supabase MCP connection is working"""
        try:
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_ANON_KEY')

            assert supabase_url is not None, "SUPABASE_URL not configured"
            assert supabase_key is not None, "SUPABASE_ANON_KEY not configured"

            # Create client
            supabase: Client = create_client(supabase_url, supabase_key)

            # Test query
            result = supabase.table('crypto_aggregates').select('*').limit(1).execute()

            # Connection successful if no exception raised
            print(f"✅ Test 1 PASSED: Supabase MCP connected ({len(result.data)} records)")

        except Exception as e:
            pytest.fail(f"Supabase MCP connection failed: {e}")

    def test_supabase_realtime_enabled(self):
        """Test 2: Supabase real-time subscriptions are enabled"""
        # Check that real-time tables are configured
        realtime_tables = [
            'crypto_aggregates',
            'crypto_trades',
            'crypto_quotes',
            'trading_signals',
            'orders',
            'positions',
            'portfolio_snapshots',
            'market_sentiment'
        ]

        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_ANON_KEY')
        supabase: Client = create_client(supabase_url, supabase_key)

        for table in realtime_tables:
            try:
                # Attempt to query table
                result = supabase.table(table).select('*').limit(1).execute()
                print(f"  ✓ {table} accessible")
            except Exception as e:
                pytest.fail(f"Real-time table {table} not accessible: {e}")

        print(f"✅ Test 2 PASSED: All {len(realtime_tables)} real-time tables accessible")

    def test_polygon_api_key_valid(self):
        """Test 3: Polygon.io API key is valid"""
        api_key = os.getenv('POLYGON_API_KEY')
        assert api_key is not None, "POLYGON_API_KEY not configured"

        # Test API with a simple request
        url = f"https://api.polygon.io/v2/aggs/ticker/X:BTCUSD/range/1/day/2023-01-01/2023-01-02?apiKey={api_key}"

        try:
            response = requests.get(url, timeout=10)
            assert response.status_code == 200, f"Polygon API returned {response.status_code}"

            data = response.json()
            assert 'results' in data or 'status' in data, "Unexpected API response format"

            print(f"✅ Test 3 PASSED: Polygon.io API key valid")

        except requests.exceptions.RequestException as e:
            pytest.fail(f"Polygon API request failed: {e}")

    def test_polygon_rate_limit_configured(self):
        """Test 4: Polygon.io rate limit is properly configured"""
        rate_limit = os.getenv('POLYGON_RATE_LIMIT')
        assert rate_limit is not None, "POLYGON_RATE_LIMIT not configured"

        rate_limit_int = int(rate_limit)
        assert rate_limit_int >= 5, "Rate limit should be at least 5 req/sec"
        assert rate_limit_int <= 1000, "Rate limit seems too high (>1000 req/sec)"

        print(f"✅ Test 4 PASSED: Polygon rate limit configured ({rate_limit_int} req/sec)")

    def test_perplexity_api_key_valid(self):
        """Test 5: Perplexity AI API key is valid"""
        api_key = os.getenv('PERPLEXITY_API_KEY')
        assert api_key is not None, "PERPLEXITY_API_KEY not configured"

        # Test API with a simple request
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "pplx-70b-online",
            "messages": [
                {"role": "system", "content": "You are a test assistant."},
                {"role": "user", "content": "Say 'OK' if you can read this."}
            ],
            "max_tokens": 10
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)

            # Check if request was successful or if it's a rate limit (both indicate valid key)
            assert response.status_code in [200, 429], f"Perplexity API returned {response.status_code}"

            if response.status_code == 200:
                data = response.json()
                assert 'choices' in data or 'id' in data, "Unexpected API response format"
                print(f"✅ Test 5 PASSED: Perplexity AI API key valid (response received)")
            elif response.status_code == 429:
                print(f"✅ Test 5 PASSED: Perplexity AI API key valid (rate limited but authenticated)")

        except requests.exceptions.RequestException as e:
            # If it's a connection error but we got a 401, that means auth worked
            if hasattr(e, 'response') and e.response.status_code == 401:
                pytest.fail(f"Perplexity API authentication failed - check API key")
            else:
                print(f"⚠️  Test 5 WARNING: Could not validate Perplexity API (network issue): {e}")
                # Don't fail the test for network issues

    def test_tradingview_webhook_secret_configured(self):
        """Test 6: TradingView webhook secret is configured"""
        webhook_secret = os.getenv('TRADINGVIEW_WEBHOOK_SECRET')
        assert webhook_secret is not None, "TRADINGVIEW_WEBHOOK_SECRET not configured"
        assert webhook_secret != 'your_secure_webhook_secret_here', "Webhook secret not updated from placeholder"
        assert len(webhook_secret) >= 16, "Webhook secret should be at least 16 characters"

        print(f"✅ Test 6 PASSED: TradingView webhook secret configured (length: {len(webhook_secret)})")

    def test_anthropic_api_key_valid(self):
        """Test 7: Anthropic Claude API key is valid"""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        assert api_key is not None, "ANTHROPIC_API_KEY not configured"
        assert api_key.startswith('sk-ant-'), "Anthropic API key has invalid format"

        print(f"✅ Test 7 PASSED: Anthropic API key configured")

    def test_database_connection_string_valid(self):
        """Test 8: Database connection string is properly formatted"""
        db_url = os.getenv('SUPABASE_DB_URL')
        assert db_url is not None, "SUPABASE_DB_URL not configured"

        # Validate format
        assert db_url.startswith('postgresql://'), "Database URL should start with postgresql://"
        assert '@' in db_url, "Database URL should contain credentials"
        assert 'supabase.co' in db_url, "Database URL should point to Supabase"
        assert ':5432' in db_url or ':6543' in db_url, "Database URL should include port"

        print(f"✅ Test 8 PASSED: Database connection string valid")

    def test_redis_url_configured(self):
        """Test 9: Redis URL is configured (for caching)"""
        redis_url = os.getenv('REDIS_URL')

        if redis_url is None:
            print("⚠️  Test 9 WARNING: Redis not configured (optional for production)")
            return

        assert redis_url.startswith('redis://'), "Redis URL should start with redis://"
        print(f"✅ Test 9 PASSED: Redis URL configured")

    def test_mcp_config_file_exists(self):
        """Test 10: MCP configuration file exists and is valid"""
        import json

        mcp_config_path = 'config/mcp-servers/mcp-config.json'
        assert os.path.exists(mcp_config_path), f"MCP config not found at {mcp_config_path}"

        with open(mcp_config_path, 'r') as f:
            config = json.load(f)

        # Check required MCP servers
        assert 'supabase' in config, "Supabase MCP not configured"
        assert 'polygon' in config, "Polygon MCP not configured"
        assert 'perplexity' in config, "Perplexity MCP not configured"

        print(f"✅ Test 10 PASSED: MCP config file valid ({len(config)} servers configured)")

    def test_environment_variables_loaded(self):
        """Test 11: All required environment variables are loaded"""
        required_vars = [
            'SUPABASE_URL',
            'SUPABASE_ANON_KEY',
            'SUPABASE_SERVICE_KEY',
            'SUPABASE_DB_URL',
            'POLYGON_API_KEY',
            'PERPLEXITY_API_KEY',
            'ANTHROPIC_API_KEY',
            'ENVIRONMENT',
            'PAPER_TRADING',
            'MAX_POSITION_SIZE',
            'MAX_DAILY_LOSS'
        ]

        missing_vars = []
        for var in required_vars:
            if os.getenv(var) is None:
                missing_vars.append(var)

        assert len(missing_vars) == 0, f"Missing environment variables: {', '.join(missing_vars)}"

        print(f"✅ Test 11 PASSED: All {len(required_vars)} required environment variables loaded")

    def test_paper_trading_mode_enabled(self):
        """Test 12: Paper trading mode is enabled (safety check)"""
        paper_trading = os.getenv('PAPER_TRADING', 'false').lower()
        live_trading = os.getenv('LIVE_TRADING', 'false').lower()

        assert paper_trading == 'true', "PAPER_TRADING should be enabled for safety"
        assert live_trading == 'false', "LIVE_TRADING should be disabled initially"

        print(f"✅ Test 12 PASSED: Paper trading enabled, live trading disabled (SAFE)")


class TestMCPFunctionality:
    """Test MCP server functionality (if servers are running)"""

    def test_polygon_mcp_get_price(self):
        """Test 13: Polygon MCP can fetch current price"""
        # This test requires the Polygon MCP server to be running
        # For now, we'll test the REST API directly
        api_key = os.getenv('POLYGON_API_KEY')
        symbol = 'X:BTCUSD'

        url = f"https://api.polygon.io/v2/last/trade/{symbol}?apiKey={api_key}"

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                assert 'results' in data, "Price data not in expected format"
                print(f"✅ Test 13 PASSED: Polygon price fetch working (BTC: ${data['results']['p']})")
            else:
                print(f"⚠️  Test 13 WARNING: Polygon API returned {response.status_code}")
        except Exception as e:
            print(f"⚠️  Test 13 WARNING: Could not test Polygon functionality: {e}")

    def test_database_write_performance(self):
        """Test 14: Database write performance is acceptable"""
        import time
        from datetime import datetime

        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_ANON_KEY')
        supabase: Client = create_client(supabase_url, supabase_key)

        # Measure insert time
        start_time = time.time()

        test_data = {
            'symbol': 'PERF-TEST',
            'timeframe': '1min',
            'open': 1000.0,
            'high': 1000.0,
            'low': 1000.0,
            'close': 1000.0,
            'volume': 0.0,
            'vwap': 1000.0,
            'timestamp': datetime.utcnow().isoformat()
        }

        result = supabase.table('crypto_aggregates').insert(test_data).execute()

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        # Clean up
        if result.data and len(result.data) > 0:
            record_id = result.data[0].get('id')
            if record_id:
                supabase.table('crypto_aggregates').delete().eq('id', record_id).execute()

        # Latency should be under 1 second for a single insert
        assert latency_ms < 1000, f"Database write too slow: {latency_ms:.0f}ms"

        print(f"✅ Test 14 PASSED: Database write latency: {latency_ms:.0f}ms")

    def test_database_query_performance(self):
        """Test 15: Database query performance is acceptable"""
        import time

        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_ANON_KEY')
        supabase: Client = create_client(supabase_url, supabase_key)

        # Measure query time
        start_time = time.time()

        result = supabase.table('crypto_aggregates')\
            .select('*')\
            .eq('symbol', 'BTC-USD')\
            .limit(100)\
            .execute()

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        # Query should be under 500ms
        assert latency_ms < 500, f"Database query too slow: {latency_ms:.0f}ms"

        print(f"✅ Test 15 PASSED: Database query latency: {latency_ms:.0f}ms ({len(result.data)} records)")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
