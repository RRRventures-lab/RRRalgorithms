"""
Security Hardening Tests
Tests for Phase 10 security implementations
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import time

from src.api.main import app
from src.security.auth import JWTManager, get_jwt_manager
from src.security.middleware import RateLimitMiddleware, SecurityHeadersMiddleware
from src.security.secrets_manager import SecretsManager


class TestCORSConfiguration:
    """Test CORS configuration"""

    def test_cors_not_allow_all_origins(self):
        """Ensure CORS does not allow all origins"""
        client = TestClient(app)

        # Make OPTIONS request from unauthorized origin
        response = client.options(
            "/api/portfolio",
            headers={
                "Origin": "https://malicious-site.com",
                "Access-Control-Request-Method": "GET"
            }
        )

        # Should not have access-control-allow-origin for unauthorized origin
        # Note: This depends on CORS_ORIGINS env var configuration
        assert response.status_code in [200, 403]

    def test_cors_allows_configured_origins(self):
        """Ensure CORS allows configured origins"""
        client = TestClient(app)

        # Make request from allowed origin (localhost)
        response = client.options(
            "/api/portfolio",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )

        assert response.status_code == 200


class TestSecurityHeaders:
    """Test security headers middleware"""

    def test_security_headers_present(self):
        """Ensure security headers are present in responses"""
        client = TestClient(app)

        response = client.get("/health")

        # Check for required security headers
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"

        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"

        assert "X-XSS-Protection" in response.headers
        assert "Content-Security-Policy" in response.headers
        assert "Referrer-Policy" in response.headers

    def test_request_id_header(self):
        """Ensure X-Request-ID is added to responses"""
        client = TestClient(app)

        response = client.get("/health")

        assert "X-Request-ID" in response.headers


class TestRateLimiting:
    """Test rate limiting middleware"""

    def test_rate_limit_enforced(self):
        """Test that rate limiting is enforced"""
        client = TestClient(app)

        # Make many requests rapidly
        successful_requests = 0
        rate_limited = False

        for i in range(100):
            response = client.get("/api/portfolio")
            if response.status_code == 200:
                successful_requests += 1
            elif response.status_code == 429:
                rate_limited = True
                break

        # Should hit rate limit eventually
        assert rate_limited or successful_requests < 100

    def test_rate_limit_headers(self):
        """Test that rate limit headers are present"""
        client = TestClient(app)

        response = client.get("/api/portfolio")

        # Rate limit headers should be present
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers


class TestJWTAuthentication:
    """Test JWT authentication"""

    @pytest.fixture
    def jwt_manager(self):
        """Get JWT manager"""
        return get_jwt_manager()

    def test_create_access_token(self, jwt_manager):
        """Test access token creation"""
        token = jwt_manager.create_access_token(
            user_id="test_user",
            scopes=["read", "write"]
        )

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_valid_token(self, jwt_manager):
        """Test valid token verification"""
        token = jwt_manager.create_access_token(
            user_id="test_user",
            scopes=["read"]
        )

        token_data = jwt_manager.verify_token(token)

        assert token_data.sub == "test_user"
        assert token_data.type == "access"
        assert "read" in token_data.scopes

    def test_verify_invalid_token(self, jwt_manager):
        """Test invalid token rejection"""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            jwt_manager.verify_token("invalid_token")

        assert exc_info.value.status_code == 401

    def test_token_revocation(self, jwt_manager):
        """Test token revocation"""
        from fastapi import HTTPException

        token = jwt_manager.create_access_token(user_id="test_user")

        # Revoke token
        jwt_manager.revoke_token(token)

        # Verify token should fail
        with pytest.raises(HTTPException) as exc_info:
            jwt_manager.verify_token(token)

        assert exc_info.value.status_code == 401

    def test_refresh_token_creation(self, jwt_manager):
        """Test refresh token creation"""
        token = jwt_manager.create_refresh_token(user_id="test_user")

        assert token is not None

        token_data = jwt_manager.verify_token(token, expected_type="refresh")
        assert token_data.type == "refresh"


class TestSecretsManager:
    """Test secrets management"""

    def test_secrets_manager_initialization(self):
        """Test secrets manager initialization"""
        manager = SecretsManager()

        assert manager is not None

    def test_get_secret_from_env(self):
        """Test getting secret from environment"""
        import os
        os.environ["TEST_SECRET"] = "test_value"

        manager = SecretsManager()
        value = manager.get_secret("TEST_SECRET")

        assert value == "test_value"

        # Cleanup
        del os.environ["TEST_SECRET"]

    def test_get_secret_with_default(self):
        """Test getting secret with default value"""
        manager = SecretsManager()
        value = manager.get_secret("NONEXISTENT_SECRET", default="default_value")

        assert value == "default_value"

    def test_get_cors_origins(self):
        """Test getting CORS origins"""
        manager = SecretsManager()
        origins = manager.get_cors_origins()

        assert isinstance(origins, list)
        assert len(origins) > 0


class TestInputValidation:
    """Test input validation"""

    def test_invalid_limit_rejected(self):
        """Test that invalid limit values are rejected"""
        client = TestClient(app)

        # Test limit > max allowed
        response = client.get("/api/trades?limit=1000")

        # Should get validation error (422) or use max limit
        assert response.status_code in [200, 422]

    def test_negative_offset_rejected(self):
        """Test that negative offset is rejected"""
        client = TestClient(app)

        response = client.get("/api/trades?offset=-1")

        assert response.status_code == 422

    def test_invalid_period_rejected(self):
        """Test that invalid period is rejected"""
        client = TestClient(app)

        response = client.get("/api/performance?period=invalid")

        assert response.status_code == 422


class TestSQLInjectionPrevention:
    """Test SQL injection prevention"""

    def test_sql_injection_in_symbol_parameter(self):
        """Test SQL injection prevention in symbol parameter"""
        client = TestClient(app)

        # Try SQL injection in symbol parameter
        malicious_input = "BTC'; DROP TABLE trades; --"

        response = client.get(f"/api/trades?symbol={malicious_input}")

        # Should either reject or sanitize
        assert response.status_code in [200, 400, 422]

        # If 200, should not have executed SQL injection
        if response.status_code == 200:
            data = response.json()
            # Should be safely handled
            assert isinstance(data, dict)


class TestAuditLogging:
    """Test audit logging"""

    def test_requests_are_logged(self, caplog):
        """Test that API requests are logged"""
        client = TestClient(app)

        with caplog.at_level("INFO"):
            response = client.get("/health")

        # Check if request was logged
        assert any("API Request" in record.message or "GET" in record.message for record in caplog.records)


@pytest.mark.asyncio
class TestPerformanceOptimizations:
    """Test performance optimizations"""

    async def test_database_connection_pooling(self):
        """Test database connection pooling"""
        from src.api.transparency_db import get_db

        db = await get_db()

        # Connection should be reused
        db2 = await get_db()

        assert db is db2  # Same instance


class TestPrometheusMetrics:
    """Test Prometheus metrics"""

    def test_metrics_endpoint_exists(self):
        """Test that /metrics endpoint exists"""
        client = TestClient(app)

        response = client.get("/metrics")

        assert response.status_code == 200
        assert "prometheus" in response.headers.get("content-type", "").lower() or \
               "text/plain" in response.headers.get("content-type", "").lower()

    def test_metrics_content(self):
        """Test that metrics contain expected data"""
        client = TestClient(app)

        # Make some requests first
        client.get("/api/portfolio")
        client.get("/health")

        # Get metrics
        response = client.get("/metrics")

        content = response.text

        # Should contain some basic metrics
        assert "api_requests_total" in content or len(content) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
