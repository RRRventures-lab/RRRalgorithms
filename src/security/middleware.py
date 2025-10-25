"""
Security Middleware for FastAPI
Provides rate limiting, security headers, and audit logging
"""

import time
import uuid
from datetime import datetime
from typing import Callable, Dict, Optional
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from fastapi import status
import logging

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses

    Headers added:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - X-XSS-Protection: 1; mode=block
    - Strict-Transport-Security: max-age=31536000; includeSubDomains
    - Content-Security-Policy: default-src 'self'
    - Referrer-Policy: strict-origin-when-cross-origin
    - Permissions-Policy: geolocation=(), microphone=(), camera=()
    """

    def __init__(self, app, enable_hsts: bool = True, csp_policy: Optional[str] = None):
        super().__init__(app)
        self.enable_hsts = enable_hsts
        self.csp_policy = csp_policy or "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        response.headers["Content-Security-Policy"] = self.csp_policy

        # HSTS - only enable in production with HTTPS
        if self.enable_hsts and request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"

        # Add request ID for tracing
        if not response.headers.get("X-Request-ID"):
            response.headers["X-Request-ID"] = str(uuid.uuid4())

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware for API endpoints

    Features:
    - Per-IP rate limiting
    - Different limits for different endpoints
    - Sliding window algorithm
    - Rate limit headers in response
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size

        # Track request times per IP
        self.request_times: Dict[str, list] = defaultdict(list)
        self.blocked_until: Dict[str, float] = {}

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request"""
        # Check X-Forwarded-For header (for proxies)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to direct client
        return request.client.host if request.client else "unknown"

    def _is_rate_limited(self, ip: str) -> tuple[bool, Optional[str], int]:
        """
        Check if IP is rate limited

        Returns:
            (is_limited, reason, remaining_requests)
        """
        current_time = time.time()

        # Check if blocked
        if ip in self.blocked_until and current_time < self.blocked_until[ip]:
            remaining = int(self.blocked_until[ip] - current_time)
            return True, f"Blocked for {remaining}s due to rate limit violation", 0

        # Clean old requests
        minute_ago = current_time - 60
        hour_ago = current_time - 3600
        second_ago = current_time - 1

        self.request_times[ip] = [
            t for t in self.request_times[ip]
            if t > hour_ago
        ]

        # Count requests
        minute_requests = sum(1 for t in self.request_times[ip] if t > minute_ago)
        hour_requests = len(self.request_times[ip])
        burst_requests = sum(1 for t in self.request_times[ip] if t > second_ago)

        # Check burst limit
        if burst_requests >= self.burst_size:
            self.blocked_until[ip] = current_time + 60  # Block for 1 minute
            return True, f"Burst limit exceeded ({self.burst_size} req/s)", 0

        # Check minute limit
        if minute_requests >= self.requests_per_minute:
            self.blocked_until[ip] = current_time + 60
            return True, f"Rate limit exceeded ({self.requests_per_minute} req/min)", 0

        # Check hour limit
        if hour_requests >= self.requests_per_hour:
            self.blocked_until[ip] = current_time + 3600
            return True, f"Rate limit exceeded ({self.requests_per_hour} req/hour)", 0

        # Not limited
        remaining = self.requests_per_minute - minute_requests - 1
        return False, None, max(0, remaining)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        ip = self._get_client_ip(request)
        is_limited, reason, remaining = self._is_rate_limited(ip)

        if is_limited:
            logger.warning(f"Rate limit exceeded for IP {ip}: {reason}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": reason,
                    "retry_after": 60
                },
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + 60),
                    "Retry-After": "60"
                }
            )

        # Record request
        self.request_times[ip].append(time.time())

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)

        return response


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """
    Audit logging middleware

    Logs:
    - All API requests
    - Response status codes
    - Request duration
    - Client IP
    - User agent
    - Authentication status
    """

    def __init__(self, app, log_body: bool = False):
        super().__init__(app)
        self.log_body = log_body

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Get client info
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "unknown")

        # Get user from auth (if available)
        user_id = "anonymous"
        if "authorization" in request.headers:
            user_id = "authenticated"  # Would extract from JWT in real implementation

        # Log request
        logger.info(
            "API Request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": client_ip,
                "user_agent": user_agent,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        # Process request
        try:
            response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000

            # Log response
            logger.info(
                "API Response",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

            # Add request ID to response
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            # Log error
            logger.error(
                "API Error",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "duration_ms": round(duration_ms, 2),
                    "timestamp": datetime.utcnow().isoformat()
                },
                exc_info=True
            )

            # Re-raise to let FastAPI handle it
            raise
