from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, Optional, Tuple
import asyncio
import logging
import time


"""
WebSocket Rate Limiter
======================

Rate limiting middleware for WebSocket connections.
Prevents DoS attacks and enforces fair usage limits.

Author: RRR Ventures
Date: 2025-10-12
"""


logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    max_connections_per_ip: int = 5
    burst_size: int = 10
    cooldown_seconds: int = 60


@dataclass
class ClientRateInfo:
    """Track rate limiting info for a client."""
    ip_address: str
    connection_count: int = 0
    request_times: list = field(default_factory=list)
    last_request_time: float = 0
    blocked_until: Optional[float] = None
    violation_count: int = 0


class RateLimiter:
    """
    Rate limiter for WebSocket connections.

    Features:
    - Per-IP connection limits
    - Request rate limiting (per minute/hour)
    - Burst protection
    - Automatic cooldown for violators
    - Gradual backoff for repeat offenders
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize rate limiter.

        Args:
            config: Rate limiting configuration
        """
        self.config = config or RateLimitConfig()
        self.clients: Dict[str, ClientRateInfo] = {}
        self.cleanup_task = None
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Start background cleanup task."""
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Rate limiter initialized")

    async def shutdown(self):
        """Stop background tasks."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

    async def check_connection_allowed(self, ip_address: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a new connection is allowed from this IP.

        Args:
            ip_address: Client IP address

        Returns:
            (allowed, reason) tuple
        """
        async with self._lock:
            client = self.clients.get(ip_address)

            if not client:
                client = ClientRateInfo(ip_address=ip_address)
                self.clients[ip_address] = client

            # Check if blocked
            if client.blocked_until and time.time() < client.blocked_until:
                remaining = int(client.blocked_until - time.time())
                return False, f"Blocked for {remaining} more seconds"

            # Check connection limit
            if client.connection_count >= self.config.max_connections_per_ip:
                return False, f"Maximum {self.config.max_connections_per_ip} connections per IP"

            client.connection_count += 1
            return True, None

    async def on_disconnect(self, ip_address: str):
        """Handle client disconnect."""
        async with self._lock:
            if ip_address in self.clients:
                self.clients[ip_address].connection_count -= 1
                if self.clients[ip_address].connection_count <= 0:
                    # Clean up if no more connections
                    del self.clients[ip_address]

    async def check_request_allowed(self, ip_address: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a request is allowed based on rate limits.

        Args:
            ip_address: Client IP address

        Returns:
            (allowed, reason) tuple
        """
        async with self._lock:
            client = self.clients.get(ip_address)

            if not client:
                # Should not happen if connection was checked
                return False, "No active connection"

            # Check if blocked
            if client.blocked_until and time.time() < client.blocked_until:
                remaining = int(client.blocked_until - time.time())
                return False, f"Rate limited for {remaining} more seconds"

            current_time = time.time()

            # Clean old request times
            minute_ago = current_time - 60
            hour_ago = current_time - 3600

            client.request_times = [
                t for t in client.request_times
                if t > hour_ago
            ]

            # Count requests in last minute and hour
            minute_requests = sum(1 for t in client.request_times if t > minute_ago)
            hour_requests = len(client.request_times)

            # Check burst limit (requests in last second)
            second_ago = current_time - 1
            burst_requests = sum(1 for t in client.request_times if t > second_ago)

            if burst_requests >= self.config.burst_size:
                client.violation_count += 1
                client.blocked_until = current_time + self._calculate_cooldown(client.violation_count)
                return False, f"Burst limit exceeded ({self.config.burst_size} requests/second)"

            # Check minute limit
            if minute_requests >= self.config.max_requests_per_minute:
                client.violation_count += 1
                client.blocked_until = current_time + self._calculate_cooldown(client.violation_count)
                return False, f"Minute limit exceeded ({self.config.max_requests_per_minute} requests/minute)"

            # Check hour limit
            if hour_requests >= self.config.max_requests_per_hour:
                client.violation_count += 1
                client.blocked_until = current_time + self._calculate_cooldown(client.violation_count)
                return False, f"Hour limit exceeded ({self.config.max_requests_per_hour} requests/hour)"

            # Request allowed
            client.request_times.append(current_time)
            client.last_request_time = current_time

            return True, None

    def _calculate_cooldown(self, violation_count: int) -> float:
        """
        Calculate cooldown period based on violation count.

        Implements exponential backoff for repeat offenders.
        """
        base_cooldown = self.config.cooldown_seconds
        # Exponential backoff: 60s, 120s, 240s, 480s, etc.
        return min(base_cooldown * (2 ** (violation_count - 1)), 3600)  # Max 1 hour

    async def _cleanup_loop(self):
        """Periodically clean up old client data."""
        while True:
            try:
                await asyncio.sleep(300)  # Clean every 5 minutes

                async with self._lock:
                    current_time = time.time()
                    inactive_threshold = current_time - 3600  # 1 hour

                    # Remove inactive clients
                    to_remove = []
                    for ip, client in self.clients.items():
                        if (client.connection_count == 0 and
                            client.last_request_time < inactive_threshold):
                            to_remove.append(ip)

                    for ip in to_remove:
                        del self.clients[ip]

                    if to_remove:
                        logger.info(f"Cleaned up {len(to_remove)} inactive clients")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    @lru_cache(maxsize=128)

    def get_stats(self) -> Dict:
        """Get rate limiter statistics."""
        return {
            'active_clients': len(self.clients),
            'blocked_clients': sum(
                1 for c in self.clients.values()
                if c.blocked_until and time.time() < c.blocked_until
            ),
            'total_connections': sum(c.connection_count for c in self.clients.values()),
            'config': {
                'max_requests_per_minute': self.config.max_requests_per_minute,
                'max_requests_per_hour': self.config.max_requests_per_hour,
                'max_connections_per_ip': self.config.max_connections_per_ip,
                'burst_size': self.config.burst_size
            }
        }


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


@lru_cache(maxsize=128)


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter