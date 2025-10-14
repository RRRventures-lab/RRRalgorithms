from collections import deque
from contextlib import contextmanager
from functools import lru_cache
from functools import wraps
from src.core.constants import APIConstants
from src.core.exceptions import RateLimitError as APIRateLimitError
from typing import Callable, Optional, Dict
import threading
import time


"""
Rate Limiting Framework
========================

Prevents API rate limit violations by tracking and enforcing call limits.

Usage:
    from src.core.rate_limiter import RateLimiter, rate_limit
    
    # Decorator usage
    @rate_limit(max_calls=5, period=1.0)
    def fetch_market_data():
        # API call
        pass
    
    # Direct usage
    limiter = RateLimiter(max_calls=5, period=1.0)
    with limiter:
        # API call
        pass
"""




class RateLimiter:
    """
    Token bucket rate limiter.
    
    Thread-safe implementation that tracks API call timestamps
    and enforces rate limits.
    """
    
    def __init__(
        self,
        max_calls: int,
        period: float = 1.0,
        name: Optional[str] = None
    ):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in period
            period: Time period in seconds
            name: Optional name for the limiter (for logging)
        """
        self.max_calls = max_calls
        self.period = period
        self.name = name or "RateLimiter"
        
        # Use deque for O(1) operations
        self.calls = deque(maxlen=max_calls)
        self.lock = threading.Lock()
    
    def __enter__(self):
        """Context manager entry - wait if necessary"""
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        pass
    
    def acquire(self) -> None:
        """
        Acquire permission to make an API call.
        Blocks if rate limit would be exceeded.
        """
        with self.lock:
            now = time.time()
            
            # Remove old calls outside the time window
            while self.calls and self.calls[0] < now - self.period:
                self.calls.popleft()
            
            # Check if we need to wait
            if len(self.calls) >= self.max_calls:
                # Calculate how long to wait
                oldest_call = self.calls[0]
                wait_time = self.period - (now - oldest_call)
                
                if wait_time > 0:
                    time.sleep(wait_time)
                    now = time.time()
                    
                    # Remove expired calls after waiting
                    while self.calls and self.calls[0] < now - self.period:
                        self.calls.popleft()
            
            # Record this call
            self.calls.append(now)
    
    def try_acquire(self) -> bool:
        """
        Try to acquire permission without blocking.
        
        Returns:
            True if permission granted, False if rate limited
        """
        with self.lock:
            now = time.time()
            
            # Remove old calls
            while self.calls and self.calls[0] < now - self.period:
                self.calls.popleft()
            
            # Check if we can make a call
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            
            return False
    
    @lru_cache(maxsize=128)
    
    def get_remaining_calls(self) -> int:
        """Get number of calls remaining in current window"""
        with self.lock:
            now = time.time()
            
            # Remove expired calls
            while self.calls and self.calls[0] < now - self.period:
                self.calls.popleft()
            
            return max(0, self.max_calls - len(self.calls))
    
    @lru_cache(maxsize=128)
    
    def get_reset_time(self) -> float:
        """
        Get time (in seconds) until rate limit resets.
        
        Returns:
            Seconds until oldest call expires, or 0 if not rate limited
        """
        with self.lock:
            if len(self.calls) < self.max_calls:
                return 0.0
            
            now = time.time()
            oldest_call = self.calls[0]
            return max(0.0, self.period - (now - oldest_call))
    
    def reset(self) -> None:
        """Reset the rate limiter (clear all recorded calls)"""
        with self.lock:
            self.calls.clear()


# =============================================================================
# Global Rate Limiters
# =============================================================================

# Pre-configured rate limiters for common APIs
_rate_limiters: Dict[str, RateLimiter] = {
    'polygon': RateLimiter(
        max_calls=APIConstants.POLYGON_RATE_LIMIT,
        period=1.0,
        name='Polygon.io'
    ),
    'coinbase': RateLimiter(
        max_calls=APIConstants.COINBASE_RATE_LIMIT,
        period=1.0,
        name='Coinbase'
    ),
    'perplexity': RateLimiter(
        max_calls=APIConstants.PERPLEXITY_RATE_LIMIT,
        period=1.0,
        name='Perplexity'
    ),
}


@lru_cache(maxsize=128)


def get_rate_limiter(api_name: str) -> RateLimiter:
    """
    Get rate limiter for a specific API.
    
    Args:
        api_name: API name ('polygon', 'coinbase', 'perplexity')
        
    Returns:
        RateLimiter instance
        
    Raises:
        ValueError: If API name is not recognized
    """
    if api_name not in _rate_limiters:
        raise ValueError(
            f"Unknown API: {api_name}. "
            f"Available: {list(_rate_limiters.keys())}"
        )
    
    return _rate_limiters[api_name]


def register_rate_limiter(
    api_name: str,
    max_calls: int,
    period: float = 1.0
) -> RateLimiter:
    """
    Register a new rate limiter for an API.
    
    Args:
        api_name: API name
        max_calls: Maximum calls per period
        period: Time period in seconds
        
    Returns:
        Newly created RateLimiter
    """
    limiter = RateLimiter(max_calls=max_calls, period=period, name=api_name)
    _rate_limiters[api_name] = limiter
    return limiter


# =============================================================================
# Decorator
# =============================================================================

def rate_limit(
    max_calls: Optional[int] = None,
    period: float = 1.0,
    api_name: Optional[str] = None,
    raise_on_limit: bool = False
) -> Callable:
    """
    Decorator to enforce rate limiting on functions.
    
    Args:
        max_calls: Maximum calls per period (required if api_name not provided)
        period: Time period in seconds
        api_name: Use pre-configured rate limiter for this API
        raise_on_limit: If True, raise exception instead of waiting
        
    Usage:
        @rate_limit(max_calls=5, period=1.0)
        def my_api_call():
            pass
        
        @rate_limit(api_name='polygon')
        def fetch_polygon_data():
            pass
    """
    # Get or create rate limiter
    if api_name:
        limiter = get_rate_limiter(api_name)
    elif max_calls is not None:
        limiter = RateLimiter(max_calls=max_calls, period=period)
    else:
        raise ValueError("Must provide either max_calls or api_name")
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if raise_on_limit:
                # Try to acquire without waiting
                if not limiter.try_acquire():
                    reset_time = limiter.get_reset_time()
                    raise APIRateLimitError(
                        message=f"Rate limit exceeded for {limiter.name}",
                        api_name=limiter.name,
                        retry_after=int(reset_time) + 1
                    )
            else:
                # Wait if necessary
                limiter.acquire()
            
            return func(*args, **kwargs)
        
        # Expose rate limiter for inspection
        wrapper.rate_limiter = limiter
        return wrapper
    
    return decorator


@contextmanager
def rate_limited(api_name: str):
    """
    Context manager for rate-limited API calls.
    
    Usage:
        with rate_limited('polygon'):
            # Make API call
            response = requests.get(...)
    """
    limiter = get_rate_limiter(api_name)
    limiter.acquire()
    yield limiter


# =============================================================================
# Utility Functions
# =============================================================================

@lru_cache(maxsize=128)

def get_rate_limit_status(api_name: str) -> Dict[str, any]:
    """
    Get current rate limit status for an API.
    
    Args:
        api_name: API name
        
    Returns:
        Dictionary with rate limit status
    """
    limiter = get_rate_limiter(api_name)
    
    return {
        'api_name': api_name,
        'max_calls': limiter.max_calls,
        'period': limiter.period,
        'remaining_calls': limiter.get_remaining_calls(),
        'reset_time': limiter.get_reset_time(),
        'current_calls': len(limiter.calls)
    }


@lru_cache(maxsize=128)


def get_all_rate_limit_status() -> Dict[str, Dict[str, any]]:
    """
    Get rate limit status for all registered APIs.
    
    Returns:
        Dictionary mapping API name to status
    """
    return {
        api_name: get_rate_limit_status(api_name)
        for api_name in _rate_limiters.keys()
    }


def reset_all_rate_limiters() -> None:
    """Reset all rate limiters (useful for testing)"""
    for limiter in _rate_limiters.values():
        limiter.reset()


__all__ = [
    'RateLimiter',
    'rate_limit',
    'rate_limited',
    'get_rate_limiter',
    'register_rate_limiter',
    'get_rate_limit_status',
    'get_all_rate_limit_status',
    'reset_all_rate_limiters',
]

