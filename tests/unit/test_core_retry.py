from src.core.retry import async_retry, sync_retry, CircuitBreaker, RetryExhausted
import asyncio
import pytest

"""
Unit tests for core.retry module.
"""



@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_retry_success():
    """Test async retry with successful function."""
    call_count = 0

    @async_retry(max_attempts=3, delay=0.01)
    async def successful_function():
        nonlocal call_count
        call_count += 1
        return "success"

    result = await successful_function()
    assert result == "success"
    assert call_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_retry_with_failure_then_success():
    """Test async retry with failure then success."""
    call_count = 0

    @async_retry(max_attempts=3, delay=0.01, exceptions=(ValueError,))
    async def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Temporary failure")
        return "success"

    result = await flaky_function()
    assert result == "success"
    assert call_count == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_retry_exhausted():
    """Test async retry when all attempts fail."""

    @async_retry(max_attempts=3, delay=0.01, exceptions=(ValueError,))
    async def always_fails():
        raise ValueError("Always fails")

    with pytest.raises(ValueError):
        await always_fails()


@pytest.mark.unit
def test_sync_retry_success():
    """Test sync retry with successful function."""
    call_count = 0

    @sync_retry(max_attempts=3, delay=0.01)
    def successful_function():
        nonlocal call_count
        call_count += 1
        return "success"

    result = successful_function()
    assert result == "success"
    assert call_count == 1


@pytest.mark.unit
def test_sync_retry_with_failure_then_success():
    """Test sync retry with failure then success."""
    call_count = 0

    @sync_retry(max_attempts=3, delay=0.01, exceptions=(ValueError,))
    def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Temporary failure")
        return "success"

    result = flaky_function()
    assert result == "success"
    assert call_count == 3


@pytest.mark.unit
def test_circuit_breaker_closed():
    """Test circuit breaker in closed state."""
    cb = CircuitBreaker(failure_threshold=3, timeout=1.0)

    def successful_function():
        return "success"

    result = cb.call(successful_function)
    assert result == "success"
    assert cb.state == "closed"


@pytest.mark.unit
def test_circuit_breaker_opens():
    """Test circuit breaker opens after threshold."""
    cb = CircuitBreaker(failure_threshold=3, timeout=1.0)

    def failing_function():
        raise ValueError("Failure")

    # Trigger failures to open circuit
    for _ in range(3):
        with pytest.raises(ValueError):
            cb.call(failing_function)

    assert cb.state == "open"

    # Next call should fail immediately without calling function
    with pytest.raises(RetryExhausted):
        cb.call(failing_function)
