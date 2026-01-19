"""
Utils - Rate Limiter
Utilities for handling API rate limits
"""

import time
import re
from typing import Callable, Any, Optional
from functools import wraps


def handle_rate_limit_error(error_msg: str) -> Optional[float]:
    """
    Parse rate limit error message for retry delay.
    
    Args:
        error_msg: Error message string
        
    Returns:
        Retry delay in seconds, or None if not a rate limit error
    """
    if "429" not in error_msg and "RESOURCE_EXHAUSTED" not in error_msg:
        return None
    
    retry_match = re.search(r'retry in (\d+\.?\d*)s', error_msg)
    if retry_match:
        return min(float(retry_match.group(1)), 60)  # Cap at 60s
    
    return 30  # Default retry delay


def rate_limited_call(
    func: Callable,
    *args,
    max_retries: int = 3,
    base_delay: float = 2.0,
    **kwargs
) -> Any:
    """
    Execute a function with automatic rate limit handling.
    
    Args:
        func: Function to call
        *args: Positional arguments
        max_retries: Maximum retry attempts
        base_delay: Base delay for exponential backoff
        **kwargs: Keyword arguments
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            error_msg = str(e)
            
            retry_delay = handle_rate_limit_error(error_msg)
            
            if retry_delay is not None:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
            else:
                # Exponential backoff for other errors
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * base_delay
                    time.sleep(wait_time)
                    continue
            
    raise last_exception


def with_rate_limit(max_retries: int = 3, base_delay: float = 2.0):
    """
    Decorator for rate limit handling.
    
    Args:
        max_retries: Maximum retry attempts
        base_delay: Base delay for exponential backoff
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return rate_limited_call(
                func, *args, 
                max_retries=max_retries, 
                base_delay=base_delay, 
                **kwargs
            )
        return wrapper
    return decorator
