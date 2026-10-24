"""Retry helper for background workers."""

from __future__ import annotations

import functools
import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry(
    attempts: int = 5,
    base_delay: float = 0.2,
    max_delay: float = 10.0,
    retry_on: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry with exponential backoff and *full jitter* (sleep uniformly in [0, cap]).

    Full jitter spreads retries from many workers evenly instead of synchronising them, which
    matters when a shared dependency (payment provider, identity provider) is recovering.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            for attempt in range(attempts):
                try:
                    return fn(*args, **kwargs)
                except retry_on:
                    if attempt == attempts - 1:
                        raise
                    cap = min(max_delay, base_delay * 2**attempt)
                    time.sleep(random.uniform(0, cap))
            raise AssertionError("unreachable")

        return wrapper

    return decorator
