"""Read-through TTL cache with probabilistic early expiration (cache-stampede protection)."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from typing import Any


class TTLCache:
    """In-process cache. Uses the XFetch algorithm: each reader may recompute a value slightly
    before it expires, with probability that grows as expiry approaches, so a hot key does not
    expire for every worker at once and stampede the database (see discussion #31)."""

    def __init__(self, ttl_seconds: float = 60.0, beta: float = 1.0) -> None:
        self.ttl = ttl_seconds
        self.beta = beta
        self._data: dict[str, tuple[Any, float, float]] = {}

    def get_or_compute(self, key: str, compute: Callable[[], Any]) -> Any:
        now = time.time()
        entry = self._data.get(key)
        if entry is not None:
            value, expires_at, delta = entry
            # XFetch: recompute early when now - delta * beta * ln(rand) >= expiry
            if now - delta * self.beta * math.log(random.random()) < expires_at:
                return value
        start = time.time()
        value = compute()
        delta = time.time() - start
        self._data[key] = (value, time.time() + self.ttl, delta)
        return value

    def invalidate(self, key: str) -> None:
        self._data.pop(key, None)
