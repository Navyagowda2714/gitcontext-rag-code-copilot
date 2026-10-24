"""Distributed token-bucket rate limiter backed by Redis.

Each API key gets a bucket of ``burst`` tokens that refills at ``per_minute / 60`` tokens per
second. The check-and-decrement runs atomically inside a Lua script so that all API pods share one
budget per key (the old in-process limiter multiplied the effective limit by the pod count).
"""

from __future__ import annotations

import time
from typing import Protocol

from shopfront.config import Settings

TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1]) or burst
local ts = tonumber(state[2]) or now
tokens = math.min(burst, tokens + (now - ts) * rate)
local allowed = 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
end
redis.call('HSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, math.ceil(burst / rate) + 1)
return {allowed, tostring(tokens)}
"""


class RedisLike(Protocol):
    def register_script(self, script: str): ...


class RateLimiter:
    def __init__(self, redis: RedisLike, per_minute: int, burst: int) -> None:
        if burst < 1 or per_minute < 1:
            raise ValueError("rate limits must be positive")
        self.rate = per_minute / 60.0
        self.burst = burst
        self._script = redis.register_script(TOKEN_BUCKET_LUA)

    @classmethod
    def from_settings(cls, settings: Settings) -> RateLimiter:
        import redis

        settings.validate()
        return cls(
            redis.Redis.from_url(settings.redis_url),
            settings.rate_limit_per_minute,
            settings.rate_limit_burst,
        )

    def allow(self, api_key: str) -> tuple[bool, float]:
        """Consume one token for ``api_key``. Returns (allowed, seconds until next token)."""
        allowed, remaining = self._script(
            keys=[f"ratelimit:{api_key}"], args=[self.rate, self.burst, time.time()]
        )
        if allowed:
            return True, 0.0
        return False, max(0.0, (1 - float(remaining)) / self.rate)
