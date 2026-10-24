"""Request middleware: bearer-token authentication and per-key rate limiting."""

from __future__ import annotations

from dataclasses import dataclass

import jwt

from shopfront.api.ratelimit import RateLimiter
from shopfront.config import settings

PUBLIC_PATHS = frozenset({"/healthz", "/readyz", "/webhooks/payments"})


@dataclass
class Request:
    path: str
    headers: dict[str, str]


@dataclass
class Response:
    status: int
    body: dict[str, object]
    headers: dict[str, str] | None = None


class AuthMiddleware:
    """Verify the bearer JWT on every non-public request.

    Returns 401 with ``WWW-Authenticate: Bearer error="invalid_token"`` when the token is expired
    or has a bad signature. Clients react to that header by refreshing (see
    ``SessionManager.invalidate`` and ``web/src/apiClient.ts``).
    """

    def __init__(self, public_key: str, audience: str = "shopfront-api") -> None:
        self._public_key = public_key
        self._audience = audience

    def __call__(self, request: Request) -> Response | None:
        if request.path in PUBLIC_PATHS:
            return None
        header = request.headers.get("authorization", "")
        if not header.lower().startswith("bearer "):
            return Response(401, {"error": "missing bearer token"})
        try:
            jwt.decode(
                header[7:], self._public_key, algorithms=["RS256"], audience=self._audience,
                leeway=30,
            )
        except jwt.ExpiredSignatureError:
            return Response(
                401, {"error": "token expired"}, {"WWW-Authenticate": 'Bearer error="invalid_token"'}
            )
        except jwt.InvalidTokenError:
            return Response(401, {"error": "invalid token"})
        return None


class RateLimitMiddleware:
    """Reject requests over the per-API-key budget with 429 and a ``Retry-After`` header."""

    def __init__(self, limiter: RateLimiter | None = None) -> None:
        self._limiter = limiter or RateLimiter.from_settings(settings)

    def __call__(self, request: Request) -> Response | None:
        key = request.headers.get("x-api-key", "anonymous")
        allowed, retry_after = self._limiter.allow(key)
        if allowed:
            return None
        return Response(
            429, {"error": "rate limit exceeded"}, {"Retry-After": str(int(retry_after) + 1)}
        )
