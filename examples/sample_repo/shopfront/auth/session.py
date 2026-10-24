"""Session management for API clients.

A ``SessionManager`` owns one OAuth token pair and hands out access tokens to callers. Access tokens
are refreshed proactively: a token is refreshed when it is within ``REFRESH_SKEW_SECONDS`` of
expiry, and only one refresh may be in flight at a time (single-flight). The identity provider
rotates refresh tokens on every use, so a second concurrent refresh would present an already-used
refresh token and be rejected with ``invalid_grant``.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable

from shopfront.auth.oauth_client import OAuthClient, ReauthRequired
from shopfront.auth.tokens import TokenPair

logger = logging.getLogger(__name__)

# Refresh this many seconds before expiry. Covers the worst clock skew measured between pods
# (23s) plus the p99 latency of the token endpoint (~9s), with headroom. See INC-2026-014.
REFRESH_SKEW_SECONDS = 60
# Random jitter added to the skew so a fleet of workers does not refresh in lockstep.
REFRESH_JITTER_SECONDS = 15
MAX_REFRESH_ATTEMPTS = 3


class SessionManager:
    """Thread-safe holder of an access/refresh token pair for one user or service account."""

    def __init__(
        self,
        client: OAuthClient,
        tokens: TokenPair,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._client = client
        self._tokens = tokens
        self._clock = clock
        self._lock = threading.Lock()
        self._jitter = random.uniform(0, REFRESH_JITTER_SECONDS)

    def get_access_token(self) -> str:
        """Return a valid access token, refreshing it first if it is about to expire."""
        self.refresh_if_needed()
        return self._tokens.access_token

    def refresh_if_needed(self) -> bool:
        """Refresh the token pair when it expires within the skew window.

        Uses double-checked locking: the expiry check runs once without the lock (fast path) and
        again after acquiring it, so when many threads notice an expiring token at the same moment
        only the first one calls the token endpoint and the others reuse its result.
        Returns True if this call performed the refresh.
        """
        if not self._needs_refresh():
            return False
        with self._lock:
            if not self._needs_refresh():
                return False
            self._tokens = self._refresh_with_retry()
            logger.info("access token refreshed; expires at %s", self._tokens.expires_at)
            return True

    def _needs_refresh(self) -> bool:
        skew = REFRESH_SKEW_SECONDS + self._jitter
        return self._tokens.expires_within(skew, now=self._clock())

    def _refresh_with_retry(self) -> TokenPair:
        """Call the token endpoint, retrying transient network errors with exponential backoff.

        An ``invalid_grant`` response is never retried: it means the refresh token was revoked or
        already rotated, so the user has to sign in again.
        """
        delay = 0.5
        for attempt in range(1, MAX_REFRESH_ATTEMPTS + 1):
            try:
                return self._client.refresh_access_token(self._tokens.refresh_token)
            except ReauthRequired:
                logger.warning("refresh token rejected (invalid_grant); re-authentication required")
                raise
            except ConnectionError:
                if attempt == MAX_REFRESH_ATTEMPTS:
                    raise
                time.sleep(delay + random.uniform(0, delay))
                delay *= 2
        raise AssertionError("unreachable")

    def invalidate(self) -> None:
        """Force the next call to refresh, e.g. after the API answered 401 for this token."""
        with self._lock:
            self._tokens = self._tokens.expired_copy()
