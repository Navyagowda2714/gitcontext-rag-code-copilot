"""Thin client for the identity provider's OAuth 2.0 token endpoint."""

from __future__ import annotations

import time

import httpx

from shopfront.auth.tokens import TokenPair, decode_expiry
from shopfront.config import settings


class ReauthRequired(Exception):
    """The refresh token is no longer valid (revoked, expired or already rotated)."""


class OAuthClient:
    def __init__(self, http: httpx.Client | None = None) -> None:
        self._http = http or httpx.Client(timeout=settings.token_timeout_seconds)

    def exchange_code(self, code: str, redirect_uri: str) -> TokenPair:
        """Authorization-code grant used once at sign-in."""
        return self._token_request(
            {"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri}
        )

    def refresh_access_token(self, refresh_token: str) -> TokenPair:
        """Refresh-token grant. The provider rotates refresh tokens: the old one is invalid after
        this call succeeds, so callers must store the returned pair atomically."""
        return self._token_request({"grant_type": "refresh_token", "refresh_token": refresh_token})

    def _token_request(self, form: dict[str, str]) -> TokenPair:
        form |= {"client_id": settings.oauth_client_id, "client_secret": settings.oauth_client_secret}
        try:
            resp = self._http.post(settings.token_url, data=form)
        except httpx.TransportError as exc:
            raise ConnectionError(str(exc)) from exc
        if resp.status_code == 400 and resp.json().get("error") == "invalid_grant":
            raise ReauthRequired(resp.json().get("error_description", "invalid_grant"))
        resp.raise_for_status()
        body = resp.json()
        access = body["access_token"]
        expires_at = decode_expiry(access) if "." in access else time.time() + body["expires_in"]
        return TokenPair(access, body["refresh_token"], expires_at)
