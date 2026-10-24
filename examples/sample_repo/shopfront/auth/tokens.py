"""Token value objects and JWT expiry parsing."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class TokenPair:
    """An OAuth access token, its rotating refresh token and the absolute expiry (unix seconds)."""

    access_token: str
    refresh_token: str
    expires_at: float

    def expires_within(self, seconds: float, now: float) -> bool:
        """True if the access token expires in less than ``seconds`` from ``now``."""
        return self.expires_at - now < seconds

    def expired_copy(self) -> TokenPair:
        return replace(self, expires_at=0.0)


def decode_expiry(jwt: str) -> float:
    """Read the ``exp`` claim from a JWT without verifying the signature.

    Only used to schedule refreshes; authorization decisions always verify the signature in
    ``shopfront.api.middleware``.
    """
    try:
        payload_b64 = jwt.split(".")[1]
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded))
        return float(claims["exp"])
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError("malformed JWT: cannot read exp claim") from exc
