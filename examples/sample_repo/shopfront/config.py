"""Service configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    token_url: str = os.getenv("SHOPFRONT_TOKEN_URL", "https://id.shopfront.dev/oauth/token")
    oauth_client_id: str = os.getenv("SHOPFRONT_OAUTH_CLIENT_ID", "")
    oauth_client_secret: str = os.getenv("SHOPFRONT_OAUTH_CLIENT_SECRET", "")
    token_timeout_seconds: float = float(os.getenv("SHOPFRONT_TOKEN_TIMEOUT", "5"))
    redis_url: str = os.getenv("SHOPFRONT_REDIS_URL", "redis://localhost:6379/0")
    # Rate limiting: sustained requests per minute and burst size per API key.
    rate_limit_per_minute: int = int(os.getenv("SHOPFRONT_RATE_LIMIT_PER_MINUTE", "600"))
    rate_limit_burst: int = int(os.getenv("SHOPFRONT_RATE_LIMIT_BURST", "50"))
    webhook_secret: str = os.getenv("SHOPFRONT_WEBHOOK_SECRET", "")
    webhook_tolerance_seconds: int = 300

    def validate(self) -> None:
        """Fail fast on dangerous values. Added after INC-2026-029 (burst set to 2 in prod)."""
        if self.rate_limit_burst < 10:
            raise ValueError("rate_limit_burst must be >= 10")
        if self.rate_limit_per_minute < self.rate_limit_burst:
            raise ValueError("rate_limit_per_minute must be >= rate_limit_burst")


settings = Settings()
