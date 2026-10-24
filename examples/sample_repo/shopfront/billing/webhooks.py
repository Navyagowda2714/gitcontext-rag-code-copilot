"""Payment-provider webhooks: signature verification and idempotent event handling."""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Protocol

from shopfront.billing.invoices import build_invoice
from shopfront.config import settings


class SignatureError(Exception):
    pass


class IdempotencyStore(Protocol):
    def add_if_absent(self, key: str, ttl_seconds: int) -> bool: ...


def verify_signature(payload: bytes, header: str, secret: str, now: float | None = None) -> None:
    """Verify ``t=<ts>,v1=<hex hmac>`` signatures and reject replays older than 5 minutes.

    The signed message is ``f"{t}.{payload}"`` and the comparison is constant-time.
    """
    parts = dict(p.split("=", 1) for p in header.split(",") if "=" in p)
    try:
        ts = int(parts["t"])
        sig = parts["v1"]
    except (KeyError, ValueError) as exc:
        raise SignatureError("malformed signature header") from exc
    now = time.time() if now is None else now
    if abs(now - ts) > settings.webhook_tolerance_seconds:
        raise SignatureError("timestamp outside tolerance; possible replay")
    expected = hmac.new(secret.encode(), f"{ts}.".encode() + payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        raise SignatureError("signature mismatch")


class WebhookHandler:
    """Process provider events exactly once.

    The provider delivers *at least once* and retries any delivery that is not acknowledged within
    10 seconds. Every event id is recorded in the idempotency store before side effects run, so a
    retried ``invoice.upcoming`` event cannot create a second invoice (INC-2026-022, PR #152).
    """

    DEDUP_TTL_SECONDS = 7 * 24 * 3600

    def __init__(self, store: IdempotencyStore, invoice_repo) -> None:
        self._store = store
        self._invoices = invoice_repo

    def handle(self, event: dict) -> str:
        event_id = event["id"]
        if not self._store.add_if_absent(f"webhook:{event_id}", self.DEDUP_TTL_SECONDS):
            return "duplicate"
        if event["type"] == "invoice.upcoming":
            data = event["data"]
            invoice = build_invoice(
                data["customer_id"],
                data["plan_price"],
                data["period_start"],
                data["period_end"],
                source_event_id=event_id,
            )
            self._invoices.save(invoice)
            return "invoiced"
        return "ignored"
