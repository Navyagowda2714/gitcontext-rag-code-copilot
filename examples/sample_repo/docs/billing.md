# Billing

## Invoices

Invoices are created from the payment provider's `invoice.upcoming` webhook. `build_invoice`
creates one line item per subscription and prorates it when the plan changed mid-cycle.

## Proration and rounding

All money math uses `Decimal`. Prorated amounts are rounded to the cent with banker's rounding
(`ROUND_HALF_EVEN`) and VAT is applied once on the subtotal. Never use floats for money.

## Webhooks

Webhooks are signed with HMAC-SHA256 over `"{timestamp}.{payload}"`. `verify_signature` rejects
deliveries whose timestamp is more than 5 minutes old to block replays.

The provider delivers events at least once and retries anything not acknowledged within 10
seconds, so handlers must be idempotent. `WebhookHandler` records each event id in Redis
(`SET NX` with a 7 day TTL) before doing any work and returns `duplicate` for repeats.
