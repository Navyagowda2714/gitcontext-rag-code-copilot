# Shopfront API

Backend for the Shopfront checkout: authentication, billing, partner rate limiting and
payment-provider webhooks.

## Layout

- `shopfront/auth` — OAuth client, token value objects, `SessionManager` (token refresh)
- `shopfront/api` — request middleware: JWT auth and Redis-backed rate limiting
- `shopfront/billing` — invoices, proration, payment webhooks
- `shopfront/storage` — caching utilities
- `shopfront/workers` — background job helpers (retry with backoff)
- `web/` — TypeScript API client used by the storefront

## Running locally

```bash
docker compose up redis
export SHOPFRONT_REDIS_URL=redis://localhost:6379/0
python -m shopfront.server
```

## Owners

Auth is owned by the Identity team (Priya Raman). Billing by the Payments team (Marcus Chen).
Rate limiting and on-call runbooks by Platform (Elena Petrova).
