# Runbook: partner traffic hitting 429s

## Symptoms

Partners report `429 rate limit exceeded`; the `ratelimit_rejections_total` metric spikes for
a few API keys.

## Diagnosis

1. Check the current limits: `SHOPFRONT_RATE_LIMIT_PER_MINUTE` (default 600) and
   `SHOPFRONT_RATE_LIMIT_BURST` (default 50) in the deployment config.
2. Inspect a key's bucket in Redis: `HGETALL ratelimit:<api-key>` shows remaining tokens and the
   last refill timestamp.
3. Compare the partner's request rate with their contract tier.

## Mitigation

- For a single partner, raise their tier via the admin API; do not raise the global limit.
- If the global config is wrong, roll back the deploy (`make rollback ENV=production`).
- The limiter is shared across pods through Redis, so restarting pods does not reset budgets.
