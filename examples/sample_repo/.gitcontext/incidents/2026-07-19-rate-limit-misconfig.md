---
id: INC-2026-029
title: Partner traffic throttled after rate-limit burst misconfiguration
severity: SEV-3
date: 2026-07-19
authors: [elena-petrova]
services: [shopfront-api]
related_prs: []
---

# Postmortem: rate-limit burst misconfiguration

## Summary

For 22 minutes the largest partner received 429 responses for roughly 70% of requests.

## Root cause

A config change meant to lower the burst for anonymous traffic set `SHOPFRONT_RATE_LIMIT_BURST=2`
globally. With a bucket of 2 tokens, any partner sending requests in parallel was throttled even
though their per-minute rate was well under the limit. Nothing validated the value at start-up.

## Resolution

Rolled back the deploy with `make rollback ENV=production`.

## Action items

- Validate rate-limit settings at boot: burst must be at least 10 and not exceed the per-minute
  rate (`Settings.validate`, done).
- Support per-key overrides so anonymous limits can be tuned separately (planned).
