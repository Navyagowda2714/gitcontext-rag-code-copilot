---
id: INC-2026-014
title: Token refresh storm logged out 18% of active users
severity: SEV-2
date: 2026-03-14
authors: [dana-whitfield, priya-raman]
services: [shopfront-api, identity]
related_prs: [134]
---

# Postmortem: token refresh storm

## Summary

For 47 minutes on 14 March 2026, about 18% of active storefront and partner sessions were logged
out and had to sign in again. The identity provider's token endpoint received roughly 40x its normal
traffic.

## Impact

- 18% of active sessions forcibly signed out; checkout conversion dropped 6% during the window.
- Two partners saw bursts of 401 errors and paused their order sync jobs.

## Root cause

`SessionManager` refreshed access tokens reactively: only after the API returned 401. A popular
partner token expired while hundreds of requests were in flight, so each request got a 401 and
started its own refresh. Refresh tokens rotate on every use, so the first refresh succeeded and
every concurrent refresh presented an already-used refresh token, received `invalid_grant`, and
the session was destroyed. Clock skew of up to 23 seconds between pods made tokens look valid on
some pods and expired on others, which widened the race.

## Timeline (UTC)

- 13:02 Error rate on `/oauth/token` alerts; invalid_grant rate at 35%.
- 13:15 On-call identifies concurrent refreshes for the same session in logs.
- 13:31 Mitigation: raised access-token lifetime from 15 to 30 minutes to reduce refresh volume.
- 13:49 Error rates back to baseline.

## What went well

Session revocation logs made the concurrent refresh pattern visible within 15 minutes.

## Action items

- Refresh tokens proactively before expiry with a skew buffer and jitter (PR #134, done).
- Serialise refreshes per session with a single-flight lock (PR #134, done).
- Alert when the invalid_grant rate exceeds 2% for 5 minutes (done).
- Run NTP drift checks on all nodes (done).
- Restore the 15 minute access-token lifetime after PR #134 shipped (done).
