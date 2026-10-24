# Authentication

## Token lifecycle

Users sign in through the identity provider with the authorization-code flow. The API receives a
short-lived access token (JWT, 15 minute lifetime) and a refresh token. Refresh tokens are
**rotating**: each successful refresh returns a new refresh token and invalidates the old one.

## Token refresh

Server-side callers use `SessionManager.get_access_token()`, which refreshes the access token
proactively once it is within 60 seconds (plus up to 15 seconds of random jitter) of expiry.
Refreshing ahead of time avoids the request failing with 401 in the first place.

Because refresh tokens are single-use, only one refresh may run at a time per session. The session
manager serialises refreshes with a lock and re-checks the expiry after acquiring it, so concurrent
callers reuse the first refresh instead of racing. The browser client in `web/src/apiClient.ts`
does the same with a shared in-flight promise.

If the identity provider answers `invalid_grant`, the refresh token was revoked or already used;
`ReauthRequired` is raised and the user must sign in again. Transient network errors are retried
up to 3 times with exponential backoff.

## Validating requests

`AuthMiddleware` verifies the RS256 signature and audience of every bearer token, with 30 seconds
of leeway for clock skew. Expired tokens get a 401 with `WWW-Authenticate: Bearer
error="invalid_token"` so clients know a refresh may help.

## History

Before March 2026 tokens were only refreshed reactively after a 401. That design caused the
token refresh storm in INC-2026-014; see PR #134 for the redesign.
