// Browser/Node API client with transparent access-token refresh.
//
// On a 401 with `WWW-Authenticate: Bearer error="invalid_token"` the client refreshes once and
// replays the request. Concurrent requests that fail while a refresh is running wait on the same
// promise instead of starting their own refresh (refresh tokens are single-use).

export interface TokenStore {
  accessToken(): string | null;
  refreshToken(): string | null;
  save(access: string, refresh: string): void;
  clear(): void;
}

let inflightRefresh: Promise<string> | null = null;

export async function refreshAccessToken(store: TokenStore, tokenUrl: string): Promise<string> {
  if (inflightRefresh) {
    return inflightRefresh;
  }
  inflightRefresh = (async () => {
    const res = await fetch(tokenUrl, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ grant_type: "refresh_token", refresh_token: store.refreshToken() ?? "" }),
    });
    if (!res.ok) {
      store.clear();
      throw new Error(`refresh failed with status ${res.status}`);
    }
    const body = await res.json();
    store.save(body.access_token, body.refresh_token);
    return body.access_token as string;
  })();
  try {
    return await inflightRefresh;
  } finally {
    inflightRefresh = null;
  }
}

export async function apiFetch(
  store: TokenStore,
  url: string,
  init: RequestInit = {},
  tokenUrl = "/oauth/token",
): Promise<Response> {
  const withAuth = (token: string | null): RequestInit => ({
    ...init,
    headers: { ...(init.headers ?? {}), Authorization: `Bearer ${token ?? ""}` },
  });
  let res = await fetch(url, withAuth(store.accessToken()));
  if (res.status === 401 && res.headers.get("WWW-Authenticate")?.includes("invalid_token")) {
    const fresh = await refreshAccessToken(store, tokenUrl);
    res = await fetch(url, withAuth(fresh));
  }
  return res;
}
