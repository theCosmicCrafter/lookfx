"""Loopback-only guard for the local server: Host allow-list plus a per-launch token.

The app binds 127.0.0.1, but any web page the user has open can still reach it
(DNS rebinding defeats the same-origin policy; a plain ``fetch`` to a loopback port
works from any origin), and the API accepts arbitrary file paths. Two checks close
that hole:

* the ``Host`` header must name the loopback interface (``127.0.0.1``,
  ``localhost`` or ``[::1]``, with any port) — a rebinding attack arrives with
  the attacker's hostname;
* every ``/api/*`` request must carry the per-launch token, either as the
  HttpOnly ``lookfx_token`` cookie set when the launcher opens ``/?token=<t>``,
  or as an ``X-LookFX-Token`` header (CLI / scripts). Cross-site requests cannot
  read the token, and ``SameSite=Strict`` keeps the cookie off them.
"""

from __future__ import annotations

import secrets
from http.cookies import SimpleCookie
from urllib.parse import parse_qs

COOKIE = "lookfx_token"
HEADER = "x-lookfx-token"
ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})


def new_token() -> str:
    return secrets.token_urlsafe(32)


def host_allowed(host: str | None) -> bool:
    """True when the Host header names this machine's loopback interface."""
    if not host:
        return False
    host = host.strip().lower()
    if host.startswith("["):                    # bracketed IPv6 literal, optional :port
        host = host.split("]", 1)[0] + "]"
    else:
        host = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    return host in ALLOWED_HOSTS


def _header(scope, name: str) -> str | None:
    want = name.encode("latin-1")
    for k, v in scope.get("headers") or ():
        if k == want:
            return v.decode("latin-1")
    return None


def _cookie_token(scope) -> str | None:
    raw = _header(scope, "cookie")
    if not raw:
        return None
    jar = SimpleCookie()
    try:
        jar.load(raw)
    except Exception:  # noqa: BLE001 — malformed cookie header: treat as absent
        return None
    m = jar.get(COOKIE)
    return m.value if m else None


def _presented_token(scope) -> str | None:
    return _header(scope, HEADER) or _cookie_token(scope)


def _tokens_match(presented: str | None, expected: str) -> bool:
    return bool(presented) and secrets.compare_digest(presented.encode(), expected.encode())


async def _send_text(send, status: int, body: str, extra_headers=()) -> None:
    data = body.encode()
    headers = [(b"content-type", b"text/plain; charset=utf-8"),
               (b"content-length", str(len(data)).encode()),
               (b"cache-control", b"no-store")]
    headers.extend(extra_headers)
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": data})


class LocalOnlyMiddleware:
    """Pure ASGI middleware (streams pass through untouched, unlike BaseHTTPMiddleware)."""

    def __init__(self, app, token: str):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        if not host_allowed(_header(scope, "host")):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            else:
                await _send_text(send, 403, "lookfx: requests are only accepted from this machine (bad Host header)")
            return
        path = scope.get("path") or "/"
        if scope["type"] == "http" and path == "/" and scope.get("method") == "GET":
            raw_qs = (scope.get("query_string") or b"").decode("latin-1")
            qs = parse_qs(raw_qs, keep_blank_values=True)
            if "token" in qs:
                # the launcher lands here once: turn the URL token into an HttpOnly cookie
                if not _tokens_match(qs["token"][0], self.token):
                    await _send_text(send, 401, "lookfx: bad token")
                    return
                cookie = f"{COOKIE}={self.token}; Path=/; HttpOnly; SameSite=Strict"
                # keep the other params (?open=...) exactly as encoded, drop only the token
                rest = "&".join(p for p in raw_qs.split("&") if p.split("=", 1)[0] != "token")
                location = "/?" + rest if rest else "/"
                await _send_text(send, 303, "", [(b"location", location.encode("latin-1")),
                                                  (b"set-cookie", cookie.encode("latin-1"))])
                return
        if path.startswith("/api/") and not _tokens_match(_presented_token(scope), self.token):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            else:
                await _send_text(send, 401, "lookfx: missing or bad token (open the app through its launcher URL)")
            return
        await self.app(scope, receive, send)
