"""Minimal per-IP rate limiting. No external dependency — a fixed-window
counter is precise enough here; this exists to blunt accidental/abusive
hammering of the ML pipeline, not to be a precision traffic-shaping tool."""

import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, default_limit: int = 60, window_seconds: int = 60, path_limits: dict | None = None):
        super().__init__(app)
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        # path_limits: {"/analysis": 20} — matched by suffix, so all tickers
        # on the same endpoint type share one counter per IP (not per ticker).
        self.path_limits = path_limits or {}
        self._hits: dict[tuple[str, str], list[float]] = defaultdict(list)

    def _route_key(self, path: str) -> tuple[str, int]:
        for suffix, limit in self.path_limits.items():
            if path.endswith(suffix):
                return suffix, limit
        return "__default__", self.default_limit

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        route_key, limit = self._route_key(request.url.path)
        now = time.time()

        key = (client_ip, route_key)
        window_start = now - self.window_seconds
        hits = [t for t in self._hits.get(key, []) if t > window_start]

        if len(hits) >= limit:
            self._hits[key] = hits
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please slow down."},
            )

        hits.append(now)
        self._hits[key] = hits

        return await call_next(request)
