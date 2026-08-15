"""Tiny in-process TTL cache — same role as st.cache_data played in app.py,
so repeated requests for the same ticker don't re-run the ML pipeline or
reload models from disk on every call."""

import functools
import time


def ttl_cache(ttl_seconds: int, should_cache=lambda value: True):
    """should_cache lets a call opt out of being cached — e.g. a transient
    upstream failure that returned empty/zeroed data shouldn't get stuck
    for the full TTL; better to let the next request retry."""

    def decorator(func):
        store: dict = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            cached = store.get(key)
            if cached is not None:
                value, expires_at = cached
                if now < expires_at:
                    return value
            value = func(*args, **kwargs)
            if should_cache(value):
                store[key] = (value, now + ttl_seconds)
            return value

        return wrapper

    return decorator
