"""Vercel Python entrypoint. Vercel's zero-config FastAPI detection looks
for an ASGI `app` here; the real application lives in api/main.py."""

from api.main import app  # noqa: F401
