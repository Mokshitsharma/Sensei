"""Clerk session-JWT verification for the paper-trading routes — the only
part of the API that needs a real user identity. Nothing else in the
backend requires auth (see api/main.py's CORS/rate-limit setup, which is
otherwise fully public/GET).

Verifies the RS256 JWT Clerk issues (sent by the frontend as
`Authorization: Bearer <token>`, from `useAuth().getToken()` /
`auth().getToken()`) against Clerk's own JWKS — no shared secret needed.
"""

import base64
import os

import jwt
from fastapi import Header, HTTPException
from jwt import PyJWKClient

from api.cache import ttl_cache


def _default_jwks_url() -> str:
    # Fallback derived from the dev Clerk publishable key already checked
    # into frontend/.env.local — publishable keys are not secret. Set
    # CLERK_JWKS_URL explicitly once you claim your own Clerk app
    # (see PLAN_2026-08-16.md) to override this.
    pk = os.environ.get("CLERK_PUBLISHABLE_KEY", "pk_test_ZW5hYmxpbmctZmlyZWZseS0xNi5jbGVyay5hY2NvdW50cy5kZXYk")
    encoded = pk.split("_", 2)[-1]
    domain = base64.b64decode(encoded + "=" * (-len(encoded) % 4)).decode().rstrip("$")
    return f"https://{domain}/.well-known/jwks.json"


_JWKS_URL = os.environ.get("CLERK_JWKS_URL", _default_jwks_url())


@ttl_cache(3600)
def _jwks_client() -> PyJWKClient:
    return PyJWKClient(_JWKS_URL)


def get_current_user_id(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()

    try:
        signing_key = _jwks_client().get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing subject")
    return user_id
