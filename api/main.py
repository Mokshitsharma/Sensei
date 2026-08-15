import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.rate_limit import RateLimitMiddleware
from api.routes import meta, stock
from api import compute

logger = logging.getLogger("sensei.prewarm")

# Keep the stocks people actually land on (home page cards) hot in cache so
# clicking one never hits the ~15-20s cold ML pipeline. Re-warms just under
# the 300s cache TTL so they stay warm indefinitely while the server runs.
_PREWARM_INTERVAL_S = 240


def _prewarm_loop():
    while True:
        for name, ticker in meta.POPULAR:
            try:
                compute.decision(ticker, "1y")
                compute.support_resistance(ticker, "1y")
                compute.trade_setup(ticker, "1y", "swing")
            except Exception:
                logger.exception("Pre-warm failed for %s", ticker)
        time.sleep(_PREWARM_INTERVAL_S)


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_prewarm_loop, daemon=True).start()
    yield


app = FastAPI(title="Sensei AI API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    RateLimitMiddleware,
    default_limit=60,
    window_seconds=60,
    path_limits={"/analysis": 20},
)

# Set ALLOWED_ORIGINS (comma-separated) once the frontend has a real domain,
# e.g. "https://sensei-ai.vercel.app,https://senseiai.app". Defaults to "*"
# so local dev keeps working without any env setup.
_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allowed_origins == "*" else _allowed_origins.split(","),
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(meta.router, prefix="/api")
app.include_router(stock.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}
