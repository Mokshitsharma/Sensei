import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import datetime as dt
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.rate_limit import RateLimitMiddleware
from api.routes import meta, stock, screener, predictions, portfolio
from api import accuracy, compute, db

logger = logging.getLogger("sensei.prewarm")

_IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

# Keep the stocks people actually land on (home page cards) hot in cache so
# clicking one never hits the ~15-20s cold ML pipeline. Re-warms just under
# the 300s cache TTL so they stay warm indefinitely while the server runs.
_PREWARM_INTERVAL_S = 240

# The gainers/losers + market-cap sections use a 30-min-cached heuristic
# forecast (compute.prediction_7d) across ~45 curated tickers — re-warm
# just under that TTL, in parallel, so /movers and /by-cap rarely hit cold.
_PREDICTION_PREWARM_INTERVAL_S = 1500


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


def _prediction_prewarm_loop():
    while True:
        tickers = list(meta._MOVERS_UNIVERSE.values())
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(compute.prediction_7d, tickers))
        time.sleep(_PREDICTION_PREWARM_INTERVAL_S)


def _seconds_until_next_ist_920() -> float:
    now = dt.datetime.now(_IST)
    target = now.replace(hour=9, minute=20, second=0, microsecond=0)
    if target <= now:
        target += dt.timedelta(days=1)
    return (target - now).total_seconds()


def _accuracy_agent_tick():
    """One agent cycle: score any predictions whose target date has
    arrived (checked first, so a prediction never gets evaluated on the
    same run it's created), then store today's new predictions."""
    try:
        n = accuracy.evaluate_due_predictions()
        if n:
            logger.info("Accuracy agent: evaluated %d due predictions", n)
    except Exception:
        logger.exception("Accuracy evaluation failed")
    try:
        accuracy.snapshot_predictions()
    except Exception:
        logger.exception("Accuracy snapshot failed")


def _accuracy_snapshot_loop():
    # Cheap (news-heuristic only, no torch/sklearn) — safe to run
    # unconditionally, unlike the heavier prewarm loops above. Runs once
    # immediately so local dev has data without waiting for a real day
    # boundary, then daily at 9:20 IST (just after market open) — daily is
    # the right cadence since prices here are daily-close granularity;
    # checking more often wouldn't see a new price anyway.
    _accuracy_agent_tick()
    while True:
        time.sleep(_seconds_until_next_ist_920())
        _accuracy_agent_tick()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()

    # Prewarm disabled for local dev — the concurrent startup burst (full ML
    # pipeline across ~50+ tickers) was getting the process hard-killed by a
    # resource limit in this environment within ~15s of boot. Re-enable
    # before deploying.
    if os.environ.get("SENSEI_ENABLE_PREWARM") == "1":
        threading.Thread(target=_prewarm_loop, daemon=True).start()
        threading.Thread(target=_prediction_prewarm_loop, daemon=True).start()

    threading.Thread(target=_accuracy_snapshot_loop, daemon=True).start()
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
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(meta.router, prefix="/api")
app.include_router(stock.router, prefix="/api")
app.include_router(screener.router, prefix="/api")
app.include_router(predictions.router, prefix="/api")
app.include_router(portfolio.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}
