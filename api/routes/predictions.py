from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException

from api import accuracy, compute
from api.routes.meta import _MOVERS_UNIVERSE
from api.serializers import to_jsonable
from api.cache import ttl_cache

router = APIRouter()

_VALID_HORIZONS = {"1d", "3d", "7d", "30d"}


@router.get("/predictions/accuracy")
def get_predictions_accuracy(horizon: str | None = None, limit: int = 100):
    if horizon is not None and horizon not in accuracy.HORIZONS:
        raise HTTPException(status_code=400, detail=f"horizon must be one of {accuracy.HORIZONS}")
    return accuracy.get_accuracy(horizon=horizon, limit=limit)


@router.get("/predictions/accuracy/breakdown")
def get_predictions_accuracy_breakdown():
    return accuracy.get_accuracy_breakdown()


@ttl_cache(1800)
def _predictions_at_horizon(horizon: str) -> list[dict]:
    tickers = list(_MOVERS_UNIVERSE.values())
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda t: compute.prediction_at_horizon(t, horizon), tickers))
    return [to_jsonable(r) for r in results if r.get("current_price", 0) > 0]


@router.get("/predictions")
def get_predictions(horizon: str = "7d", direction: str = "ALL"):
    if horizon not in _VALID_HORIZONS:
        raise HTTPException(status_code=400, detail=f"horizon must be one of {sorted(_VALID_HORIZONS)}")
    if direction not in ("UP", "DOWN", "FLAT", "ALL"):
        raise HTTPException(status_code=400, detail="direction must be UP, DOWN, FLAT, or ALL")

    rows = _predictions_at_horizon(horizon)
    if direction != "ALL":
        rows = [r for r in rows if r["direction"] == direction]
    rows.sort(key=lambda r: r["expected_move_pct"], reverse=True)
    return rows
