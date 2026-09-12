"""Prediction-accuracy agent: for a curated ticker universe and a set of
tracked horizons, stores today's price + the AI's predicted price, then —
once each prediction's target date actually arrives — fetches the real
price at that time and scores whether the call was directionally correct.
Nothing is ever evaluated early: a 7-day call is only checked 7 days out,
not the next morning."""

import datetime as dt
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from api import compute, db
from api.routes.meta import _MOVERS_UNIVERSE

logger = logging.getLogger("sensei.accuracy")

# Horizons tracked by the agent — mirrors the AI Prediction tab's options.
HORIZONS = ["1d", "7d", "30d"]
_HORIZON_DAYS = {"1d": 1, "7d": 7, "30d": 30}

# Must match the FLAT threshold in src/domain/news_price_model.py — a
# prediction called FLAT there means "expected move under 0.3%", so the
# actual outcome needs the same band to be judged fairly. Without this,
# "actual == FLAT" required an exact 0.0% move (never happens in practice),
# so every FLAT call was scored wrong regardless of how right it was.
_FLAT_THRESHOLD_PCT = 0.3


def _today_iso() -> str:
    return dt.date.today().isoformat()


def _target_date(horizon: str) -> str:
    return (dt.date.today() + dt.timedelta(days=_HORIZON_DAYS[horizon])).isoformat()


def snapshot_predictions() -> int:
    """Stores today's price + predicted price for every ticker, at every
    tracked horizon. No-op per horizon if today's already logged for it."""
    today = _today_iso()
    tickers = list(_MOVERS_UNIVERSE.values())
    written = 0

    for horizon in HORIZONS:
        with db.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM prediction_log WHERE predicted_at = ? AND horizon = ? LIMIT 1",
                (today, horizon),
            )
            if cur.fetchone():
                continue

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda t: compute.prediction_at_horizon(t, horizon), tickers))

        target_date = _target_date(horizon)
        rows = [
            (
                r["ticker"], r["company"], horizon, today, target_date,
                r["current_price"], r["predicted_price"], r["direction"], r["confidence"],
            )
            for r in results
            if r.get("current_price", 0) > 0
        ]
        if not rows:
            continue

        with db.cursor() as cur:
            cur.executemany(
                """INSERT INTO prediction_log
                   (ticker, company, horizon, predicted_at, target_date,
                    price_at_prediction, predicted_price, direction, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        written += len(rows)
        logger.info("Snapshotted %d %s predictions for %s (target %s)", len(rows), horizon, today, target_date)

    return written


def evaluate_due_predictions() -> int:
    """The other half of the agent: finds every stored prediction whose
    target date has arrived and hasn't been scored yet, fetches the real
    price now, and records whether the call was correct."""
    today = _today_iso()
    with db.cursor() as cur:
        cur.execute(
            """SELECT id, ticker, price_at_prediction, direction
               FROM prediction_log
               WHERE target_date <= ? AND evaluated_at IS NULL""",
            (today,),
        )
        due = [dict(r) for r in cur.fetchall()]

    if not due:
        return 0

    updates = []
    for entry in due:
        quote = compute.stock_quote(entry["ticker"])
        if not quote:
            continue
        actual_price = quote["value"]
        actual_move_pct = (
            (actual_price - entry["price_at_prediction"]) / entry["price_at_prediction"] * 100
            if entry["price_at_prediction"]
            else 0.0
        )
        if abs(actual_move_pct) < _FLAT_THRESHOLD_PCT:
            actual_direction = "FLAT"
        elif actual_move_pct > 0:
            actual_direction = "UP"
        else:
            actual_direction = "DOWN"
        correct = actual_direction == entry["direction"]
        updates.append((actual_price, actual_move_pct, int(correct), today, entry["id"]))

    with db.cursor() as cur:
        cur.executemany(
            """UPDATE prediction_log
               SET actual_price = ?, actual_move_pct = ?, correct = ?, evaluated_at = ?
               WHERE id = ?""",
            updates,
        )
    logger.info("Evaluated %d due predictions", len(updates))
    return len(updates)


def get_accuracy(horizon: Optional[str] = None, limit: int = 100) -> dict:
    """Track record across every evaluated prediction (optionally scoped
    to one horizon) — the score only ever reflects calls that have
    actually reached their target date."""
    where = "WHERE evaluated_at IS NOT NULL"
    params: tuple = ()
    if horizon:
        where += " AND horizon = ?"
        params = (horizon,)

    with db.cursor() as cur:
        cur.execute(f"SELECT COUNT(*), SUM(correct) FROM prediction_log {where}", params)
        total, correct = cur.fetchone()
        total = total or 0
        correct = correct or 0

        cur.execute(
            f"""SELECT ticker, company, horizon, predicted_at, target_date,
                       price_at_prediction, predicted_price, direction,
                       actual_price, actual_move_pct, correct
                FROM prediction_log {where}
                ORDER BY evaluated_at DESC, target_date DESC
                LIMIT ?""",
            params + (limit,),
        )
        rows = [dict(r) for r in cur.fetchall()]

    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM prediction_log WHERE evaluated_at IS NULL")
        pending = cur.fetchone()[0]

    return {
        "total": total,
        "correct": correct,
        "pct_correct": (correct / total * 100) if total else None,
        "pending": pending,
        "rows": rows,
    }


def get_accuracy_breakdown() -> dict:
    """Aggregate view for presenting the track record honestly: accuracy
    per horizon (each with its own sample size, since a small-sample
    horizon shouldn't be read the same as a well-sampled one), plus a
    cumulative accuracy-over-time series so the trend — not a single
    point-in-time number — is what gets shown."""
    by_horizon = []
    for horizon in HORIZONS:
        with db.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*), SUM(correct) FROM prediction_log WHERE evaluated_at IS NOT NULL AND horizon = ?",
                (horizon,),
            )
            total, correct = cur.fetchone()
            total = total or 0
            correct = correct or 0
            cur.execute(
                "SELECT COUNT(*) FROM prediction_log WHERE evaluated_at IS NULL AND horizon = ?",
                (horizon,),
            )
            pending = cur.fetchone()[0]
        by_horizon.append({
            "horizon": horizon,
            "total": total,
            "correct": correct,
            "pct_correct": (correct / total * 100) if total else None,
            "pending": pending,
        })

    with db.cursor() as cur:
        cur.execute(
            """SELECT evaluated_at, COUNT(*) as n, SUM(correct) as correct
               FROM prediction_log
               WHERE evaluated_at IS NOT NULL
               GROUP BY evaluated_at
               ORDER BY evaluated_at ASC"""
        )
        daily = [dict(r) for r in cur.fetchall()]

    trend = []
    cum_total = 0
    cum_correct = 0
    for day in daily:
        cum_total += day["n"]
        cum_correct += day["correct"] or 0
        trend.append({
            "date": day["evaluated_at"],
            "day_total": day["n"],
            "day_correct": day["correct"] or 0,
            "cumulative_total": cum_total,
            "cumulative_correct": cum_correct,
            "cumulative_pct_correct": round(cum_correct / cum_total * 100, 1) if cum_total else None,
        })

    return {"by_horizon": by_horizon, "trend": trend}
