"""SQLite persistence for the prediction-accuracy tracker and paper trading.
A single file on disk is fine here — this runs as one long-lived uvicorn
process (see Dockerfile), not something serverless/multi-instance."""

import os
import sqlite3
from contextlib import contextmanager

_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sensei.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    company TEXT NOT NULL,
    horizon TEXT NOT NULL DEFAULT '7d',
    predicted_at TEXT NOT NULL,
    target_date TEXT NOT NULL DEFAULT '',
    price_at_prediction REAL NOT NULL,
    predicted_price REAL NOT NULL,
    direction TEXT NOT NULL,
    confidence TEXT NOT NULL,
    evaluated_at TEXT,
    actual_price REAL,
    actual_move_pct REAL,
    correct INTEGER
);
CREATE INDEX IF NOT EXISTS idx_prediction_log_date ON prediction_log(predicted_at);

CREATE TABLE IF NOT EXISTS paper_accounts (
    user_id TEXT PRIMARY KEY,
    cash_balance REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_positions (
    user_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    quantity REAL NOT NULL,
    avg_entry_price REAL NOT NULL,
    PRIMARY KEY (user_id, ticker)
);

CREATE TABLE IF NOT EXISTS paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    executed_at TEXT NOT NULL
);
"""

_connection: sqlite3.Connection | None = None


def _get_connection() -> sqlite3.Connection:
    global _connection
    if _connection is None:
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        _connection = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _connection.execute("PRAGMA journal_mode=WAL")
        _connection.row_factory = sqlite3.Row
    return _connection


def init_db() -> None:
    conn = _get_connection()
    conn.executescript(_SCHEMA)
    _migrate_prediction_log(conn)
    conn.commit()


def _migrate_prediction_log(conn: sqlite3.Connection) -> None:
    """Self-healing schema upgrade for prediction_log — adds the
    horizon/target-date/evaluation columns to a database created by an
    older version of this app, backfilling sane values for existing rows
    (they were all implicitly 7-day predictions before this column
    existed)."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(prediction_log)")}
    if "horizon" not in cols:
        conn.execute("ALTER TABLE prediction_log ADD COLUMN horizon TEXT NOT NULL DEFAULT '7d'")
    if "target_date" not in cols:
        conn.execute("ALTER TABLE prediction_log ADD COLUMN target_date TEXT NOT NULL DEFAULT ''")
        conn.execute(
            "UPDATE prediction_log SET target_date = date(predicted_at, '+7 days') WHERE target_date = ''"
        )
    if "evaluated_at" not in cols:
        conn.execute("ALTER TABLE prediction_log ADD COLUMN evaluated_at TEXT")
    if "actual_price" not in cols:
        conn.execute("ALTER TABLE prediction_log ADD COLUMN actual_price REAL")
    if "actual_move_pct" not in cols:
        conn.execute("ALTER TABLE prediction_log ADD COLUMN actual_move_pct REAL")
    if "correct" not in cols:
        conn.execute("ALTER TABLE prediction_log ADD COLUMN correct INTEGER")

    # Index depends on target_date, which may have just been added above —
    # must run after the ALTER TABLEs, not in the initial CREATE-TABLE pass.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prediction_log_target ON prediction_log(target_date, evaluated_at)"
    )


@contextmanager
def cursor():
    """Yields a cursor and commits on clean exit. Caller owns the SQL."""
    conn = _get_connection()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
