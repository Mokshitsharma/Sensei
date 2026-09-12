import datetime as dt

from fastapi import APIRouter, Depends, HTTPException

from api import compute, db
from api.auth import get_current_user_id

router = APIRouter(prefix="/portfolio")

_STARTING_CASH = 100_000.0


def _ensure_account(user_id: str) -> None:
    with db.cursor() as cur:
        cur.execute("SELECT 1 FROM paper_accounts WHERE user_id = ?", (user_id,))
        if cur.fetchone() is None:
            cur.execute(
                "INSERT INTO paper_accounts (user_id, cash_balance, created_at) VALUES (?, ?, ?)",
                (user_id, _STARTING_CASH, dt.datetime.utcnow().isoformat()),
            )


@router.get("")
def get_portfolio(user_id: str = Depends(get_current_user_id)):
    _ensure_account(user_id)
    with db.cursor() as cur:
        cur.execute("SELECT cash_balance FROM paper_accounts WHERE user_id = ?", (user_id,))
        cash_balance = cur.fetchone()["cash_balance"]

        cur.execute(
            "SELECT ticker, quantity, avg_entry_price FROM paper_positions WHERE user_id = ? AND quantity > 0",
            (user_id,),
        )
        positions_rows = [dict(r) for r in cur.fetchall()]

        cur.execute(
            """SELECT COALESCE(SUM(CASE WHEN side = 'SELL' THEN quantity * price ELSE -quantity * price END), 0) AS realized
               FROM paper_trades WHERE user_id = ?""",
            (user_id,),
        )
        realized_pnl = cur.fetchone()["realized"]

    positions = []
    for pos in positions_rows:
        quote = compute.stock_quote(pos["ticker"])
        current_price = quote["value"] if quote else pos["avg_entry_price"]
        cost_basis = pos["quantity"] * pos["avg_entry_price"]
        market_value = pos["quantity"] * current_price
        unrealized_pnl = market_value - cost_basis
        positions.append({
            "ticker": pos["ticker"],
            "company": compute.company_for(pos["ticker"]),
            "quantity": pos["quantity"],
            "avg_entry_price": pos["avg_entry_price"],
            "current_price": current_price,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": (unrealized_pnl / cost_basis * 100) if cost_basis else 0.0,
        })

    return {
        "cash_balance": cash_balance,
        "positions": positions,
        "realized_pnl": realized_pnl,
    }


@router.get("/history")
def get_history(user_id: str = Depends(get_current_user_id)):
    with db.cursor() as cur:
        cur.execute(
            """SELECT id, ticker, side, quantity, price, executed_at
               FROM paper_trades WHERE user_id = ? ORDER BY executed_at DESC""",
            (user_id,),
        )
        return [dict(r) for r in cur.fetchall()]


@router.post("/trade")
def place_trade(body: dict, user_id: str = Depends(get_current_user_id)):
    ticker = body.get("ticker")
    side = body.get("side")
    quantity = body.get("quantity")

    if ticker not in compute.TICKER_TO_COMPANY:
        raise HTTPException(status_code=404, detail=f"Unknown ticker: {ticker}")
    if side not in ("BUY", "SELL"):
        raise HTTPException(status_code=400, detail="side must be BUY or SELL")
    if not isinstance(quantity, (int, float)) or quantity <= 0:
        raise HTTPException(status_code=400, detail="quantity must be a positive number")

    quote = compute.stock_quote(ticker)
    if not quote:
        raise HTTPException(status_code=502, detail="Live price unavailable, try again")
    price = quote["value"]

    _ensure_account(user_id)
    with db.cursor() as cur:
        cur.execute("SELECT cash_balance FROM paper_accounts WHERE user_id = ?", (user_id,))
        cash_balance = cur.fetchone()["cash_balance"]

        cur.execute(
            "SELECT quantity, avg_entry_price FROM paper_positions WHERE user_id = ? AND ticker = ?",
            (user_id, ticker),
        )
        pos = cur.fetchone()
        held_qty = pos["quantity"] if pos else 0.0
        avg_price = pos["avg_entry_price"] if pos else 0.0

        cost = price * quantity
        if side == "BUY":
            if cost > cash_balance:
                raise HTTPException(status_code=400, detail="Insufficient cash balance")
            new_qty = held_qty + quantity
            new_avg = ((held_qty * avg_price) + cost) / new_qty
            cash_balance -= cost
        else:
            if quantity > held_qty:
                raise HTTPException(status_code=400, detail="Insufficient shares held")
            new_qty = held_qty - quantity
            new_avg = avg_price if new_qty > 0 else 0.0
            cash_balance += cost

        cur.execute(
            "UPDATE paper_accounts SET cash_balance = ? WHERE user_id = ?",
            (cash_balance, user_id),
        )
        cur.execute(
            """INSERT INTO paper_positions (user_id, ticker, quantity, avg_entry_price)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(user_id, ticker) DO UPDATE SET quantity = excluded.quantity, avg_entry_price = excluded.avg_entry_price""",
            (user_id, ticker, new_qty, new_avg),
        )
        executed_at = dt.datetime.utcnow().isoformat()
        cur.execute(
            """INSERT INTO paper_trades (user_id, ticker, side, quantity, price, executed_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, ticker, side, quantity, price, executed_at),
        )
        trade_id = cur.lastrowid

    return {
        "id": trade_id, "ticker": ticker, "side": side,
        "quantity": quantity, "price": price, "executed_at": executed_at,
    }
