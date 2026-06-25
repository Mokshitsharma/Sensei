# training/train_all.py

"""
Batch training orchestrator — trains all 500 stocks (or a filtered subset).

Usage:
    python -m training.train_all                     # all stocks
    python -m training.train_all --tier large        # large-cap only
    python -m training.train_all --sector "Banking & Finance"
    python -m training.train_all --limit 10          # first N stocks
    python -m training.train_all --skip-trained      # skip already trained
    python -m training.train_all --workers 4         # parallel workers
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.nifty500 import NIFTY_500
from training.registry import is_trained
from training.train_stock import train_stock

# Try to import tqdm; fall back to a no-op wrapper if unavailable
try:
    from tqdm import tqdm as _tqdm

    def _progress_bar(iterable, total, desc):
        return _tqdm(iterable, total=total, desc=desc, unit="stock", ncols=90)

except ImportError:
    def _progress_bar(iterable, total, desc):
        print(f"{desc} ({total} stocks)")
        return iterable


_REPORT_PATH = Path(__file__).parent / "training_report.json"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_stocks(
    tier: Optional[str],
    sector: Optional[str],
    limit: Optional[int],
    skip_trained: bool,
) -> List[Dict]:
    """
    Build the ordered list of stocks to train.

    Returns:
        List of ``{"name": str, "symbol": str, "tier": str, "sector": str}``.
    """
    candidates = []
    for name, info in NIFTY_500.items():
        if tier and info["tier"] != tier:
            continue
        if sector and info["sector"] != sector:
            continue
        candidates.append({
            "name": name,
            "symbol": info["symbol"],
            "tier": info["tier"],
            "sector": info["sector"],
        })

    if skip_trained:
        candidates = [c for c in candidates if not is_trained(c["symbol"])]

    if limit is not None:
        candidates = candidates[:limit]

    return candidates


def _train_one(stock: Dict, timeframe: str, force: bool) -> Dict:
    """
    Wrapper that calls :func:`train_stock` and attaches metadata.

    Returns:
        ``train_stock`` result dict augmented with ``name`` and ``sector``.
    """
    result = train_stock(
        symbol=stock["symbol"],
        timeframe=timeframe,
        force=force,
    )
    result["name"] = stock["name"]
    result["sector"] = stock["sector"]
    result["tier"] = stock["tier"]
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_all(
    tier: Optional[str] = None,
    sector: Optional[str] = None,
    limit: Optional[int] = None,
    skip_trained: bool = False,
    max_workers: int = 2,
    timeframe: str = "5y",
    force: bool = False,
) -> Dict:
    """
    Train models for all stocks matching the given filters.

    Args:
        tier:         Filter by market cap tier — ``"large"``, ``"mid"``, or ``"small"``.
        sector:       Filter by sector name (exact match, case-sensitive).
        limit:        Limit to the first *N* matching stocks.
        skip_trained: Skip symbols that already have all four models.
        max_workers:  Maximum parallel training threads.  Keep this low (1-2)
                      to avoid OOM on machines with limited RAM/VRAM.
        timeframe:    Historical period to download (passed to ``train_stock``).
        force:        Retrain even if models already exist.

    Returns:
        Summary dictionary::

            {
                "started_at": ISO timestamp,
                "finished_at": ISO timestamp,
                "elapsed_seconds": float,
                "total": int,
                "succeeded": int,
                "failed": int,
                "results": [ {per-stock result dict}, ... ],
            }
    """
    stocks = _select_stocks(tier, sector, limit, skip_trained)
    total = len(stocks)

    if total == 0:
        print("No stocks matched the given filters.")
        return {
            "started_at": datetime.utcnow().isoformat(),
            "finished_at": datetime.utcnow().isoformat(),
            "elapsed_seconds": 0.0,
            "total": 0,
            "succeeded": 0,
            "failed": 0,
            "results": [],
        }

    print(
        f"\nStarting batch training — {total} stocks | "
        f"workers={max_workers} | timeframe={timeframe}"
    )
    if tier:
        print(f"  Tier filter   : {tier}")
    if sector:
        print(f"  Sector filter : {sector}")
    if skip_trained:
        print("  Skipping already-trained symbols.")
    print()

    started_at = datetime.utcnow()
    t0 = time.monotonic()

    all_results: List[Dict] = []
    succeeded = 0
    failed = 0

    # Submit all jobs to the thread pool
    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for stock in stocks:
            future = executor.submit(_train_one, stock, timeframe, force)
            futures[future] = stock

        # Collect results with progress bar
        bar = _progress_bar(
            as_completed(futures),
            total=total,
            desc="Training",
        )

        for future in bar:
            stock = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                # Unexpected error outside train_stock (e.g. import issue)
                result = {
                    "symbol": stock["symbol"],
                    "name": stock["name"],
                    "sector": stock["sector"],
                    "tier": stock["tier"],
                    "success": False,
                    "models_trained": [],
                    "errors": {"fatal": str(exc)},
                    "output_dir": "",
                }

            all_results.append(result)

            if result.get("success"):
                succeeded += 1
            else:
                failed += 1

            # Update tqdm postfix if available
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(
                    ok=succeeded,
                    fail=failed,
                    sym=stock["symbol"],
                )

    elapsed = time.monotonic() - t0
    finished_at = datetime.utcnow()

    summary = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "results": all_results,
    }

    # --- Write training report ---
    try:
        _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_REPORT_PATH, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nTraining report written to: {_REPORT_PATH}")
    except OSError as exc:
        print(f"\nWarning: could not write training report — {exc}")

    # --- Print final summary ---
    print(
        f"\n{'='*60}\n"
        f"Batch training complete.\n"
        f"  Total    : {total}\n"
        f"  Succeeded: {succeeded}\n"
        f"  Failed   : {failed}\n"
        f"  Elapsed  : {elapsed:.1f}s\n"
        f"{'='*60}\n"
    )

    if failed > 0:
        print("Failed stocks:")
        for r in all_results:
            if not r.get("success"):
                errors = r.get("errors", {})
                print(f"  {r['symbol']} — {list(errors.keys())}")

    return summary


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-train models for all NIFTY 500 stocks (or a subset)."
    )
    parser.add_argument(
        "--tier",
        choices=["large", "mid", "small"],
        default=None,
        help="Train only stocks of this market cap tier.",
    )
    parser.add_argument(
        "--sector",
        default=None,
        help="Train only stocks in this sector (exact name match).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N matching stocks.",
    )
    parser.add_argument(
        "--skip-trained",
        action="store_true",
        help="Skip stocks that already have all four model files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        metavar="N",
        help="Maximum parallel training threads (default: 2).",
    )
    parser.add_argument(
        "--timeframe",
        default="5y",
        help="Historical data period per stock (default: 5y).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if models already exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary = train_all(
        tier=args.tier,
        sector=args.sector,
        limit=args.limit,
        skip_trained=args.skip_trained,
        max_workers=args.workers,
        timeframe=args.timeframe,
        force=args.force,
    )
    sys.exit(0 if summary["failed"] == 0 else 1)
