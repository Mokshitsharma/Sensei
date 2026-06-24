# training/train_stock.py

"""
Train all four model types for a single NSE stock symbol.

Models saved to:
    models/{SYMBOL_DIR}/lstm.pt
    models/{SYMBOL_DIR}/tcn.pt
    models/{SYMBOL_DIR}/ppo.zip
    models/{SYMBOL_DIR}/ml.joblib

Usage:
    python -m training.train_stock --symbol HDFCBANK.NS --timeframe 5y
    python -m training.train_stock --symbol RELIANCE.NS --force
"""

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, random_split
import joblib

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.prices import load_prices
from src.domain.indicators import add_indicators
from src.ml.features import build_features
from src.ml.model import build_model, train_model as fit_model
from src.dl.dataset import TimeSeriesDataset
from src.dl.lstm import LSTMPricePredictor, train_lstm
from src.dl.temporal_cnn import TemporalCNN
from src.rl.env import TradingEnv
from src.rl.agent import PPOTradingAgent
from training.registry import symbol_to_dir, is_trained, MODELS_DIR

# ---------------------------------------------------------------------------
# Feature columns used by DL and RL models
# ---------------------------------------------------------------------------
_FEATURE_COLS = [
    "rsi_norm",
    "ema_spread",
    "macd_diff",
    "atr_pct",
    "volatility_10",
    "return_1",
    "return_5",
    "range_position",
]

_ML_FEATURE_COLS = [
    "rsi_norm",
    "ema_spread",
    "macd_diff",
    "atr_pct",
    "volatility_10",
    "return_1",
    "return_5",
    "return_10",
    "range_position",
]


# ---------------------------------------------------------------------------
# Internal per-model training functions
# ---------------------------------------------------------------------------

def _train_ml(df, output_dir: Path) -> str:
    """
    Train a Random Forest regressor on ``future_return_5d``.

    Returns:
        Absolute path to the saved .joblib file.

    Raises:
        Exception on failure.
    """
    valid_cols = [c for c in _ML_FEATURE_COLS if c in df.columns]
    target = "future_return_5d"

    data = df[valid_cols + [target]].dropna()
    X = data[valid_cols].values
    y = data[target].values

    model = build_model("random_forest", task="regression")
    fit_model(model, X, y)

    path = output_dir / "ml.joblib"
    joblib.dump(model, str(path))
    return str(path)


def _train_lstm(df, output_dir: Path, device: str = "cpu") -> str:
    """
    Train the LSTM model on the prepared feature dataframe.

    Returns:
        Absolute path to the saved .pt file.
    """
    valid_cols = [c for c in _FEATURE_COLS if c in df.columns]
    dataset = TimeSeriesDataset(
        df=df.dropna().reset_index(drop=True),
        feature_cols=valid_cols,
        target_col="future_return_5d",
        seq_len=30,
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = LSTMPricePredictor(num_features=len(valid_cols))
    train_lstm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        device=device,
    )

    path = output_dir / "lstm.pt"
    torch.save(model.state_dict(), str(path))
    return str(path)


def _train_tcn(df, output_dir: Path, device: str = "cpu") -> str:
    """
    Train the Temporal CNN model.

    Returns:
        Absolute path to the saved .pt file.
    """
    valid_cols = [c for c in _FEATURE_COLS if c in df.columns]
    dataset = TimeSeriesDataset(
        df=df.dropna().reset_index(drop=True),
        feature_cols=valid_cols,
        target_col="future_return_5d",
        seq_len=30,
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = TemporalCNN(num_features=len(valid_cols))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    for epoch in range(1, 31):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    path = output_dir / "tcn.pt"
    torch.save(model.state_dict(), str(path))
    return str(path)


def _train_ppo(df, output_dir: Path) -> str:
    """
    Train the PPO RL agent.

    Returns:
        Absolute path to the saved .zip file.
    """
    valid_cols = [c for c in _FEATURE_COLS if c in df.columns]
    clean_df = df.dropna(subset=valid_cols).reset_index(drop=True)

    env = TradingEnv(df=clean_df, feature_cols=valid_cols)
    agent = PPOTradingAgent(env=env)
    agent.train(timesteps=50_000)

    path = output_dir / "ppo.zip"
    agent.save(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_stock(
    symbol: str,
    timeframe: str = "5y",
    force: bool = False,
) -> Dict:
    """
    Train all four model types for a given NSE stock symbol.

    Steps:
        1. Fetch historical price data.
        2. Compute technical indicators.
        3. Build ML/DL feature set.
        4. Train ML (Random Forest).
        5. Train LSTM.
        6. Train Temporal CNN.
        7. Train PPO RL agent.

    Each model is trained independently — a failure in one does not abort others.

    Args:
        symbol:    Yahoo Finance ticker (e.g. ``"HDFCBANK.NS"``).
        timeframe: Historical period to download (e.g. ``"5y"``, ``"2y"``).
        force:     If ``True``, retrain even if models already exist.

    Returns:
        Dictionary::

            {
                "symbol": str,
                "success": bool,          # True if at least one model trained
                "models_trained": [...],  # names of successfully trained models
                "errors": {name: str},    # errors per model
                "output_dir": str,
            }
    """
    result: Dict = {
        "symbol": symbol,
        "success": False,
        "models_trained": [],
        "errors": {},
        "output_dir": "",
    }

    # --- Check if already trained ---
    if not force and is_trained(symbol):
        print(f"[{symbol}] Models already exist. Use --force to retrain.")
        result["success"] = True
        result["output_dir"] = str(MODELS_DIR / symbol_to_dir(symbol))
        return result

    # --- Create output directory ---
    output_dir = MODELS_DIR / symbol_to_dir(symbol)
    output_dir.mkdir(parents=True, exist_ok=True)
    result["output_dir"] = str(output_dir)

    # --- Fetch and prepare data ---
    print(f"[{symbol}] Fetching price data ({timeframe})...")
    try:
        df = load_prices(symbol, timeframe)
        df = add_indicators(df)
        df = build_features(df)
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        print(f"[{symbol}] Dataset: {len(df)} rows, {len(df.columns)} columns.")
    except Exception as exc:
        error_msg = f"Data loading failed: {exc}"
        print(f"[{symbol}] ERROR — {error_msg}")
        result["errors"]["data"] = error_msg
        return result

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- ML model ---
    print(f"[{symbol}] Training Random Forest ML model...")
    try:
        path = _train_ml(df, output_dir)
        result["models_trained"].append("ml")
        print(f"[{symbol}] ML saved: {path}")
    except Exception as exc:
        msg = traceback.format_exc()
        result["errors"]["ml"] = str(exc)
        print(f"[{symbol}] ML training failed: {exc}")

    # --- LSTM ---
    print(f"[{symbol}] Training LSTM model...")
    try:
        path = _train_lstm(df, output_dir, device=device)
        result["models_trained"].append("lstm")
        print(f"[{symbol}] LSTM saved: {path}")
    except Exception as exc:
        result["errors"]["lstm"] = str(exc)
        print(f"[{symbol}] LSTM training failed: {exc}")

    # --- TCN ---
    print(f"[{symbol}] Training Temporal CNN model...")
    try:
        path = _train_tcn(df, output_dir, device=device)
        result["models_trained"].append("tcn")
        print(f"[{symbol}] TCN saved: {path}")
    except Exception as exc:
        result["errors"]["tcn"] = str(exc)
        print(f"[{symbol}] TCN training failed: {exc}")

    # --- PPO ---
    print(f"[{symbol}] Training PPO RL agent...")
    try:
        path = _train_ppo(df, output_dir)
        result["models_trained"].append("ppo")
        print(f"[{symbol}] PPO saved: {path}")
    except Exception as exc:
        result["errors"]["ppo"] = str(exc)
        print(f"[{symbol}] PPO training failed: {exc}")

    result["success"] = len(result["models_trained"]) > 0
    trained_count = len(result["models_trained"])
    error_count = len(result["errors"])
    print(
        f"[{symbol}] Done — {trained_count}/4 models trained, "
        f"{error_count} error(s)."
    )
    return result


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train all 4 model types for a single NSE stock."
    )
    parser.add_argument(
        "--symbol",
        required=True,
        help="Yahoo Finance ticker, e.g. HDFCBANK.NS",
    )
    parser.add_argument(
        "--timeframe",
        default="5y",
        help="Historical period to download (default: 5y)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if models already exist",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    outcome = train_stock(
        symbol=args.symbol,
        timeframe=args.timeframe,
        force=args.force,
    )
    import json
    print(json.dumps(outcome, indent=2))
    sys.exit(0 if outcome["success"] else 1)
