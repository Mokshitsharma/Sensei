# src/dl/train.py

from typing import Iterable, Optional
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split

from src.data.nifty50 import NIFTY_50
from src.training.pooled_data import build_pooled_dataset

from src.dl.dataset import TimeSeriesDataset
from src.dl.lstm import LSTMPricePredictor, train_lstm, save_model as save_lstm
from src.dl.temporal_cnn import TemporalCNN, save_model as save_tcn


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = ["rsi_norm", "ema_spread", "macd_diff", "atr_pct"]


def train_dl_models_pooled(
    tickers: Optional[Iterable[str]] = None,
    timeframe: str = "2y",
    seq_len: int = 30,
    batch_size: int = 64,
    epochs: int = 30,
    device: str = "cpu",
    model_name: str = "general",
) -> None:
    """Trains one LSTM and one Temporal CNN on data pooled across every
    given ticker (defaults to all NIFTY 50). Each ticker's rolling
    sequence windows are built independently (via its own
    TimeSeriesDataset) and only combined at the dataset level — a training
    sequence never spans two different stocks' price histories."""
    tickers = list(tickers or NIFTY_50.values())

    print(f"Building pooled dataset from {len(tickers)} tickers...")
    pairs = build_pooled_dataset(tickers, timeframe=timeframe, min_rows=seq_len + 10)
    if not pairs:
        raise RuntimeError("No usable ticker data — check network/yfinance access")

    per_ticker_datasets = [
        TimeSeriesDataset(
            df=df,
            feature_cols=FEATURE_COLS,
            target_col="future_return_5d",
            seq_len=seq_len,
        )
        for _, df in pairs
    ]
    dataset = ConcatDataset(per_ticker_datasets)
    print(f"Pooled {len(dataset)} sequences from {len(pairs)}/{len(tickers)} tickers")

    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    num_features = len(FEATURE_COLS)

    # -----------------------------
    # LSTM Training
    # -----------------------------
    print("Training LSTM...")
    lstm = LSTMPricePredictor(num_features=num_features)
    train_lstm(
        model=lstm,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        device=device,
    )

    lstm_path = MODEL_DIR / f"lstm_{model_name}.pt"
    save_lstm(lstm, str(lstm_path))
    print(f"LSTM saved to {lstm_path}")

    # -----------------------------
    # Temporal CNN Training
    # -----------------------------
    print("Training Temporal CNN...")
    tcn = TemporalCNN(num_features=num_features)
    tcn.to(device)

    optimizer = torch.optim.Adam(tcn.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    for epoch in range(1, epochs + 1):
        tcn.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            preds = tcn(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        tcn.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = tcn(x)
                val_loss += criterion(preds, y).item()

        print(
            f"Epoch {epoch:03d} | "
            f"Train MSE: {train_loss / len(train_loader):.6f} | "
            f"Val MSE: {val_loss / len(val_loader):.6f}"
        )

    tcn_path = MODEL_DIR / f"tcn_{model_name}.pt"
    save_tcn(tcn, str(tcn_path))
    print(f"TCN saved to {tcn_path}")


if __name__ == "__main__":
    train_dl_models_pooled(
        timeframe="2y",
        epochs=30,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
