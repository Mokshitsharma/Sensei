# src/dl/train.py

import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from src.data.prices import load_prices
from src.domain.indicators import add_indicators
from src.ml.features import build_features

from src.dl.dataset import TimeSeriesDataset
from src.dl.lstm import LSTMPricePredictor, train_lstm, save_model as save_lstm
from src.dl.temporal_cnn import TemporalCNN, save_model as save_tcn


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def train_dl_models(
    ticker: str,
    timeframe: str = "5y",
    seq_len: int = 30,
    batch_size: int = 64,
    epochs: int = 30,
    device: str = "cpu",
) -> None:
    """
    Train LSTM and Temporal CNN models for 5-day return prediction.
    """

    # -----------------------------
    # Load & prepare data
    # -----------------------------
    df = load_prices(ticker, timeframe)
    df = add_indicators(df)
    df = build_features(df)
    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "rsi_norm",
        "ema_spread",
        "macd_diff",
        "atr_pct",
    ]

    dataset = TimeSeriesDataset(
        df=df,
        feature_cols=feature_cols,
        target_col="future_return_5d",
        seq_len=seq_len,
    )

    # -----------------------------
    # Train / Validation split
    # -----------------------------
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    num_features = len(feature_cols)

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

    lstm_path = MODEL_DIR / f"lstm_{ticker.replace('.', '_')}.pt"
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

    tcn_path = MODEL_DIR / f"tcn_{ticker.replace('.', '_')}.pt"
    save_tcn(tcn, str(tcn_path))
    print(f"TCN saved to {tcn_path}")


if __name__ == "__main__":
    train_dl_models(
        ticker="HDFCBANK.NS",
        timeframe="5y",
        epochs=40,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )