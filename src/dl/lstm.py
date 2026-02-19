# src/dl/lstm.py

import torch
import torch.nn as nn
from typing import Tuple


class LSTMPricePredictor(nn.Module):
    """
    LSTM-based time series model for predicting future returns.

    Task:
        Regression → predict next 5-day return

    Input shape:
        (batch_size, sequence_length, num_features)

    Output:
        (batch_size, 1)
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (B, T, F)

        Returns:
            Tensor of shape (B, 1)
        """
        _, (hidden, _) = self.lstm(x)

        last_hidden = hidden[-1]  # last layer hidden state
        out = self.regressor(last_hidden)

        return out


def train_lstm(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cpu",
) -> None:
    """
    Trains LSTM model using MSE loss.
    """

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                val_loss += criterion(preds, y).item()

        print(
            f"Epoch {epoch:03d} | "
            f"Train MSE: {train_loss / len(train_loader):.6f} | "
            f"Val MSE: {val_loss / len(val_loader):.6f}"
        )


def predict(
    model: nn.Module,
    sequence: torch.Tensor,
    device: str = "cpu",
) -> float:
    """
    Predicts next 5-day return from a single sequence.

    Args:
        sequence: Tensor (1, T, F)

    Returns:
        Predicted return (float)
    """
    model.eval()
    model.to(device)

    with torch.no_grad():
        pred = model(sequence.to(device))

    return float(pred.item())


def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_model(
    path: str,
    num_features: int,
    device: str = "cpu",
) -> LSTMPricePredictor:
    model = LSTMPricePredictor(num_features=num_features)
    import os
    if not os.path.exists(path) or os.path.getsize(path) == 0:
         raise ValueError(f"Invalid model file: {path}")
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model