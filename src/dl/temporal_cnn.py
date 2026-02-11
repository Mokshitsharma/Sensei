# src/dl/temporal_cnn.py

import torch
import torch.nn as nn
from typing import List


class Chomp1d(nn.Module):
    """
    Removes extra padding to ensure causality.
    """

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Single TCN residual block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalCNN(nn.Module):
    """
    Temporal Convolutional Network for return prediction.

    Input:
        (batch, seq_len, num_features)

    Output:
        (batch, 1)
    """

    def __init__(
        self,
        num_features: int,
        channels: List[int] = [32, 32, 64],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        layers = []
        in_ch = num_features

        for i, out_ch in enumerate(channels):
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
            )
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)

        self.regressor = nn.Sequential(
            nn.Linear(in_ch, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor (B, T, F)

        Returns:
            Tensor (B, 1)
        """
        x = x.transpose(1, 2)        # (B, F, T)
        y = self.tcn(x)              # (B, C, T)
        last_step = y[:, :, -1]      # causal last timestep
        return self.regressor(last_step)


def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_model(
    path: str,
    num_features: int,
    device: str = "cpu",
) -> TemporalCNN:
    model = TemporalCNN(num_features=num_features)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model