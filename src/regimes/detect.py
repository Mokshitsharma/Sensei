# src/regimes/detect.py

import pandas as pd
from typing import Dict

from src.regimes.hmm import MarketRegimeHMM


def detect_current_regime(df: pd.DataFrame) -> Dict[str, object]:
    """
    Detect current market regime using HMM.

    Returns:
        {
            "regime": "BULL" | "BEAR" | "SIDEWAYS",
            "confidence": float,
        }
    """

    hmm = MarketRegimeHMM()
    hmm.fit(df)

    regime = hmm.predict(df)

    return {
        "regime": regime,
        "confidence": 0.7,  # HMMs don't output true probabilities cleanly
    }


def detect_regime_series(df: pd.DataFrame) -> pd.Series:
    """
    Detect regime at each timestep.

    Useful for:
        - regime-aware backtesting
        - visualization
    """

    hmm = MarketRegimeHMM()
    hmm.fit(df)

    return hmm.predict_series(df)