# src/utils/config.py

from pathlib import Path


# =============================
# Project Paths
# =============================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
EXPORT_DIR = PROJECT_ROOT / "exports"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)


# =============================
# Data Settings
# =============================

DEFAULT_TIMEFRAME = "1y"
INTRADAY_TIMEFRAMES = ["5m", "15m", "30m", "1h"]

PRICE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]


# =============================
# Feature Engineering
# =============================

FEATURE_COLUMNS = [
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

TARGET_RETURN = "future_return_5d"
TARGET_DIRECTION = "future_direction_5d"

SEQUENCE_LENGTH = 30


# =============================
# ML Settings
# =============================

ML_MODEL_NAME = "ml_return_model"
ML_TASK = "regression"

TRAIN_TEST_SPLIT = 0.7
RANDOM_SEED = 42


# =============================
# Deep Learning (LSTM / TCN)
# =============================

DL_EPOCHS = 40
DL_BATCH_SIZE = 64
DL_LEARNING_RATE = 1e-3

LSTM_MODEL_NAME = "lstm_model"
TCN_MODEL_NAME = "tcn_model"


# =============================
# Reinforcement Learning (PPO)
# =============================

RL_GAMMA = 0.99
RL_LR = 3e-4
RL_ENTROPY_COEF = 0.01
RL_TOTAL_TIMESTEPS = 200_000

ACTIONS = ["HOLD", "BUY", "SELL"]


# =============================
# Backtesting
# =============================

INITIAL_CAPITAL = 100_000.0
POSITION_SIZE = 1.0
TRANSACTION_COST = 0.0005


# =============================
# Regime Detection
# =============================

REGIME_TYPES = ["BULL", "BEAR", "SIDEWAYS"]


# =============================
# News & NLP
# =============================

MAX_NEWS_ITEMS = 5
NEWS_SENTIMENT_POS_THRESHOLD = 0.2
NEWS_SENTIMENT_NEG_THRESHOLD = -0.2


# =============================
# Utility Helpers
# =============================

def model_path(name: str) -> Path:
    """
    Resolve model file path.
    """
    return MODEL_DIR / name


def export_path(name: str) -> Path:
    """
    Resolve export file path.
    """
    return EXPORT_DIR / name