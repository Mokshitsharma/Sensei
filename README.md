# 📈 Sensei AI Trading System

> **Intelligent Multi-Model Trading Engine for NIFTY 50 Stocks**  
> Combining ML, Deep Learning, Reinforcement Learning, HMM, and NLP for production-grade trading signals

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.32+-red.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 Overview

Sensei AI is an **end-to-end intelligent trading system** that generates explainable BUY/SELL/HOLD signals by orchestrating six complementary AI paradigms:

- 📊 **Machine Learning** - Logistic Regression, Random Forest, Gradient Boosting (P(UP) prediction)
- 🧠 **LSTM Networks** - 2-layer sequential model for 5-day return forecasting
- ⚡ **Temporal CNN** - Dilated causal convolutions for temporal patterns
- 🔄 **Reinforcement Learning** - PPO agent trained on portfolio value optimization
- 📈 **Hidden Markov Models** - Market regime detection (BULL/BEAR classification)
- 📰 **NLP Sentiment** - News feed analysis for contextual signal adjustment

The system produces **ensemble BUY/SELL/HOLD signals** with confidence scores (0-100%) backed by explainable reasoning.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- 4GB RAM (minimum)
- Internet connection (data fetching)

### Installation

```bash
# Clone repository
git clone https://github.com/Mokshitsharma/Sensei.git
cd Sensei

# Create virtual environment
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

Open browser: **http://localhost:8501**

---

## 📚 Architecture

### System Diagram

```
Market Data (OHLCV)
    ↓
[Feature Engineering] (12+ features)
    ↓
┌─────────────────────────────────────────┐
│  Parallel Inference (6 Models)          │
├─────────────────────────────────────────┤
│ • ML Ensemble    → P(UP)                │
│ • LSTM           → 5d Return Forecast   │
│ • TCN            → Temporal Confirmation│
│ • HMM            → Market Regime (B/B)  │
│ • PPO            → RL Action            │
│ • News Sentiment → Contextual Signal    │
└────────────────┬────────────────────────┘
                 ↓
         [Decision Engine]
      Weighted Score Aggregation
                 ↓
         BUY / SELL / HOLD
         + Confidence (%)
         + Explanation
                 ↓
         [Backtest Engine]
      Sharpe Ratio, Max Drawdown
                 ↓
      [Streamlit Dashboard UI]
```

### Feature Engineering (ML Layer)

Sensei generates **12+ features** from price + indicators:

```python
# Normalized Technical Features
- rsi_norm          : Normalized RSI (0-1)
- ema_spread        : (EMA20 - EMA50) / Close
- macd_diff         : MACD - Signal Line

# Volatility & Momentum
- atr_pct           : ATR / Close
- volatility_10     : 10-day rolling std of returns
- return_1/5/10     : 1, 5, 10-day returns

# Price Positioning
- range_position    : (Close - 20Low) / (20High - 20Low)

# Supervised Targets (no leakage)
- future_return_5d  : 5-day ahead return
- future_direction_5d: Binary (UP/DOWN)
```

### Model Specifications

#### **Machine Learning (Ensemble)**
| Component | Config |
|-----------|--------|
| Logistic Regression | max_iter=1000, solver=lbfgs |
| Random Forest | n_estimators=300, max_depth=6 |
| Gradient Boosting | n_estimators=300, lr=0.05, max_depth=3 |
| **Output** | **P(UP) ∈ [0, 1]** |

#### **LSTM (Temporal Forecasting)**
```python
Input:   (Batch, 30 days, 4 features)
         └─ Features: rsi_norm, ema_spread, macd_diff, atr_pct

Architecture:
  └─ LSTM(input=4, hidden=64, layers=2, dropout=0.2)
  └─ Regressor(64 → 32 → 1)
  
Loss:    MSE
Output:  5-day return forecast ∈ ℝ
```

#### **Temporal CNN (Causal Pattern Detection)**
```python
Input:   (Batch, 30 days, 4 features)

Architecture:
  └─ TemporalBlock(4 → 32, kernel=3, dilation=1)
  └─ TemporalBlock(32 → 32, kernel=3, dilation=2)
  └─ TemporalBlock(32 → 64, kernel=3, dilation=4)
  └─ Regressor(64 → 32 → 1)

Output:  5-day return ∈ ℝ
```

#### **Hidden Markov Model (Regime Detection)**
```python
States:           2 (BULL, BEAR)
Observation:      Daily log-returns
Covariance Type:  Diagonal (stable)
Fit Method:       Expectation-Maximization

Logic:
  Mean(State_0) > Mean(State_1)  →  BULL
  Else                           →  BEAR
```

#### **Reinforcement Learning (PPO Agent)**
```python
Environment:      Custom TradingEnv (Gym-compatible)
  Actions:        [HOLD=0, BUY=1, SELL=2]
  Observations:   Technical features (4D)
  Reward:         ΔPortfolioValue - 0.1 × Drawdown

Agent:           PPO (Stable-Baselines3)
  Timesteps:      Configurable
  Policy:         MlpPolicy
  Learning Rate:  1e-4

Output:          Action ∈ {BUY, SELL, HOLD}
```

### Decision Engine Logic

```python
score = 0.0

# ML Ensemble (±1 point)
if P(UP) > 0.6:     score += 1   # Bullish
elif P(UP) < 0.4:   score -= 1   # Bearish

# LSTM (±1 point)
if return_5d > 0:   score += 1   # Positive forecast
else:               score -= 1

# TCN (±0.5 points)
if tcn_return > 0:  score += 0.5 # Confirms upside

# HMM Regime (±1 point)
if regime == BULL:  score += 1
elif regime == BEAR: score -= 1

# PPO Agent (±1 point)
if ppo_action == BUY:   score += 1
elif ppo_action == SELL: score -= 1

# News Sentiment (±1 point)
if sentiment > +0.2:    score += 1  # Positive
elif sentiment < -0.2:  score -= 1  # Negative

# Final Decision
THRESHOLD_BUY  = score ≥ +2
THRESHOLD_SELL = score ≤ -2
DEFAULT        = HOLD

confidence = min(|score| / 5.0, 1.0)  # Normalize to [0, 1]
```

---

## 📊 Features

### ✨ Multi-Model Signal Generation
- **6 parallel inference engines** for robustness
- **Ensemble voting** with weighted aggregation
- **Confidence scoring** (0-100%) per decision
- **Explainable AI** - lists contributing factors

### 📈 Technical Analysis
- **Indicators**: RSI, MACD, EMA, ATR, Bollinger Bands
- **Cross-validation**: 10-fold CV on training data
- **Feature importance**: Captured in tree-based models

### 🏆 Backtesting Suite
- **Equity curve** visualization
- **Sharpe ratio** calculation
- **Max drawdown** analysis
- **Total return** metrics
- **Win rate** statistics

### 💻 Professional Dashboard
- **Groww-style light UI** (modern Indian broker aesthetic)
- **Real-time decision card** (color-coded BUY/SELL/HOLD)
- **Company fundamentals** (Market Cap, P/E, ROE, 52W range)
- **AI metrics row** (ML prob, LSTM/TCN forecasts, market regime)
- **Interactive charts** (lightweight-charts integration)
- **Strategy performance** - backtest results visualization

### 📰 NLP Layer
- **News sentiment analysis** from RSS feeds
- **Signal bias adjustment** based on sentiment
- **Contextual decision making**

---

## 📁 Project Structure

```
Sensei/
├── app.py                      # Main Streamlit entry point
├── requirements.txt            # Dependencies
├── runtime.txt                 # Python 3.12+
│
├── models/                     # Pre-trained artifacts
│   ├── lstm_HDFCBANK_NS.pt    # LSTM weights (217 KB)
│   ├── tcn_HDFCBANK_NS.pt     # TCN weights (139 KB)
│   ├── ml_return_model.joblib # ML ensemble (1.6 MB)
│   └── ppo_hdfc.zip           # PPO agent (138 KB)
│
└── src/
    ├── data/                   # Data layer
    │   ├── nifty50.py         # Stock universe
    │   ├── prices.py          # yfinance wrapper
    │   └── news.py            # Sentiment feeds
    │
    ├── domain/                 # Domain logic
    │   ├── indicators.py       # Technical indicators
    │   ├── signals.py          # Rule-based signals
    │   ├── fundamentals.py     # Company metrics
    │   ├── patterns.py         # Chart patterns
    │   ├── backtest.py         # Backtest engine
    │   └── export.py           # Data export
    │
    ├── ml/                     # Machine Learning
    │   ├── features.py         # Feature engineering
    │   ├── model.py            # Model factory
    │   ├── train.py            # Training pipeline
    │   ├── predict.py          # Inference
    │   ├── evaluation.py        # Metrics
    │   └── explain.py          # Interpretability
    │
    ├── dl/                     # Deep Learning
    │   ├── lstm.py             # LSTM model + training
    │   ├── temporal_cnn.py     # TCN model + training
    │   ├── dataset.py          # Time series dataset
    │   └── train.py            # DL training loops
    │
    ├── regimes/                # Regime Detection
    │   ├── hmm.py              # HMM implementation
    │   ├── clustering.py        # K-means regimes
    │   └── detect.py           # Regime inference
    │
    ├── rl/                     # Reinforcement Learning
    │   ├── env.py              # TradingEnv (Gym)
    │   ├── agent.py            # PPO wrapper
    │   ├── train.py            # PPO training
    │   └── evaluate.py         # RL backtesting
    │
    ├── pipeline/               # Orchestration
    │   ├── signal_pipeline.py  # Full inference
    │   └── decision_engine.py  # Final decision
    │
    ├── backtest/               # Backtesting
    │   ├── engine.py           # Backtest simulator
    │   └── metrics.py          # Portfolio metrics
    │
    ├── charts/                 # Visualization
    │   └── lightweight.py      # Chart rendering
    │
    └── utils/                  # Utilities
        ├── config.py           # Constants
        ├── logger.py           # Logging
        ├── metrics.py          # Performance metrics
        └── data.py             # Data helpers
```

---

## 🔧 Usage

### Basic Trading Signal

```python
from src.pipeline.signal_pipeline import run_signal_pipeline
from src.pipeline.decision_engine import make_final_decision
from src.data.prices import load_prices
from src.domain.fundamentals import load_fundamentals
from src.data.news import get_news_signal

# Load data
price_df = load_prices("HDFCBANK.NS", timeframe="1y")
fundamentals = load_fundamentals("HDFCBANK.NS")
news = get_news_signal("HDFC Bank")

# Run inference
signals = run_signal_pipeline(
    price_df=price_df,
    fundamentals=fundamentals,
    company="HDFC Bank",
    lstm_model_path="models/lstm_HDFCBANK_NS.pt",
    tcn_model_path="models/tcn_HDFCBANK_NS.pt",
    ppo_model_path="models/ppo_hdfc.zip"
)

# Final decision
decision = make_final_decision(
    signals=signals,
    news_sentiment=news["sentiment_score"]
)

print(f"Action: {decision['action']}")
print(f"Confidence: {decision['confidence']*100:.1f}%")
print(f"Explanation: {decision['explanation']}")
# Output:
# Action: BUY
# Confidence: 75.2%
# Explanation: ML model is bullish | LSTM predicts positive return | Market regime is bullish
```

### Backtesting Strategy

```python
from src.backtest.engine import run_backtest
from src.backtest.metrics import calculate_metrics

# Simulate strategy on historical data
backtest_df = run_backtest(price_df, signal_series)
metrics = calculate_metrics(backtest_df["equity"])

print(f"Total Return: {metrics['total_return']*100:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
```

---

## 📊 Model Performance

| Model | Data | Task | Status |
|-------|------|------|--------|
| ML Ensemble | NIFTY 50 | P(UP) | ✅ Trained |
| LSTM (HDFC) | 5yr OHLCV | 5d Return | ✅ Trained |
| TCN (HDFC) | 5yr OHLCV | 5d Return | ✅ Trained |
| HMM | Real-time | Regime | ✅ Fitted on-the-fly |
| PPO (HDFC) | Simulated env | Trading | ✅ Trained |
| News Sentiment | RSS feeds | Sentiment | ✅ Real-time |

**Note:** Models pre-trained on **HDFC Bank (HDFCBANK.NS)**. Extend to other NIFTY 50 stocks by retraining on respective historical data.

---

## 🛠️ Dependencies

```
streamlit==1.32.2           # Web UI
pandas==2.1.4               # Data manipulation
numpy==1.26.4               # Numerical computing
scikit-learn==1.3.2         # ML models
matplotlib==3.8.2           # Plotting (legacy)
yfinance==0.2.37            # Price data
joblib==1.3.2               # Model persistence
hmmlearn==0.3.2             # HMM library
nsepython==0.0.972          # NSE India data
feedparser==6.0.11          # RSS news feeds
streamlit-lightweight-charts==0.0.26  # Interactive charts
requests==2.31.0            # HTTP client
beautifulsoup4==4.12.3      # HTML parsing
lxml==5.1.0                 # XML parsing
python-dateutil==2.8.2      # Date utilities
pytz==2024.1                # Timezone handling
tqdm==4.66.2                # Progress bars
torch==2.0+                 # Deep Learning (PyTorch)
stable-baselines3==2.0+     # RL algorithms
gymnasium==0.28+            # RL environments
```

---

## ⚙️ Configuration

Edit `src/utils/config.py` to customize:

```python
# Model paths
LSTM_MODEL_PATH = "models/lstm_HDFCBANK_NS.pt"
TCN_MODEL_PATH = "models/tcn_HDFCBANK_NS.pt"
PPO_MODEL_PATH = "models/ppo_hdfc.zip"

# Trading environment
INITIAL_BALANCE = 100_000.0
TRANSACTION_COST = 0.001  # 0.1% per trade

# Training hyperparameters
LSTM_EPOCHS = 30
LSTM_LR = 1e-3
PPO_TIMESTEPS = 100_000

# Data
SEQUENCE_LENGTH = 30
LOOKBACK_DAYS = 250
FORECAST_HORIZON = 5
```

---

## 📈 Dashboard Screenshots

### 1. Decision Card
```
┌─────────────────────────────────┐
│ Final Decision: BUY             │
│ Confidence: 72.5%               │
└─────────────────────────────────┘
```

### 2. Stock Header
```
HDFC Bank
₹2,834.50 (BULL market)
```

### 3. AI Metrics
```
ML Prob (UP):    0.68
LSTM Return:     0.0234
TCN Return:      0.0198
Market Regime:   BULL
```

### 4. Backtest Results
```
Total Return:    18.75%
Sharpe Ratio:    1.43
Max Drawdown:    12.3%
```

---

## 🔬 Backtesting Engine

Sensei includes a simple but effective backtesting simulator:

```python
def run_backtest(price_df: pd.DataFrame, signals: pd.Series):
    """
    Simulates trading strategy on historical data.
    
    Inputs:
      - price_df: OHLCV data
      - signals: BUY/SELL/HOLD series
    
    Returns:
      - equity curve
      - trades log
      - performance metrics
    """
```

**No look-ahead bias:** Signals use only past data (close[t-1], indicators[t-1], etc.)

---

## 🚀 Deployment

### Local
```bash
streamlit run app.py
```

### Heroku
```bash
git push heroku main  # Auto-deploys via Procfile
```

### Docker
```bash
docker build -t sensei .
docker run -p 8501:8501 sensei
```

---

## ⚠️ Disclaimer

**⚠️ IMPORTANT: NOT FINANCIAL ADVICE**

- This project is for **educational and research purposes only**
- Trading involves **substantial risk of loss**
- Past performance does not guarantee future results
- **Never risk capital you cannot afford to lose**
- Consult a financial advisor before trading
- Sensei signals should supplement, not replace, professional advice

---

## 📚 References

- [PyTorch LSTM Tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
- [Temporal Convolutional Networks](https://arxiv.org/abs/1803.01271)
- [Hidden Markov Models for Finance](https://en.wikipedia.org/wiki/Hidden_Markov_model)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)
- [Feature Engineering for ML](https://scikit-learn.org/stable/modules/preprocessing.html)

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/xyz`)
3. Commit changes (`git commit -m "Add xyz"`)
4. Push branch (`git push origin feature/xyz`)
5. Open Pull Request

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details

---

## ✉️ Contact

**Mokshit Sharma**  
[GitHub](https://github.com/Mokshitsharma) | [Email](mailto:your-email@example.com)

---

**Last Updated:** March 2026

> Built with ❤️ for advancing AI-driven trading research
