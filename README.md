# 🧠 Sensei AI — Intelligent Stock Analysis Platform

> **Institutional-grade AI trading intelligence for Indian equity markets, built for retail investors.**

[![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.3-purple?style=flat-square)](https://stable-baselines3.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 🎯 What is Sensei?

Sensei AI is a real-time stock analysis platform that unifies **five AI model families** — classical machine learning, LSTM, Temporal CNN, Proximal Policy Optimization (PPO), and Hidden Markov Models — with **FinBERT news NLP**, **multi-method Support & Resistance detection**, and a **plain-English Market Narrator** to deliver actionable, explainable trading decisions for all 50 Nifty stocks.

---

## ✨ Features

### 🤖 Multi-Model AI Decision Engine

The heart of Sensei is a scored voting system across 6 signal layers:

| Signal | Model | Output |
|---|---|---|
| Technical Analysis | RSI, MACD, EMA crossover | Rule-based signal |
| Classical ML | Random Forest + SHAP | UP probability + feature attribution |
| Deep Learning 1 | LSTM (2-layer, PyTorch) | 5-day return forecast |
| Deep Learning 2 | Temporal CNN (causal, dilated) | 5-day return forecast |
| Reinforcement Learning | PPO Agent (Stable-Baselines3) | BUY / SELL / HOLD action |
| Regime Detection | Hidden Markov Model | BULL / BEAR market state |

All signals are aggregated into a final **BUY / SELL / HOLD** decision with a confidence score.

### 📰 News Intelligence Engine

- Live headlines via **Google News RSS** (up to 10 per stock)
- **FinBERT** (ProsusAI) financial sentiment with keyword-VADER fallback
- Domain-weighted scoring:
  - Earnings headlines: **1.5×**
  - Regulatory news: **1.3×**
  - Management changes: **1.2×**
  - Macro news: **0.9×**
  - General: **0.6×**
- Automatic top bullish / top bearish headline identification
- **News-driven price forecast**: direction, predicted price, ±range, confidence

### 📊 Support & Resistance Engine (5 Methods)

| Method | Description |
|---|---|
| Classic Pivot Points | Calculated from previous session H/L/C |
| Fibonacci Retracements | Drawn from recent swing high → swing low |
| Camarilla Pivots | Tighter intraday S/R levels |
| Swing High/Low Detection | Rolling pivot algorithm on daily OHLCV |
| Volume Profile Peaks | Price zones with above-average volume |

Returns **top-3 supports** and **top-3 resistances**, each with strength rating (`Strong / Moderate / Weak`) and historical touch count.

### 🧠 AI Analyst Report (6 Tabs)

The **Market Narrator** translates every model number into investor-readable language:

- 📈 **Trend** — EMA crossover, regime, 5-day momentum, price range position
- ⚡ **Momentum** — RSI (overbought/oversold), MACD divergence, 1-day return
- 🌊 **Volatility** — ATR %, 10-day realized volatility assessment
- 🤖 **AI Models** — LSTM/TCN/PPO/ML consensus narrative
- 🔍 **SHAP Drivers** — horizontal bar chart + text explaining top prediction factors
- 📰 **News** — sentiment direction, magnitude, and market impact explanation

### 📐 Trade Setup Generator

| Mode | Data | Output |
|---|---|---|
| ⚡ Intraday | 15-min OHLCV (5-day) | Entry zone, Stop Loss, T1, T2, R:R, candlestick bias |
| 📅 Swing | Daily OHLCV | Multi-day setup with ATR-based levels and trade plan |

### 📈 Strategy Backtesting

- Equity curve visualization
- **Sharpe Ratio**, **Maximum Drawdown**, **Total Return**

---

## 🛠️ Tech Stack

| Category | Technology | Version |
|---|---|---|
| Web UI | Streamlit | 1.32.2 |
| Interactive Charts | streamlit-lightweight-charts | 0.7.20 |
| Deep Learning | PyTorch | 2.2.2 |
| Reinforcement Learning | Stable-Baselines3 | 2.3.2 |
| RL Environment | Gymnasium | — |
| Classical ML | Scikit-learn | 1.3.2 |
| Explainability | SHAP | ≥0.44.0 |
| Regime Detection | hmmlearn | 0.3.2 |
| NLP (Financial) | HuggingFace Transformers (FinBERT) | — |
| News Fetching | feedparser | 6.0.11 |
| Market Data | yfinance | 0.2.37 |
| NSE Data | nsepython | 0.0.972 |
| Data Manipulation | pandas | 2.1.4 |
| Numerical Computing | numpy | 1.26.4 |
| Visualization | Matplotlib | 3.8.2 |
| Model Serialization | joblib | 1.3.2 |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.13
- pip
- ~2GB disk space (for model weights and dependencies including PyTorch)

### Installation

```bash
# Clone the repository
git clone https://github.com/Mokshitsharma/Sensei.git
cd Sensei

# Install all dependencies
pip install -r requirements.txt
```

> ⚠️ PyTorch 2.2 and Stable-Baselines3 are large dependencies. Installation may take a few minutes.

### Run the App

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

### Usage

1. Select a stock from the **Nifty 50 dropdown** in the sidebar
2. Choose a **timeframe** (1y, 2y, or 5y)
3. Click **Run Analysis**
4. Explore the AI decision, price chart, fundamentals, backtest, AI report, news, S/R levels, and trade setups

### Dev Container (Optional)

If you use VS Code with the Dev Containers extension:

```bash
code .
# Ctrl+Shift+P → "Dev Containers: Reopen in Container"
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI                             │
└──────────────────────┬──────────────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │      DATA LAYER         │
          │  yfinance · nsepython   │
          │  Google News RSS        │
          └────────────┬────────────┘
                       │
          ┌────────────▼──────────────────────────────────┐
          │              SIGNAL PIPELINE                   │
          │                                                │
          │  Rule-based TA   Random Forest + SHAP          │
          │  LSTM (PyTorch)  Temporal CNN (PyTorch)        │
          │  PPO Agent       HMM Regime Detection          │
          └────────────┬───────────────────────────────────┘
                       │
          ┌────────────▼──────────────────┐
          │       NEWS NLP LAYER           │
          │  FinBERT sentiment scoring     │
          │  Impact weighting              │
          │  News-driven price forecast    │
          └────────────┬──────────────────┘
                       │
          ┌────────────▼──────────────────┐
          │       DECISION ENGINE          │
          │  Scored voting (−5 to +5)      │
          │  BUY ≥2 · SELL ≤−2 · HOLD    │
          └────────────┬──────────────────┘
                       │
          ┌────────────▼──────────────────┐
          │       MARKET NARRATOR          │
          │  SHAP attribution → text       │
          │  6-tab plain-English report    │
          └────────────┬──────────────────┘
                       │
          ┌────────────▼──────────────────────────────────┐
          │              UI RENDER                         │
          │  Decision · Chart · Fundamentals · Backtest    │
          │  AI Report · News · S&R · Trade Setup          │
          └────────────────────────────────────────────────┘
```

---

## 🧮 Decision Engine Logic

```python
Score range: -5.0 to +5.0

Signal             Condition      Score
─────────────────────────────────────────
ML Prob UP         > 60%          +1.0
ML Prob UP         < 40%          -1.0
LSTM Forecast      > 0            +1.0
LSTM Forecast      < 0            -1.0
TCN Forecast       > 0            +0.5
HMM Regime         BULL           +1.0
HMM Regime         BEAR           -1.0
PPO Action         BUY            +1.0
PPO Action         SELL           -1.0
News Sentiment     > 0.20         +1.0
News Sentiment     < -0.20        -1.0

Final: BUY if score ≥ 2  |  SELL if score ≤ -2  |  HOLD otherwise
Confidence = min(|score| / 5.0, 1.0)
```

---

## 📁 Project Structure

```
Sensei/
├── app.py                          # Streamlit UI + orchestration
├── requirements.txt
├── runtime.txt                     # Python 3.13
├── config/
│   ├── settings.yaml               # Feature columns, model config
│   └── logging.yaml
├── models/                         # Pre-trained weights
│   ├── lstm_HDFCBANK_NS.pt         # LSTM weights
│   ├── tcn_HDFCBANK_NS.pt          # TCN weights
│   ├── ppo_hdfc.zip                # PPO agent
│   └── ml_return_model.joblib      # Random Forest
└── src/
    ├── data/                       # Data ingestion
    │   ├── nifty50.py              # Ticker dictionary (50 stocks)
    │   ├── prices.py               # OHLCV loader
    │   ├── news.py                 # News NLP pipeline
    │   └── providers/              # Yahoo + NSE provider abstraction
    ├── domain/                     # Business logic
    │   ├── indicators.py           # RSI, MACD, EMA, ATR
    │   ├── signals.py              # Rule-based signal
    │   ├── support_resistance.py   # 5-method S/R engine
    │   ├── setup_engine.py         # Trade setup builder
    │   └── news_price_model.py     # News → price forecast
    ├── ml/                         # Classical ML
    │   ├── features.py             # 9-feature engineering
    │   ├── model.py                # Random Forest loader
    │   └── shap_explain.py         # SHAP TreeExplainer
    ├── dl/                         # Deep Learning
    │   ├── lstm.py                 # LSTMPricePredictor
    │   ├── temporal_cnn.py         # TemporalCNN (TCN)
    │   └── train.py                # Training script
    ├── rl/                         # Reinforcement Learning
    │   ├── env.py                  # Gymnasium TradingEnv
    │   ├── agent.py                # PPOTradingAgent
    │   └── train.py                # PPO training
    ├── regimes/                    # Regime detection
    │   └── hmm.py                  # GaussianHMM (2-state)
    ├── pipeline/                   # Orchestration
    │   ├── signal_pipeline.py      # All model inference
    │   └── decision_engine.py      # Scoring + BUY/SELL/HOLD
    ├── explainability/
    │   └── narrator.py             # Signals → plain English
    ├── backtest/
    │   ├── engine.py
    │   └── metrics.py
    └── utils/
        ├── config.py
        ├── logger.py
        └── metrics.py
```

---

## 📊 Model Details

### LSTM Price Predictor
```
Architecture: 2-layer LSTM → Linear(64→32) → ReLU → Linear(32→1)
Input: (batch, 30 days, 4 features)
Output: Predicted 5-day return (regression)
Features: RSI_norm, EMA_spread, MACD_diff, ATR_%
Loss: MSE | Optimizer: Adam
```

### Temporal CNN (TCN)
```
Architecture: 3 TemporalBlocks with dilated causal convolutions
Dilations: [1, 2, 4] | Channels: [32, 32, 64] | Kernel: 3
Each block: Conv1d → Chomp1d → ReLU → Dropout → Residual connection
Output: Linear(64→32) → ReLU → Linear(32→1)
```

### PPO Reinforcement Learning
```
Environment: Gymnasium TradingEnv
Actions: {0=HOLD, 1=BUY, 2=SELL}
Observation: 4 technical features at time t
Reward: ΔNAV − drawdown_penalty (×0.1)
Initial portfolio: ₹1,00,000 | Transaction cost: 0.1%
```

### Hidden Markov Model
```
Library: hmmlearn GaussianHMM
States: 2 (BULL / BEAR) | Covariance: diagonal
Features: daily percentage returns (stationary)
Outlier filter: |returns| < 50% | Iterations: 200
```

---

## ⚠️ Limitations & Known Issues

- Pre-trained LSTM, TCN, and PPO models are trained on **HDFCBANK.NS only**. The system loads these same weights for all other Nifty 50 stocks — predictions will be less accurate for other tickers until per-stock training is done.
- The backtest uses a simplified signal series (positive/negative returns) rather than the actual ML-generated signals. Use backtesting results directionally only.
- FinBERT requires a HuggingFace model download on first run (~500MB). Ensure internet connectivity.
- Intraday trade setups require the Yahoo intraday data endpoint to be accessible; this may fail intermittently.

---

## 🗺️ Roadmap

- [ ] **Per-stock model training** — automated training pipeline for all 50 Nifty stocks
- [ ] **Real-time streaming** — WebSocket-based live price updates
- [ ] **Portfolio optimizer** — Markowitz / Kelly criterion multi-stock view
- [ ] **Realistic backtest** — slippage, partial fills, position sizing
- [ ] **Alert system** — Telegram / email notifications on signal changes
- [ ] **Transformer model** — Temporal Fusion Transformer for longer-horizon forecasting
- [ ] **Options intelligence** — OI, PCR, IV skew integration
- [ ] **Model drift monitoring** — rolling accuracy tracking + retraining triggers

---

## ⚠️ Disclaimer

Sensei AI is built for **educational and research purposes only**. It is **not** financial advice. Do not make real investment decisions based solely on this tool's output. Always consult a SEBI-registered financial advisor.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

For large changes, please open an issue first to discuss the direction.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p>Built with ❤️ for Indian retail investors</p>
  <p>⭐ Star this repo if you find it useful</p>
</div>
