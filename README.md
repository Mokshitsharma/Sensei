# 🧠 Sensei AI — Intelligent Stock Analysis Platform

> **Institutional-grade AI trading intelligence for all 50 Nifty stocks, built for retail investors.**

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?style=flat-square&logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange?style=flat-square&logo=pytorch)
![Stable-Baselines3](https://img.shields.io/badge/SB3-2.3-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Problem Statement

Retail investors in India have no access to the same level of AI-powered trading intelligence that institutional players use. They rely on tips, basic charts, and gut feeling — with no unified, explainable system that combines technical analysis, deep learning, reinforcement learning, and real-time news sentiment into a single decision. The result is uninformed trades, poor risk management, and missed opportunities across the Nifty 50.

---

## 💡 My Solution

Sensei AI is a real-time stock analysis platform that unifies **five AI model families** with **financial NLP**, **multi-method Support & Resistance detection**, and a **plain-English Market Narrator** to deliver actionable, explainable BUY / SELL / HOLD decisions for all 50 Nifty stocks.

A **scored voting decision engine** (range: −5.0 to +5.0) aggregates signals from:
- Classical ML (Random Forest + SHAP)
- Deep Learning (LSTM + Temporal CNN via PyTorch)
- Reinforcement Learning (PPO Agent via Stable-Baselines3)
- Regime Detection (Hidden Markov Model)
- Financial NLP (FinBERT sentiment on live Google News RSS)

The output includes confidence score, SHAP-based feature attribution, intraday/swing trade setups, backtesting metrics, and a 6-tab AI Analyst Report — all in a Streamlit web app.

---

## 📊 Metrics

| Model | Type | Output |
|---|---|---|
| Random Forest | Classical ML | UP probability (0–1) + SHAP attribution |
| LSTM (2-layer, PyTorch) | Deep Learning | 5-day return forecast |
| Temporal CNN (causal, dilated) | Deep Learning | 5-day return forecast |
| PPO Agent (Stable-Baselines3) | Reinforcement Learning | BUY / SELL / HOLD action |
| GaussianHMM (hmmlearn) | Regime Detection | BULL / BEAR market state |
| FinBERT (HuggingFace) | Financial NLP | Weighted sentiment score (−1 to +1) |

**Backtest Metrics:**
- Total Return (%)
- Sharpe Ratio
- Maximum Drawdown (%)

**Decision Engine:**
- Score ≥ +2.0 → BUY
- Score ≤ −2.0 → SELL
- Otherwise → HOLD
- Confidence = min(|score| / 5.0, 1.0)

---

## 🛠️ Skills & Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.13 |
| Web UI | Streamlit 1.32, streamlit-lightweight-charts |
| Deep Learning | PyTorch 2.2 (LSTM, Temporal CNN) |
| Reinforcement Learning | Stable-Baselines3 2.3, Gymnasium |
| Classical ML | Scikit-learn 1.3 (Random Forest) |
| Explainability | SHAP ≥ 0.44 (TreeExplainer) |
| Financial NLP | HuggingFace Transformers (FinBERT by ProsusAI) |
| Regime Detection | hmmlearn 0.3 (GaussianHMM) |
| Market Data | yfinance 0.2, nsepython 0.0.972 |
| News | feedparser 6.0 (Google News RSS) |
| Data | Pandas 2.1, NumPy 1.26 |
| Visualization | Matplotlib 3.8 |
| Model Serialization | joblib 1.3 |
| Dev Environment | VS Code Dev Containers |

---

## 📂 Dataset Details

| Source | Type | Coverage |
|---|---|---|
| Yahoo Finance (yfinance) | OHLCV daily + 15-min intraday | All 50 Nifty stocks |
| NSE Python (nsepython) | NSE live fundamentals (PE, ROE, 52W range, Market Cap) | All 50 Nifty stocks |
| Google News RSS | Live financial headlines (up to 10 per stock) | Real-time |

**Features engineered for ML models (9 features):**
RSI (normalized), EMA spread, MACD divergence, ATR %, 10-day volatility, 1-day return, 5-day return, 10-day return, price range position.

**LSTM/TCN input:** (batch, 30 days, 4 features) — RSI_norm, EMA_spread, MACD_diff, ATR_%

---

## 🗂️ Folder Structure

```
Sensei/
├── app.py                          # Streamlit UI + orchestration (585 lines)
├── requirements.txt                # All dependencies
├── runtime.txt                     # Python 3.13
├── config/
│   ├── settings.yaml               # Feature columns, model config
│   └── logging.yaml
├── models/                         # Pre-trained model weights
│   ├── lstm_HDFCBANK_NS.pt         # LSTM weights (PyTorch)
│   ├── tcn_HDFCBANK_NS.pt          # Temporal CNN weights
│   ├── ppo_hdfc.zip                # PPO agent (SB3)
│   └── ml_return_model.joblib      # Random Forest
└── src/
    ├── data/                       # Data ingestion
    │   ├── nifty50.py              # Ticker dictionary (50 stocks)
    │   ├── prices.py               # OHLCV loader
    │   ├── news.py                 # News NLP pipeline
    │   └── providers/              # Yahoo + NSE abstraction
    ├── domain/                     # Business logic
    │   ├── indicators.py           # RSI, MACD, EMA, ATR
    │   ├── signals.py              # Rule-based signals
    │   ├── support_resistance.py   # 5-method S/R engine
    │   ├── setup_engine.py         # Intraday + Swing trade setup
    │   └── news_price_model.py     # News → price impact forecast
    ├── ml/                         # Classical ML
    │   ├── features.py             # Feature engineering (9 features)
    │   ├── model.py                # Random Forest loader
    │   └── shap_explain.py         # SHAP TreeExplainer
    ├── dl/                         # Deep Learning
    │   ├── lstm.py                 # LSTMPricePredictor (2-layer)
    │   ├── temporal_cnn.py         # TemporalCNN (causal, dilated)
    │   └── train.py                # Training script
    ├── rl/                         # Reinforcement Learning
    │   ├── env.py                  # Gymnasium TradingEnv
    │   ├── agent.py                # PPOTradingAgent
    │   └── train.py                # PPO training
    ├── regimes/
    │   └── hmm.py                  # GaussianHMM (2-state: BULL/BEAR)
    ├── pipeline/
    │   ├── signal_pipeline.py      # All model inference orchestration
    │   └── decision_engine.py      # Scoring + BUY/SELL/HOLD logic
    ├── explainability/
    │   └── narrator.py             # Signals → plain-English 6-tab report
    ├── backtest/
    │   ├── engine.py               # Backtest runner
    │   └── metrics.py              # Sharpe, Drawdown, Return
    ├── charts/
    │   └── lightweight.py          # Price chart renderer
    └── utils/
        ├── config.py
        ├── logger.py
        └── metrics.py
```

---

## ⚙️ System Architecture

```
Step 1 → User selects Nifty 50 stock + timeframe via Streamlit sidebar
Step 2 → Data Layer fetches OHLCV (yfinance), fundamentals (nsepython), live news (Google RSS)
Step 3 → Feature Engineering computes 9 ML features (RSI, EMA, MACD, ATR, returns, volatility)
Step 4 → Signal Pipeline runs all 5 AI models in parallel:
          → Random Forest → UP probability + SHAP attribution
          → LSTM (PyTorch) → 5-day return forecast
          → Temporal CNN (PyTorch) → 5-day return forecast
          → PPO Agent (SB3) → BUY/SELL/HOLD action
          → GaussianHMM → BULL/BEAR regime
Step 5 → News NLP Pipeline: FinBERT scores each headline → domain-weighted sentiment score
Step 6 → Decision Engine aggregates all signals into score (−5 to +5) → final BUY/SELL/HOLD + confidence
Step 7 → Market Narrator generates 6-tab plain-English AI Analyst Report
Step 8 → Support & Resistance Engine runs 5 methods → top 3 supports + top 3 resistances
Step 9 → Trade Setup Generator builds intraday (15-min) + swing (daily) setups with Entry/SL/Target/R:R
Step 10 → Backtest Engine runs strategy → Sharpe Ratio, Max Drawdown, Total Return
Step 11 → Streamlit UI renders all outputs: Decision Card, Price Chart, Metrics, News, S/R, Trade Setups
```

---

## 🔍 Why This Tech Stack?

| Choice | Reason |
|---|---|
| **Streamlit** | Fastest way to build interactive data apps in Python — no frontend dev needed |
| **PyTorch** | Flexible deep learning framework for custom LSTM + Temporal CNN architectures |
| **Stable-Baselines3** | Best production-ready RL library for Gymnasium environments |
| **FinBERT** | Domain-specific BERT model pre-trained on financial text — far superior to generic sentiment models |
| **SHAP** | Industry-standard explainability; TreeExplainer is optimized for Random Forest |
| **hmmlearn** | Lightweight, reliable HMM library; GaussianHMM is ideal for continuous return sequences |
| **yfinance + nsepython** | Free, reliable Indian stock market data sources for NSE/BSE |
| **feedparser** | Lightweight RSS parser for real-time Google News integration |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.13
- pip
- ~2GB disk space (PyTorch + model weights)

### Installation
```bash
git clone https://github.com/Mokshitsharma/Sensei.git
cd Sensei
pip install -r requirements.txt
```

### Run
```bash
streamlit run app.py
```
Open browser at **http://localhost:8501**

### Usage
1. Select a stock from the Nifty 50 dropdown
2. Choose a timeframe (1y / 2y / 5y)
3. Click **Run Analysis**
4. Explore: AI Decision → Chart → Fundamentals → Backtest → AI Report → News → S&R → Trade Setups

---

## 🔮 Future Improvements

1. **Per-stock model training** — automated training pipeline for all 50 Nifty stocks (current models trained on HDFCBANK only)
2. **Real-time WebSocket streaming** — live price tick updates instead of on-demand fetch
3. **Portfolio optimizer** — Markowitz / Kelly Criterion multi-stock allocation view
4. **Transformer model** — Temporal Fusion Transformer for longer-horizon (10–30 day) forecasting
5. **Alert system** — Telegram / email push notifications on signal changes
6. **Realistic backtest** — slippage, partial fills, position sizing, and commission modelling
7. **Options intelligence** — OI data, PCR, IV skew integration for derivatives traders

---

## ⚠️ Disclaimer

Sensei AI is built for **educational and research purposes only**. It is **not financial advice**. Do not make real investment decisions based solely on this tool. Always consult a SEBI-registered financial advisor.

---

## 👤 Author

**Mokshit Sharma**
B.Tech + M.Tech | AI & Data Science | DAVV, Indore
📧 sharman48520@gmail.com | 🌐 [mokshitsharma27.vercel.app](https://mokshitsharma27.vercel.app)
🔗 [LinkedIn](https://linkedin.com/in/mokshit-sharma-75b5ab305) | 
💻 [GitHub](https://github.com/Mokshitsharma)

---

⭐ Star this repo if you find it useful!
