---
title: Sensei AI API
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

<!-- readme-refresh 2026-11 -->
<div align="center">

# 📈 Sensei

**Trading-intelligence research for Nifty 50 stocks, for education**

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" /> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" /> <img src="https://img.shields.io/badge/FinBERT-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="FinBERT" /> <img src="https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge" alt="LightGBM" /> <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn" /> <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" /> <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />

</div>

## ✨ Highlights

- Five model families vote on one explainable BUY, SELL or HOLD signal
- SHAP feature attribution and FinBERT news sentiment
- Paper trading, a screener and a public prediction-accuracy record
- For education only, not investment advice

<!-- body -->
> **Educational stock analysis platform exploring multi-model AI for Nifty 50 stocks.**

---

## 📌 Problem Statement

Retail market analysis often relies on fragmented tools or single indicators without a unified framework. Sensei AI is an educational exploration into combining technical analysis, deep learning, reinforcement learning, regime detection, and news sentiment into a single explainable interface. Rigorous walk-forward backtests show that this multi-model approach does not outperform simple buy-and-hold; it serves as a transparent educational benchmark rather than an alpha generator.

---

## 💡 My Solution

Sensei AI is a stock analysis platform that unifies **five AI model families** with **financial NLP**, **multi-method Support & Resistance detection**, and a **plain-English Market Narrator** to deliver actionable, explainable BUY / SELL / HOLD decisions for all 50 Nifty stocks.

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

### Walk-Forward Out-of-Sample Backtest Results

<!-- RESULTS:START -->
Walk-forward out-of-sample backtest, 2018-01-01 to 2026-09-30 (2161 trading days, 45 Nifty 50 tickers, 20 bps per side). 95% CIs: stationary bootstrap (10,000 resamples, mean block 21 days, seed 20261026). Run `20261027-111713-backtest-walkforward`.

| Strategy | Sharpe [95% CI] | Max drawdown [95% CI] | Total return | Sharpe minus B&H [95% CI] |
|---|---|---|---|---|
| Buy-and-hold (equal weight) | 0.93 [0.19, 1.78] | -36.2% [-58.6%, -16.6%] | 282.3% | - |
| Random Forest | 0.79 [0.07, 1.61] | -34.4% [-55.9%, -15.1%] | 161.9% | -0.14 [-0.35, 0.10] |
| RF + HMM | 0.20 [-0.46, 1.00] | -19.8% [-36.9%, -6.9%] | 9.5% | -0.73 [-1.23, -0.19] |

- RF 5-day directional hit rate: 52.9% [52.0%, 53.7%] (n = 97,020 ticker-days); always-predict-up baseline: 53.2% [52.3%, 54.2%]; difference -0.4% [-0.7%, -0.1%].
- Binomial test vs always-up baseline (53.2%, non-overlapping 5-day windows, n = 19,440): hit rate 53.5%, p = 0.455.
- Random Forest: permutation test vs randomly shifted signals p = 0.552 (shuffled signals, zero cost: p = 0.469); per-ticker Sharpe minus B&H -0.10 [-0.12, -0.08], wins 3/45, Wilcoxon p = 6.25e-12. **Verdict: does NOT beat buy-and-hold (Sharpe diff CI includes or is below 0).**
- RF + HMM: permutation test vs randomly shifted signals p = 0.222 (shuffled signals, zero cost: p = 0.525); per-ticker Sharpe minus B&H -0.53 [-0.63, -0.43], wins 1/45, Wilcoxon p = 1.14e-13. **Verdict: does NOT beat buy-and-hold (Sharpe diff CI includes or is below 0).**

Full numbers: `results/runs/20261027-111713-backtest-walkforward/report.md` and `metrics.json`.
<!-- RESULTS:END -->

### Limitations

- **Survivorship bias:** The backtest universe applies today's Nifty 50 constituents retrospectively from 2018 to 2026 (and only the 45 tickers with continuous price history across the window), ignoring historical index additions and deletions.
- **Execution cost assumptions:** A flat 20 bps cost per side is assumed with no slippage or market impact modeled; real-world execution in volatile conditions would degrade net performance further.
- **No demonstrated edge over buy-and-hold:** The strategy fails to achieve a statistically significant edge over equal-weight buy-and-hold; the paired Sharpe difference 95% bootstrap confidence interval includes zero and is predominantly negative.
- **Educational and research use only:** Not investment advice, a recommendation, or an offer to buy/sell securities. Past backtest results do not guarantee future live performance.

### Price-Target Win Rate (goal: 80%)

**Result: the 80% goal is not met.** No horizon reached 80% on the held-out test.

A price target is a level k × σ√h away from today's close (σ = 21-day daily volatility, k ∈ {0.5, 1, 1.5, 2}), on the up or down side. It wins when the daily high (or low) touches it within h trading days. One LightGBM classifier per (side, k) predicts the touch; isotonic calibration and a threshold picked on cross-validation decide what is published. Guards: targets closer than 0.5σ√h never count; baselines are the touch rate at the same distance with no model; coverage is reported with every win rate; at most 50 Optuna trials per horizon; the test window was evaluated once with frozen configs.

| Horizon | CV win rate (coverage) | Test win rate (coverage) | Test baseline | Test edge [95% CI] |
|---|---|---|---|---|
| 1 day | 80.5% (8.0%) | 75.7% (5.2%) | 56.8% | +18.9 pp [+14.5, +22.0] |
| 1 week | not reachable on CV | nothing published | – | – |
| 2 weeks | 80.9% (0.8%) | 71.1% (5.3%) | 60.9% | +10.2 pp [+2.9, +16.6] |
| 1 month | 82.8% (1.0%) | 73.2% (9.2%) | 62.2% | +11.0 pp [−3.3, +20.3] |

- Test: 2024-09-25 to 2026-11-09, 58 stocks. CV: combinatorial purged CV (6 groups, 15 splits, purge + embargo = h) on earlier data.
- Every published target is at the smallest distance, k = 0.5 (0.6% to 2.2% from the close on test). Larger k never reached 80% on CV.
- The model beats the same-distance baseline (CI above zero at 1 day and 2 weeks), but by less than the 80% bar needs.
- CV win rates are optimistic: thresholds were picked on the same out-of-fold rows. On test, coverage rose and win rates fell.
- "Close within ±X of the predicted price" and "close inside the predicted range" have no edge over same-distance or same-width baselines.
- A forward test now logs each day's targets before the outcome is known (`scripts/targets_forward.py`).

Code: `scripts/targets_run.py`, `src/targets/`. Full tables: `results/targets/RESULTS.md`.

### Pipeline Latency Benchmark

Measured with `scripts/benchmark_pipeline.py` (cached offline run on 2026-10-27; Intel64 Family 6 Model 142 Stepping 9, GenuineIntel, 4 logical CPUs, Windows 10):

| Path | Samples (n) | Median p50 (s) | p95 (s) |
|---|---|---|---|
| Cold (model cache empty) | 5 | 0.41 s | 0.67 s |
| Warm (model cache populated) | 5 | 0.20 s | 0.21 s |

Median speed-up from caching: 2.0x. Raw data saved in `results/benchmark_pipeline.json` and latest latency run.

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

---

<div align="center"><sub>Built by <a href="https://github.com/Mokshitsharma">Mokshit Sharma</a> · <a href="https://mokshitsharma27.vercel.app/">Portfolio</a> · <a href="https://www.linkedin.com/in/mokshit-sharma-75b5ab305">LinkedIn</a></sub></div>
