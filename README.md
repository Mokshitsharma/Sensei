# 📊 Sensei — Indian Stock Intelligence Platform

Sensei is an **Indian equity analysis and decision-support platform** focused on **NIFTY 50 stocks**.  
It combines **price action, technical indicators, chart patterns, fundamentals, intraday scalping logic, and backtesting** to generate **explainable BUY / HOLD / SELL signals**.

> ⚠️ **Disclaimer**  
> This project is for **educational and analytical purposes only**.  
> It is **not** a live trading system and does **not** provide financial advice.

---

## ✨ Key Highlights

- 🇮🇳 Indian market focused (NIFTY 50 only)
- 📈 TradingView-style interactive charts (Lightweight Charts)
- ⚡ Intraday scalping (5m / 15m)
- 🧠 Explainable swing trading signals with confidence
- 🧪 Built-in historical backtesting
- 🧩 Clean, modular, production-grade architecture

---

## 🧠 What Does Sensei Do?

Sensei answers one core question:

> **“Based on price, indicators, patterns, and fundamentals — should I BUY, HOLD, or SELL this stock?”**

The system is designed as a **decision-support tool**, not an automated trading bot.

---

## 🏗 How the System Works

### 1️⃣ Data Layer
- Fetches **historical and intraday price data**
- Fetches **fundamental metrics** (PE, EPS, ROE)
- Normalizes NSE symbols automatically

### 2️⃣ Analysis (Domain Layer)
- Technical indicators: EMA, RSI, MACD
- Pattern detection: Golden Cross, Death Cross, Breakouts
- Intraday scalping logic (Entry, Stop Loss, Target)
- Swing trading signal scoring with confidence
- Historical backtesting with performance metrics

### 3️⃣ Visualization Layer
- Candlestick charts using **Lightweight Charts**
- EMA overlays
- Timeframe-aware rendering (daily vs intraday)
- Streamlit-based UI

---

## 🧱 Project Architecture

```

Sensei/
├── app.py                     # Streamlit entry point
├── src/
│   ├── data/
│   │   ├── prices.py           # Price data loader
│   │   └── nifty50.py          # NIFTY 50 universe
│   ├── domain/
│   │   ├── indicators.py       # RSI, MACD, EMA
│   │   ├── patterns.py         # Chart pattern detection
│   │   ├── signals.py          # BUY / HOLD / SELL logic
│   │   ├── intraday.py         # Intraday scalping logic
│   │   ├── backtest.py         # Strategy backtesting
│   │   └── fundamentals.py     # PE, EPS, ROE
│   └── charts/
│       └── lightweight.py      # Chart rendering
└── README.md

```

**Design principle:** clean separation of concerns  
`data → domain → charts → UI`

---

## 📈 Features

### 📊 Price & Technical Analysis
- Candlestick charts
- EMA (20 / 50)
- RSI
- MACD
- Golden Cross / Death Cross
- Breakout detection

### ⚡ Intraday Scalping (5m / 15m)
- Real-time intraday candles
- Entry price
- Stop loss
- Target price
- No-trade filtering

### 🧠 Swing Trading Signals
- BUY / HOLD / SELL
- Confidence score
- Explainable reasoning behind signals

### 🧪 Backtesting
- Historical signal evaluation
- Win rate, return %, max drawdown
- Equity curve logic
- CSV export for analysis

### 📊 Fundamentals (Indian Stocks)
- PE Ratio
- EPS
- ROE

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| UI | Streamlit |
| Charts | Lightweight Charts |
| Data | yfinance |
| Analysis | Pandas, NumPy |
| Architecture | Modular / Layered |

---

## ▶️ How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/sensei.git
cd sensei
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

The app will be available at:

```
http://localhost:8501
```

---

## 🧭 How to Use the App

1. Select a **NIFTY 50 stock**
2. Choose a **timeframe**

   * Intraday: `5m`, `15m`
   * Swing: `6mo`, `1y`, `2y`, `5y`
3. Analyze:

   * Price & EMA trends
   * Intraday setups (if applicable)
   * Swing signal, confidence & reasoning
   * Backtest performance

---

## 🎯 Use Cases

* Learning technical analysis
* Practicing quant & trading logic
* Stock research & screening
* Portfolio / interview project
* Strategy experimentation (non-live)

---

## 🔮 Future Improvements

* Replace yfinance with broker-grade APIs (Zerodha Kite, TrueData)
* Market-hours & holiday awareness
* Slippage & brokerage modeling
* Multi-timeframe confirmation
* Portfolio-level position sizing
* Trade logging & persistence
* Alerts (Telegram / Email)
* Authentication & SaaS deployment

---

## 📌 Final Notes

Sensei is designed to demonstrate:

* Strong Python engineering
* Clean system design
* Practical understanding of financial markets
* Explainable analytics over black-box signals

If you find this project useful, consider ⭐ starring the repository."# Sensei1" 
