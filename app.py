# app.py

import streamlit as st

from src.data.nifty50 import NIFTY_50
from src.data.prices import load_prices
from src.domain.fundamentals import load_fundamentals
from src.data.news import get_news_signal

from src.pipeline.signal_pipeline import run_signal_pipeline
from src.pipeline.decision_engine import make_final_decision

from src.charts.lightweight import render_price_chart
from src.utils.logger import get_logger
from src.utils.config import (
    DEFAULT_TIMEFRAME,
    FEATURE_COLUMNS,
)

# -----------------------------
# Setup
# -----------------------------
st.set_page_config(
    page_title="AI Trading System",
    layout="wide",
)

logger = get_logger("app")

st.title("📈 AI-Driven Trading System")
st.caption(
    "ML + LSTM + Temporal CNN + Regime Detection + PPO + News NLP"
)

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Configuration")

company = st.sidebar.selectbox(
    "Select NIFTY 50 Stock",
    list(NIFTY_50.keys()),
)

ticker = NIFTY_50[company]

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["5m", "15m", "30m", "1h", "1y", "2y", "5y"],
    index=4,
)
if timeframe.endswith(("m", "h")):
    st.warning("Intraday NSE data is limited. Results may be approximate.")

run_button = st.sidebar.button("Run Analysis 🚀")

# -----------------------------
# Main Execution
# -----------------------------
if run_button:
    try:
        logger.info(f"Running analysis for {ticker}")

        with st.spinner("Loading price data..."):
            price_df = load_prices(ticker, timeframe)

        with st.spinner("Loading fundamentals..."):
            fundamentals = load_fundamentals(ticker)

        with st.spinner("Fetching news..."):
            news = get_news_signal(company)

        with st.spinner("Running AI models..."):
            signals = run_signal_pipeline(
                price_df=price_df,
                fundamentals=fundamentals,
                company=company,
                lstm_model_path="models/lstm.pt",
                tcn_model_path="models/tcn.pt",
                ppo_model_path="models/ppo.zip",
            )

        decision = make_final_decision(
            signals=signals,
            news_sentiment=news["sentiment_score"],
        )

        # -----------------------------
        # Layout
        # -----------------------------
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Price Chart")
            render_price_chart(price_df)

        with col2:
            st.subheader("Final Decision")

            st.metric(
                "Action",
                decision["action"],
                f"Confidence: {decision['confidence'] * 100:.1f}%",
            )

            st.write("### Model Signals")
            st.write(f"**ML Probability (Up):** {signals['ml_prob_up']:.2f}")
            st.write(f"**LSTM 5-Day Return:** {signals['lstm_return']:.3f}")
            st.write(f"**TCN 5-Day Return:** {signals['tcn_return']:.3f}")
            st.write(f"**Market Regime:** {signals['regime']}")
            st.write(f"**PPO Action:** {signals['ppo_action']}")

            st.write("### Fundamentals")
            st.write(f"Current Price: ₹{fundamentals['current_price']}")
            st.write(f"Market Cap: {fundamentals['market_cap']:,}")
            st.write(f"ROE: {fundamentals['roe']:.2f}")
            st.write(
                f"52-Week Range: "
                f"{fundamentals['52_week_low']} – {fundamentals['52_week_high']}"
            )

            st.write("### News Impact")
            st.write(news["summary"])
            st.write(f"Sentiment Score: {news['sentiment_score']}")

            st.write("### Explanation")
            st.write(decision["explanation"])

    except Exception as e:
        logger.exception("App error")
        st.error(str(e))

else:
    st.info("Select a stock and click **Run Analysis**")