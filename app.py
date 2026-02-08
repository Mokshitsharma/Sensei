# app.py
import streamlit as st

from src.data.prices import load_prices
from src.data.nifty50 import NIFTY_50
from src.domain.fundamentals import get_fundamentals
from src.domain.indicators import add_indicators
from src.domain.patterns import detect_patterns
from src.domain.signals import generate_signal
from src.domain.intraday import intraday_scalping_signal
from src.domain.backtest import backtest_strategy
from src.charts.lightweight import render_price_chart


st.set_page_config(
    page_title="Sensei — Indian Stock Intelligence",
    page_icon="📈",
    layout="wide",
)

st.title("📊 Sensei — Indian Stock Intelligence")
st.warning(
    "This is an educational analytics project, not financial advice or live trading signals."
)

with st.sidebar:
    company = st.selectbox(
        "NIFTY 50 Stock",
        list(NIFTY_50.keys()),
        index=list(NIFTY_50.keys()).index("TCS"),
    )

    timeframe = st.selectbox(
        "Timeframe",
        ["5m", "15m", "6mo", "1y", "2y", "5y"],
        index=3,
    )

ticker = NIFTY_50[company]

# ---------------------------
# Data
# ---------------------------
price_df = add_indicators(load_prices(ticker, timeframe))
fundamentals = get_fundamentals(ticker)
patterns = detect_patterns(price_df)
latest = price_df.iloc[-1]

# ---------------------------
# Fundamentals (ALWAYS SHOWN)
# ---------------------------
st.subheader(f"📊 Fundamentals — {company}")

c1, c2, c3 = st.columns(3)
c1.metric("PE", fundamentals["pe"] or "—")
c2.metric("EPS", fundamentals["eps"] or "—")
c3.metric("ROE %", fundamentals["roe"] or "—")

# ---------------------------
# Chart (ALWAYS SHOWN)
# ---------------------------
st.subheader("📈 Price & Trend")
render_price_chart(price_df, patterns)

# ---------------------------
# INTRADAY MODE
# ---------------------------
if timeframe in ("5m", "15m"):
    st.subheader("⚡ Intraday Scalping Setup")

    scalp = intraday_scalping_signal(latest)

    if scalp["signal"] == "NO TRADE":
        st.info("No intraday setup right now.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Signal", scalp["signal"])
        c2.metric("Entry", f"₹{scalp['entry']}")
        c3.metric("Target", f"₹{scalp['target']}")
        c4.metric("Stop Loss", f"₹{scalp['stop_loss']}")

# ---------------------------
# SWING MODE
# ---------------------------
else:
    st.subheader("🧠 Swing Signal")

    signal_data = generate_signal(
        indicators={
            "rsi": latest["rsi"],
            "macd": latest["macd"],
            "macd_signal": latest["macd_signal"],
        },
        patterns=patterns,
        fundamentals=fundamentals,
    )

    st.metric("Signal", signal_data["signal"])
    st.metric("Confidence", f"{signal_data['confidence']}%")

    for r in signal_data["reasons"]:
        st.success(r)

    st.subheader("🧪 Backtest Performance")

    stats = backtest_strategy(price_df, fundamentals)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades", stats["trades"])
    c2.metric("Win Rate", f"{stats['win_rate']}%")
    c3.metric("Return", f"{stats['return_pct']}%")
    c4.metric("Max Drawdown", f"{stats['max_drawdown']}%")
