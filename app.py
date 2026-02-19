import streamlit as st
import pandas as pd

from src.data.nifty50 import NIFTY_50
from src.data.prices import load_prices
from src.domain.fundamentals import load_fundamentals
from src.data.news import get_news_signal
from src.pipeline.signal_pipeline import run_signal_pipeline
from src.pipeline.decision_engine import make_final_decision
from src.backtest.engine import run_backtest
from src.backtest.metrics import calculate_metrics
from src.charts.lightweight import render_price_chart


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Sensei AI",
    layout="wide",
)

# =====================================================
# CUSTOM LIGHT CSS (Groww Style)
# =====================================================

st.markdown("""
<style>
.main {
    background-color: #ffffff;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.metric-card {
    padding: 20px;
    border-radius: 12px;
    background-color:#0a2a47;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}

.buy { color: #00b386; font-weight: 600; }
.sell { color: #ff4d4f; font-weight: 600; }
.hold { color: #999999; font-weight: 600; }

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.title("Sensei AI Trading Dashboard")
st.caption("AI-powered market intelligence")

st.markdown("---")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Market Selection")

company = st.sidebar.selectbox(
    "Choose Stock",
    list(NIFTY_50.keys())
)

ticker = NIFTY_50[company]

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["1y", "2y", "5y"]
)

run = st.sidebar.button("Run Analysis")

# =====================================================
# MAIN LOGIC
# =====================================================

if run:

    with st.spinner("Running AI engine..."):
        price_df = load_prices(ticker, timeframe)
        fundamentals = load_fundamentals(ticker)
        news = get_news_signal(company)

        signals = run_signal_pipeline(
            price_df=price_df,
            fundamentals=fundamentals,
            company=company,
            lstm_model_path="models/lstm_HDFCBANK_NS.pt",
            tcn_model_path="models/tcn_HDFCBANK_NS.pt",
            ppo_model_path="models/ppo_hdfc.zip",
        )

        decision = make_final_decision(
            signals=signals,
            news_sentiment=news["sentiment_score"],
        )

    # =====================================================
    # DECISION CARD (Centered)
    # =====================================================

    action = decision["action"]

    if action == "BUY":
        css_class = "buy"
    elif action == "SELL":
        css_class = "sell"
    else:
        css_class = "hold"

    st.markdown(
        f"""
        <div class="metric-card">
            <h2>Final Decision: <span class="{css_class}">{action}</span></h2>
            <p>Confidence: {decision['confidence']*100:.1f}%</p>
        </div>
        """,
        unsafe_allow_html=True
        # colour of final decision written in box is white for fial decision, changing it to black colour
        
    )

    st.markdown("---")

    # =====================================================
    # PRICE CHART (Full Width)
    # =====================================================

    # =========================
    # STOCK HEADER (Broker Style)
    # =========================

    current_price = fundamentals["current_price"]

    price_color = "#09b300" if signals["regime"] == "BULL" else "#ff4d4f"

    st.markdown(
        f"""
        <div style="margin-bottom:10px;">
            <h2 style="margin:0;">{company}</h2>
            <h3 style="margin:0; color:{price_color};">
                ₹ {current_price:,.2f}
            </h3>
        </div>
    """,
    unsafe_allow_html=True
)

    st.markdown("---")

    st.subheader("Price Chart")
    render_price_chart(price_df)


    st.markdown("---")

    # =====================================================
    # KEY METRICS ROW
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("ML Prob (UP)", f"{signals['ml_prob_up']:.2f}")
    col2.metric("LSTM Return", f"{signals['lstm_return']:.3f}")
    col3.metric("TCN Return", f"{signals['tcn_return']:.3f}")
    col4.metric("Market Regime", signals["regime"])

    st.markdown("---")

    # =========================
    # FUNDAMENTALS
    # =========================

    st.subheader("Company Overview")

    f1, f2, f3, f4 = st.columns(4)

    f1.metric("Current Price", f"₹{fundamentals['current_price']}")
    market_cap_cr = fundamentals["market_cap"] / 1e7
    f2.metric(
        "Market Cap",
        f"₹ {market_cap_cr:,.0f} Cr"
    )

    f3.metric("ROE", f"{fundamentals['roe']:.2f}")
    f4.metric(
        "52W Range",
        f"{fundamentals['52_week_low']} – {fundamentals['52_week_high']}"
)


    st.markdown("---")

    # =====================================================
    # BACKTEST
    # =====================================================

    st.subheader("Strategy Backtest")

    signals_series = pd.Series(
        ["BUY" if x > 0 else "SELL"
         for x in price_df["close"].pct_change().fillna(0)]
    )

    backtest_df = run_backtest(price_df, signals_series)
    metrics = calculate_metrics(backtest_df["equity"])

    st.line_chart(backtest_df["equity"])

    b1, b2, b3 = st.columns(3)
    b1.metric("Total Return", f"{metrics['total_return']*100:.2f}%")
    b2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    b3.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")

    st.markdown("---")

    st.subheader("AI Explanation")
    st.info(decision["explanation"])

else:
    st.info("Select a stock and click Run Analysis")