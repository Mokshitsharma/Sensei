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
from src.domain.setup_engine import build_intraday_setup, build_swing_setup
from src.domain.support_resistance import get_support_resistance
from src.domain.news_price_model import predict_news_price_impact
from src.domain.setup_engine import _daily_atr


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


# =====================================================
# HELPER: Render a trade setup card
# =====================================================

def _render_setup_card(setup: dict, company: str):
    if setup.get("error") and setup["entry_zone"] == (0, 0):
        st.warning(setup["plan"])
        return

    bias = setup.get("bias", "NEUTRAL")
    bias_color = {"BULLISH": "#00b386", "BEARISH": "#ff4d4f"}.get(bias, "#f0a500")
    bias_bg    = {"BULLISH": "#e6fff7", "BEARISH": "#fff0f0"}.get(bias, "#fffbe6")

    # Bias banner
    st.markdown(
        f"""<div style="background:{bias_bg};border-left:4px solid {bias_color};
        padding:10px 16px;border-radius:6px;margin-bottom:12px;">
        <span style="font-weight:600;color:{bias_color};font-size:16px;">{bias}</span>
        &nbsp;<span style="color:#888;font-size:13px;">— {setup.get('pattern','—')}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # Entry / SL / Targets row
    c1, c2, c3, c4, c5 = st.columns(5)
    entry_low, entry_high = setup["entry_zone"]
    c1.metric("Entry Zone", f"₹{entry_low:.2f}–{entry_high:.2f}")
    c2.metric("Stop Loss",  f"₹{setup['stop_loss']:.2f}")
    c3.metric("Target 1",   f"₹{setup['target_1']:.2f}")
    c4.metric("Target 2",   f"₹{setup['target_2']:.2f}")
    rr = setup["risk_reward"]
    rr_color = "normal" if rr >= 2 else "inverse"
    c5.metric("Risk / Reward", f"{rr:.1f}x", delta=None)

    # Key levels
    if setup.get("key_levels"):
        st.markdown("**Key levels**")
        level_cols = st.columns(len(setup["key_levels"]))
        for idx, (label, val) in enumerate(setup["key_levels"].items()):
            level_cols[idx].metric(label, f"₹{val:,.2f}")

    # Trade plan
    st.markdown("---")
    st.markdown(f"**Trade plan** — valid: _{setup.get('validity', '—')}_")
    st.info(setup["plan"])


if run:

    with st.spinner("Running AI engine..."):
        price_df = load_prices(ticker, timeframe)
        fundamentals = load_fundamentals(ticker)
        news = get_news_signal(company, ticker=ticker, max_items=10)

        # Load intraday data for trade setup (best-effort)
        try:
            from src.data.providers.yahoo import YahooProvider
            _yahoo = YahooProvider()
            intraday_df = _yahoo.fetch_intraday_ohlcv(ticker, interval="15m", lookback_days=5)
        except Exception:
            intraday_df = pd.DataFrame()

        signals = run_signal_pipeline(
            price_df=price_df,
            fundamentals=fundamentals,
            company=company,
            lstm_model_path="models/lstm_HDFCBANK_NS.pt",
            tcn_model_path="models/tcn_HDFCBANK_NS.pt",
            ppo_model_path="models/ppo_hdfc.zip",
        )

        # Support & Resistance
        sr_data = get_support_resistance(price_df)

        # News-driven price prediction
        try:
            _atr_val = _daily_atr(price_df)
        except Exception:
            _atr_val = float(price_df["close"].std()) * 0.1
        news_price_pred = predict_news_price_impact(
            current_price=fundamentals.get("current_price", float(price_df["close"].iloc[-1])),
            news_result=news,
            atr=_atr_val,
            horizon="3d",
        )

        decision = make_final_decision(
            signals=signals,
            news_sentiment=news["sentiment_score"],
            shap_values=signals.get("shap_values"),
            feature_values=signals.get("feature_values"),
            company=company,
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

    # =====================================================
    # AI ANALYST REPORT
    # =====================================================

    st.subheader("🧠 AI Analyst Report")

    narrative = decision.get("narrative", {})

    if narrative:

        # Headline verdict
        action_colors = {"BUY": "#00b386", "SELL": "#ff4d4f", "HOLD": "#f0a500"}
        headline_color = action_colors.get(action, "#333333")

        st.markdown(
            f"<h4 style='color:{headline_color};'>{narrative.get('headline','')}</h4>",
            unsafe_allow_html=True,
        )

        # Tabbed report
        tab_trend, tab_momentum, tab_vol, tab_ai, tab_shap, tab_news = st.tabs([
            "📈 Trend", "⚡ Momentum", "🌊 Volatility",
            "🤖 AI Models", "🔍 SHAP Drivers", "📰 News"
        ])

        with tab_trend:
            st.markdown(narrative.get("trend", "—"))

        with tab_momentum:
            st.markdown(narrative.get("momentum", "—"))

        with tab_vol:
            st.markdown(narrative.get("volatility", "—"))

        with tab_ai:
            st.markdown(narrative.get("ai_models", "—"))

        with tab_shap:
            shap_ranked = decision.get("shap_ranked", [])

            if shap_ranked:
                import matplotlib
                import matplotlib.pyplot as plt
                matplotlib.use("Agg")

                _LABELS = {
                    "rsi_norm":      "RSI",
                    "ema_spread":    "EMA Crossover",
                    "macd_diff":     "MACD Divergence",
                    "atr_pct":       "ATR (Volatility)",
                    "volatility_10": "10-day Volatility",
                    "return_1":      "1-day Return",
                    "return_5":      "5-day Return",
                    "return_10":     "10-day Return",
                    "range_position":"Price Range Position",
                }

                features = [_LABELS.get(r["feature"], r["feature"]) for r in shap_ranked]
                values   = [r["shap"] for r in shap_ranked]
                colors   = ["#00b386" if v > 0 else "#ff4d4f" for v in values]

                fig, ax = plt.subplots(figsize=(8, max(3, len(features) * 0.5)))
                ax.barh(features[::-1], values[::-1], color=colors[::-1])
                ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
                ax.set_xlabel("SHAP Value  (positive = bullish, negative = bearish)")
                ax.set_title(f"Feature Impact on ML Prediction — {company}", fontsize=12, pad=10)
                ax.tick_params(axis="y", labelsize=9)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.markdown("---")

            st.markdown(narrative.get("shap_story", "SHAP data not available."))

        with tab_news:
            st.markdown(narrative.get("news", "—"))

        st.markdown("---")
        st.markdown("**Summary**")
        st.info(narrative.get("summary", "—"))

    else:
        st.info(decision.get("explanation", "No explanation available."))

    with st.expander("🔧 Raw Model Signals"):
        raw_signals = {
            "ML Prob UP":     f"{signals['ml_prob_up']*100:.1f}%",
            "LSTM Forecast":  f"{signals['lstm_return']*100:.3f}%",
            "TCN Forecast":   f"{signals['tcn_return']*100:.3f}%",
            "HMM Regime":     signals["regime"],
            "PPO Action":     signals["ppo_action"],
            "Ensemble Score": f"{decision['score']:.1f} / 5.0",
        }
        st.table(pd.DataFrame.from_dict(raw_signals, orient="index", columns=["Value"]))

    # =====================================================
    # NEWS INTELLIGENCE
    # =====================================================

    st.markdown("---")
    st.subheader("📰 News Intelligence")

    news_details = news.get("details", [])
    bull_c = news.get("bull_count", 0)
    bear_c = news.get("bear_count", 0)
    neut_c = news.get("neutral_count", 0)
    total_c = bull_c + bear_c + neut_c

    if total_c > 0:
        # Sentiment summary bar
        n1, n2, n3, n4 = st.columns(4)
        w_score = news.get("weighted_score", 0)
        score_color = "#00b386" if w_score > 0.1 else "#ff4d4f" if w_score < -0.1 else "#f0a500"
        n1.metric("Weighted Score", f"{w_score:+.3f}", help="Impact-weighted sentiment (-1 to +1)")
        n2.metric("Bullish Headlines", f"{bull_c}", delta=None)
        n3.metric("Bearish Headlines", f"{bear_c}", delta=None)
        n4.metric("Neutral Headlines", f"{neut_c}", delta=None)

        st.markdown(f"**Overall:** {news.get('summary', '')}")

        # Top bullish / top bearish
        col_bull, col_bear = st.columns(2)
        with col_bull:
            top_b = news.get("top_bullish")
            if top_b:
                st.markdown(
                    f"""<div style="background:#e6fff7;border-left:4px solid #00b386;
                    padding:10px 14px;border-radius:6px;">
                    <div style="color:#00b386;font-weight:600;font-size:12px;margin-bottom:4px;">
                    TOP BULLISH — {top_b.get('impact_type','')}</div>
                    <div style="font-size:13px;">{top_b['headline']}</div>
                    <div style="color:#888;font-size:11px;margin-top:4px;">{top_b.get('source','')} · {top_b.get('published','')}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        with col_bear:
            top_br = news.get("top_bearish")
            if top_br:
                st.markdown(
                    f"""<div style="background:#fff0f0;border-left:4px solid #ff4d4f;
                    padding:10px 14px;border-radius:6px;">
                    <div style="color:#ff4d4f;font-weight:600;font-size:12px;margin-bottom:4px;">
                    TOP BEARISH — {top_br.get('impact_type','')}</div>
                    <div style="font-size:13px;">{top_br['headline']}</div>
                    <div style="color:#888;font-size:11px;margin-top:4px;">{top_br.get('source','')} · {top_br.get('published','')}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # All headlines table
        with st.expander(f"All {total_c} headlines"):
            label_colors = {"POSITIVE": "#00b386", "NEGATIVE": "#ff4d4f", "NEUTRAL": "#f0a500"}
            impact_icons = {
                "Earnings": "💰", "Regulatory": "⚖️", "Management": "👔",
                "Macro": "🌍", "Sector": "🏭", "General": "📄",
            }
            for item in news_details:
                lc = label_colors.get(item["label"], "#888")
                icon = impact_icons.get(item.get("impact_type", "General"), "📄")
                url = item.get("url", "")
                link = f'<a href="{url}" target="_blank" style="color:#4a90e2;">↗</a>' if url else ""
                st.markdown(
                    f"""<div style="border-bottom:1px solid #eee;padding:8px 0;">
                    <span style="background:{lc};color:white;border-radius:3px;
                    padding:1px 6px;font-size:11px;font-weight:600;">{item['label']}</span>
                    &nbsp;<span style="font-size:11px;color:#888;">{icon} {item.get('impact_type','')}
                    · conf {item['confidence']:.0%}</span><br>
                    <span style="font-size:13px;">{item['headline']}</span>
                    &nbsp;{link}<br>
                    <span style="color:#aaa;font-size:11px;">{item.get('source','')} · {item.get('published','')}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # News-driven price prediction
        st.markdown("---")
        st.markdown("**📊 News-driven Price Forecast**")
        pred = news_price_pred
        pred_dir = pred.get("direction", "FLAT")
        pred_color = {"UP": "#00b386", "DOWN": "#ff4d4f"}.get(pred_dir, "#f0a500")
        conf_label = pred.get("confidence", "LOW")
        conf_badge = {"HIGH": "🟢 High", "MEDIUM": "🟡 Medium", "LOW": "🔴 Low"}.get(conf_label, conf_label)

        pc1, pc2, pc3, pc4, pc5 = st.columns(5)
        pc1.metric("Direction", pred_dir)
        pc2.metric("Predicted Price", f"₹{pred.get('predicted_price',0):,.2f}")
        pc3.metric("Range Low",  f"₹{pred.get('price_low',0):,.2f}")
        pc4.metric("Range High", f"₹{pred.get('price_high',0):,.2f}")
        pc5.metric("Expected Move", f"{pred.get('expected_move_pct',0):+.2f}%")

        st.caption(f"Confidence: {conf_badge} · Horizon: {pred.get('horizon_label','')}")
        st.info(pred.get("explanation", ""))

    else:
        st.info("No news found for this stock.")

    # =====================================================
    # SUPPORT & RESISTANCE
    # =====================================================

    st.markdown("---")
    st.subheader("📊 Support & Resistance Levels")

    sr = sr_data
    supports    = sr.get("supports", [])
    resistances = sr.get("resistances", [])
    pivot_data  = sr.get("pivot_data", {})

    strength_colors = {"Strong": "#00b386", "Moderate": "#f0a500", "Weak": "#aaaaaa"}
    strength_bg     = {"Strong": "#e6fff7", "Moderate": "#fffbe6", "Weak": "#f5f5f5"}

    sr_col1, sr_col2 = st.columns(2)

    with sr_col1:
        st.markdown("**🛡️ Support Levels** (nearest first)")
        for i, lv in enumerate(supports, 1):
            sc = strength_colors.get(lv["strength"], "#888")
            sb = strength_bg.get(lv["strength"], "#f5f5f5")
            methods_str = ", ".join(lv.get("methods", [])[:3])
            st.markdown(
                f"""<div style="background:{sb};border-left:4px solid {sc};
                padding:10px 14px;border-radius:6px;margin-bottom:8px;">
                <span style="font-weight:700;font-size:15px;">S{i}: ₹{lv['price']:,.2f}</span>
                &nbsp;<span style="background:{sc};color:white;border-radius:3px;
                padding:1px 6px;font-size:11px;">{lv['strength']}</span><br>
                <span style="color:#888;font-size:11px;">{methods_str} · {lv.get('touches',0)} touches</span>
                </div>""",
                unsafe_allow_html=True,
            )

    with sr_col2:
        st.markdown("**🔺 Resistance Levels** (nearest first)")
        for i, lv in enumerate(resistances, 1):
            rc = strength_colors.get(lv["strength"], "#888")
            rb = strength_bg.get(lv["strength"], "#f5f5f5")
            methods_str = ", ".join(lv.get("methods", [])[:3])
            st.markdown(
                f"""<div style="background:{rb};border-left:4px solid {rc};
                padding:10px 14px;border-radius:6px;margin-bottom:8px;">
                <span style="font-weight:700;font-size:15px;">R{i}: ₹{lv['price']:,.2f}</span>
                &nbsp;<span style="background:{rc};color:white;border-radius:3px;
                padding:1px 6px;font-size:11px;">{lv['strength']}</span><br>
                <span style="color:#888;font-size:11px;">{methods_str} · {lv.get('touches',0)} touches</span>
                </div>""",
                unsafe_allow_html=True,
            )

    # Pivot table
    if pivot_data:
        with st.expander("Classic Pivot Points (previous session)"):
            
            piv_df = pd.DataFrame.from_dict(pivot_data, orient="index", columns=["Price (₹)"])
            piv_df["Price (₹)"] = piv_df["Price (₹)"].map(lambda x: f"₹{x:,.2f}")
            st.table(piv_df)

    # =====================================================
    # TRADE SETUP
    # =====================================================

    st.markdown("---")
    st.subheader("📐 Trade Setup")

    setup_tab_intra, setup_tab_swing = st.tabs(["⚡ Intraday (15-min)", "📅 Swing (Daily)"])

    with setup_tab_intra:
        if not intraday_df.empty:
            intra_setup = build_intraday_setup(intraday_df, price_df, ticker)
        else:
            intra_setup = {"error": "Intraday data unavailable", "mode": "Intraday",
                           "bias": "NEUTRAL", "entry_zone": (0,0), "stop_loss": 0,
                           "target_1": 0, "target_2": 0, "risk_reward": 0,
                           "pattern": "—", "key_levels": {}, "validity": "—",
                           "plan": "Intraday data could not be loaded."}

        _render_setup_card(intra_setup, company)

    with setup_tab_swing:
        swing_setup = build_swing_setup(price_df, ticker)
        _render_setup_card(swing_setup, company)

else:
    st.info("Select a stock and click Run Analysis")