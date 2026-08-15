import os

# torch and scikit-learn/numpy each bundle their own Intel OpenMP runtime on
# Windows; loading both in one process aborts the process with no Python
# traceback ("OMP: Error #15"). Must be set before those libraries import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import src.utils.numpy_compat  # noqa: F401  (must run before any model unpickling)

import streamlit as st
import pandas as pd

from src.data.nifty50 import NIFTY_50
from src.data.prices import load_prices
from src.domain.fundamentals import load_fundamentals
from src.data.news import get_news_signal
from src.data.providers.yahoo import YahooProvider
from src.pipeline.signal_pipeline import run_signal_pipeline
from src.pipeline.decision_engine import make_final_decision
from src.backtest.engine import run_backtest
from src.backtest.metrics import calculate_metrics
from src.charts.lightweight import render_price_chart
from src.domain.setup_engine import build_intraday_setup, build_swing_setup, _daily_atr
from src.domain.support_resistance import get_support_resistance
from src.domain.news_price_model import predict_news_price_impact


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Sensei AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

GREEN = "#00b386"
RED = "#ff4d4f"
AMBER = "#f0a500"
PURPLE = "#5B2EFF"
PURPLE_DARK = "#4321C9"
TEXT_MUTED = "#6b7280"
BORDER = "#eef0f2"

POPULAR_STOCKS = list(NIFTY_50.items())[:8]

# =====================================================
# GLOBAL THEME (Groww-inspired)
# =====================================================

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header[data-testid="stHeader"] {{background: transparent; height: 0;}}
[data-testid="stSidebar"] {{width: 0px !important; min-width: 0px !important;}}
[data-testid="collapsedControl"] {{display: none;}}
[data-testid="stToolbar"] {{visibility: hidden;}}

.main {{
    background-color: #ffffff;
}}

.block-container {{
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1300px;
}}

/* Native metric widgets restyled as clean cards */
div[data-testid="stMetric"] {{
    background: #ffffff;
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 14px 16px;
}}
div[data-testid="stMetricLabel"] {{
    font-size: 12px;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: .03em;
    font-weight: 600;
}}
div[data-testid="stMetricValue"] {{
    font-size: 20px;
    font-weight: 700;
    color: #111827;
}}

hr {{ border-color: {BORDER}; margin: 1.6rem 0; }}

h2, h3 {{ color: #111827; font-weight: 700; letter-spacing: -0.01em; }}

.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    height: 40px;
    border-radius: 8px 8px 0 0;
    color: {TEXT_MUTED};
    font-weight: 600;
    font-size: 14px;
}}
.stTabs [aria-selected="true"] {{
    color: {PURPLE} !important;
    border-bottom: 2px solid {PURPLE} !important;
}}

.stButton>button {{
    background: {PURPLE};
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
}}
.stButton>button:hover {{
    background: {PURPLE_DARK};
    color: white;
}}

/* Stock picker — pill-shaped search bar look */
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
    border-radius: 24px !important;
    border: 1px solid {BORDER} !important;
    background: #f7f7fb !important;
    font-size: 15px;
}}
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {{
    border-color: {PURPLE} !important;
}}

/* Openable bars (st.expander) — clean card look */
div[data-testid="stExpander"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    box-shadow: none !important;
    margin-bottom: 12px;
    overflow: hidden;
}}
div[data-testid="stExpander"] summary {{
    font-weight: 700 !important;
    font-size: 15px !important;
    color: #111827 !important;
    padding: 14px 16px !important;
}}

/* Bordered containers (stock cards) */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 12px !important;
    transition: box-shadow .15s ease, border-color .15s ease;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    border-color: {PURPLE} !important;
    box-shadow: 0 4px 14px rgba(91,46,255,0.12);
}}

.buy {{ color: {GREEN}; font-weight: 700; }}
.sell {{ color: {RED}; font-weight: 700; }}
.hold {{ color: {AMBER}; font-weight: 700; }}
</style>
""", unsafe_allow_html=True)


# =====================================================
# CACHED DATA / COMPUTE LAYER
# (keeps the app fast — reruns triggered by tab clicks or
# widget interaction hit cache instead of refetching data /
# reloading ML models from disk)
# =====================================================

@st.cache_data(ttl=300, show_spinner=False)
def _cached_prices(ticker: str, timeframe: str):
    return load_prices(ticker, timeframe)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_fundamentals(ticker: str):
    return load_fundamentals(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_news(company: str, ticker: str):
    return get_news_signal(company, ticker=ticker, max_items=10)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_intraday(ticker: str):
    try:
        return YahooProvider().fetch_intraday_ohlcv(ticker, interval="15m", lookback_days=5)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def _cached_support_resistance(price_df: pd.DataFrame):
    return get_support_resistance(price_df)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_signal_pipeline(ticker: str, company: str, price_df: pd.DataFrame, fundamentals: dict):
    return run_signal_pipeline(
        price_df=price_df,
        fundamentals=fundamentals,
        company=company,
        lstm_model_path="models/lstm_HDFCBANK_NS.pt",
        tcn_model_path="models/tcn_HDFCBANK_NS.pt",
        ppo_model_path="models/ppo_hdfc.zip",
    )


@st.cache_data(ttl=120, show_spinner=False)
def _cached_index_quote(symbol: str):
    try:
        df = YahooProvider().fetch_daily_ohlcv(symbol)
        if df.empty or len(df) < 2:
            return None
        last = float(df["close"].iloc[-1])
        prev = float(df["close"].iloc[-2])
        change = last - prev
        pct = (change / prev * 100) if prev else 0.0
        return {"value": last, "change": change, "pct": pct}
    except Exception:
        return None


_cached_stock_quote = _cached_index_quote  # same shape — last close vs prior close


# =====================================================
# UI HELPERS
# =====================================================

def _stat_card(label: str, value: str, color: str = "#111827", sub: str | None = None) -> str:
    sub_html = f'<div style="font-size:11px;color:{TEXT_MUTED};margin-top:2px;">{sub}</div>' if sub else ""
    return f"""
    <div style="background:#fff;border:1px solid {BORDER};border-radius:10px;padding:14px 16px;height:100%;">
        <div style="font-size:12px;color:{TEXT_MUTED};font-weight:600;text-transform:uppercase;letter-spacing:.03em;">{label}</div>
        <div style="font-size:20px;font-weight:700;color:{color};margin-top:4px;">{value}</div>
        {sub_html}
    </div>
    """


def _stat_row(items: list[tuple]) -> None:
    """items: list of (label, value, color, sub) — color/sub optional."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value = item[0], item[1]
        color = item[2] if len(item) > 2 else "#111827"
        sub = item[3] if len(item) > 3 else None
        with col:
            st.markdown(_stat_card(label, value, color, sub), unsafe_allow_html=True)


def _section_header(title: str) -> None:
    st.markdown(
        f'<div style="font-size:19px;font-weight:700;color:#111827;margin:6px 0 14px 0;">{title}</div>',
        unsafe_allow_html=True,
    )


def _render_brand_header() -> None:
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,{PURPLE} 0%,#7C4DFF 100%);
             border-radius:16px;padding:20px 28px;margin-bottom:16px;
             display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
            <div style="display:flex;align-items:center;gap:10px;">
                <span style="font-size:24px;">⚡</span>
                <span style="font-size:21px;font-weight:800;color:#fff;letter-spacing:-0.02em;">Sensei AI</span>
            </div>
            <div style="color:#e6e0ff;font-size:13px;font-weight:500;">AI-powered market intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_index_strip() -> None:
    indices = {
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        "BANK NIFTY": "^NSEBANK",
    }
    cols = st.columns(len(indices))
    for col, (name, symbol) in zip(cols, indices.items()):
        quote = _cached_index_quote(symbol)
        with col:
            if quote is None:
                st.markdown(
                    f'<div style="padding:10px 0;"><span style="font-size:13px;color:{TEXT_MUTED};font-weight:600;">{name}</span></div>',
                    unsafe_allow_html=True,
                )
                continue
            up = quote["change"] >= 0
            color = GREEN if up else RED
            arrow = "▲" if up else "▼"
            st.markdown(
                f"""
                <div style="padding:10px 0;">
                    <span style="font-size:13px;color:{TEXT_MUTED};font-weight:600;">{name}</span>
                    &nbsp;<span style="font-size:14px;font-weight:700;color:#111827;">{quote['value']:,.2f}</span>
                    &nbsp;<span style="font-size:12px;color:{color};font-weight:600;">{arrow} {abs(quote['change']):,.2f} ({quote['pct']:+.2f}%)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _go_to_detail(ticker: str, company: str, timeframe: str = "1y") -> None:
    st.session_state["active_ticker"] = ticker
    st.session_state["active_company"] = company
    st.session_state["active_timeframe"] = timeframe
    st.session_state["view"] = "detail"


def _render_setup_card(setup: dict, company: str) -> None:
    if setup.get("error") and setup["entry_zone"] == (0, 0):
        st.warning(setup["plan"])
        return

    bias = setup.get("bias", "NEUTRAL")
    bias_color = {"BULLISH": GREEN, "BEARISH": RED}.get(bias, AMBER)
    bias_bg = {"BULLISH": "#e6fff7", "BEARISH": "#fff0f0"}.get(bias, "#fffbe6")

    st.markdown(
        f"""<div style="background:{bias_bg};border-left:4px solid {bias_color};
        padding:10px 16px;border-radius:6px;margin-bottom:12px;">
        <span style="font-weight:700;color:{bias_color};font-size:16px;">{bias}</span>
        &nbsp;<span style="color:{TEXT_MUTED};font-size:13px;">— {setup.get('pattern', '—')}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    entry_low, entry_high = setup["entry_zone"]
    rr = setup["risk_reward"]
    _stat_row([
        ("Entry Zone", f"₹{entry_low:.2f}–{entry_high:.2f}"),
        ("Stop Loss", f"₹{setup['stop_loss']:.2f}", RED),
        ("Target 1", f"₹{setup['target_1']:.2f}", GREEN),
        ("Target 2", f"₹{setup['target_2']:.2f}", GREEN),
        ("Risk / Reward", f"{rr:.1f}x", GREEN if rr >= 2 else AMBER),
    ])

    if setup.get("key_levels"):
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        st.markdown("**Key levels**")
        items = [(label, f"₹{val:,.2f}") for label, val in setup["key_levels"].items()]
        _stat_row(items)

    st.markdown("---")
    st.markdown(f"**Trade plan** — valid: _{setup.get('validity', '—')}_")
    st.info(setup["plan"])


# =====================================================
# EXPLORE VIEW (home) — ticker strip, search, stock cards
# =====================================================

def _render_explore_view() -> None:
    _render_index_strip()
    st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)

    nav1, nav2, nav3 = st.columns([3, 1.2, 1])
    with nav1:
        company = st.selectbox(
            "Stock", list(NIFTY_50.keys()), label_visibility="collapsed",
            index=None, placeholder="🔍  Search a stock…", key="explore_company",
        )
    with nav2:
        timeframe = st.selectbox(
            "Timeframe", ["1y", "2y", "5y"], label_visibility="collapsed", key="explore_timeframe",
        )
    with nav3:
        if st.button("Run Analysis", use_container_width=True, key="explore_run", disabled=not company):
            _go_to_detail(NIFTY_50[company], company, timeframe)
            st.rerun()

    st.markdown("---")

    _section_header("Popular Stocks")
    cols = st.columns(4)
    for i, (name, tkr) in enumerate(POPULAR_STOCKS):
        with cols[i % 4]:
            with st.container(border=True):
                quote = _cached_stock_quote(tkr)
                st.markdown(f"**{name}**")
                if quote:
                    up = quote["change"] >= 0
                    color = GREEN if up else RED
                    st.markdown(
                        f"""<div style="margin:2px 0 10px 0;">
                        <span style="font-size:16px;font-weight:700;">₹{quote['value']:,.2f}</span>
                        &nbsp;<span style="font-size:12px;color:{color};font-weight:600;">
                        {quote['change']:+.2f} ({quote['pct']:+.2f}%)</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div style="margin:2px 0 10px 0;color:{TEXT_MUTED};font-size:12px;">Price unavailable</div>',
                        unsafe_allow_html=True,
                    )
                if st.button("View →", key=f"card_{tkr}", use_container_width=True):
                    _go_to_detail(tkr, name)
                    st.rerun()

    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
    st.caption("More stocks, IPOs and mutual funds are coming soon.")


# =====================================================
# STOCK DETAIL VIEW — opens when a card / search result
# is clicked. Sticky AI-signal panel + tabbed research feed
# with each section as an openable bar (st.expander).
# =====================================================

def _render_detail_view() -> None:
    ticker = st.session_state["active_ticker"]
    company = st.session_state["active_company"]
    timeframe = st.session_state["active_timeframe"]

    if st.button("← Back to Explore"):
        st.session_state["view"] = "explore"
        st.rerun()

    with st.spinner("Running AI engine..."):
        price_df = _cached_prices(ticker, timeframe)
        fundamentals = _cached_fundamentals(ticker)
        news = _cached_news(company, ticker)
        intraday_df = _cached_intraday(ticker)

        signals = _cached_signal_pipeline(ticker, company, price_df, fundamentals)
        sr_data = _cached_support_resistance(price_df)

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

        if not intraday_df.empty:
            intra_setup = build_intraday_setup(intraday_df, price_df, ticker)
        else:
            intra_setup = {
                "error": "Intraday data unavailable", "mode": "Intraday",
                "bias": "NEUTRAL", "entry_zone": (0, 0), "stop_loss": 0,
                "target_1": 0, "target_2": 0, "risk_reward": 0,
                "pattern": "—", "key_levels": {}, "validity": "—",
                "plan": "Intraday data could not be loaded.",
            }
        swing_setup = build_swing_setup(price_df, ticker)

    action = decision["action"]

    # =========================
    # STOCK HEADER + CHART (always visible, like Groww's main pane)
    # =========================

    current_price = fundamentals["current_price"]
    price_color = GREEN if signals["regime"] == "BULL" else RED

    st.markdown(
        f"""
        <div style="margin-bottom:10px;">
            <h2 style="margin:0;">{company}</h2>
            <h3 style="margin:0; color:{price_color};">₹ {current_price:,.2f}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_price_chart(price_df)
    st.markdown("---")

    # =========================
    # MAIN LAYOUT — tabbed research feed (left) + sticky AI signal panel (right)
    # =========================

    left, right = st.columns([2, 1], gap="large")

    with right:
        theme = {
            "BUY":  {"bg": "#e6fff7", "fg": GREEN, "border": GREEN},
            "SELL": {"bg": "#fff0f0", "fg": RED, "border": RED},
            "HOLD": {"bg": "#fff8e6", "fg": "#c98a00", "border": AMBER},
        }.get(action, {"bg": "#f5f5f5", "fg": "#333", "border": "#ccc"})

        entry_low, entry_high = swing_setup.get("entry_zone", (0, 0))
        levels_html = ""
        if not (swing_setup.get("error") and swing_setup["entry_zone"] == (0, 0)):
            levels_html = (
                f'<div style="margin-top:14px;padding-top:14px;border-top:1px solid {BORDER};">'
                f'<div style="font-size:12px;color:{TEXT_MUTED};font-weight:700;text-transform:uppercase;letter-spacing:.03em;margin-bottom:10px;">Swing Trade Levels</div>'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
                f'<span style="font-size:13px;color:{TEXT_MUTED};">Entry Zone</span>'
                f'<span style="font-size:13px;font-weight:700;">₹{entry_low:.2f}–{entry_high:.2f}</span></div>'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
                f'<span style="font-size:13px;color:{TEXT_MUTED};">Stop Loss</span>'
                f'<span style="font-size:13px;font-weight:700;color:{RED};">₹{swing_setup["stop_loss"]:.2f}</span></div>'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
                f'<span style="font-size:13px;color:{TEXT_MUTED};">Target 1</span>'
                f'<span style="font-size:13px;font-weight:700;color:{GREEN};">₹{swing_setup["target_1"]:.2f}</span></div>'
                f'<div style="display:flex;justify-content:space-between;">'
                f'<span style="font-size:13px;color:{TEXT_MUTED};">Risk / Reward</span>'
                f'<span style="font-size:13px;font-weight:700;">{swing_setup["risk_reward"]:.1f}x</span></div>'
                f'</div>'
            )

        headline_text = decision.get('narrative', {}).get('headline') or decision.get('explanation', '')
        panel_html = (
            f'<div style="position:sticky;top:20px;">'
            f'<div style="background:#fff;border:1px solid {BORDER};border-radius:14px;padding:20px;">'
            f'<div style="font-size:12px;color:{TEXT_MUTED};font-weight:700;text-transform:uppercase;letter-spacing:.04em;">AI Recommendation</div>'
            f'<div style="display:flex;align-items:baseline;gap:10px;margin-top:6px;">'
            f'<span style="font-size:28px;font-weight:800;color:{theme["fg"]};line-height:1;">{action}</span>'
            f'<span style="font-size:13px;color:{TEXT_MUTED};">· {decision["confidence"] * 100:.1f}% confidence</span></div>'
            f'<div style="background:{theme["bg"]};border-left:4px solid {theme["border"]};border-radius:8px;padding:10px 12px;margin-top:12px;font-size:12px;color:#333;">{headline_text}</div>'
            f'{levels_html}'
            f'</div>'
            f'</div>'
        )
        st.markdown(panel_html, unsafe_allow_html=True)

    with left:
        tab_overview, tab_technicals, tab_news, tab_setup = st.tabs(
            ["Overview", "Technicals", "News", "Trade Setup"]
        )

        # ---------------- OVERVIEW ----------------
        with tab_overview:
            with st.expander("Key Metrics", expanded=True):
                _stat_row([
                    ("ML Prob (UP)", f"{signals['ml_prob_up']:.2f}"),
                    ("LSTM Return", f"{signals['lstm_return']:.3f}"),
                    ("TCN Return", f"{signals['tcn_return']:.3f}"),
                    ("Market Regime", signals["regime"], GREEN if signals["regime"] == "BULL" else RED),
                ])

            with st.expander("Company Fundamentals", expanded=True):
                market_cap_cr = fundamentals["market_cap"] / 1e7
                _stat_row([
                    ("Current Price", f"₹{fundamentals['current_price']}"),
                    ("Market Cap", f"₹ {market_cap_cr:,.0f} Cr"),
                    ("ROE", f"{fundamentals['roe']:.2f}"),
                    ("52W Range", f"{fundamentals['52_week_low']} – {fundamentals['52_week_high']}"),
                ])

            with st.expander("Strategy Backtest"):
                signals_series = pd.Series(
                    ["BUY" if x > 0 else "SELL"
                     for x in price_df["close"].pct_change().fillna(0)]
                )
                backtest_df = run_backtest(price_df, signals_series)
                metrics = calculate_metrics(backtest_df["equity"])

                st.line_chart(backtest_df["equity"])
                _stat_row([
                    ("Total Return", f"{metrics['total_return'] * 100:.2f}%", GREEN if metrics['total_return'] >= 0 else RED),
                    ("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}"),
                    ("Max Drawdown", f"{metrics['max_drawdown'] * 100:.2f}%", RED),
                ])

        # ---------------- TECHNICALS ----------------
        with tab_technicals:
            narrative = decision.get("narrative", {})

            with st.expander("AI Analyst Report", expanded=True):
                if narrative:
                    action_colors = {"BUY": GREEN, "SELL": RED, "HOLD": AMBER}
                    headline_color = action_colors.get(action, "#333333")

                    st.markdown(
                        f"<h4 style='color:{headline_color};'>{narrative.get('headline', '')}</h4>",
                        unsafe_allow_html=True,
                    )

                    tab_trend, tab_momentum, tab_vol, tab_ai, tab_shap, tab_news_n = st.tabs([
                        "Trend", "Momentum", "Volatility", "AI Models", "SHAP Drivers", "News"
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
                                "rsi_norm": "RSI",
                                "ema_spread": "EMA Crossover",
                                "macd_diff": "MACD Divergence",
                                "atr_pct": "ATR (Volatility)",
                                "volatility_10": "10-day Volatility",
                                "return_1": "1-day Return",
                                "return_5": "5-day Return",
                                "return_10": "10-day Return",
                                "range_position": "Price Range Position",
                            }
                            features = [_LABELS.get(r["feature"], r["feature"]) for r in shap_ranked]
                            values = [r["shap"] for r in shap_ranked]
                            colors = [GREEN if v > 0 else RED for v in values]

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

                    with tab_news_n:
                        st.markdown(narrative.get("news", "—"))

                    st.markdown("---")
                    st.markdown("**Summary**")
                    st.info(narrative.get("summary", "—"))
                else:
                    st.info(decision.get("explanation", "No explanation available."))

            with st.expander("Raw Model Signals"):
                raw_signals = {
                    "ML Prob UP": f"{signals['ml_prob_up'] * 100:.1f}%",
                    "LSTM Forecast": f"{signals['lstm_return'] * 100:.3f}%",
                    "TCN Forecast": f"{signals['tcn_return'] * 100:.3f}%",
                    "HMM Regime": signals["regime"],
                    "PPO Action": signals["ppo_action"],
                    "Ensemble Score": f"{decision['score']:.1f} / 5.0",
                }
                st.table(pd.DataFrame.from_dict(raw_signals, orient="index", columns=["Value"]))

            with st.expander("Support & Resistance Levels"):
                supports = sr_data.get("supports", [])
                resistances = sr_data.get("resistances", [])
                pivot_data = sr_data.get("pivot_data", {})

                strength_colors = {"Strong": GREEN, "Moderate": AMBER, "Weak": "#aaaaaa"}
                strength_bg = {"Strong": "#e6fff7", "Moderate": "#fffbe6", "Weak": "#f5f5f5"}

                sr_col1, sr_col2 = st.columns(2)
                with sr_col1:
                    st.markdown("**Support Levels** (nearest first)")
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
                            <span style="color:{TEXT_MUTED};font-size:11px;">{methods_str} · {lv.get('touches', 0)} touches</span>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                with sr_col2:
                    st.markdown("**Resistance Levels** (nearest first)")
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
                            <span style="color:{TEXT_MUTED};font-size:11px;">{methods_str} · {lv.get('touches', 0)} touches</span>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                if pivot_data:
                    st.markdown("**Classic Pivot Points** (previous session)")
                    piv_df = pd.DataFrame.from_dict(pivot_data, orient="index", columns=["Price (₹)"])
                    piv_df["Price (₹)"] = piv_df["Price (₹)"].map(lambda x: f"₹{x:,.2f}")
                    st.table(piv_df)

        # ---------------- NEWS ----------------
        with tab_news:
            news_details = news.get("details", [])
            bull_c = news.get("bull_count", 0)
            bear_c = news.get("bear_count", 0)
            neut_c = news.get("neutral_count", 0)
            total_c = bull_c + bear_c + neut_c

            if total_c > 0:
                w_score = news.get("weighted_score", 0)
                score_color = GREEN if w_score > 0.1 else RED if w_score < -0.1 else AMBER
                _stat_row([
                    ("Weighted Score", f"{w_score:+.3f}", score_color),
                    ("Bullish Headlines", f"{bull_c}", GREEN),
                    ("Bearish Headlines", f"{bear_c}", RED),
                    ("Neutral Headlines", f"{neut_c}"),
                ])

                st.markdown(f"**Overall:** {news.get('summary', '')}")

                col_bull, col_bear = st.columns(2)
                with col_bull:
                    top_b = news.get("top_bullish")
                    if top_b:
                        st.markdown(
                            f"""<div style="background:#e6fff7;border-left:4px solid {GREEN};
                            padding:10px 14px;border-radius:6px;">
                            <div style="color:{GREEN};font-weight:600;font-size:12px;margin-bottom:4px;">
                            TOP BULLISH — {top_b.get('impact_type', '')}</div>
                            <div style="font-size:13px;">{top_b['headline']}</div>
                            <div style="color:{TEXT_MUTED};font-size:11px;margin-top:4px;">{top_b.get('source', '')} · {top_b.get('published', '')}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                with col_bear:
                    top_br = news.get("top_bearish")
                    if top_br:
                        st.markdown(
                            f"""<div style="background:#fff0f0;border-left:4px solid {RED};
                            padding:10px 14px;border-radius:6px;">
                            <div style="color:{RED};font-weight:600;font-size:12px;margin-bottom:4px;">
                            TOP BEARISH — {top_br.get('impact_type', '')}</div>
                            <div style="font-size:13px;">{top_br['headline']}</div>
                            <div style="color:{TEXT_MUTED};font-size:11px;margin-top:4px;">{top_br.get('source', '')} · {top_br.get('published', '')}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                with st.expander(f"All {total_c} headlines"):
                    label_colors = {"POSITIVE": GREEN, "NEGATIVE": RED, "NEUTRAL": AMBER}
                    impact_icons = {
                        "Earnings": "💰", "Regulatory": "⚖️", "Management": "👔",
                        "Macro": "🌍", "Sector": "🏭", "General": "📄",
                    }
                    for item in news_details:
                        lc = label_colors.get(item["label"], "#888")
                        icon = impact_icons.get(item.get("impact_type", "General"), "📄")
                        url = item.get("url", "")
                        link = f'<a href="{url}" target="_blank" style="color:{PURPLE};">↗</a>' if url else ""
                        st.markdown(
                            f"""<div style="border-bottom:1px solid #eee;padding:8px 0;">
                            <span style="background:{lc};color:white;border-radius:3px;
                            padding:1px 6px;font-size:11px;font-weight:600;">{item['label']}</span>
                            &nbsp;<span style="font-size:11px;color:{TEXT_MUTED};">{icon} {item.get('impact_type', '')}
                            · conf {item['confidence']:.0%}</span><br>
                            <span style="font-size:13px;">{item['headline']}</span>
                            &nbsp;{link}<br>
                            <span style="color:#aaa;font-size:11px;">{item.get('source', '')} · {item.get('published', '')}</span>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                st.markdown("---")
                st.markdown("**News-driven Price Forecast**")
                pred = news_price_pred
                pred_dir = pred.get("direction", "FLAT")
                conf_label = pred.get("confidence", "LOW")
                conf_badge = {"HIGH": "🟢 High", "MEDIUM": "🟡 Medium", "LOW": "🔴 Low"}.get(conf_label, conf_label)

                _stat_row([
                    ("Direction", pred_dir, GREEN if pred_dir == "UP" else RED if pred_dir == "DOWN" else AMBER),
                    ("Predicted Price", f"₹{pred.get('predicted_price', 0):,.2f}"),
                    ("Range Low", f"₹{pred.get('price_low', 0):,.2f}"),
                    ("Range High", f"₹{pred.get('price_high', 0):,.2f}"),
                    ("Expected Move", f"{pred.get('expected_move_pct', 0):+.2f}%"),
                ])

                st.caption(f"Confidence: {conf_badge} · Horizon: {pred.get('horizon_label', '')}")
                st.info(pred.get("explanation", ""))
            else:
                st.info("No news found for this stock.")

        # ---------------- TRADE SETUP ----------------
        with tab_setup:
            setup_tab_intra, setup_tab_swing = st.tabs(["Intraday (15-min)", "Swing (Daily)"])
            with setup_tab_intra:
                _render_setup_card(intra_setup, company)
            with setup_tab_swing:
                _render_setup_card(swing_setup, company)


# =====================================================
# APP ENTRY POINT
# =====================================================

_render_brand_header()

if "view" not in st.session_state:
    st.session_state["view"] = "explore"

if st.session_state["view"] == "detail" and st.session_state.get("active_ticker"):
    _render_detail_view()
else:
    _render_explore_view()
