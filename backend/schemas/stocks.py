from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class StockListItem(BaseModel):
    symbol: str
    name: str
    sector: str
    tier: str  # "large" | "mid" | "small"


class SignalBreakdown(BaseModel):
    rule_signal: str
    ml_prob_up: float
    lstm_return: float
    tcn_return: float
    regime: str
    ppo_action: str


class ShapFeature(BaseModel):
    feature: str
    value: float
    direction: str


class Fundamentals(BaseModel):
    current_price: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    roe: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    debt_to_equity: Optional[float] = None
    book_value: Optional[float] = None


class BacktestMetrics(BaseModel):
    total_return: float
    sharpe_ratio: float
    max_drawdown: float


class SupportResistance(BaseModel):
    supports: List[float]
    resistances: List[float]


class CandleBar(BaseModel):
    time: int        # Unix timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


class NewsItem(BaseModel):
    headline: str
    source: str
    published: str
    url: str
    label: str
    confidence: float
    impact_type: str


class AnalysisRequest(BaseModel):
    timeframe: str = "1y"  # "1y" | "2y" | "5y"


class AnalysisResult(BaseModel):
    symbol: str
    name: str
    timeframe: str
    action: str           # "BUY" | "SELL" | "HOLD"
    confidence: float
    score: float
    explanation: str
    signals: SignalBreakdown
    news_sentiment: float
    shap_ranked: List[ShapFeature]
    narrative: Dict[str, str]
    fundamentals: Fundamentals
    backtest: BacktestMetrics
    support_resistance: SupportResistance
    trade_setups: Dict[str, Any]
    chart_data: List[CandleBar]
    news: List[NewsItem]
    model_source: str     # "trained" | "generic"
    timestamp: str
