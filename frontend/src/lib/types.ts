export type Quote = {
  value: number;
  change: number;
  pct: number;
} | null;

export type StockListItem = {
  name: string;
  ticker: string;
};

export type PopularStock = StockListItem & {
  quote: Quote;
};

export type PriceRecord = {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

export type PriceResponse = {
  ticker: string;
  company: string;
  timeframe: string;
  records: PriceRecord[];
};

export type Fundamentals = {
  current_price: number;
  market_cap: number;
  book_value: number;
  debt_to_equity: number;
  roe: number;
  "52_week_high": number;
  "52_week_low": number;
};

export type Signals = {
  rule_signal: unknown;
  ml_prob_up: number;
  lstm_return: number;
  tcn_return: number;
  regime: string;
  ppo_action: string;
  shap_values: unknown;
  feature_values: unknown;
};

export type NarrativeShapItem = { feature: string; shap: number };

export type Narrative = {
  headline?: string;
  trend?: string;
  momentum?: string;
  volatility?: string;
  ai_models?: string;
  shap_story?: string;
  news?: string;
  summary?: string;
};

export type Decision = {
  action: "BUY" | "SELL" | "HOLD";
  confidence: number;
  score: number;
  explanation?: string;
  narrative?: Narrative;
  shap_ranked?: NarrativeShapItem[];
};

export type NewsPriceForecast = {
  direction: "UP" | "DOWN" | "FLAT";
  predicted_price: number;
  price_low: number;
  price_high: number;
  expected_move_pct: number;
  confidence: "HIGH" | "MEDIUM" | "LOW";
  horizon_label: string;
  explanation: string;
};

export type SrLevel = {
  price: number;
  strength: "Strong" | "Moderate" | "Weak";
  methods: string[];
  touches: number;
};

export type SupportResistance = {
  supports: SrLevel[];
  resistances: SrLevel[];
  pivot_data: Record<string, number>;
};

export type TradeSetup = {
  mode: string;
  bias: "BULLISH" | "BEARISH" | "NEUTRAL";
  price: number;
  entry_zone: [number, number];
  stop_loss: number;
  target_1: number;
  target_2: number;
  risk_reward: number;
  pattern: string;
  key_levels: Record<string, number>;
  validity: string;
  plan: string;
  error?: string | null;
};

export type Analysis = {
  ticker: string;
  company: string;
  signals: Signals;
  decision: Decision;
  news_price_forecast: NewsPriceForecast;
  support_resistance: SupportResistance;
  setup: {
    intraday: TradeSetup;
    swing: TradeSetup;
  };
};

export type NewsHeadline = {
  label: "POSITIVE" | "NEGATIVE" | "NEUTRAL";
  headline: string;
  source?: string;
  published?: string;
  url?: string;
  impact_type?: string;
  confidence: number;
};

export type NewsResponse = {
  sentiment_score: number;
  weighted_score: number;
  bull_count: number;
  bear_count: number;
  neutral_count: number;
  summary: string;
  top_bullish?: NewsHeadline;
  top_bearish?: NewsHeadline;
  details: NewsHeadline[];
};

export type BacktestResponse = {
  equity_curve: number[];
  metrics: {
    total_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
  };
};

export type Indices = Record<string, Quote>;

export type CapSegment = "large" | "mid" | "small";

export type MoverPrediction = {
  ticker: string;
  company: string;
  current_price: number;
  today_change_pct: number;
  predicted_price: number;
  expected_move_pct: number;
  confidence: "HIGH" | "MEDIUM" | "LOW";
  direction: "UP" | "DOWN" | "FLAT";
  horizon_label: string;
};

export type MoversResponse = {
  gainers: MoverPrediction[];
  losers: MoverPrediction[];
};

export type PredictionHorizon = "1d" | "7d" | "30d";
export type OutlookHorizon = PredictionHorizon | "90d" | "180d" | "365d" | "730d";

export type ScreenFilter =
  | "rsi_oversold"
  | "rsi_overbought"
  | "macd_bullish"
  | "macd_bearish"
  | "near_52w_high"
  | "near_52w_low";

export type ScreenInfo = {
  id: ScreenFilter;
  label: string;
  bias: "Bullish" | "Bearish";
};

export type ScreenerRow = {
  ticker: string;
  company: string;
  price: number;
  rsi: number;
  macd_bullish: boolean;
  pct_from_52w_high: number | null;
  pct_from_52w_low: number | null;
};

export type Outlook = {
  ticker: string;
  company: string;
  horizon: OutlookHorizon;
  horizon_label: string;
  current_price: number;
  mode: "quantitative" | "qualitative";
  decision: {
    action: "BUY" | "SELL" | "HOLD";
    confidence: number;
    narrative?: Narrative;
  };
  shap_ranked: NarrativeShapItem[];
  regime: string;
  fundamentals: Fundamentals;
  news_summary: {
    sentiment_score: number;
    bull_count: number;
    bear_count: number;
    top_bullish?: NewsHeadline;
    top_bearish?: NewsHeadline;
  };
  // quantitative mode only
  predicted_price?: number;
  expected_move_pct?: number;
  direction?: "UP" | "DOWN" | "FLAT";
  price_confidence?: "HIGH" | "MEDIUM" | "LOW";
  // qualitative mode only
  disclaimer?: string;
};

export type AccuracyHorizon = "1d" | "7d" | "30d";

export type AccuracyRow = {
  ticker: string;
  company: string;
  horizon: AccuracyHorizon;
  predicted_at: string;
  target_date: string;
  price_at_prediction: number;
  predicted_price: number;
  direction: "UP" | "DOWN" | "FLAT";
  actual_price: number;
  actual_move_pct: number;
  correct: 0 | 1;
};

export type AccuracySummary = {
  total: number;
  correct: number;
  pct_correct: number | null;
  pending: number;
  rows: AccuracyRow[];
};

export type AccuracyByHorizon = {
  horizon: AccuracyHorizon;
  total: number;
  correct: number;
  pct_correct: number | null;
  pending: number;
};

export type AccuracyTrendPoint = {
  date: string;
  day_total: number;
  day_correct: number;
  cumulative_total: number;
  cumulative_correct: number;
  cumulative_pct_correct: number | null;
};

export type AccuracyBreakdown = {
  by_horizon: AccuracyByHorizon[];
  trend: AccuracyTrendPoint[];
};

export type PortfolioPosition = {
  ticker: string;
  company: string;
  quantity: number;
  avg_entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
};

export type Portfolio = {
  cash_balance: number;
  positions: PortfolioPosition[];
  realized_pnl: number;
};

export type PaperTrade = {
  id: number;
  ticker: string;
  side: "BUY" | "SELL";
  quantity: number;
  price: number;
  executed_at: string;
};
