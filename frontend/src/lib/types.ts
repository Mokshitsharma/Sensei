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
