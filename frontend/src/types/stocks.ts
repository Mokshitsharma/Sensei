export interface StockListItem {
  symbol: string;
  name: string;
  sector: string;
  tier: 'large' | 'mid' | 'small';
}

export interface SignalBreakdown {
  rule_signal: string;
  ml_prob_up: number;
  lstm_return: number;
  tcn_return: number;
  regime: string;
  ppo_action: string;
}

export interface ShapFeature {
  feature: string;
  value: number;
  direction: string;
}

export interface Fundamentals {
  current_price?: number;
  market_cap?: number;
  pe_ratio?: number;
  roe?: number;
  week_52_high?: number;
  week_52_low?: number;
  debt_to_equity?: number;
  book_value?: number;
}

export interface BacktestMetrics {
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
}

export interface SupportResistance {
  supports: number[];
  resistances: number[];
}

export interface CandleBar {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface NewsItem {
  headline: string;
  source: string;
  published: string;
  url: string;
  label: string;
  confidence: number;
  impact_type: string;
}

export interface AnalysisResult {
  symbol: string;
  name: string;
  timeframe: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  score: number;
  explanation: string;
  signals: SignalBreakdown;
  news_sentiment: number;
  shap_ranked: ShapFeature[];
  narrative: Record<string, string>;
  fundamentals: Fundamentals;
  backtest: BacktestMetrics;
  support_resistance: SupportResistance;
  trade_setups: Record<string, unknown>;
  chart_data: CandleBar[];
  news: NewsItem[];
  model_source: 'trained' | 'generic';
  timestamp: string;
}
