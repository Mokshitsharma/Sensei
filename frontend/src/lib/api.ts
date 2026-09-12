import type {
  AccuracyBreakdown,
  AccuracyHorizon,
  AccuracySummary,
  Analysis,
  BacktestResponse,
  CapSegment,
  Fundamentals,
  Indices,
  MoverPrediction,
  MoversResponse,
  NewsResponse,
  Outlook,
  OutlookHorizon,
  PaperTrade,
  Portfolio,
  PopularStock,
  PredictionHorizon,
  PriceResponse,
  Quote,
  ScreenFilter,
  ScreenInfo,
  ScreenerRow,
  StockListItem,
  TradeSetup,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function get<T>(path: string, token?: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    cache: "no-store",
    headers: token ? { Authorization: `Bearer ${token}` } : undefined,
  });
  if (!res.ok) {
    throw new Error(`API ${path} failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown, token: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`API ${path} failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  indices: () => get<Indices>("/api/indices"),
  stocks: () => get<StockListItem[]>("/api/stocks"),
  popularStocks: () => get<PopularStock[]>("/api/stocks/popular"),
  quote: (ticker: string) => get<Quote>(`/api/stocks/${encodeURIComponent(ticker)}/quote`),
  price: (ticker: string, timeframe = "1y") =>
    get<PriceResponse>(
      `/api/stocks/${encodeURIComponent(ticker)}/price?timeframe=${timeframe}`
    ),
  fundamentals: (ticker: string) =>
    get<Fundamentals>(`/api/stocks/${encodeURIComponent(ticker)}/fundamentals`),
  news: (ticker: string) =>
    get<NewsResponse>(`/api/stocks/${encodeURIComponent(ticker)}/news`),
  analysis: (ticker: string, timeframe = "1y") =>
    get<Analysis>(
      `/api/stocks/${encodeURIComponent(ticker)}/analysis?timeframe=${timeframe}`
    ),
  setup: (ticker: string, mode: "intraday" | "swing", timeframe = "1y") =>
    get<TradeSetup>(
      `/api/stocks/${encodeURIComponent(ticker)}/setup?mode=${mode}&timeframe=${timeframe}`
    ),
  backtest: (ticker: string, timeframe = "1y") =>
    get<BacktestResponse>(
      `/api/stocks/${encodeURIComponent(ticker)}/backtest?timeframe=${timeframe}`
    ),
  movers: () => get<MoversResponse>("/api/stocks/movers"),
  byCap: (segment: CapSegment) =>
    get<MoverPrediction[]>(`/api/stocks/by-cap?segment=${segment}`),
  screens: () => get<ScreenInfo[]>("/api/screener/screens"),
  screener: (filter: ScreenFilter, segment: CapSegment | "all" = "all") =>
    get<ScreenerRow[]>(`/api/screener?filter=${filter}&segment=${segment}`),
  predictionsAccuracy: (horizon?: AccuracyHorizon, limit = 100) =>
    get<AccuracySummary>(
      `/api/predictions/accuracy?limit=${limit}${horizon ? `&horizon=${horizon}` : ""}`
    ),
  predictionsAccuracyBreakdown: () =>
    get<AccuracyBreakdown>("/api/predictions/accuracy/breakdown"),
  predictions: (horizon: PredictionHorizon, direction: "UP" | "DOWN" | "FLAT" | "ALL" = "ALL") =>
    get<MoverPrediction[]>(`/api/predictions?horizon=${horizon}&direction=${direction}`),
  outlook: (ticker: string, horizon: OutlookHorizon) =>
    get<Outlook>(`/api/stocks/${encodeURIComponent(ticker)}/outlook?horizon=${horizon}`),
  portfolio: (token: string) => get<Portfolio>("/api/portfolio", token),
  portfolioHistory: (token: string) =>
    get<PaperTrade[]>("/api/portfolio/history", token),
  trade: (
    token: string,
    body: { ticker: string; side: "BUY" | "SELL"; quantity: number }
  ) => post<PaperTrade>("/api/portfolio/trade", body, token),
};
