import type {
  Analysis,
  BacktestResponse,
  Fundamentals,
  Indices,
  NewsResponse,
  PopularStock,
  PriceResponse,
  Quote,
  StockListItem,
  TradeSetup,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
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
};
