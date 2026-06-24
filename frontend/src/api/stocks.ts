import { apiClient } from './client'
import type { StockListItem, AnalysisResult, CandleBar } from '../types/stocks'

export interface FetchStocksParams {
  tier?: string
  sector?: string
  q?: string
  limit?: number
}

export async function fetchStocks(params?: FetchStocksParams): Promise<StockListItem[]> {
  const { data } = await apiClient.get<StockListItem[]>('/stocks', { params })
  return data
}

export async function searchStocks(q: string): Promise<StockListItem[]> {
  const { data } = await apiClient.get<StockListItem[]>('/stocks/search', { params: { q } })
  return data
}

export async function fetchSectors(): Promise<string[]> {
  const { data } = await apiClient.get<string[]>('/stocks/sectors')
  return data
}

export async function analyzeStock(symbol: string, timeframe: string): Promise<AnalysisResult> {
  const { data } = await apiClient.post<AnalysisResult>(`/stocks/${symbol}/analyze`, { timeframe })
  return data
}

export async function fetchChartData(symbol: string, timeframe: string): Promise<CandleBar[]> {
  const result = await analyzeStock(symbol, timeframe)
  return result.chart_data
}
