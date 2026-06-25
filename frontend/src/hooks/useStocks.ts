import { useQuery, useMutation } from '@tanstack/react-query'
import { fetchStocks, searchStocks, fetchSectors, analyzeStock } from '../api/stocks'
import type { FetchStocksParams } from '../api/stocks'

export function useStockList(params?: FetchStocksParams) {
  return useQuery({
    queryKey: ['stocks', params],
    queryFn: () => fetchStocks(params),
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

export function useStockSearch(q: string) {
  return useQuery({
    queryKey: ['stocks', 'search', q],
    queryFn: () => searchStocks(q),
    enabled: q.length >= 2,
    staleTime: 30 * 1000, // 30 seconds
  })
}

export function useSectors() {
  return useQuery({
    queryKey: ['sectors'],
    queryFn: fetchSectors,
    staleTime: 10 * 60 * 1000, // 10 minutes
  })
}

export function useAnalyzeStock() {
  return useMutation({
    mutationFn: ({ symbol, timeframe }: { symbol: string; timeframe: string }) =>
      analyzeStock(symbol, timeframe),
  })
}
