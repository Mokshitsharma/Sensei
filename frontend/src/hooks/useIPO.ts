import { useQuery } from '@tanstack/react-query'
import { fetchIPOs, fetchIPODetail } from '../api/ipo'

export function useIPOList(status?: string) {
  return useQuery({
    queryKey: ['ipos', status],
    queryFn: () => fetchIPOs(status),
    staleTime: 2 * 60 * 1000, // 2 minutes
  })
}

export function useIPODetail(id: string | null) {
  return useQuery({
    queryKey: ['ipo', id],
    queryFn: () => fetchIPODetail(id!),
    enabled: !!id,
    staleTime: 5 * 60 * 1000,
  })
}
