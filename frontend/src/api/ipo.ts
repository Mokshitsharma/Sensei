import { apiClient } from './client'
import type { IPOListItem, IPODetail } from '../types/ipo'

export async function fetchIPOs(status?: string): Promise<IPOListItem[]> {
  const params = status && status !== 'all' ? { status } : {}
  const { data } = await apiClient.get<IPOListItem[]>('/ipo', { params })
  return data
}

export async function fetchIPODetail(id: string): Promise<IPODetail> {
  const { data } = await apiClient.get<IPODetail>(`/ipo/${id}`)
  return data
}
