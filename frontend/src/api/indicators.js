/**
 * Economic Indicators API hooks — React Query wrappers for FRED endpoints.
 *
 * - useLatestIndicators()         — Latest value + delta for each series
 * - useIndicatorSeries(seriesId)  — Time series for a single indicator
 */
import { useQuery } from '@tanstack/react-query'
import api from './client'

export function useLatestIndicators() {
  return useQuery({
    queryKey: ['indicators', 'latest'],
    queryFn: async () => {
      const { data } = await api.get('/indicators/latest')
      return data
    },
    staleTime: 10 * 60 * 1000,
  })
}

export function useIndicatorSeries(seriesId, startDate, endDate) {
  return useQuery({
    queryKey: ['indicators', seriesId, startDate, endDate],
    queryFn: async () => {
      const { data } = await api.get(`/indicators/${seriesId}`, {
        params: { start_date: startDate, end_date: endDate },
      })
      return data
    },
    staleTime: 10 * 60 * 1000,
    enabled: !!seriesId,
  })
}
