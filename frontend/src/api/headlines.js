/**
 * Headlines API hooks — React Query wrappers for RSS headline endpoints.
 *
 * - useRecentHeadlines(source, daysBack) — Recent headlines, filterable by source
 */
import { useQuery } from '@tanstack/react-query'
import api from './client'

export function useRecentHeadlines(source = null, daysBack = 2, limit = 100) {
  return useQuery({
    queryKey: ['headlines', 'recent', source, daysBack, limit],
    queryFn: async () => {
      const params = { days_back: daysBack, limit }
      if (source) params.source = source
      const { data } = await api.get('/headlines/recent', { params })
      return data
    },
    staleTime: 5 * 60 * 1000,
  })
}
