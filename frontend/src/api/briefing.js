/**
 * Briefing API hooks — React Query wrappers for AI briefing endpoints.
 *
 * - useBriefingSummary(daysBack) — AI-generated situational awareness summary
 */
import { useQuery } from '@tanstack/react-query'
import api from './client'

export function useBriefingSummary(daysBack = 3) {
  return useQuery({
    queryKey: ['briefing', 'summary', daysBack],
    queryFn: async () => {
      const { data } = await api.get('/briefing/summary', {
        params: { days_back: daysBack },
        timeout: 180000, // 3 min — first-load of Gemma 4 26B can be slow
      })
      return data
    },
    staleTime: 10 * 60 * 1000, // Cache for 10 minutes
    retry: 1,
  })
}
