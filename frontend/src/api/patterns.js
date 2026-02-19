/**
 * Historical Patterns API hooks — React Query wrappers for frequency analysis.
 *
 * - usePattern(symbol, params)            — Single event group pattern
 * - useAllPatterns(symbol, start, end)    — All event group + country patterns
 */
import { useQuery } from '@tanstack/react-query'
import api from './client'

export function usePattern(symbol, params = {}) {
  return useQuery({
    queryKey: ['patterns', symbol, params],
    queryFn: async () => {
      const { data } = await api.get(
        `/patterns/${encodeURIComponent(symbol)}`,
        { params },
      )
      return data
    },
    enabled: !!symbol,
  })
}

export function useAllPatterns(symbol, start_date, end_date, min_occurrences = 10) {
  return useQuery({
    queryKey: ['patterns', symbol, 'all', start_date, end_date, min_occurrences],
    queryFn: async () => {
      const { data } = await api.get(
        `/patterns/${encodeURIComponent(symbol)}/all`,
        { params: { start_date, end_date, min_occurrences } },
      )
      return data
    },
    enabled: !!symbol,
  })
}
