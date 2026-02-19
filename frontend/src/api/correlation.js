import { useQuery } from '@tanstack/react-query'
import api from './client'

export function useCorrelations(symbol, start_date, end_date, method = 'pearson') {
  return useQuery({
    queryKey: ['correlation', symbol, start_date, end_date, method],
    queryFn: async () => {
      const { data } = await api.get(
        `/correlation/${encodeURIComponent(symbol)}`,
        { params: { start_date, end_date, method } },
      )
      return data
    },
    enabled: !!symbol,
  })
}

export function useRollingCorrelation(symbol, event_metric = 'conflict_count', start_date, end_date, window_days = 30) {
  return useQuery({
    queryKey: ['correlation', symbol, 'rolling', event_metric, start_date, end_date, window_days],
    queryFn: async () => {
      const { data } = await api.get(
        `/correlation/${encodeURIComponent(symbol)}/rolling`,
        { params: { event_metric, start_date, end_date, window_days } },
      )
      return data
    },
    enabled: !!symbol,
  })
}

export function useTopCorrelations(start_date, end_date, limit = 20, symbols) {
  return useQuery({
    queryKey: ['correlation', 'top', start_date, end_date, limit, symbols],
    queryFn: async () => {
      const { data } = await api.get('/correlation/top', {
        params: { start_date, end_date, limit, symbols },
      })
      return data
    },
    staleTime: 30 * 60 * 1000,
  })
}

export function useCorrelationHeatmap(symbols, start_date, end_date, method = 'pearson') {
  return useQuery({
    queryKey: ['correlation', 'heatmap', symbols, start_date, end_date, method],
    queryFn: async () => {
      const { data } = await api.get('/correlation/heatmap', {
        params: { symbols, start_date, end_date, method },
      })
      return data
    },
    enabled: !!symbols,
  })
}
