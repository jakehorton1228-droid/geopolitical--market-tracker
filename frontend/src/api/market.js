import { useQuery } from '@tanstack/react-query'
import api from './client'

export function useSymbols() {
  return useQuery({
    queryKey: ['market', 'symbols'],
    queryFn: async () => {
      const { data } = await api.get('/market/symbols/flat')
      return data
    },
    staleTime: 5 * 60 * 1000,
  })
}

export function useMarketData(symbol, start_date, end_date) {
  return useQuery({
    queryKey: ['market', symbol, start_date, end_date],
    queryFn: async () => {
      const { data } = await api.get(`/market/${encodeURIComponent(symbol)}`, {
        params: { start_date, end_date },
      })
      return data
    },
    enabled: !!symbol,
  })
}

export function useMarketWithEvents(symbol, start_date, end_date, min_mentions = 5) {
  return useQuery({
    queryKey: ['market', symbol, 'with-events', start_date, end_date, min_mentions],
    queryFn: async () => {
      const { data } = await api.get(
        `/market/${encodeURIComponent(symbol)}/with-events`,
        { params: { start_date, end_date, min_mentions } },
      )
      return data
    },
    enabled: !!symbol,
  })
}

export function useLatestPrice(symbol) {
  return useQuery({
    queryKey: ['market', symbol, 'latest'],
    queryFn: async () => {
      const { data } = await api.get(`/market/${encodeURIComponent(symbol)}/latest`)
      return data
    },
    enabled: !!symbol,
  })
}
