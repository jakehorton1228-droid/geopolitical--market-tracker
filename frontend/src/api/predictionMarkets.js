/**
 * Prediction Markets API hooks — React Query wrappers for Polymarket endpoints.
 *
 * - usePredictionMarkets()          — All markets with latest snapshot
 * - usePredictionMovers(daysBack)   — Markets with biggest probability changes
 * - useMarketHistory(marketId)      — Probability time series for one market
 */
import { useQuery } from '@tanstack/react-query'
import api from './client'

export function usePredictionMarkets(limit = 50) {
  return useQuery({
    queryKey: ['prediction-markets', limit],
    queryFn: async () => {
      const { data } = await api.get('/prediction-markets', {
        params: { limit },
      })
      return data
    },
    staleTime: 5 * 60 * 1000,
  })
}

export function usePredictionMovers(daysBack = 7, limit = 10) {
  return useQuery({
    queryKey: ['prediction-markets', 'movers', daysBack, limit],
    queryFn: async () => {
      const { data } = await api.get('/prediction-markets/movers', {
        params: { days_back: daysBack, limit },
      })
      return data
    },
    staleTime: 5 * 60 * 1000,
  })
}

export function useMarketHistory(marketId) {
  return useQuery({
    queryKey: ['prediction-markets', 'history', marketId],
    queryFn: async () => {
      const { data } = await api.get(`/prediction-markets/${marketId}/history`)
      return data
    },
    staleTime: 5 * 60 * 1000,
    enabled: !!marketId,
  })
}
