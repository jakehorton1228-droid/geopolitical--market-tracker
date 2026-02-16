import { useQuery, useMutation } from '@tanstack/react-query'
import api from './client'

export function usePredictLogistic() {
  return useMutation({
    mutationFn: async (input) => {
      const { data } = await api.post('/predictions/logistic', input)
      return data
    },
  })
}

export function useModelSummary(symbol, start_date, end_date) {
  return useQuery({
    queryKey: ['predictions', 'summary', symbol, start_date, end_date],
    queryFn: async () => {
      const { data } = await api.get(
        `/predictions/logistic/${encodeURIComponent(symbol)}/summary`,
        { params: { start_date, end_date } },
      )
      return data
    },
    enabled: !!symbol,
  })
}
