import { useQuery } from '@tanstack/react-query'
import api from './client'

export function useEvents(params = {}) {
  return useQuery({
    queryKey: ['events', params],
    queryFn: async () => {
      const { data } = await api.get('/events', { params })
      return data
    },
    staleTime: 5 * 60 * 1000,
  })
}

export function useEventCount(start_date, end_date) {
  return useQuery({
    queryKey: ['events', 'count', start_date, end_date],
    queryFn: async () => {
      const { data } = await api.get('/events/count', {
        params: { start_date, end_date },
      })
      return data
    },
    staleTime: 5 * 60 * 1000,
  })
}

export function useEventsByCountry(start_date, end_date) {
  return useQuery({
    queryKey: ['events', 'by-country', start_date, end_date],
    queryFn: async () => {
      const { data } = await api.get('/events/by-country', {
        params: { start_date, end_date },
      })
      return data
    },
    staleTime: 5 * 60 * 1000,
  })
}

export function useEventsMap(params = {}) {
  return useQuery({
    queryKey: ['events', 'map', params],
    queryFn: async () => {
      const { data } = await api.get('/events/map', { params })
      return data
    },
    staleTime: 5 * 60 * 1000,
  })
}
