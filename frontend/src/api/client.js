/**
 * Shared Axios HTTP client for all API calls.
 *
 * Base URL: /api (proxied to FastAPI backend via nginx)
 * Timeout: 120 seconds (analysis endpoints can be slow on large datasets)
 *
 * All API hook modules (events.js, market.js, etc.) import this client.
 */
import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 120000,
})

export default api
