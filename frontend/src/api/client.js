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
  // Serialize array params as repeated keys (?s=a&s=b) rather than the
  // default bracket form (?s[]=a) — this is the format FastAPI expects for
  // `list[str]` query parameters. `indexes: null` = repeat, no brackets.
  paramsSerializer: { indexes: null },
})

export default api
