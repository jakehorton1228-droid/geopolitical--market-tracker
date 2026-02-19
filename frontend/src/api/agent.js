import { useState, useCallback, useRef } from 'react'
import api from './client'

/**
 * Custom hook for managing agent chat state and API calls.
 */
export function useAgentChat() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const abortRef = useRef(null)

  const sendMessage = useCallback(async (text) => {
    const userMsg = { role: 'user', content: text }
    setMessages(prev => [...prev, userMsg])
    setIsLoading(true)
    setError(null)

    // Build history from previous messages (for multi-turn)
    const history = messages.map(m => ({
      role: m.role,
      content: m.content,
    }))

    try {
      const { data } = await api.post('/agent/chat', {
        message: text,
        history,
      }, { timeout: 120000 }) // 2 min timeout for agent calls

      const assistantMsg = {
        role: 'assistant',
        content: data.response,
        tool_calls: data.tool_calls || [],
        model: data.model,
      }
      setMessages(prev => [...prev, assistantMsg])
    } catch (err) {
      const detail = err.response?.data?.detail || err.message
      setError(detail)
      // Add error as a system message so user sees it
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${detail}`,
        isError: true,
      }])
    } finally {
      setIsLoading(false)
    }
  }, [messages])

  const clearHistory = useCallback(() => {
    setMessages([])
    setError(null)
  }, [])

  return { messages, isLoading, error, sendMessage, clearHistory }
}
