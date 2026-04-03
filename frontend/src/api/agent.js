/**
 * AI Agent API hook — multi-agent intelligence pipeline chat.
 *
 * - useAgentChat() — Full chat state management:
 *   - messages[]     — Conversation history (user + assistant messages)
 *   - sendMessage()  — POST to /api/agent/chat (LangGraph pipeline)
 *   - isLoading      — True while agent is processing
 *   - error          — Last error message
 *   - clearHistory() — Reset conversation
 */
import { useState, useCallback } from 'react'
import api from './client'

export function useAgentChat() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  const sendMessage = useCallback(async (text) => {
    const userMsg = { role: 'user', content: text }
    setMessages(prev => [...prev, userMsg])
    setIsLoading(true)
    setError(null)

    const history = messages.map(m => ({
      role: m.role,
      content: m.content,
    }))

    try {
      const { data } = await api.post('/agent/chat', {
        message: text,
        history,
      }, { timeout: 180000 })

      const assistantMsg = {
        role: 'assistant',
        content: data.response,
        agents_used: data.agents_used || [],
        iterations: data.iterations || 0,
      }
      setMessages(prev => [...prev, assistantMsg])
    } catch (err) {
      const detail = err.response?.data?.detail || err.message
      setError(detail)
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
