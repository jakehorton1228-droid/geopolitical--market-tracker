/**
 * AI Agent API hooks — single-agent and multi-agent chat.
 *
 * - useAgentChat(multiAgent) — Full chat state management:
 *   - messages[]     — Conversation history (user + assistant messages)
 *   - sendMessage()  — POST to /api/agent/chat or /api/agent/chat/multi
 *   - isLoading      — True while agent is processing
 *   - error          — Last error message
 *   - clearHistory() — Reset conversation
 *   - multiAgent     — Whether to use the multi-agent LangGraph pipeline
 *
 * Single-agent: one Claude with all 15 tools (fast, simple).
 * Multi-agent: Supervisor → Collection → Analysis → Dissemination (thorough, slower).
 */
import { useState, useCallback, useRef } from 'react'
import api from './client'

export function useAgentChat(multiAgent = false) {
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

    const endpoint = multiAgent ? '/agent/chat/multi' : '/agent/chat'
    // Multi-agent takes longer (3 Claude calls + tools) — 3 min timeout
    const timeout = multiAgent ? 180000 : 120000

    try {
      const { data } = await api.post(endpoint, {
        message: text,
        history,
      }, { timeout })

      const assistantMsg = {
        role: 'assistant',
        content: data.response,
        tool_calls: data.tool_calls || [],
        model: data.model || '',
        // Multi-agent metadata
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
  }, [messages, multiAgent])

  const clearHistory = useCallback(() => {
    setMessages([])
    setError(null)
  }, [])

  return { messages, isLoading, error, sendMessage, clearHistory }
}
