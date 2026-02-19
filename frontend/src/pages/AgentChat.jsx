/**
 * AI Agent Chat page — conversational interface to the Claude-powered analyst.
 *
 * Features:
 * - Full chat UI with user/assistant message bubbles
 * - Markdown rendering for agent responses (tables, lists, code blocks)
 * - Tool call chips showing which analysis tools the agent used
 * - Suggestion buttons for common questions
 * - Animated thinking indicator while agent processes
 * - Auto-scroll to latest message
 *
 * The agent uses Claude's tool use capability to call 10 internal
 * analysis functions (events, correlations, patterns, predictions, anomalies).
 */
import { useState, useRef, useEffect } from 'react'
import Markdown from 'react-markdown'
import { useAgentChat } from '../api/agent'

const SUGGESTIONS = [
  'What are the strongest event-market correlations?',
  'What happened in Russia this month?',
  'Should I be worried about gold prices?',
  'Run a prediction for crude oil (CL=F)',
  'Detect anomalies for SPY over the last 90 days',
  'How does conflict affect natural gas?',
]

function ToolChips({ tools }) {
  if (!tools || tools.length === 0) return null
  return (
    <div className="flex flex-wrap gap-1.5 mt-2">
      {tools.map((t, i) => (
        <span
          key={i}
          className="text-[10px] bg-accent-blue/10 text-accent-blue px-2 py-0.5 rounded-full"
        >
          {t.tool}
        </span>
      ))}
    </div>
  )
}

function MessageBubble({ msg }) {
  const isUser = msg.role === 'user'
  const isError = msg.isError

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[80%] rounded-xl px-4 py-3 ${
          isUser
            ? 'bg-accent-blue text-white'
            : isError
              ? 'bg-red-500/10 border border-red-500/30 text-text-primary'
              : 'bg-bg-secondary border border-border text-text-primary'
        }`}
      >
        {isUser ? (
          <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
        ) : (
          <div className="prose-agent text-sm">
            <Markdown>{msg.content}</Markdown>
          </div>
        )}
        {!isUser && <ToolChips tools={msg.tool_calls} />}
      </div>
    </div>
  )
}

function ThinkingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="bg-bg-secondary border border-border rounded-xl px-4 py-3">
        <div className="flex items-center gap-2 text-sm text-text-secondary">
          <div className="flex gap-1">
            <span className="w-1.5 h-1.5 bg-accent-blue rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
            <span className="w-1.5 h-1.5 bg-accent-blue rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
            <span className="w-1.5 h-1.5 bg-accent-blue rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
          </div>
          Analyzing with tools...
        </div>
      </div>
    </div>
  )
}

export default function AgentChat() {
  const { messages, isLoading, sendMessage, clearHistory } = useAgentChat()
  const [input, setInput] = useState('')
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  const handleSubmit = (e) => {
    e.preventDefault()
    const text = input.trim()
    if (!text || isLoading) return
    setInput('')
    sendMessage(text)
  }

  const handleSuggestion = (text) => {
    if (isLoading) return
    sendMessage(text)
  }

  return (
    <div className="flex flex-col h-[calc(100vh-3rem)]">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-border">
        <div>
          <h2 className="text-2xl font-bold text-text-primary">AI Analyst</h2>
          <p className="text-sm text-text-secondary mt-0.5">
            Claude-powered geopolitical market analysis
          </p>
        </div>
        {messages.length > 0 && (
          <button
            onClick={clearHistory}
            className="text-xs text-text-secondary hover:text-text-primary bg-bg-secondary border border-border px-3 py-1.5 rounded-lg transition-colors"
          >
            Clear chat
          </button>
        )}
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto py-4 space-y-4">
        {messages.length === 0 && !isLoading ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="text-4xl mb-4">&#9789;</div>
            <h3 className="text-lg font-medium text-text-primary mb-2">
              Geopolitical Market Analyst
            </h3>
            <p className="text-sm text-text-secondary max-w-md mb-6">
              Ask me about geopolitical events, market correlations, historical patterns,
              predictions, or anomalies. I have access to 10 years of GDELT event data
              and 33 financial instruments.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-lg w-full">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => handleSuggestion(s)}
                  className="text-left text-xs text-text-secondary hover:text-text-primary bg-bg-secondary hover:bg-bg-tertiary border border-border rounded-lg px-3 py-2 transition-colors"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg, i) => (
              <MessageBubble key={i} msg={msg} />
            ))}
            {isLoading && <ThinkingIndicator />}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input bar */}
      <form onSubmit={handleSubmit} className="pt-4 border-t border-border">
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about events, markets, correlations..."
            disabled={isLoading}
            className="flex-1 bg-bg-secondary border border-border rounded-lg px-4 py-2.5 text-sm text-text-primary placeholder:text-text-secondary/50 focus:outline-none focus:border-accent-blue disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="bg-accent-blue hover:bg-accent-blue/90 disabled:opacity-40 text-white px-5 py-2.5 rounded-lg text-sm font-medium transition-colors"
          >
            Send
          </button>
        </div>
        <p className="text-[10px] text-text-secondary text-center mt-2">
          AI-powered analysis for educational purposes only. Not financial advice.
        </p>
      </form>
    </div>
  )
}
