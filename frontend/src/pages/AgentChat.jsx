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
import { motion, AnimatePresence } from 'framer-motion'
import Markdown from 'react-markdown'
import { useAgentChat } from '../api/agent'
import { fadeInUp, staggerContainer, staggerItem } from '../utils/animations'

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
        <motion.span
          key={i}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: i * 0.05, duration: 0.2 }}
          className="text-[10px] bg-accent-blue/10 text-accent-blue px-2 py-0.5 rounded-full"
        >
          {t.tool}
        </motion.span>
      ))}
    </div>
  )
}

function MessageBubble({ msg, index }) {
  const isUser = msg.role === 'user'
  const isError = msg.isError

  return (
    <motion.div
      initial={{ opacity: 0, y: 10, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
    >
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
    </motion.div>
  )
}

function ThinkingIndicator() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="flex justify-start"
    >
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
    </motion.div>
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
      <motion.div {...fadeInUp} className="flex items-center justify-between pb-4 border-b border-border">
        <div>
          <h2 className="text-2xl font-bold text-text-primary">AI Analyst</h2>
          <p className="text-sm text-text-secondary mt-0.5">
            Claude-powered geopolitical market analysis
          </p>
        </div>
        {messages.length > 0 && (
          <motion.button
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={clearHistory}
            className="text-xs text-text-secondary hover:text-text-primary bg-bg-secondary border border-border px-3 py-1.5 rounded-lg transition-colors"
          >
            Clear chat
          </motion.button>
        )}
      </motion.div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto py-4 space-y-4">
        <AnimatePresence>
          {messages.length === 0 && !isLoading ? (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center h-full text-center"
            >
              <div className="text-4xl mb-4">&#9789;</div>
              <h3 className="text-lg font-medium text-text-primary mb-2">
                Geopolitical Market Analyst
              </h3>
              <p className="text-sm text-text-secondary max-w-md mb-6">
                Ask me about geopolitical events, market correlations, historical patterns,
                predictions, or anomalies. I have access to 10 years of GDELT event data
                and 33 financial instruments.
              </p>
              <motion.div
                className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-lg w-full"
                variants={staggerContainer.variants}
                initial="initial"
                animate="animate"
              >
                {SUGGESTIONS.map((s) => (
                  <motion.button
                    key={s}
                    variants={staggerItem.variants}
                    whileHover={{ scale: 1.02, y: -1 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => handleSuggestion(s)}
                    className="text-left text-xs text-text-secondary hover:text-text-primary bg-bg-secondary hover:bg-bg-tertiary border border-border rounded-lg px-3 py-2 transition-colors"
                  >
                    {s}
                  </motion.button>
                ))}
              </motion.div>
            </motion.div>
          ) : (
            <>
              {messages.map((msg, i) => (
                <MessageBubble key={i} msg={msg} index={i} />
              ))}
              {isLoading && <ThinkingIndicator />}
            </>
          )}
        </AnimatePresence>
        <div ref={messagesEndRef} />
      </div>

      {/* Input bar */}
      <motion.form
        {...fadeInUp}
        transition={{ delay: 0.2, duration: 0.4 }}
        onSubmit={handleSubmit}
        className="pt-4 border-t border-border"
      >
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
          <motion.button
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            type="submit"
            disabled={isLoading || !input.trim()}
            className="bg-accent-blue hover:bg-accent-blue/90 disabled:opacity-40 text-white px-5 py-2.5 rounded-lg text-sm font-medium transition-colors"
          >
            Send
          </motion.button>
        </div>
        <p className="text-[10px] text-text-secondary text-center mt-2">
          AI-powered analysis for educational purposes only. Not financial advice.
        </p>
      </motion.form>
    </div>
  )
}
