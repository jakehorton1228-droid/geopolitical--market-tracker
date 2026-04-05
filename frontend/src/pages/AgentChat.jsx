/**
 * AI Analyst Chat page — conversational interface to the intelligence pipeline.
 *
 * Features:
 * - Full chat UI with user/assistant message bubbles
 * - Markdown rendering for agent responses (tables, lists, code blocks)
 * - Agent pipeline chips showing which stages ran (collection, analysis, dissemination)
 * - Suggestion buttons for common questions
 * - Animated thinking indicator while pipeline processes
 * - Auto-scroll to latest message
 *
 * Uses the LangGraph multi-agent pipeline:
 *   Collection (deterministic) -> Analysis (deterministic) -> Dissemination (Llama 3 via Ollama)
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

function AgentChips({ agents }) {
  if (!agents || agents.length === 0) return null
  const colors = {
    collection: 'bg-accent-green/10 text-accent-green',
    analysis: 'bg-accent-amber/10 text-accent-amber',
    dissemination: 'bg-accent-blue/10 text-accent-blue',
  }
  return (
    <div className="flex flex-wrap gap-1.5 mt-2">
      {agents.map((a, i) => (
        <motion.span
          key={a}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: i * 0.1, duration: 0.2 }}
          className={`text-[10px] px-2 py-0.5 rounded-full ${colors[a] || 'bg-white/5 text-text-secondary'}`}
        >
          {a}
        </motion.span>
      ))}
    </div>
  )
}

function MessageBubble({ msg }) {
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
              : 'glass-panel text-text-primary'
        }`}
      >
        {isUser ? (
          <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
        ) : (
          <div className="prose-agent text-sm">
            <Markdown>{msg.content}</Markdown>
          </div>
        )}
        {!isUser && <AgentChips agents={msg.agents_used} />}
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
      <div className="glass-panel px-4 py-3">
        <div className="flex items-center gap-2 text-sm text-text-secondary">
          <div className="flex gap-1">
            <span className="w-1.5 h-1.5 bg-accent-blue rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
            <span className="w-1.5 h-1.5 bg-accent-blue rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
            <span className="w-1.5 h-1.5 bg-accent-blue rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
          </div>
          Intelligence pipeline running...
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
            Ask questions in plain English — the AI queries real data, runs analysis, and writes a response grounded in what it finds.
          </p>
        </div>
        <div className="flex items-center gap-3">
          {messages.length > 0 && (
            <motion.button
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={clearHistory}
              className="text-xs text-text-secondary hover:text-text-primary glass-panel px-3 py-1.5 transition-colors"
            >
              Clear chat
            </motion.button>
          )}
        </div>
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
                    className="text-left text-xs text-text-secondary hover:text-text-primary glass-panel glass-panel-hover px-3 py-2 transition-colors"
                  >
                    {s}
                  </motion.button>
                ))}
              </motion.div>
            </motion.div>
          ) : (
            <>
              {messages.map((msg, i) => (
                <MessageBubble key={i} msg={msg} />
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
            className="flex-1 bg-glass border border-glass-border rounded-lg px-4 py-2.5 text-sm text-text-primary placeholder:text-text-secondary/50 focus:outline-none focus:border-accent-blue/50 focus:shadow-[0_0_12px_rgba(59,130,246,0.1)] disabled:opacity-50 backdrop-blur-xl transition-all"
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
