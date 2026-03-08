/**
 * Prediction Markets page — browse geopolitical prediction markets from Polymarket.
 *
 * Displays a sortable table of all tracked markets showing:
 * - Question text and parent event
 * - Current probability (yes_price) as a visual bar
 * - 24h trading volume
 * - Expandable row with probability trend chart
 *
 * Uses Framer Motion for:
 * - Staggered row entrance animation
 * - Smooth expand/collapse of the chart panel (AnimatePresence + layoutId)
 */
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { usePredictionMarkets, useMarketHistory } from '../api/predictionMarkets'
import { staggerContainer, staggerItem, fadeInUp } from '../utils/animations'
import { COLORS } from '../utils/constants'
import SkeletonChart from '../components/shared/SkeletonChart'

/** Format a probability as a percentage string. */
function formatProb(p) {
  return `${(p * 100).toFixed(1)}%`
}

/** Format large numbers with K/M suffixes. */
function formatVolume(v) {
  if (!v) return '$0'
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`
  if (v >= 1_000) return `$${(v / 1_000).toFixed(0)}K`
  return `$${v.toFixed(0)}`
}

/** Expandable chart section for a single market. */
function MarketChart({ marketId }) {
  const { data: history, isLoading } = useMarketHistory(marketId)

  if (isLoading) {
    return <SkeletonChart height="h-40" message="Loading probability history..." />
  }

  if (!history || history.length === 0) {
    return (
      <p className="text-xs text-text-secondary text-center py-4">
        No historical data yet — snapshots build up over daily runs.
      </p>
    )
  }

  const chartData = history.map((s) => ({
    date: s.snapshot_date,
    probability: s.yes_price * 100,
  }))

  return (
    <div className="h-40">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData}>
          <XAxis
            dataKey="date"
            tick={{ fill: COLORS.gray, fontSize: 10 }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fill: COLORS.gray, fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => `${v}%`}
            width={40}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: COLORS.bgSecondary,
              border: `1px solid ${COLORS.border}`,
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={(v) => [`${v.toFixed(1)}%`, 'Probability']}
          />
          <Line
            type="monotone"
            dataKey="probability"
            stroke={COLORS.blue}
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

/** Single market row with expand/collapse. */
function MarketRow({ market, isExpanded, onToggle }) {
  return (
    <motion.div variants={staggerItem.variants} layout>
      {/* Clickable row */}
      <button
        onClick={onToggle}
        className="w-full text-left grid grid-cols-[1fr_120px_100px_32px] gap-3 items-center px-4 py-3 hover:bg-bg-tertiary/30 transition-colors border-b border-border/30"
      >
        {/* Question + event */}
        <div className="min-w-0">
          <p className="text-sm text-text-primary truncate">{market.question}</p>
          {market.event_title && (
            <p className="text-xs text-text-secondary truncate mt-0.5">
              {market.event_title}
            </p>
          )}
        </div>

        {/* Probability bar */}
        <div className="flex items-center gap-2">
          <div className="flex-1 h-2 bg-bg-tertiary rounded-full overflow-hidden">
            <motion.div
              className="h-full rounded-full"
              style={{
                backgroundColor:
                  market.yes_price > 0.6
                    ? COLORS.green
                    : market.yes_price > 0.3
                    ? COLORS.amber
                    : COLORS.red,
              }}
              initial={{ width: 0 }}
              animate={{ width: `${market.yes_price * 100}%` }}
              transition={{ duration: 0.8, ease: 'easeOut' }}
            />
          </div>
          <span className="text-xs font-mono text-text-primary w-12 text-right">
            {formatProb(market.yes_price)}
          </span>
        </div>

        {/* Volume */}
        <span className="text-xs font-mono text-text-secondary text-right">
          {formatVolume(market.volume_24h)}
        </span>

        {/* Expand indicator */}
        <motion.span
          className="text-text-secondary text-sm"
          animate={{ rotate: isExpanded ? 180 : 0 }}
          transition={{ duration: 0.2 }}
        >
          ▾
        </motion.span>
      </button>

      {/* Expandable chart */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: 'easeInOut' }}
            className="overflow-hidden bg-bg-tertiary/20 px-4 py-3 border-b border-border/30"
          >
            <MarketChart marketId={market.market_id} />
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

export default function PredictionMarkets() {
  const { data: markets, isLoading } = usePredictionMarkets(100)
  const [expandedId, setExpandedId] = useState(null)
  const [sortBy, setSortBy] = useState('volume') // 'volume' | 'probability' | 'question'

  const sortedMarkets = markets
    ? [...markets].sort((a, b) => {
        if (sortBy === 'volume') return (b.volume_24h || 0) - (a.volume_24h || 0)
        if (sortBy === 'probability') return b.yes_price - a.yes_price
        if (sortBy === 'question') return a.question.localeCompare(b.question)
        return 0
      })
    : []

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div {...fadeInUp}>
        <h2 className="text-2xl font-bold text-text-primary">Prediction Markets</h2>
        <p className="text-sm text-text-secondary mt-1">
          Geopolitical prediction market odds from Polymarket — click any row to see the probability trend
        </p>
      </motion.div>

      {/* Sort controls */}
      <motion.div {...fadeInUp} className="flex gap-2">
        {[
          { key: 'volume', label: 'Volume' },
          { key: 'probability', label: 'Probability' },
          { key: 'question', label: 'A-Z' },
        ].map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setSortBy(key)}
            className={`px-3 py-1.5 text-xs rounded-lg transition-colors ${
              sortBy === key
                ? 'bg-accent-blue text-white'
                : 'bg-bg-secondary text-text-secondary hover:text-text-primary border border-border'
            }`}
          >
            {label}
          </button>
        ))}
        <span className="text-xs text-text-secondary self-center ml-2">
          {sortedMarkets.length} markets
        </span>
      </motion.div>

      {/* Markets table */}
      <div className="bg-bg-secondary border border-border rounded-xl overflow-hidden">
        {/* Column headers */}
        <div className="grid grid-cols-[1fr_120px_100px_32px] gap-3 px-4 py-2 border-b border-border text-xs text-text-secondary uppercase tracking-wide">
          <span>Market</span>
          <span className="text-right">Probability</span>
          <span className="text-right">24h Volume</span>
          <span></span>
        </div>

        {/* Rows */}
        {isLoading ? (
          <div className="space-y-0 animate-pulse">
            {[...Array(8)].map((_, i) => (
              <div key={i} className="px-4 py-3 border-b border-border/30">
                <div className="h-4 w-3/4 bg-bg-tertiary rounded mb-1" />
                <div className="h-3 w-1/3 bg-bg-tertiary rounded" />
              </div>
            ))}
          </div>
        ) : sortedMarkets.length > 0 ? (
          <motion.div
            variants={staggerContainer.variants}
            initial="initial"
            animate="animate"
          >
            {sortedMarkets.map((market) => (
              <MarketRow
                key={market.market_id}
                market={market}
                isExpanded={expandedId === market.market_id}
                onToggle={() =>
                  setExpandedId(
                    expandedId === market.market_id ? null : market.market_id
                  )
                }
              />
            ))}
          </motion.div>
        ) : (
          <div className="px-4 py-8 text-center text-text-secondary text-sm">
            No prediction market data yet. Run the ingestion pipeline to fetch Polymarket data.
          </div>
        )}
      </div>
    </div>
  )
}
