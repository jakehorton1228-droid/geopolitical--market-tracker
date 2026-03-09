/**
 * Intelligence Briefing — flagship dashboard fusing all 5 data sources.
 *
 * 5-Panel Layout:
 * 1. FRED Macro Strip — Economic indicators with animated counters
 * 2. Prediction Market Movers — Biggest probability changes (24h)
 * 3. Fused Event/Price Timeline — Events overlaid on price chart
 * 4. News Headlines — Recent headlines from RSS feeds
 * 5. Risk Radar — Countries with highest event intensity
 */
import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceDot,
} from 'recharts'
import AnimatedNumber from '../components/shared/AnimatedNumber'
import SkeletonCard from '../components/shared/SkeletonCard'
import SkeletonChart from '../components/shared/SkeletonChart'
import { useLatestIndicators } from '../api/indicators'
import { usePredictionMovers, usePredictionMarkets } from '../api/predictionMarkets'
import { useRecentHeadlines } from '../api/headlines'
import { useEventsByCountry } from '../api/events'
import { useMarketWithEvents } from '../api/market'
import { fadeInUp, staggerContainer, staggerItem } from '../utils/animations'

function daysAgo(n) {
  const d = new Date()
  d.setDate(d.getDate() - n)
  return d.toISOString().split('T')[0]
}

const SERIES_LABELS = {
  GDP: { label: 'GDP', unit: '$B', decimals: 1 },
  CPIAUCSL: { label: 'CPI', unit: '', decimals: 1 },
  UNRATE: { label: 'Unemployment', unit: '%', decimals: 1 },
  DFF: { label: 'Fed Funds Rate', unit: '%', decimals: 2 },
  DGS10: { label: '10Y Treasury', unit: '%', decimals: 2 },
  UMCSENT: { label: 'Consumer Sentiment', unit: '', decimals: 1 },
}

function deltaColor(delta) {
  if (delta > 0) return 'text-accent-green'
  if (delta < 0) return 'text-accent-red'
  return 'text-text-secondary'
}

function deltaPrefix(delta) {
  return delta > 0 ? '+' : ''
}

// Sentiment color placeholder (Phase D will add real sentiment)
function headlineSentiment() {
  return 'text-text-primary'
}

export default function IntelligenceBriefing() {
  const [range] = useState({
    start: daysAgo(30),
    end: daysAgo(0),
  })
  const [selectedSymbol] = useState('CL=F') // Oil as default for timeline

  // Data hooks
  const { data: indicators, isLoading: indicatorsLoading } = useLatestIndicators()
  const { data: movers, isLoading: moversLoading } = usePredictionMovers(1, 8)
  const { data: allMarkets, isLoading: marketsLoading } = usePredictionMarkets(8)
  const { data: headlines, isLoading: headlinesLoading } = useRecentHeadlines(null, 3, 10)
  const { data: countryEvents, isLoading: countryLoading } = useEventsByCountry(range.start, range.end)
  const { data: priceEvents, isLoading: priceLoading } = useMarketWithEvents(
    selectedSymbol,
    range.start,
    range.end,
  )

  // Risk radar: top 8 countries by event count
  const riskCountries = useMemo(() => {
    if (!countryEvents) return []
    return [...countryEvents]
      .sort((a, b) => b.count - a.count)
      .slice(0, 8)
  }, [countryEvents])

  // Fused timeline data - API returns flat array with price + event data per day
  const chartData = useMemo(() => {
    if (!priceEvents || !Array.isArray(priceEvents) || priceEvents.length === 0) return []
    return priceEvents.map((p) => ({
      date: p.date,
      price: p.close,
      hasEvent: (p.event_count ?? 0) > 0,
      eventCount: p.event_count ?? 0,
    }))
  }, [priceEvents])

  const eventDots = useMemo(() => {
    if (!chartData.length) return []
    return chartData.filter((d) => d.hasEvent)
  }, [chartData])

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div {...fadeInUp}>
        <h2 className="text-2xl font-bold text-text-primary">Intelligence Briefing</h2>
        <p className="text-sm text-text-secondary mt-1">
          Real-time fusion of geopolitical events, markets, economic indicators, news, and predictions
        </p>
      </motion.div>

      {/* Panel 1: FRED Macro Strip */}
      <motion.div {...fadeInUp} transition={{ delay: 0.1 }}>
        <div className="bg-bg-secondary border border-border rounded-xl p-4">
          <h3 className="text-xs font-medium text-text-secondary uppercase tracking-wide mb-3">
            Economic Indicators
          </h3>
          {indicatorsLoading ? (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 animate-pulse">
              {[...Array(6)].map((_, i) => (
                <div key={i} className="bg-bg-tertiary rounded-lg p-3">
                  <div className="h-3 w-16 bg-bg-primary rounded mb-2" />
                  <div className="h-6 w-12 bg-bg-primary rounded" />
                </div>
              ))}
            </div>
          ) : indicators?.length > 0 ? (
            <motion.div
              className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3"
              variants={staggerContainer.variants}
              initial="initial"
              animate="animate"
            >
              {indicators.map((ind) => {
                const config = SERIES_LABELS[ind.series_id] || { label: ind.series_id, unit: '', decimals: 1 }
                return (
                  <motion.div
                    key={ind.series_id}
                    variants={staggerItem.variants}
                    className="bg-bg-tertiary rounded-lg p-3"
                  >
                    <p className="text-[10px] text-text-secondary uppercase tracking-wide mb-1">
                      {config.label}
                    </p>
                    <p className="text-lg font-bold text-text-primary">
                      <AnimatedNumber
                        value={ind.value}
                        decimals={config.decimals}
                        suffix={config.unit ? ` ${config.unit}` : ''}
                      />
                    </p>
                    {ind.delta != null && (
                      <p className={`text-[10px] font-mono mt-0.5 ${deltaColor(ind.delta)}`}>
                        {deltaPrefix(ind.delta)}{ind.delta.toFixed(config.decimals)}
                      </p>
                    )}
                  </motion.div>
                )
              })}
            </motion.div>
          ) : (
            <p className="text-text-secondary text-sm text-center py-4">
              No FRED data available.
            </p>
          )}
        </div>
      </motion.div>

      {/* Two-column layout for middle panels */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Panel 2: Prediction Markets (movers if available, otherwise top by volume) */}
        <motion.div {...fadeInUp} transition={{ delay: 0.2 }}>
          <div className="bg-bg-secondary border border-border rounded-xl p-4 h-full">
            <h3 className="text-xs font-medium text-text-secondary uppercase tracking-wide mb-3">
              {movers?.length > 0 ? 'Prediction Market Movers (24h)' : 'Top Prediction Markets'}
            </h3>
            {(moversLoading || marketsLoading) ? (
              <div className="space-y-2 animate-pulse">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="flex items-center gap-3">
                    <div className="h-4 flex-1 bg-bg-tertiary rounded" />
                    <div className="h-4 w-16 bg-bg-tertiary rounded" />
                  </div>
                ))}
              </div>
            ) : movers?.length > 0 ? (
              <motion.div
                className="space-y-2"
                variants={staggerContainer.variants}
                initial="initial"
                animate="animate"
              >
                {movers.map((m, i) => {
                  const change = m.probability_change ?? 0
                  const isUp = change > 0
                  return (
                    <motion.div
                      key={m.market_id || i}
                      variants={staggerItem.variants}
                      className="flex items-center gap-3 bg-bg-tertiary rounded-lg px-3 py-2"
                    >
                      <div className="flex-1 min-w-0">
                        <p className="text-sm text-text-primary truncate">
                          {m.question || m.title}
                        </p>
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        <span className="text-sm font-mono text-text-primary">
                          {((m.probability ?? 0) * 100).toFixed(0)}%
                        </span>
                        <span
                          className={`text-xs font-mono ${isUp ? 'text-accent-green' : 'text-accent-red'}`}
                        >
                          {isUp ? '▲' : '▼'} {Math.abs(change * 100).toFixed(1)}%
                        </span>
                      </div>
                    </motion.div>
                  )
                })}
              </motion.div>
            ) : allMarkets?.length > 0 ? (
              <motion.div
                className="space-y-2"
                variants={staggerContainer.variants}
                initial="initial"
                animate="animate"
              >
                {allMarkets.map((m, i) => (
                  <motion.div
                    key={m.market_id || i}
                    variants={staggerItem.variants}
                    className="flex items-center gap-3 bg-bg-tertiary rounded-lg px-3 py-2"
                  >
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-text-primary truncate">
                        {m.question || m.event_title}
                      </p>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <span className="text-sm font-mono text-text-primary">
                        {((m.yes_price ?? 0) * 100).toFixed(0)}%
                      </span>
                      <span className="text-[10px] text-text-secondary">
                        ${(m.volume_24h / 1000).toFixed(0)}k vol
                      </span>
                    </div>
                  </motion.div>
                ))}
              </motion.div>
            ) : (
              <p className="text-text-secondary text-sm text-center py-4">
                No prediction market data available.
              </p>
            )}
          </div>
        </motion.div>

        {/* Panel 4: News Headlines */}
        <motion.div {...fadeInUp} transition={{ delay: 0.3 }}>
          <div className="bg-bg-secondary border border-border rounded-xl p-4 h-full">
            <h3 className="text-xs font-medium text-text-secondary uppercase tracking-wide mb-3">
              Latest Headlines
            </h3>
            {headlinesLoading ? (
              <div className="space-y-2 animate-pulse">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="h-4 bg-bg-tertiary rounded" />
                ))}
              </div>
            ) : headlines?.length > 0 ? (
              <motion.div
                className="space-y-2"
                variants={staggerContainer.variants}
                initial="initial"
                animate="animate"
              >
                {headlines.slice(0, 8).map((h, i) => (
                  <motion.div
                    key={h.id || i}
                    variants={staggerItem.variants}
                    className="flex items-start gap-2"
                  >
                    <span className="text-[10px] text-text-secondary uppercase shrink-0 mt-0.5 w-14">
                      {h.source}
                    </span>
                    <p className={`text-sm ${headlineSentiment()} leading-tight`}>
                      {h.headline}
                    </p>
                  </motion.div>
                ))}
              </motion.div>
            ) : (
              <p className="text-text-secondary text-sm text-center py-4">
                No headlines available.
              </p>
            )}
          </div>
        </motion.div>
      </div>

      {/* Panel 3: Fused Event/Price Timeline */}
      <motion.div {...fadeInUp} transition={{ delay: 0.4 }}>
        <div className="bg-bg-secondary border border-border rounded-xl p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-xs font-medium text-text-secondary uppercase tracking-wide">
              Fused Timeline: {selectedSymbol} + Events
            </h3>
            <span className="text-[10px] text-text-secondary">
              Last 30 days • Red dots = high-impact events
            </span>
          </div>
          {priceLoading ? (
            <SkeletonChart height="h-64" />
          ) : chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={256}>
              <LineChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <XAxis
                  dataKey="date"
                  tick={{ fill: '#9ca3af', fontSize: 10 }}
                  tickFormatter={(d) => d.slice(5)}
                  axisLine={{ stroke: '#374151' }}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: '#9ca3af', fontSize: 10 }}
                  domain={['auto', 'auto']}
                  axisLine={false}
                  tickLine={false}
                  width={50}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1f2937',
                    border: '1px solid #374151',
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  labelStyle={{ color: '#9ca3af' }}
                  formatter={(value, name) => [
                    `$${value.toFixed(2)}`,
                    name === 'price' ? selectedSymbol : name,
                  ]}
                />
                <Line
                  type="monotone"
                  dataKey="price"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={false}
                />
                {eventDots.map((dot, i) => (
                  <ReferenceDot
                    key={i}
                    x={dot.date}
                    y={dot.price}
                    r={5}
                    fill="#ef4444"
                    stroke="#ef4444"
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-64 flex items-center justify-center text-text-secondary text-sm">
              No price/event data available for {selectedSymbol}.
            </div>
          )}
        </div>
      </motion.div>

      {/* Panel 5: Risk Radar */}
      <motion.div {...fadeInUp} transition={{ delay: 0.5 }}>
        <div className="bg-bg-secondary border border-border rounded-xl p-4">
          <h3 className="text-xs font-medium text-text-secondary uppercase tracking-wide mb-3">
            Risk Radar — Hottest Countries (30d)
          </h3>
          {countryLoading ? (
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3 animate-pulse">
              {[...Array(8)].map((_, i) => (
                <div key={i} className="h-16 bg-bg-tertiary rounded-lg" />
              ))}
            </div>
          ) : riskCountries.length > 0 ? (
            <motion.div
              className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3"
              variants={staggerContainer.variants}
              initial="initial"
              animate="animate"
            >
              {riskCountries.map((c, i) => {
                // Heat color based on rank
                const heat = 1 - i / riskCountries.length
                const bgColor = `rgba(239, 68, 68, ${0.1 + heat * 0.3})`
                return (
                  <motion.div
                    key={c.country_code}
                    variants={staggerItem.variants}
                    className="rounded-lg p-3 text-center border border-border"
                    style={{ backgroundColor: bgColor }}
                  >
                    <p className="text-lg font-bold text-text-primary">
                      {c.country_code}
                    </p>
                    <p className="text-xs text-text-secondary mt-1">
                      {c.count.toLocaleString()} events
                    </p>
                  </motion.div>
                )
              })}
            </motion.div>
          ) : (
            <p className="text-text-secondary text-sm text-center py-4">
              No event data available.
            </p>
          )}
        </div>
      </motion.div>
    </div>
  )
}
