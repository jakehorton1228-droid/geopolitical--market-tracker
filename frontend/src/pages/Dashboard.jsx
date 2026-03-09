/**
 * Dashboard page — overview of the geopolitical market tracker.
 *
 * Displays:
 * - Metric cards: total events (since 2016), tracked symbols, strongest correlation, data sources
 * - FRED economic indicator strip with animated counters
 * - Top correlations bar chart (from precomputed cache)
 * - Recent high-impact events table (sorted by mentions)
 */
import { useState } from 'react'
import { motion } from 'framer-motion'
import MetricCard from '../components/cards/MetricCard'
import TopCorrelationsBar from '../components/charts/TopCorrelationsBar'
import AnimatedNumber from '../components/shared/AnimatedNumber'
import SkeletonCard from '../components/shared/SkeletonCard'
import SkeletonChart from '../components/shared/SkeletonChart'
import { useEventCount, useEvents } from '../api/events'
import { useSymbols } from '../api/market'
import { useTopCorrelations } from '../api/correlation'
import { useLatestIndicators } from '../api/indicators'
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

export default function Dashboard() {
  const [range] = useState({ start: '2016-01-01', end: daysAgo(0) })

  const { data: eventCount, isLoading: countLoading } = useEventCount(range.start, range.end)
  const { data: symbols, isLoading: symbolsLoading } = useSymbols()
  const { data: topCorr, isLoading: corrLoading } = useTopCorrelations(
    range.start, range.end, 10,
  )
  const { data: recentEvents, isLoading: eventsLoading } = useEvents({
    min_mentions: 20,
    limit: 15,
    start_date: range.start,
    end_date: range.end,
  })
  const { data: indicators, isLoading: indicatorsLoading } = useLatestIndicators()

  const strongestCorr = topCorr?.[0]

  return (
    <div className="space-y-6">
      <motion.div {...fadeInUp}>
        <h2 className="text-2xl font-bold text-text-primary">Dashboard</h2>
        <p className="text-sm text-text-secondary mt-1">
          Overview of geopolitical events and market correlations
        </p>
      </motion.div>

      {/* Metric Cards */}
      <motion.div
        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"
        variants={staggerContainer.variants}
        initial="initial"
        animate="animate"
      >
        <motion.div variants={staggerItem.variants}>
          {countLoading ? <SkeletonCard /> : (
            <MetricCard
              label="Events (since 2016)"
              value={eventCount?.count?.toLocaleString() ?? '0'}
              subtext="GDELT geopolitical events"
            />
          )}
        </motion.div>
        <motion.div variants={staggerItem.variants}>
          {symbolsLoading ? <SkeletonCard /> : (
            <MetricCard
              label="Symbols Tracked"
              value={symbols?.length ?? '0'}
              subtext="Commodities, currencies, ETFs"
            />
          )}
        </motion.div>
        <motion.div variants={staggerItem.variants}>
          {corrLoading ? <SkeletonCard /> : (
            <MetricCard
              label="Strongest Correlation"
              value={strongestCorr ? strongestCorr.correlation.toFixed(3) : '—'}
              subtext={strongestCorr ? `${strongestCorr.symbol} x ${strongestCorr.event_metric.replace(/_/g, ' ')} (Pearson)` : ''}
              color={strongestCorr?.correlation > 0 ? '#10b981' : '#ef4444'}
            />
          )}
        </motion.div>
        <motion.div variants={staggerItem.variants}>
          <MetricCard
            label="Data Sources"
            value="5"
            subtext="GDELT + Yahoo + RSS + FRED + Polymarket"
          />
        </motion.div>
      </motion.div>

      {/* FRED Economic Indicators */}
      <motion.div {...fadeInUp} transition={{ delay: 0.2, duration: 0.4 }}>
        <div className="bg-bg-secondary border border-border rounded-xl p-4">
          <h3 className="text-sm font-medium text-text-primary mb-3">
            Economic Indicators (FRED)
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
          ) : indicators && indicators.length > 0 ? (
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
                    <p className="text-[9px] text-text-secondary mt-0.5">
                      {ind.date}
                    </p>
                  </motion.div>
                )
              })}
            </motion.div>
          ) : (
            <p className="text-text-secondary text-sm text-center py-4">
              No FRED data yet. Run the ingestion pipeline to fetch economic indicators.
            </p>
          )}
        </div>
      </motion.div>

      {/* Top Correlations */}
      <motion.div {...fadeInUp} transition={{ delay: 0.3, duration: 0.4 }}>
        {corrLoading ? (
          <SkeletonChart height="h-72" message="Computing correlations across 33 symbols..." />
        ) : topCorr && topCorr.length > 0 ? (
          <TopCorrelationsBar data={topCorr} />
        ) : (
          <div className="bg-bg-secondary border border-border rounded-xl p-8 text-center text-text-secondary">
            No correlation data available. Ingest events and market data first.
          </div>
        )}
      </motion.div>

      {/* Recent High-Impact Events */}
      <motion.div {...fadeInUp} transition={{ delay: 0.4, duration: 0.4 }}>
        <div className="bg-bg-secondary border border-border rounded-xl p-4">
          <h3 className="text-sm font-medium text-text-primary mb-3">
            Recent High-Impact Events
          </h3>
          {eventsLoading ? (
            <div className="space-y-2 animate-pulse">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="flex gap-4">
                  <div className="h-4 w-20 bg-bg-tertiary rounded" />
                  <div className="h-4 w-32 bg-bg-tertiary rounded" />
                  <div className="h-4 flex-1 bg-bg-tertiary rounded" />
                  <div className="h-4 w-12 bg-bg-tertiary rounded" />
                </div>
              ))}
            </div>
          ) : recentEvents && recentEvents.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-text-secondary border-b border-border">
                    <th className="text-left p-2">Date</th>
                    <th className="text-left p-2">Location</th>
                    <th className="text-left p-2">Actors</th>
                    <th className="text-right p-2">Goldstein</th>
                    <th className="text-right p-2">Mentions</th>
                  </tr>
                </thead>
                <tbody>
                  {recentEvents.map((e, i) => (
                    <motion.tr
                      key={e.id}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: i * 0.03, duration: 0.3 }}
                      className="border-b border-border/30 hover:bg-bg-tertiary/30"
                    >
                      <td className="p-2 text-text-secondary">{e.event_date}</td>
                      <td className="p-2">{e.action_geo_name || e.action_geo_country_code || '—'}</td>
                      <td className="p-2 text-text-secondary">
                        {e.actor1_name || '?'} → {e.actor2_name || '?'}
                      </td>
                      <td className="p-2 text-right font-mono" style={{
                        color: (e.goldstein_scale ?? 0) < 0 ? '#ef4444' : '#10b981',
                      }}>
                        {e.goldstein_scale?.toFixed(1) ?? '—'}
                      </td>
                      <td className="p-2 text-right font-mono">{e.num_mentions ?? '—'}</td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-text-secondary text-sm text-center py-4">
              No high-impact events found. Try ingesting data first.
            </p>
          )}
        </div>
      </motion.div>
    </div>
  )
}
