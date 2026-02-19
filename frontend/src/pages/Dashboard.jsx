/**
 * Dashboard page — overview of the geopolitical market tracker.
 *
 * Displays:
 * - Metric cards: total events (since 2016), tracked symbols, strongest correlation, data sources
 * - Top correlations bar chart (from precomputed cache)
 * - Recent high-impact events table (sorted by mentions)
 */
import { useState } from 'react'
import MetricCard from '../components/cards/MetricCard'
import TopCorrelationsBar from '../components/charts/TopCorrelationsBar'
import SkeletonCard from '../components/shared/SkeletonCard'
import SkeletonChart from '../components/shared/SkeletonChart'
import { useEventCount, useEvents } from '../api/events'
import { useSymbols } from '../api/market'
import { useTopCorrelations } from '../api/correlation'

function daysAgo(n) {
  const d = new Date()
  d.setDate(d.getDate() - n)
  return d.toISOString().split('T')[0]
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

  const strongestCorr = topCorr?.[0]

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-text-primary">Dashboard</h2>
        <p className="text-sm text-text-secondary mt-1">
          Overview of geopolitical events and market correlations
        </p>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {countLoading ? <SkeletonCard /> : (
          <MetricCard
            label="Events (since 2016)"
            value={eventCount?.count?.toLocaleString() ?? '0'}
            subtext="GDELT geopolitical events"
          />
        )}
        {symbolsLoading ? <SkeletonCard /> : (
          <MetricCard
            label="Symbols Tracked"
            value={symbols?.length ?? '0'}
            subtext="Commodities, currencies, ETFs"
          />
        )}
        {corrLoading ? <SkeletonCard /> : (
          <MetricCard
            label="Strongest Correlation"
            value={strongestCorr ? strongestCorr.correlation.toFixed(3) : '—'}
            subtext={strongestCorr ? `${strongestCorr.symbol} x ${strongestCorr.event_metric.replace(/_/g, ' ')} (Pearson)` : ''}
            color={strongestCorr?.correlation > 0 ? '#10b981' : '#ef4444'}
          />
        )}
        <MetricCard
          label="Data Sources"
          value="2"
          subtext="GDELT + Yahoo Finance"
        />
      </div>

      {/* Top Correlations */}
      {corrLoading ? (
        <SkeletonChart height="h-72" message="Computing correlations across 33 symbols..." />
      ) : topCorr && topCorr.length > 0 ? (
        <TopCorrelationsBar data={topCorr} />
      ) : (
        <div className="bg-bg-secondary border border-border rounded-xl p-8 text-center text-text-secondary">
          No correlation data available. Ingest events and market data first.
        </div>
      )}

      {/* Recent High-Impact Events */}
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
                {recentEvents.map((e) => (
                  <tr key={e.id} className="border-b border-border/30 hover:bg-bg-tertiary/30">
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
                  </tr>
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
    </div>
  )
}
