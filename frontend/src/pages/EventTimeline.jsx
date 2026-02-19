/**
 * Event Timeline page — price charts overlaid with geopolitical event markers.
 *
 * Shows a combined line + scatter chart where:
 * - Line: daily closing price for the selected symbol
 * - Dots: geopolitical events colored by type (red=conflict, green=cooperation)
 * - Hover: event details (actor, location, Goldstein score, mentions)
 *
 * Events are filtered to countries relevant to the selected symbol
 * using the SYMBOL_COUNTRY_MAP from the backend.
 */
import { useState } from 'react'
import SymbolSelector from '../components/shared/SymbolSelector'
import DateRangePicker from '../components/shared/DateRangePicker'
import PriceEventOverlay from '../components/charts/PriceEventOverlay'
import LoadingSpinner from '../components/shared/LoadingSpinner'
import { useMarketWithEvents } from '../api/market'
import { EVENT_GROUP_CONFIG } from '../lib/constants'

function daysAgo(n) {
  const d = new Date()
  d.setDate(d.getDate() - n)
  return d.toISOString().split('T')[0]
}

export default function EventTimeline() {
  const [symbol, setSymbol] = useState('CL=F')
  const [startDate, setStartDate] = useState(daysAgo(180))
  const [endDate, setEndDate] = useState(daysAgo(0))

  const { data, isLoading } = useMarketWithEvents(symbol, startDate, endDate)

  const eventDays = data?.filter((d) => d.event_count > 0) ?? []

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-text-primary">Event Timeline</h2>
        <p className="text-sm text-text-secondary mt-1">
          Price chart overlaid with geopolitical events
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-4">
        <SymbolSelector value={symbol} onChange={setSymbol} />
        <DateRangePicker
          startDate={startDate}
          endDate={endDate}
          onStartChange={setStartDate}
          onEndChange={setEndDate}
        />
      </div>

      {isLoading ? (
        <LoadingSpinner message={`Loading ${symbol} data...`} />
      ) : data && data.length > 0 ? (
        <PriceEventOverlay data={data} symbol={symbol} />
      ) : (
        <div className="bg-bg-secondary border border-border rounded-xl p-8 text-center text-text-secondary">
          No data available for {symbol}. Try ingesting data first.
        </div>
      )}

      {eventDays.length > 0 && (
        <div className="bg-bg-secondary border border-border rounded-xl p-4">
          <h3 className="text-sm font-medium text-text-primary mb-3">
            Event Days ({eventDays.length})
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-text-secondary border-b border-border">
                  <th className="text-left p-2">Date</th>
                  <th className="text-right p-2">Close</th>
                  <th className="text-right p-2">Return</th>
                  <th className="text-right p-2">Events</th>
                  <th className="text-right p-2">Goldstein</th>
                  <th className="text-right p-2">Mentions</th>
                  <th className="text-left p-2">Top Event</th>
                </tr>
              </thead>
              <tbody>
                {eventDays.slice().reverse().slice(0, 50).map((d) => (
                  <tr key={d.date} className="border-b border-border/30 hover:bg-bg-tertiary/30">
                    <td className="p-2 text-text-secondary">{d.date}</td>
                    <td className="p-2 text-right font-mono">${d.close.toFixed(2)}</td>
                    <td className="p-2 text-right font-mono" style={{
                      color: (d.daily_return ?? 0) >= 0 ? '#10b981' : '#ef4444',
                    }}>
                      {d.daily_return != null ? `${(d.daily_return * 100).toFixed(2)}%` : '—'}
                    </td>
                    <td className="p-2 text-right">{d.event_count}</td>
                    <td className="p-2 text-right font-mono" style={{
                      color: (d.avg_goldstein ?? 0) < 0 ? '#ef4444' : '#10b981',
                    }}>
                      {d.avg_goldstein?.toFixed(1) ?? '—'}
                    </td>
                    <td className="p-2 text-right">{d.total_mentions ?? 0}</td>
                    <td className="p-2 text-text-secondary">
                      {d.top_event ? (
                        <span className="flex items-center gap-1">
                          <span
                            className="w-1.5 h-1.5 rounded-full inline-block"
                            style={{
                              backgroundColor:
                                EVENT_GROUP_CONFIG[d.top_event.group]?.color ?? '#9ca3af',
                            }}
                          />
                          {d.top_event.description || d.top_event.group}
                        </span>
                      ) : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
