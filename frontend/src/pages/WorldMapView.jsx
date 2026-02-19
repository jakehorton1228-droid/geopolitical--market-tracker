/**
 * World Map page — choropleth visualization of geopolitical event intensity.
 *
 * Shows a zoomable world map where countries are colored by event count.
 * Clicking a country shows drill-down details: event breakdown by type,
 * average Goldstein score, and total media mentions.
 *
 * Uses react-simple-maps for the map rendering and Natural Earth TopoJSON.
 */
import { useState, useMemo } from 'react'
import {
  ComposableMap, Geographies, Geography, ZoomableGroup,
} from 'react-simple-maps'
import DateRangePicker from '../components/shared/DateRangePicker'
import LoadingSpinner from '../components/shared/LoadingSpinner'
import { useEventsMap } from '../api/events'
import { COLORS } from '../lib/constants'

const GEO_URL = 'https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json'

function daysAgo(n) {
  const d = new Date()
  d.setDate(d.getDate() - n)
  return d.toISOString().split('T')[0]
}

function eventIntensityColor(count) {
  if (count > 500) return '#ef4444'
  if (count > 200) return '#f59e0b'
  if (count > 50) return '#3b82f6'
  if (count > 10) return '#6366f1'
  return '#374151'
}

export default function WorldMapView() {
  const [startDate, setStartDate] = useState(daysAgo(365))
  const [endDate, setEndDate] = useState(daysAgo(0))
  const [selected, setSelected] = useState(null)

  const { data: mapData, isLoading } = useEventsMap({
    start_date: startDate,
    end_date: endDate,
  })

  // Build a lookup by country code for coloring the map
  const countryLookup = useMemo(() => {
    const lookup = {}
    if (mapData) {
      mapData.forEach((c) => {
        lookup[c.country_code] = c
      })
    }
    return lookup
  }, [mapData])

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-text-primary">World Map</h2>
        <p className="text-sm text-text-secondary mt-1">
          Geopolitical event intensity by country
        </p>
      </div>

      <DateRangePicker
        startDate={startDate}
        endDate={endDate}
        onStartChange={setStartDate}
        onEndChange={setEndDate}
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Map */}
        <div className="lg:col-span-2 bg-bg-secondary border border-border rounded-xl p-4">
          {isLoading ? (
            <LoadingSpinner message="Loading map data..." />
          ) : (
            <ComposableMap
              projectionConfig={{ scale: 147, center: [0, 20] }}
              style={{ width: '100%', height: 'auto' }}
            >
              <ZoomableGroup>
                <Geographies geography={GEO_URL}>
                  {({ geographies }) =>
                    geographies.map((geo) => {
                      const iso = geo.properties.ISO_A3 || geo.properties.ADM0_A3
                      const countryData = countryLookup[iso]
                      const count = countryData?.event_count ?? 0

                      return (
                        <Geography
                          key={geo.rsmKey}
                          geography={geo}
                          onClick={() => countryData && setSelected(countryData)}
                          fill={count > 0 ? eventIntensityColor(count) : '#1f2937'}
                          stroke={COLORS.border}
                          strokeWidth={0.5}
                          style={{
                            hover: { fill: '#6366f1', cursor: count > 0 ? 'pointer' : 'default' },
                            pressed: { fill: '#4f46e5' },
                          }}
                        />
                      )
                    })
                  }
                </Geographies>
              </ZoomableGroup>
            </ComposableMap>
          )}

          {/* Legend */}
          <div className="flex gap-4 mt-3 text-[10px] text-text-secondary justify-center">
            {[
              { label: '10+', color: '#6366f1' },
              { label: '50+', color: '#3b82f6' },
              { label: '200+', color: '#f59e0b' },
              { label: '500+', color: '#ef4444' },
            ].map(({ label, color }) => (
              <span key={label} className="flex items-center gap-1">
                <span className="w-3 h-3 rounded" style={{ backgroundColor: color }} />
                {label}
              </span>
            ))}
          </div>
        </div>

        {/* Detail Panel */}
        <div className="bg-bg-secondary border border-border rounded-xl p-4">
          <h3 className="text-sm font-medium text-text-primary mb-3">Country Detail</h3>
          {selected ? (
            <div className="space-y-3">
              <div>
                <p className="text-2xl font-bold">{selected.country_code}</p>
                <p className="text-xs text-text-secondary">{selected.event_count} events</p>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-bg-tertiary rounded-lg p-2">
                  <p className="text-text-secondary">Avg Goldstein</p>
                  <p className="font-bold" style={{
                    color: selected.avg_goldstein < 0 ? COLORS.red : COLORS.green,
                  }}>
                    {selected.avg_goldstein.toFixed(2)}
                  </p>
                </div>
                <div className="bg-bg-tertiary rounded-lg p-2">
                  <p className="text-text-secondary">Mentions</p>
                  <p className="font-bold">{selected.total_mentions.toLocaleString()}</p>
                </div>
                <div className="bg-bg-tertiary rounded-lg p-2">
                  <p className="text-text-secondary">Conflict</p>
                  <p className="font-bold text-accent-red">{selected.conflict_count}</p>
                </div>
                <div className="bg-bg-tertiary rounded-lg p-2">
                  <p className="text-text-secondary">Cooperation</p>
                  <p className="font-bold text-accent-green">{selected.cooperation_count}</p>
                </div>
              </div>
            </div>
          ) : (
            <p className="text-text-secondary text-sm">
              Click a country on the map to see details.
            </p>
          )}

          {/* Top Countries List */}
          {mapData && mapData.length > 0 && (
            <div className="mt-4">
              <p className="text-xs text-text-secondary mb-2">Top Countries</p>
              <div className="space-y-1">
                {mapData.slice(0, 10).map((c) => (
                  <button
                    key={c.country_code}
                    onClick={() => setSelected(c)}
                    className={`w-full flex justify-between items-center px-2 py-1 rounded text-xs transition-colors ${
                      selected?.country_code === c.country_code
                        ? 'bg-accent-blue/20 text-accent-blue'
                        : 'hover:bg-bg-tertiary text-text-secondary'
                    }`}
                  >
                    <span>{c.country_code}</span>
                    <span className="font-mono">{c.event_count}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
