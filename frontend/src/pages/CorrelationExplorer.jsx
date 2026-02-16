import { useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, AreaChart, Area,
} from 'recharts'
import SymbolSelector from '../components/shared/SymbolSelector'
import DateRangePicker from '../components/shared/DateRangePicker'
import CorrelationHeatmap from '../components/charts/CorrelationHeatmap'
import LoadingSpinner from '../components/shared/LoadingSpinner'
import { useCorrelations, useRollingCorrelation, useCorrelationHeatmap } from '../api/correlation'
import { correlationColor, formatCorrelation } from '../lib/formatters'
import { COLORS, DEFAULT_SYMBOLS } from '../lib/constants'

function daysAgo(n) {
  const d = new Date()
  d.setDate(d.getDate() - n)
  return d.toISOString().split('T')[0]
}

export default function CorrelationExplorer() {
  const [symbol, setSymbol] = useState('CL=F')
  const [startDate, setStartDate] = useState(daysAgo(365))
  const [endDate, setEndDate] = useState(daysAgo(0))
  const [selectedMetric, setSelectedMetric] = useState('conflict_count')
  const [method, setMethod] = useState('pearson')

  const { data: correlations, isLoading: corrLoading } = useCorrelations(
    symbol, startDate, endDate, method,
  )
  const { data: rolling, isLoading: rollingLoading } = useRollingCorrelation(
    symbol, selectedMetric, startDate, endDate,
  )
  const { data: heatmap, isLoading: heatmapLoading } = useCorrelationHeatmap(
    DEFAULT_SYMBOLS.slice(0, 6).join(','), startDate, endDate, method,
  )

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-text-primary">Correlation Explorer</h2>
        <p className="text-sm text-text-secondary mt-1">
          Analyze how geopolitical event metrics correlate with market returns
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4">
        <SymbolSelector value={symbol} onChange={setSymbol} />
        <DateRangePicker
          startDate={startDate}
          endDate={endDate}
          onStartChange={setStartDate}
          onEndChange={setEndDate}
        />
        <select
          value={method}
          onChange={(e) => setMethod(e.target.value)}
          className="bg-bg-tertiary border border-border rounded-lg px-3 py-1.5 text-sm text-text-primary"
        >
          <option value="pearson">Pearson</option>
          <option value="spearman">Spearman</option>
        </select>
      </div>

      {/* Per-Metric Correlation Cards */}
      <div className="bg-bg-secondary border border-border rounded-xl p-4">
        <h3 className="text-sm font-medium text-text-primary mb-3">
          {symbol} — Correlation per Event Metric
        </h3>
        {corrLoading ? (
          <LoadingSpinner />
        ) : correlations && correlations.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {correlations.map((c) => (
              <button
                key={c.event_metric}
                onClick={() => setSelectedMetric(c.event_metric)}
                className={`text-left p-3 rounded-lg border transition-colors ${
                  selectedMetric === c.event_metric
                    ? 'border-accent-blue bg-accent-blue/10'
                    : 'border-border/50 hover:border-border'
                }`}
              >
                <p className="text-xs text-text-secondary">
                  {c.event_metric.replace(/_/g, ' ')}
                </p>
                <p
                  className="text-lg font-bold font-mono"
                  style={{ color: correlationColor(c.correlation) }}
                >
                  {formatCorrelation(c.correlation)}
                </p>
                <p className="text-[10px] text-text-secondary mt-1">
                  p={c.p_value < 0.001 ? '<0.001' : c.p_value.toFixed(3)} | n={c.n_observations}
                </p>
              </button>
            ))}
          </div>
        ) : (
          <p className="text-text-secondary text-sm text-center py-4">
            No correlation data available for {symbol}.
          </p>
        )}
      </div>

      {/* Rolling Correlation */}
      <div className="bg-bg-secondary border border-border rounded-xl p-4">
        <h3 className="text-sm font-medium text-text-primary mb-3">
          Rolling Correlation: {symbol} x {selectedMetric.replace(/_/g, ' ')}
        </h3>
        {rollingLoading ? (
          <LoadingSpinner />
        ) : rolling && rolling.data.length > 0 ? (
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={rolling.data} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
              <XAxis
                dataKey="date"
                tick={{ fill: COLORS.gray, fontSize: 11 }}
                tickFormatter={(d) => d.slice(5)}
              />
              <YAxis tick={{ fill: COLORS.gray, fontSize: 11 }} domain={[-1, 1]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: COLORS.bgSecondary,
                  border: `1px solid ${COLORS.border}`,
                  borderRadius: 8,
                  fontSize: 12,
                }}
              />
              <Area dataKey="upper_ci" stroke="none" fill={COLORS.blue} fillOpacity={0.1} />
              <Area dataKey="lower_ci" stroke="none" fill={COLORS.blue} fillOpacity={0.1} />
              <Line type="monotone" dataKey="correlation" stroke={COLORS.blue} strokeWidth={2} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-text-secondary text-sm text-center py-4">
            Insufficient data for rolling correlation.
          </p>
        )}
      </div>

      {/* Heatmap */}
      {heatmapLoading ? (
        <LoadingSpinner message="Computing heatmap..." />
      ) : heatmap ? (
        <CorrelationHeatmap data={heatmap} />
      ) : null}
    </div>
  )
}
