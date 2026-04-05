/**
 * Correlation Explorer page — interactive analysis of event-market correlations.
 *
 * Features:
 * - Per-symbol correlation table (goldstein, mentions, tone, conflict, cooperation vs returns)
 * - Rolling correlation timeseries chart with confidence intervals
 * - Correlation heatmap across multiple symbols
 * - Symbol and date range selectors
 */
import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, AreaChart, Area,
} from 'recharts'
import SymbolSelector from '../components/shared/SymbolSelector'
import DateRangePicker from '../components/shared/DateRangePicker'
import CorrelationHeatmap from '../components/charts/CorrelationHeatmap'
import LoadingSpinner from '../components/shared/LoadingSpinner'
import PageHelp from '../components/shared/PageHelp'
import InfoTooltip from '../components/shared/InfoTooltip'
import { useCorrelations, useRollingCorrelation, useCorrelationHeatmap } from '../api/correlation'
import { correlationColor, formatCorrelation } from '../utils/formatters'
import { COLORS, DEFAULT_SYMBOLS } from '../utils/constants'
import { fadeInUp, staggerContainer, staggerItem } from '../utils/animations'
import { GLOSSARY } from '../utils/glossary'

function daysAgo(n) {
  const d = new Date()
  d.setDate(d.getDate() - n)
  return d.toISOString().split('T')[0]
}

export default function CorrelationExplorer() {
  const [symbol, setSymbol] = useState('CL=F')
  const [startDate, setStartDate] = useState(daysAgo(3650))
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
      <motion.div {...fadeInUp}>
        <h2 className="text-2xl font-bold text-text-primary">Correlation Explorer</h2>
        <p className="text-sm text-text-secondary mt-1">
          Do geopolitical events move markets? Pick an asset and see which event types have the strongest statistical relationship with its price.
        </p>
      </motion.div>

      <PageHelp
        description="A correlation measures how two things move together. +1 means they always rise and fall in sync; -1 means they move inversely; 0 means no relationship. For event-market data, correlations are usually small — even 0.15 can be meaningful."
        lookFor={[
          'Values above 0.2 (positive) or below -0.2 (negative) — worth paying attention to',
          'Low p-values (p < 0.05) — statistically significant, unlikely to be random',
          'Large sample sizes (n > 100) — more reliable than tiny samples',
          'Stable rolling correlations — if the line is flat, the relationship is consistent; if it swings, the relationship shifts with market regimes',
        ]}
        terms={[GLOSSARY.correlation, GLOSSARY.pvalue, GLOSSARY.nObservations, GLOSSARY.confidenceInterval]}
      />

      {/* Controls */}
      <motion.div {...fadeInUp} transition={{ delay: 0.1, duration: 0.4 }} className="flex flex-wrap items-center gap-4">
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
          className="bg-glass border border-glass-border rounded-lg px-3 py-1.5 text-sm text-text-primary backdrop-blur-xl focus:outline-none focus:border-accent-blue/50 transition-all"
        >
          <option value="pearson">Pearson</option>
          <option value="spearman">Spearman</option>
        </select>
      </motion.div>

      {/* Per-Metric Correlation Cards */}
      <motion.div {...fadeInUp} transition={{ delay: 0.2, duration: 0.4 }}>
        <div className="glass-panel p-4 border-gradient-top">
          <h3 className="section-label mb-3 flex items-center gap-2">
            <span>{symbol} — Event Metrics vs Price Moves</span>
            <InfoTooltip {...GLOSSARY.correlation} />
            <span className="text-[10px] text-text-secondary normal-case tracking-normal ml-2">
              click a card to see how the relationship evolved over time
            </span>
          </h3>
          {corrLoading ? (
            <LoadingSpinner />
          ) : correlations && correlations.length > 0 ? (
            <motion.div
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3"
              variants={staggerContainer.variants}
              initial="initial"
              animate="animate"
            >
              {correlations.map((c) => {
                const abs = Math.abs(c.correlation)
                const strength = abs >= 0.3 ? 'strong' : abs >= 0.15 ? 'moderate' : abs >= 0.05 ? 'weak' : 'negligible'
                const sig = c.p_value < 0.05
                return (
                  <motion.button
                    key={c.event_metric}
                    variants={staggerItem.variants}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => setSelectedMetric(c.event_metric)}
                    className={`text-left p-3 rounded-lg border transition-colors ${
                      selectedMetric === c.event_metric
                        ? 'border-accent-blue bg-accent-blue/10'
                        : 'border-border/50 hover:border-border'
                    }`}
                  >
                    <p className="text-xs text-text-secondary capitalize">
                      {c.event_metric.replace(/_/g, ' ')}
                    </p>
                    <p
                      className="text-lg font-bold font-mono"
                      style={{ color: correlationColor(c.correlation) }}
                    >
                      {formatCorrelation(c.correlation)}
                    </p>
                    <p className="text-[10px] text-text-secondary mt-1">
                      <span className="capitalize">{strength}</span>
                      {sig ? ' · significant' : ' · not significant'}
                    </p>
                    <p className="text-[9px] text-text-secondary/70 mt-0.5 font-mono">
                      p={c.p_value < 0.001 ? '<0.001' : c.p_value.toFixed(3)} · n={c.n_observations}
                    </p>
                  </motion.button>
                )
              })}
            </motion.div>
          ) : (
            <p className="text-text-secondary text-sm text-center py-4">
              No correlation data available for {symbol}.
            </p>
          )}
        </div>
      </motion.div>

      {/* Rolling Correlation */}
      <motion.div {...fadeInUp} transition={{ delay: 0.3, duration: 0.4 }}>
        <div className="glass-panel p-4 border-gradient-top">
          <h3 className="section-label mb-3 flex items-center gap-2">
            <span>Relationship Over Time: {symbol} × {selectedMetric.replace(/_/g, ' ')}</span>
            <InfoTooltip {...GLOSSARY.confidenceInterval} />
            <span className="text-[10px] text-text-secondary normal-case tracking-normal ml-2">
              flat line = stable relationship; swings = shifts with market regime
            </span>
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
                    color: '#e2e8f0',
                  }}
                  labelStyle={{ color: '#94a3b8' }}
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
      </motion.div>

      {/* Heatmap */}
      <motion.div {...fadeInUp} transition={{ delay: 0.4, duration: 0.4 }}>
        {heatmapLoading ? (
          <LoadingSpinner message="Computing heatmap..." />
        ) : heatmap ? (
          <CorrelationHeatmap data={heatmap} />
        ) : null}
      </motion.div>
    </div>
  )
}
