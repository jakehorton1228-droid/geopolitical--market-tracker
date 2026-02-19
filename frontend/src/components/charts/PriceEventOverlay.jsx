/** Combined line+scatter chart: price line with geopolitical event dot overlay. */
import {
  ComposedChart, Line, Scatter, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid,
} from 'recharts'
import { COLORS } from '../../lib/constants'

function eventColor(row) {
  if ((row.conflict_count ?? 0) > (row.cooperation_count ?? 0)) return COLORS.red
  if ((row.cooperation_count ?? 0) > (row.conflict_count ?? 0)) return COLORS.green
  return COLORS.amber
}

export default function PriceEventOverlay({ data, symbol }) {
  const withEvents = data
    .filter((d) => d.event_count > 0)
    .map((d) => ({
      ...d,
      eventDot: d.close,
      dotColor: eventColor(d),
    }))

  return (
    <div className="bg-bg-secondary border border-border rounded-xl p-4">
      <h3 className="text-sm font-medium text-text-primary mb-3">
        {symbol} — Price + Events
      </h3>
      <ResponsiveContainer width="100%" height={350}>
        <ComposedChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
          <XAxis
            dataKey="date"
            tick={{ fill: COLORS.gray, fontSize: 11 }}
            tickFormatter={(d) => d.slice(5)}
          />
          <YAxis
            tick={{ fill: COLORS.gray, fontSize: 11 }}
            domain={['auto', 'auto']}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: COLORS.bgSecondary,
              border: `1px solid ${COLORS.border}`,
              borderRadius: 8,
              fontSize: 12,
            }}
            labelStyle={{ color: COLORS.gray }}
            formatter={(value, name) => {
              if (name === 'close') return [`$${value.toFixed(2)}`, 'Price']
              return [value, name]
            }}
          />
          <Line
            type="monotone"
            dataKey="close"
            stroke={COLORS.blue}
            strokeWidth={1.5}
            dot={false}
          />
          <Scatter
            data={withEvents}
            dataKey="eventDot"
            fill={COLORS.red}
            shape={(props) => {
              const { cx, cy, payload } = props
              const r = Math.min(3 + (payload.event_count || 0) * 0.3, 8)
              return (
                <circle
                  cx={cx}
                  cy={cy}
                  r={r}
                  fill={payload.dotColor}
                  fillOpacity={0.7}
                  stroke={payload.dotColor}
                  strokeWidth={1}
                />
              )
            }}
          />
        </ComposedChart>
      </ResponsiveContainer>
      <div className="flex gap-4 mt-2 text-[10px] text-text-secondary justify-center">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-accent-red inline-block" />
          Conflict
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-accent-green inline-block" />
          Cooperation
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-accent-amber inline-block" />
          Mixed
        </span>
      </div>
    </div>
  )
}
