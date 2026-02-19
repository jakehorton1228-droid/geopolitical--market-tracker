/** Horizontal bar chart showing strongest event-market correlation pairs. */
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, Cell,
} from 'recharts'
import { COLORS } from '../../lib/constants'

export default function TopCorrelationsBar({ data }) {
  const chartData = data.map((d) => ({
    label: `${d.symbol} x ${d.event_metric.replace(/_/g, ' ')}`,
    correlation: d.correlation,
  }))

  return (
    <div className="bg-bg-secondary border border-border rounded-xl p-4">
      <h3 className="text-sm font-medium text-text-primary mb-3">
        Strongest Event-Market Correlations
      </h3>
      <ResponsiveContainer width="100%" height={Math.max(200, chartData.length * 28)}>
        <BarChart data={chartData} layout="vertical" margin={{ left: 150, right: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} horizontal={false} />
          <XAxis
            type="number"
            tick={{ fill: COLORS.gray, fontSize: 11 }}
            domain={[-0.5, 0.5]}
          />
          <YAxis
            type="category"
            dataKey="label"
            tick={{ fill: COLORS.gray, fontSize: 10 }}
            width={140}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: COLORS.bgSecondary,
              border: `1px solid ${COLORS.border}`,
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={(value) => [value.toFixed(4), 'Correlation']}
          />
          <Bar dataKey="correlation" radius={[0, 4, 4, 0]}>
            {chartData.map((entry, index) => (
              <Cell
                key={index}
                fill={entry.correlation >= 0 ? COLORS.green : COLORS.red}
                fillOpacity={0.8}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
