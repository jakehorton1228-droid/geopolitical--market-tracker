/** Color-coded heatmap grid: symbols (rows) x event metrics (columns) with correlation values. */
import { correlationColor, formatCorrelation } from '../../lib/formatters'

export default function CorrelationHeatmap({ data }) {
  const { symbols, event_metrics, matrix } = data

  return (
    <div className="bg-bg-secondary border border-border rounded-xl p-4 overflow-x-auto">
      <h3 className="text-sm font-medium text-text-primary mb-3">Correlation Matrix</h3>
      <table className="text-xs w-full">
        <thead>
          <tr>
            <th className="text-left p-1 text-text-secondary">Symbol</th>
            {event_metrics.map((m) => (
              <th key={m} className="p-1 text-text-secondary text-center whitespace-nowrap">
                {m.replace(/_/g, ' ')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {symbols.map((sym, symIdx) => (
            <tr key={sym} className="border-t border-border/50">
              <td className="p-1 font-medium text-text-primary">{sym}</td>
              {event_metrics.map((m, metricIdx) => {
                const val = matrix[symIdx]?.[metricIdx] ?? 0
                return (
                  <td
                    key={m}
                    className="p-1 text-center font-mono"
                    style={{ color: correlationColor(val) }}
                  >
                    {formatCorrelation(val)}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
