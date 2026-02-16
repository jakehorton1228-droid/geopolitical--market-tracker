import { formatPercent } from '../../lib/formatters'

export default function PatternCard({ pattern }) {
  const upColor =
    pattern.up_percentage >= 55
      ? 'text-accent-green'
      : pattern.up_percentage <= 45
        ? 'text-accent-red'
        : 'text-text-secondary'

  return (
    <div className="bg-bg-secondary border border-border rounded-xl p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-text-secondary uppercase tracking-wide">
          {pattern.event_filter}
        </span>
        {pattern.is_significant && (
          <span className="text-[10px] bg-accent-blue/20 text-accent-blue px-1.5 py-0.5 rounded">
            Significant
          </span>
        )}
      </div>

      <p className={`text-xl font-bold ${upColor}`}>
        UP {pattern.up_percentage.toFixed(0)}% of the time
      </p>

      <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
        <div>
          <p className="text-text-secondary">Occurrences</p>
          <p className="font-medium">{pattern.total_occurrences}</p>
        </div>
        <div>
          <p className="text-text-secondary">Avg Return</p>
          <p className="font-medium">{formatPercent(pattern.avg_return_all)}</p>
        </div>
        <div>
          <p className="text-text-secondary">p-value</p>
          <p className="font-medium">
            {pattern.p_value < 0.001 ? '<0.001' : pattern.p_value.toFixed(3)}
          </p>
        </div>
      </div>

      {/* Up/Down bar */}
      <div className="mt-3 h-2 rounded-full bg-bg-tertiary overflow-hidden flex">
        <div
          className="bg-accent-green h-full"
          style={{ width: `${pattern.up_percentage}%` }}
        />
        <div
          className="bg-accent-red h-full"
          style={{ width: `${100 - pattern.up_percentage}%` }}
        />
      </div>
      <div className="flex justify-between text-[10px] text-text-secondary mt-1">
        <span>{pattern.up_count} up</span>
        <span>{pattern.down_count} down</span>
      </div>
    </div>
  )
}
