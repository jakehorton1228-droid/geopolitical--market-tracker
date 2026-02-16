const PRESETS = [
  { label: '30D', days: 30 },
  { label: '90D', days: 90 },
  { label: '180D', days: 180 },
  { label: '1Y', days: 365 },
]

function daysAgo(n) {
  const d = new Date()
  d.setDate(d.getDate() - n)
  return d.toISOString().split('T')[0]
}

function today() {
  return new Date().toISOString().split('T')[0]
}

export default function DateRangePicker({ startDate, endDate, onStartChange, onEndChange }) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex gap-1">
        {PRESETS.map(({ label, days }) => (
          <button
            key={label}
            onClick={() => {
              onStartChange(daysAgo(days))
              onEndChange(today())
            }}
            className="px-2 py-1 text-xs rounded bg-bg-tertiary text-text-secondary hover:text-text-primary hover:bg-border transition-colors"
          >
            {label}
          </button>
        ))}
      </div>
      <input
        type="date"
        value={startDate}
        onChange={(e) => onStartChange(e.target.value)}
        className="bg-bg-tertiary border border-border rounded px-2 py-1 text-xs text-text-primary"
      />
      <span className="text-text-secondary text-xs">to</span>
      <input
        type="date"
        value={endDate}
        onChange={(e) => onEndChange(e.target.value)}
        className="bg-bg-tertiary border border-border rounded px-2 py-1 text-xs text-text-primary"
      />
    </div>
  )
}
