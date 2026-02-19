/** Compact metric card with label, large value, and optional subtext. Used on Dashboard. */
export default function MetricCard({ label, value, subtext, color }) {
  return (
    <div className="bg-bg-secondary border border-border rounded-xl p-4">
      <p className="text-xs text-text-secondary uppercase tracking-wide">{label}</p>
      <p className="text-2xl font-bold mt-1" style={color ? { color } : undefined}>
        {value}
      </p>
      {subtext && (
        <p className="text-xs text-text-secondary mt-1">{subtext}</p>
      )}
    </div>
  )
}
