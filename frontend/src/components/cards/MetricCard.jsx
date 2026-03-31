/** Compact metric card with glassmorphism, label, large value, and optional subtext. */
export default function MetricCard({ label, value, subtext, color }) {
  return (
    <div className="glass-panel glass-panel-hover p-4">
      <p className="text-[10px] text-text-secondary uppercase tracking-wider font-mono">{label}</p>
      <p className="text-2xl font-bold mt-1" style={color ? { color } : undefined}>
        {value}
      </p>
      {subtext && (
        <p className="text-xs text-text-secondary mt-1">{subtext}</p>
      )}
    </div>
  )
}
