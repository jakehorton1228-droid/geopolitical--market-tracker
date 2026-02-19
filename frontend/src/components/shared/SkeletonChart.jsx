export default function SkeletonChart({ height = 'h-64', message = 'Loading...' }) {
  return (
    <div className="bg-bg-secondary border border-border rounded-xl p-4 animate-pulse">
      <div className="h-4 w-40 bg-bg-tertiary rounded mb-4" />
      <div className={`${height} bg-bg-tertiary rounded flex items-center justify-center`}>
        <p className="text-sm text-text-secondary animate-pulse">{message}</p>
      </div>
    </div>
  )
}
