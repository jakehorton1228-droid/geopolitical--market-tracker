export default function SkeletonCard() {
  return (
    <div className="bg-bg-secondary border border-border rounded-xl p-4 animate-pulse">
      <div className="h-3 w-20 bg-bg-tertiary rounded mb-3" />
      <div className="h-7 w-16 bg-bg-tertiary rounded mb-2" />
      <div className="h-3 w-28 bg-bg-tertiary rounded" />
    </div>
  )
}
