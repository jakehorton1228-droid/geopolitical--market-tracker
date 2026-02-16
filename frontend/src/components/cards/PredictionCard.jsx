import { signalColor } from '../../lib/formatters'

export default function PredictionCard({ prediction }) {
  const color = signalColor(prediction.probability_up)

  return (
    <div className="bg-bg-secondary border border-border rounded-xl p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-text-secondary uppercase tracking-wide">
          Logistic Regression
        </span>
        <span className="text-xs text-text-secondary">
          Acc: {(prediction.accuracy * 100).toFixed(0)}%
        </span>
      </div>

      <div className="flex items-center gap-3">
        <span className="text-2xl font-bold" style={{ color }}>
          {prediction.prediction}
        </span>
        <span className="text-lg text-text-secondary">
          {(prediction.probability_up * 100).toFixed(0)}% UP
        </span>
      </div>

      {/* Probability bar */}
      <div className="mt-3 h-2 rounded-full bg-bg-tertiary overflow-hidden flex">
        <div
          className="h-full transition-all"
          style={{
            width: `${prediction.probability_up * 100}%`,
            backgroundColor: color,
          }}
        />
      </div>

      {/* Top feature contributions */}
      {prediction.feature_contributions.length > 0 && (
        <div className="mt-3 space-y-1">
          <p className="text-xs text-text-secondary">Key Drivers:</p>
          {prediction.feature_contributions.slice(0, 3).map((fc) => (
            <div key={fc.feature} className="flex justify-between text-xs">
              <span className="text-text-secondary">
                {fc.feature.replace(/_/g, ' ')}
              </span>
              <span className={fc.contribution > 0 ? 'text-accent-green' : 'text-accent-red'}>
                {fc.contribution > 0 ? '+' : ''}{fc.contribution.toFixed(3)}
              </span>
            </div>
          ))}
        </div>
      )}

      <p className="text-[10px] text-text-secondary mt-3 italic">
        {prediction.disclaimer}
      </p>
    </div>
  )
}
