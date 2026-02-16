import { useState } from 'react'
import SymbolSelector from '../components/shared/SymbolSelector'
import PatternCard from '../components/cards/PatternCard'
import PredictionCard from '../components/cards/PredictionCard'
import LoadingSpinner from '../components/shared/LoadingSpinner'
import { useAllPatterns } from '../api/patterns'
import { useModelSummary } from '../api/predictions'
import { usePredictLogistic } from '../api/predictions'
import { DEFAULT_SYMBOLS } from '../lib/constants'

function daysAgo(n) {
  const d = new Date()
  d.setDate(d.getDate() - n)
  return d.toISOString().split('T')[0]
}

export default function Signals() {
  const [symbol, setSymbol] = useState('CL=F')
  const startDate = daysAgo(365)
  const endDate = daysAgo(0)

  const { data: patterns, isLoading: patternsLoading } = useAllPatterns(
    symbol, startDate, endDate, 10,
  )
  const { data: modelSummary, isLoading: summaryLoading } = useModelSummary(
    symbol, startDate, endDate,
  )

  const predict = usePredictLogistic()

  const handlePredict = () => {
    predict.mutate({
      symbol,
      goldstein_mean: -3.0,
      goldstein_min: -8.0,
      goldstein_max: 1.0,
      mentions_total: 150,
      avg_tone: -2.5,
      conflict_count: 5,
      cooperation_count: 2,
    })
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-text-primary">Signals</h2>
        <p className="text-sm text-text-secondary mt-1">
          Historical patterns (Level 1) and logistic regression predictions (Level 2)
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-4">
        <SymbolSelector value={symbol} onChange={setSymbol} />

        <button
          onClick={handlePredict}
          disabled={predict.isPending}
          className="px-4 py-1.5 bg-accent-blue text-white text-sm rounded-lg hover:bg-accent-blue/80 transition-colors disabled:opacity-50"
        >
          {predict.isPending ? 'Predicting...' : 'Run Prediction (Sample Data)'}
        </button>
      </div>

      {/* Level 2: Logistic Regression Prediction */}
      {predict.data && (
        <div>
          <h3 className="text-sm font-medium text-text-secondary uppercase tracking-wide mb-3">
            Level 2 — Logistic Regression Prediction
          </h3>
          <PredictionCard prediction={predict.data} />
        </div>
      )}

      {predict.isError && (
        <div className="bg-accent-red/10 border border-accent-red/30 rounded-xl p-4 text-sm text-accent-red">
          Prediction failed: {predict.error?.response?.data?.detail || predict.error?.message || 'Unknown error'}
        </div>
      )}

      {/* Model Summary */}
      {summaryLoading ? (
        <LoadingSpinner message="Loading model summary..." />
      ) : modelSummary ? (
        <div className="bg-bg-secondary border border-border rounded-xl p-4">
          <h3 className="text-sm font-medium text-text-primary mb-3">
            Model Summary — {symbol}
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <div className="bg-bg-tertiary rounded-lg p-2">
              <p className="text-text-secondary">Accuracy</p>
              <p className="text-lg font-bold">{(modelSummary.accuracy * 100).toFixed(1)}%</p>
            </div>
            <div className="bg-bg-tertiary rounded-lg p-2">
              <p className="text-text-secondary">Training Samples</p>
              <p className="text-lg font-bold">{modelSummary.n_training_samples}</p>
            </div>
            <div className="bg-bg-tertiary rounded-lg p-2">
              <p className="text-text-secondary">UP Ratio</p>
              <p className="text-lg font-bold">{(modelSummary.up_ratio * 100).toFixed(0)}%</p>
            </div>
            <div className="bg-bg-tertiary rounded-lg p-2">
              <p className="text-text-secondary">Accuracy Std</p>
              <p className="text-lg font-bold">{(modelSummary.accuracy_std * 100).toFixed(1)}%</p>
            </div>
          </div>

          {modelSummary.feature_importance && (
            <div className="mt-3">
              <p className="text-xs text-text-secondary mb-2">Feature Importance (|coefficient|)</p>
              <div className="space-y-1">
                {Object.entries(modelSummary.feature_importance).map(([name, val]) => (
                  <div key={name} className="flex items-center gap-2 text-xs">
                    <span className="text-text-secondary w-32 shrink-0">
                      {name.replace(/_/g, ' ')}
                    </span>
                    <div className="flex-1 h-2 bg-bg-tertiary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-accent-blue rounded-full"
                        style={{
                          width: `${Math.min(val * 100, 100)}%`,
                        }}
                      />
                    </div>
                    <span className="font-mono text-text-secondary w-12 text-right">
                      {val.toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : null}

      {/* Level 1: Historical Patterns */}
      <div>
        <h3 className="text-sm font-medium text-text-secondary uppercase tracking-wide mb-3">
          Level 1 — Historical Frequency Patterns
        </h3>
        {patternsLoading ? (
          <LoadingSpinner message="Computing patterns..." />
        ) : patterns && patterns.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {patterns.map((p, i) => (
              <PatternCard key={i} pattern={p} />
            ))}
          </div>
        ) : (
          <div className="bg-bg-secondary border border-border rounded-xl p-8 text-center text-text-secondary">
            No patterns found for {symbol}. Needs sufficient event + market data overlap.
          </div>
        )}
      </div>
    </div>
  )
}
