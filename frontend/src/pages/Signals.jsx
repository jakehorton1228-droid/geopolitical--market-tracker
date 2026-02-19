import { useState } from 'react'
import SymbolSelector from '../components/shared/SymbolSelector'
import PatternCard from '../components/cards/PatternCard'
import PredictionCard from '../components/cards/PredictionCard'
import LoadingSpinner from '../components/shared/LoadingSpinner'
import { useAllPatterns } from '../api/patterns'
import { useModelSummary } from '../api/predictions'
import { usePredictLogistic } from '../api/predictions'
import { DEFAULT_SYMBOLS } from '../lib/constants'

const DATA_START = '2016-01-01'

function todayStr() {
  return new Date().toISOString().split('T')[0]
}

function accuracyColor(acc) {
  if (acc >= 0.55) return 'text-accent-green'
  if (acc >= 0.52) return 'text-yellow-400'
  return 'text-accent-red'
}

export default function Signals() {
  const [symbol, setSymbol] = useState('CL=F')
  const [showMethodology, setShowMethodology] = useState(false)
  const startDate = DATA_START
  const endDate = todayStr()

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

      {/* How This Works — collapsible methodology panel */}
      <div className="bg-bg-secondary border border-border rounded-xl">
        <button
          onClick={() => setShowMethodology(!showMethodology)}
          className="w-full flex items-center justify-between px-4 py-3 text-sm text-text-secondary hover:text-text-primary transition-colors"
        >
          <span className="font-medium">How This Works</span>
          <span className="text-xs">{showMethodology ? '▲ Hide' : '▼ Show'}</span>
        </button>
        {showMethodology && (
          <div className="px-4 pb-4 space-y-3 text-xs text-text-secondary border-t border-border pt-3">
            <div>
              <p className="font-medium text-text-primary mb-1">Level 1 — Historical Frequency Patterns</p>
              <p>
                Counts how often the market moved UP or DOWN on days when specific event types
                occurred. This is not a predictive model — it shows conditional probability from
                historical data. A ~50% result means the event has no consistent directional bias.
              </p>
            </div>
            <div>
              <p className="font-medium text-text-primary mb-1">Level 2 — Logistic Regression</p>
              <p>
                Trains a binary classification model (sklearn LogisticRegression) on 7 event-based
                features to predict market direction. Evaluated with 5-fold cross-validation.
                Features: Goldstein mean/min/max, total mentions, average tone, conflict count,
                cooperation count.
              </p>
            </div>
            <div>
              <p className="font-medium text-text-primary mb-1">What makes a pattern significant?</p>
              <p>
                A p-value &lt; 0.05 (marked with a blue badge) means the directional bias is
                statistically unlikely to be random chance. A one-sample t-test checks if the
                mean return on event days differs significantly from zero.
              </p>
            </div>
          </div>
        )}
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

      {/* Data coverage banner */}
      <div className="flex flex-wrap items-center gap-4 text-xs text-text-secondary bg-bg-secondary border border-border rounded-lg px-4 py-2">
        <span>
          <span className="text-text-primary font-medium">Date Range:</span>{' '}
          Jan 2016 — {new Date().toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}
        </span>
        <span className="text-border">|</span>
        <span>
          <span className="text-text-primary font-medium">Training Samples:</span>{' '}
          {modelSummary ? modelSummary.n_training_samples.toLocaleString() : '—'} trading days
        </span>
        <span className="text-border">|</span>
        <span>
          <span className="text-text-primary font-medium">Sources:</span>{' '}
          GDELT events + Yahoo Finance
        </span>
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
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-text-primary">
              Model Summary — {symbol}
            </h3>
            <span className="text-[10px] text-text-secondary">
              Trained on {startDate} to {endDate}
            </span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <div className="bg-bg-tertiary rounded-lg p-2">
              <p className="text-text-secondary">Accuracy</p>
              <p className={`text-lg font-bold ${accuracyColor(modelSummary.accuracy)}`}>
                {(modelSummary.accuracy * 100).toFixed(1)}%
              </p>
              <p className="text-[10px] text-text-secondary mt-0.5">Baseline (random): 50%</p>
            </div>
            <div className="bg-bg-tertiary rounded-lg p-2">
              <p className="text-text-secondary">Training Samples</p>
              <p className="text-lg font-bold">{modelSummary.n_training_samples.toLocaleString()}</p>
              <p className="text-[10px] text-text-secondary mt-0.5">Days with events + market data</p>
            </div>
            <div className="bg-bg-tertiary rounded-lg p-2">
              <p className="text-text-secondary">UP Ratio</p>
              <p className="text-lg font-bold">{(modelSummary.up_ratio * 100).toFixed(0)}%</p>
              <p className="text-[10px] text-text-secondary mt-0.5">% of days market went up</p>
            </div>
            <div className="bg-bg-tertiary rounded-lg p-2">
              <p className="text-text-secondary">Accuracy Std</p>
              <p className="text-lg font-bold">{(modelSummary.accuracy_std * 100).toFixed(1)}%</p>
              <p className="text-[10px] text-text-secondary mt-0.5">Variation across 5 CV folds</p>
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
          <LoadingSpinner message="Computing patterns across 10 years of data..." />
        ) : patterns && patterns.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {patterns.map((p, i) => (
              <PatternCard
                key={i}
                pattern={p}
                totalTradingDays={modelSummary?.n_training_samples}
              />
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
