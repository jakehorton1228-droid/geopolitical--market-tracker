/**
 * Signals page — market direction predictions with methodology transparency.
 *
 * Two levels of prediction:
 * - Level 1 (Historical Frequency): Conditional probability counting.
 *   "When violent conflict events occur, oil went UP 72% of the time."
 * - Level 2 (Logistic Regression): 7-feature model with cross-validation.
 *   "Based on today's event profile, probability of UP: 64%."
 *
 * Includes collapsible methodology panel explaining how each model works,
 * data coverage banner (date range, training samples), and model summary
 * with accuracy color-coding.
 */
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import SymbolSelector from '../components/shared/SymbolSelector'
import PatternCard from '../components/cards/PatternCard'
import PredictionCard from '../components/cards/PredictionCard'
import LoadingSpinner from '../components/shared/LoadingSpinner'
import { useAllPatterns } from '../api/patterns'
import { useModelSummary } from '../api/predictions'
import { usePredictLogistic } from '../api/predictions'
import { fadeInUp, staggerContainer, staggerItem, scaleIn } from '../utils/animations'
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
      <motion.div {...fadeInUp}>
        <h2 className="text-2xl font-bold text-text-primary">Signals</h2>
        <p className="text-sm text-text-secondary mt-1">
          Historical patterns (Level 1) and logistic regression predictions (Level 2)
        </p>
      </motion.div>

      {/* How This Works — collapsible methodology panel */}
      <motion.div {...fadeInUp} transition={{ delay: 0.1, duration: 0.4 }}>
        <div className="bg-bg-secondary border border-border rounded-xl">
          <button
            onClick={() => setShowMethodology(!showMethodology)}
            className="w-full flex items-center justify-between px-4 py-3 text-sm text-text-secondary hover:text-text-primary transition-colors"
          >
            <span className="font-medium">How This Works</span>
            <motion.span
              className="text-xs"
              animate={{ rotate: showMethodology ? 180 : 0 }}
              transition={{ duration: 0.2 }}
            >
              ▾
            </motion.span>
          </button>
          <AnimatePresence>
            {showMethodology && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.3, ease: 'easeInOut' }}
                className="overflow-hidden"
              >
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
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>

      <motion.div {...fadeInUp} transition={{ delay: 0.15, duration: 0.4 }} className="flex flex-wrap items-center gap-4">
        <SymbolSelector value={symbol} onChange={setSymbol} />

        <motion.button
          whileHover={{ scale: 1.03 }}
          whileTap={{ scale: 0.97 }}
          onClick={handlePredict}
          disabled={predict.isPending}
          className="px-4 py-1.5 bg-accent-blue text-white text-sm rounded-lg hover:bg-accent-blue/80 transition-colors disabled:opacity-50"
        >
          {predict.isPending ? 'Predicting...' : 'Run Prediction (Sample Data)'}
        </motion.button>
      </motion.div>

      {/* Data coverage banner */}
      <motion.div {...fadeInUp} transition={{ delay: 0.2, duration: 0.4 }}>
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
      </motion.div>

      {/* Level 2: Logistic Regression Prediction */}
      <AnimatePresence>
        {predict.data && (
          <motion.div {...scaleIn}>
            <h3 className="text-sm font-medium text-text-secondary uppercase tracking-wide mb-3">
              Level 2 — Logistic Regression Prediction
            </h3>
            <PredictionCard prediction={predict.data} />
          </motion.div>
        )}
      </AnimatePresence>

      {predict.isError && (
        <motion.div {...fadeInUp}>
          <div className="bg-accent-red/10 border border-accent-red/30 rounded-xl p-4 text-sm text-accent-red">
            Prediction failed: {predict.error?.response?.data?.detail || predict.error?.message || 'Unknown error'}
          </div>
        </motion.div>
      )}

      {/* Model Summary */}
      <motion.div {...fadeInUp} transition={{ delay: 0.25, duration: 0.4 }}>
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
            <motion.div
              className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs"
              variants={staggerContainer.variants}
              initial="initial"
              animate="animate"
            >
              <motion.div variants={staggerItem.variants} className="bg-bg-tertiary rounded-lg p-2">
                <p className="text-text-secondary">Accuracy</p>
                <p className={`text-lg font-bold ${accuracyColor(modelSummary.accuracy)}`}>
                  {(modelSummary.accuracy * 100).toFixed(1)}%
                </p>
                <p className="text-[10px] text-text-secondary mt-0.5">Baseline (random): 50%</p>
              </motion.div>
              <motion.div variants={staggerItem.variants} className="bg-bg-tertiary rounded-lg p-2">
                <p className="text-text-secondary">Training Samples</p>
                <p className="text-lg font-bold">{modelSummary.n_training_samples.toLocaleString()}</p>
                <p className="text-[10px] text-text-secondary mt-0.5">Days with events + market data</p>
              </motion.div>
              <motion.div variants={staggerItem.variants} className="bg-bg-tertiary rounded-lg p-2">
                <p className="text-text-secondary">UP Ratio</p>
                <p className="text-lg font-bold">{(modelSummary.up_ratio * 100).toFixed(0)}%</p>
                <p className="text-[10px] text-text-secondary mt-0.5">% of days market went up</p>
              </motion.div>
              <motion.div variants={staggerItem.variants} className="bg-bg-tertiary rounded-lg p-2">
                <p className="text-text-secondary">Accuracy Std</p>
                <p className="text-lg font-bold">{(modelSummary.accuracy_std * 100).toFixed(1)}%</p>
                <p className="text-[10px] text-text-secondary mt-0.5">Variation across 5 CV folds</p>
              </motion.div>
            </motion.div>

            {modelSummary.feature_importance && (
              <div className="mt-3">
                <p className="text-xs text-text-secondary mb-2">Feature Importance (|coefficient|)</p>
                <div className="space-y-1">
                  {Object.entries(modelSummary.feature_importance).map(([name, val], i) => (
                    <motion.div
                      key={name}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.05, duration: 0.3 }}
                      className="flex items-center gap-2 text-xs"
                    >
                      <span className="text-text-secondary w-32 shrink-0">
                        {name.replace(/_/g, ' ')}
                      </span>
                      <div className="flex-1 h-2 bg-bg-tertiary rounded-full overflow-hidden">
                        <motion.div
                          className="h-full bg-accent-blue rounded-full"
                          initial={{ width: 0 }}
                          animate={{ width: `${Math.min(val * 100, 100)}%` }}
                          transition={{ delay: 0.3 + i * 0.05, duration: 0.6, ease: 'easeOut' }}
                        />
                      </div>
                      <span className="font-mono text-text-secondary w-12 text-right">
                        {val.toFixed(3)}
                      </span>
                    </motion.div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : null}
      </motion.div>

      {/* Level 1: Historical Patterns */}
      <motion.div {...fadeInUp} transition={{ delay: 0.3, duration: 0.4 }}>
        <h3 className="text-sm font-medium text-text-secondary uppercase tracking-wide mb-3">
          Level 1 — Historical Frequency Patterns
        </h3>
        {patternsLoading ? (
          <LoadingSpinner message="Computing patterns across 10 years of data..." />
        ) : patterns && patterns.length > 0 ? (
          <motion.div
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
            variants={staggerContainer.variants}
            initial="initial"
            animate="animate"
          >
            {patterns.map((p, i) => (
              <motion.div key={i} variants={staggerItem.variants}>
                <PatternCard
                  pattern={p}
                  totalTradingDays={modelSummary?.n_training_samples}
                />
              </motion.div>
            ))}
          </motion.div>
        ) : (
          <div className="bg-bg-secondary border border-border rounded-xl p-8 text-center text-text-secondary">
            No patterns found for {symbol}. Needs sufficient event + market data overlap.
          </div>
        )}
      </motion.div>
    </div>
  )
}
