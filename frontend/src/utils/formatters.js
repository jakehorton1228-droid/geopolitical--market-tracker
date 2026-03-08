/** Format a number as percentage */
export function formatPercent(value, decimals = 1) {
  return `${value >= 0 ? '+' : ''}${value.toFixed(decimals)}%`
}

/** Format a number as currency-like price */
export function formatPrice(value) {
  if (value >= 1000) return value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
  if (value >= 1) return value.toFixed(2)
  return value.toFixed(4)
}

/** Format a correlation value */
export function formatCorrelation(value) {
  const sign = value >= 0 ? '+' : ''
  return `${sign}${value.toFixed(3)}`
}

/** Format a p-value */
export function formatPValue(value) {
  if (value < 0.001) return '<0.001'
  if (value < 0.01) return value.toFixed(3)
  return value.toFixed(2)
}

/** Color for correlation value */
export function correlationColor(value) {
  const abs = Math.abs(value)
  if (abs > 0.3) return value > 0 ? '#10b981' : '#ef4444'
  if (abs > 0.1) return value > 0 ? '#6ee7b7' : '#fca5a5'
  return '#9ca3af'
}

/** Color for signal strength */
export function signalColor(probability) {
  if (probability >= 0.65) return '#10b981'
  if (probability >= 0.55) return '#6ee7b7'
  if (probability <= 0.35) return '#ef4444'
  if (probability <= 0.45) return '#fca5a5'
  return '#9ca3af'
}
