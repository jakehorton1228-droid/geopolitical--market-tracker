/** Dropdown selector for tracked financial symbols, grouped by category. */
import { useSymbols } from '../../api/market'
import { DEFAULT_SYMBOLS } from '../../lib/constants'

export default function SymbolSelector({ value, onChange }) {
  const { data: symbols } = useSymbols()

  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="bg-bg-tertiary border border-border rounded-lg px-3 py-1.5 text-sm text-text-primary focus:outline-none focus:ring-1 focus:ring-accent-blue"
    >
      <optgroup label="Quick Select">
        {DEFAULT_SYMBOLS.map((s) => (
          <option key={s} value={s}>{s}</option>
        ))}
      </optgroup>
      {symbols && (
        <optgroup label="All Symbols">
          {symbols
            .filter((s) => !DEFAULT_SYMBOLS.includes(s.symbol))
            .map((s) => (
              <option key={s.symbol} value={s.symbol}>
                {s.symbol} — {s.name}
              </option>
            ))}
        </optgroup>
      )}
    </select>
  )
}
