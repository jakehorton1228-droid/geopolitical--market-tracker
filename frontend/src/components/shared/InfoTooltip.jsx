/**
 * InfoTooltip — a small ? icon that shows a definition on hover.
 *
 * USAGE:
 *   import { InfoTooltip } from '../shared/InfoTooltip'
 *   import { GLOSSARY } from '../../utils/glossary'
 *
 *   // With a glossary term
 *   <InfoTooltip {...GLOSSARY.correlation} />
 *
 *   // With inline text
 *   <InfoTooltip term="Strongest Link" short="The highest correlation we found." />
 *
 *   // Can wrap a label
 *   <label>Goldstein Scale <InfoTooltip {...GLOSSARY.goldstein} /></label>
 */
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

export function InfoTooltip({ term, short, long, className = '' }) {
  const [open, setOpen] = useState(false)

  return (
    <span
      className={`relative inline-flex items-center ${className}`}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      <button
        type="button"
        aria-label={`Info about ${term}`}
        className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-white/10 hover:bg-white/20 text-text-secondary hover:text-text-primary text-[10px] font-semibold transition-colors cursor-help"
        onClick={(e) => {
          e.preventDefault()
          setOpen((prev) => !prev)
        }}
      >
        ?
      </button>

      <AnimatePresence>
        {open && (
          <motion.span
            initial={{ opacity: 0, y: 4, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 4, scale: 0.96 }}
            transition={{ duration: 0.15 }}
            className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 z-50 w-64 pointer-events-none"
          >
            <span
              className="block px-3 py-2 text-xs rounded-lg shadow-2xl border border-white/15"
              style={{ backgroundColor: '#0f172a', color: '#f1f5f9' }}
            >
              <span className="block font-semibold mb-1" style={{ color: '#f8fafc' }}>{term}</span>
              <span className="block leading-snug" style={{ color: '#cbd5e1' }}>{short}</span>
              {long && (
                <span
                  className="block text-[10px] mt-1.5 leading-relaxed"
                  style={{ color: '#94a3b8' }}
                >
                  {long}
                </span>
              )}
            </span>
          </motion.span>
        )}
      </AnimatePresence>
    </span>
  )
}

export default InfoTooltip
