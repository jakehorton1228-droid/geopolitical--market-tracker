/**
 * PageHelp — collapsible "How to read this page" panel.
 *
 * Shows at the top of each page to explain what the page is for,
 * what to look for, and define terms used on that specific page.
 *
 * Collapsed by default so it doesn't get in the way of repeat users.
 *
 * USAGE:
 *   <PageHelp
 *     title="What is this page?"
 *     description="This page shows..."
 *     lookFor={['Strong correlations above 0.2', 'Events with high mention counts']}
 *     terms={[GLOSSARY.correlation, GLOSSARY.pvalue]}
 *   />
 */
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

export function PageHelp({ title = 'How to read this page', description, lookFor = [], terms = [] }) {
  const [open, setOpen] = useState(false)

  return (
    <motion.div
      initial={{ opacity: 0, y: -4 }}
      animate={{ opacity: 1, y: 0 }}
      className="mb-4"
    >
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        className="flex items-center gap-2 text-xs text-text-secondary hover:text-text-primary transition-colors group"
      >
        <span className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-accent-blue/10 text-accent-blue text-[10px] font-semibold">
          i
        </span>
        <span className="font-medium">{title}</span>
        <span className={`transition-transform ${open ? 'rotate-90' : ''}`}>→</span>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div
              className="mt-3 p-4 border-l-2 border-accent-blue/60 rounded-r-lg text-sm space-y-3"
              style={{ backgroundColor: 'rgba(15, 23, 42, 0.85)', color: '#cbd5e1' }}
            >
              {description && (
                <p className="leading-relaxed">{description}</p>
              )}

              {lookFor.length > 0 && (
                <div>
                  <div className="text-[11px] font-semibold text-text-primary uppercase tracking-wider mb-1.5">
                    What to look for
                  </div>
                  <ul className="space-y-1 text-xs">
                    {lookFor.map((item, i) => (
                      <li key={i} className="flex gap-2">
                        <span className="text-accent-blue">•</span>
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {terms.length > 0 && (
                <div>
                  <div className="text-[11px] font-semibold text-text-primary uppercase tracking-wider mb-1.5">
                    Terms used on this page
                  </div>
                  <dl className="space-y-2 text-xs">
                    {terms.map((t, i) => (
                      <div key={i}>
                        <dt className="font-semibold text-text-primary">{t.term}</dt>
                        <dd className="text-text-secondary mt-0.5">{t.short}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

export default PageHelp
