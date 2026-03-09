import { Outlet, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import Sidebar from './Sidebar'
import { pageTransition } from '../../utils/animations'

/**
 * AppShell — main layout wrapper with animated page transitions.
 *
 * AnimatePresence watches for route changes. When the route changes:
 * 1. The OLD page plays its `exit` animation (fade out + slide up)
 * 2. The NEW page plays its `initial` → `animate` animation (fade in + slide down)
 *
 * The `key` prop on motion.div tells AnimatePresence which component
 * changed — when the key changes, it treats it as a new component.
 * We use `location.pathname` as the key so each route gets its own animation.
 */
export default function AppShell() {
  const location = useLocation()

  return (
    <div className="flex h-screen bg-bg-primary text-text-primary overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-y-auto p-6 bg-dot-grid">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            {...pageTransition}
            className="h-full"
          >
            <Outlet />
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  )
}
