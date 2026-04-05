import { NavLink } from 'react-router-dom'

// Navigation follows a natural user flow:
//   1. Briefing    — What's happening? (the daily narrative, default landing page)
//   2. World Map   — Where?
//   3. Timeline    — When and how did markets respond?
//   4. Correlations — Which event types reliably move which markets?
//   5. Signals     — What do the models predict?
//   6. Predictions — What does the crowd (Polymarket) think?
//   7. AI Analyst  — Ask anything
//   8. Dashboard   — Platform overview (system stats)
const NAV_ITEMS = [
  { to: '/briefing', label: 'Briefing', icon: '◉' },
  { to: '/map', label: 'World Map', icon: '⊕' },
  { to: '/timeline', label: 'Timeline', icon: '⊟' },
  { to: '/correlation', label: 'Correlations', icon: '⊘' },
  { to: '/signals', label: 'Signals', icon: '⊗' },
  { to: '/markets', label: 'Predictions', icon: '◎' },
  { to: '/agent', label: 'AI Analyst', icon: '◈' },
  { to: '/dashboard', label: 'Dashboard', icon: '⊞' },
]

export default function Sidebar() {
  return (
    <aside className="w-56 flex flex-col shrink-0 border-r border-glass-border bg-glass backdrop-blur-xl">
      {/* Brand */}
      <div className="p-4 border-b border-glass-border">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-accent-blue/20 flex items-center justify-center">
            <span className="text-accent-blue text-sm font-bold">G</span>
          </div>
          <div>
            <h1 className="text-sm font-bold text-text-primary tracking-tight">
              GeoMarket
            </h1>
            <p className="text-[10px] text-text-secondary leading-none">
              Intelligence Platform
            </p>
          </div>
        </div>
      </div>

      {/* Status indicator */}
      <div className="px-4 py-2 border-b border-glass-border">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-accent-green pulse-live" />
          <span className="text-[10px] text-text-secondary uppercase tracking-wider font-mono">
            Systems Online
          </span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-2 space-y-0.5">
        {NAV_ITEMS.map(({ to, label, icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all duration-200 ${
                isActive
                  ? 'bg-accent-blue/10 text-accent-blue font-medium border border-accent-blue/20 shadow-[0_0_12px_rgba(59,130,246,0.1)]'
                  : 'text-text-secondary hover:text-text-primary hover:bg-bg-tertiary/30 border border-transparent'
              }`
            }
          >
            <span className="text-base opacity-70">{icon}</span>
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="p-3 border-t border-glass-border">
        <div className="flex items-center justify-center gap-1.5 flex-wrap">
          {['GDELT', 'Yahoo', 'RSS', 'FRED', 'Poly'].map((src) => (
            <span
              key={src}
              className="text-[9px] font-mono text-text-secondary/60 bg-bg-tertiary/30 px-1.5 py-0.5 rounded"
            >
              {src}
            </span>
          ))}
        </div>
      </div>
    </aside>
  )
}
