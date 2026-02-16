import { NavLink } from 'react-router-dom'

const NAV_ITEMS = [
  { to: '/', label: 'Dashboard', icon: '⊞' },
  { to: '/correlation', label: 'Correlations', icon: '⊘' },
  { to: '/timeline', label: 'Timeline', icon: '⊟' },
  { to: '/map', label: 'World Map', icon: '⊕' },
  { to: '/signals', label: 'Signals', icon: '⊗' },
]

export default function Sidebar() {
  return (
    <aside className="w-56 bg-bg-secondary border-r border-border flex flex-col shrink-0">
      <div className="p-4 border-b border-border">
        <h1 className="text-lg font-bold text-text-primary tracking-tight">
          GeoMarket
        </h1>
        <p className="text-xs text-text-secondary mt-0.5">
          Geopolitical Tracker
        </p>
      </div>

      <nav className="flex-1 p-2 space-y-1">
        {NAV_ITEMS.map(({ to, label, icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                isActive
                  ? 'bg-bg-tertiary text-accent-blue font-medium'
                  : 'text-text-secondary hover:text-text-primary hover:bg-bg-tertiary/50'
              }`
            }
          >
            <span className="text-base">{icon}</span>
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="p-3 border-t border-border">
        <p className="text-[10px] text-text-secondary text-center">
          Data: GDELT + Yahoo Finance
        </p>
      </div>
    </aside>
  )
}
