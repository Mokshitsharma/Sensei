import { NavLink } from 'react-router-dom'
import { Brain, Home, TrendingUp, Star } from 'lucide-react'
import clsx from 'clsx'

const navItems = [
  { to: '/', icon: Home, label: 'Dashboard', exact: true },
  { to: '/stocks', icon: TrendingUp, label: 'Stocks' },
  { to: '/ipo', icon: Star, label: 'IPO Tracker' },
]

export default function Sidebar() {
  return (
    <aside
      className="flex flex-col w-60 h-full border-r"
      style={{
        backgroundColor: 'var(--bg-surface)',
        borderColor: 'var(--border)',
        minWidth: '240px',
      }}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-5 py-5 border-b" style={{ borderColor: 'var(--border)' }}>
        <div
          className="flex items-center justify-center w-9 h-9 rounded-lg"
          style={{ backgroundColor: 'var(--accent)' }}
        >
          <Brain size={20} color="#fff" />
        </div>
        <div>
          <div className="font-bold text-base" style={{ color: 'var(--text-primary)' }}>
            Sensei AI
          </div>
          <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Stock Intelligence
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {navItems.map(({ to, icon: Icon, label, exact }) => (
          <NavLink
            key={to}
            to={to}
            end={exact}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                isActive
                  ? 'text-white'
                  : 'hover:text-white'
              )
            }
            style={({ isActive }) => ({
              backgroundColor: isActive ? 'var(--accent)' : 'transparent',
              color: isActive ? '#fff' : 'var(--text-secondary)',
            })}
          >
            {({ isActive }) => (
              <>
                <Icon size={18} color={isActive ? '#fff' : 'var(--text-secondary)'} />
                {label}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Bottom badge */}
      <div className="px-5 py-4 border-t" style={{ borderColor: 'var(--border)' }}>
        <div
          className="flex items-center justify-between px-3 py-2 rounded-lg text-xs"
          style={{ backgroundColor: 'var(--bg-elevated)', color: 'var(--text-secondary)' }}
        >
          <span>NSE Coverage</span>
          <span
            className="font-semibold px-1.5 py-0.5 rounded text-xs"
            style={{ backgroundColor: 'var(--accent)', color: '#fff' }}
          >
            500 Stocks
          </span>
        </div>
      </div>
    </aside>
  )
}
