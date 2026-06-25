import { useLocation } from 'react-router-dom'

const pageTitles: Record<string, { title: string; subtitle: string }> = {
  '/': { title: 'Dashboard', subtitle: 'Overview of your analysis' },
  '/stocks': { title: 'Stock Analysis', subtitle: 'Powered by 5 AI Models' },
  '/ipo': { title: 'IPO Intelligence', subtitle: 'Track and evaluate new listings' },
}

export default function Header() {
  const location = useLocation()

  // Match by prefix for dynamic routes
  const matchedKey = Object.keys(pageTitles)
    .filter((k) => k !== '/')
    .find((k) => location.pathname.startsWith(k)) ?? '/'

  const pageKey = location.pathname === '/' ? '/' : matchedKey
  const page = pageTitles[pageKey] ?? { title: 'Sensei AI', subtitle: 'Powered by 5 AI Models' }

  return (
    <header
      className="flex items-center justify-between px-6 py-4 border-b"
      style={{
        backgroundColor: 'var(--bg-surface)',
        borderColor: 'var(--border)',
      }}
    >
      <div>
        <h1 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
          {page.title}
        </h1>
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
          {page.subtitle}
        </p>
      </div>
      <div className="flex items-center gap-3">
        <span
          className="text-xs font-bold px-2 py-1 rounded"
          style={{ backgroundColor: 'var(--accent)', color: '#fff', letterSpacing: '0.05em' }}
        >
          BETA
        </span>
      </div>
    </header>
  )
}
