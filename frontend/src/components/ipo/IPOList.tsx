import { useState } from 'react'
import type { IPOListItem } from '../../types/ipo'
import IPOCard from './IPOCard'
import clsx from 'clsx'

interface IPOListProps {
  ipos: IPOListItem[]
  onViewDetails: (id: string) => void
}

const FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'active', label: 'Active' },
  { key: 'upcoming', label: 'Upcoming' },
  { key: 'listed', label: 'Listed' },
]

export default function IPOList({ ipos, onViewDetails }: IPOListProps) {
  const [activeFilter, setActiveFilter] = useState('all')

  const filtered = activeFilter === 'all'
    ? ipos
    : ipos.filter((ipo) => ipo.status === activeFilter)

  const counts: Record<string, number> = {
    all: ipos.length,
    active: ipos.filter((i) => i.status === 'active').length,
    upcoming: ipos.filter((i) => i.status === 'upcoming').length,
    listed: ipos.filter((i) => i.status === 'listed').length,
  }

  return (
    <div className="flex flex-col gap-5">
      {/* Filter tabs */}
      <div className="flex gap-2 flex-wrap">
        {FILTERS.map((f) => (
          <button
            key={f.key}
            onClick={() => setActiveFilter(f.key)}
            className={clsx(
              'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors'
            )}
            style={{
              backgroundColor: activeFilter === f.key ? 'var(--accent)' : 'var(--bg-surface)',
              color: activeFilter === f.key ? '#fff' : 'var(--text-secondary)',
              border: `1px solid ${activeFilter === f.key ? 'var(--accent)' : 'var(--border)'}`,
            }}
          >
            {f.label}
            <span
              className="text-xs px-1.5 py-0.5 rounded-full"
              style={{
                backgroundColor: activeFilter === f.key ? 'rgba(255,255,255,0.2)' : 'var(--bg-elevated)',
                color: activeFilter === f.key ? '#fff' : 'var(--text-muted)',
              }}
            >
              {counts[f.key]}
            </span>
          </button>
        ))}
      </div>

      {/* Grid */}
      {filtered.length === 0 ? (
        <div
          className="rounded-xl border p-12 text-center"
          style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border)' }}
        >
          <p className="text-base mb-1 font-medium" style={{ color: 'var(--text-secondary)' }}>
            No IPOs found
          </p>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            There are no {activeFilter !== 'all' ? activeFilter : ''} IPOs at the moment.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {filtered.map((ipo) => (
            <IPOCard key={ipo.id} ipo={ipo} onViewDetails={onViewDetails} />
          ))}
        </div>
      )}
    </div>
  )
}
