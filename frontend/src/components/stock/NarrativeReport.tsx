import { useState } from 'react'
import clsx from 'clsx'

interface NarrativeReportProps {
  narrative: Record<string, string>
}

const TAB_KEYS = [
  { key: 'market_summary', label: 'Market Summary' },
  { key: 'technical_analysis', label: 'Technical' },
  { key: 'fundamental_analysis', label: 'Fundamental' },
  { key: 'risk_assessment', label: 'Risk' },
  { key: 'news_impact', label: 'News Impact' },
  { key: 'trade_setup', label: 'Trade Setup' },
]

export default function NarrativeReport({ narrative }: NarrativeReportProps) {
  // Find first tab that has content
  const availableTabs = TAB_KEYS.filter((t) => narrative[t.key])
  const [activeTab, setActiveTab] = useState(availableTabs[0]?.key || TAB_KEYS[0].key)

  if (!narrative || Object.keys(narrative).length === 0) {
    return null
  }

  const content = narrative[activeTab]

  return (
    <div
      className="rounded-xl border p-5"
      style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border)' }}
    >
      <h3 className="font-semibold text-sm uppercase tracking-wider mb-4" style={{ color: 'var(--text-muted)' }}>
        AI Narrative Report
      </h3>

      {/* Tab bar */}
      <div
        className="flex gap-1 mb-4 flex-wrap border-b pb-3"
        style={{ borderColor: 'var(--border)' }}
      >
        {TAB_KEYS.map((tab) => {
          const hasContent = !!narrative[tab.key]
          const isActive = activeTab === tab.key
          return (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              disabled={!hasContent}
              className={clsx(
                'px-3 py-1.5 rounded text-xs font-medium transition-colors relative',
                !hasContent && 'opacity-30 cursor-not-allowed',
                isActive ? 'text-white' : 'hover:text-white'
              )}
              style={{
                color: isActive ? '#fff' : 'var(--text-secondary)',
                backgroundColor: isActive ? 'var(--accent)' : 'transparent',
              }}
            >
              {tab.label}
            </button>
          )
        })}
      </div>

      {/* Content */}
      <div className="text-sm leading-relaxed whitespace-pre-wrap" style={{ color: 'var(--text-secondary)' }}>
        {content || (
          <span style={{ color: 'var(--text-muted)' }}>No content available for this section.</span>
        )}
      </div>
    </div>
  )
}
