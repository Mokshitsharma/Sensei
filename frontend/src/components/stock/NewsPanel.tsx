import type { NewsItem } from '../../types/stocks'
import { ExternalLink } from 'lucide-react'
import Badge from '../common/Badge'

interface NewsPanelProps {
  news: NewsItem[]
  sentiment_score: number
}

function sentimentBadge(score: number): { label: string; variant: 'buy' | 'sell' | 'hold' } {
  if (score >= 0.1) return { label: `Positive (${score.toFixed(2)})`, variant: 'buy' }
  if (score <= -0.1) return { label: `Negative (${score.toFixed(2)})`, variant: 'sell' }
  return { label: `Neutral (${score.toFixed(2)})`, variant: 'hold' }
}

function labelVariant(label: string): 'buy' | 'sell' | 'hold' | 'muted' {
  const l = label?.toUpperCase()
  if (l === 'POSITIVE') return 'buy'
  if (l === 'NEGATIVE') return 'sell'
  if (l === 'NEUTRAL') return 'hold'
  return 'muted'
}

function formatDate(dateStr: string): string {
  try {
    const d = new Date(dateStr)
    return d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })
  } catch {
    return dateStr
  }
}

export default function NewsPanel({ news, sentiment_score }: NewsPanelProps) {
  const { label: sentimentLabel, variant: sentimentVariant } = sentimentBadge(sentiment_score)

  return (
    <div
      className="rounded-xl border p-5"
      style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border)' }}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-sm uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
          News & Sentiment
        </h3>
        <Badge variant={sentimentVariant}>
          Overall: {sentimentLabel}
        </Badge>
      </div>

      {news.length === 0 ? (
        <div className="text-center py-8" style={{ color: 'var(--text-muted)' }}>
          No news available
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          {news.map((item, i) => (
            <div
              key={i}
              className="flex flex-col gap-2 p-3 rounded-lg"
              style={{ backgroundColor: 'var(--bg-elevated)' }}
            >
              <div className="flex items-start justify-between gap-3">
                <p className="text-sm leading-snug" style={{ color: 'var(--text-primary)' }}>
                  {item.headline}
                </p>
                {item.url && (
                  <a
                    href={item.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex-shrink-0 mt-0.5"
                  >
                    <ExternalLink size={13} color="var(--text-muted)" />
                  </a>
                )}
              </div>
              <div className="flex items-center flex-wrap gap-2 text-xs">
                <span style={{ color: 'var(--text-muted)' }}>{item.source}</span>
                <span style={{ color: 'var(--border)' }}>·</span>
                <span style={{ color: 'var(--text-muted)' }}>{formatDate(item.published)}</span>
                <Badge variant={labelVariant(item.label)} size="sm">
                  {item.label}
                </Badge>
                {item.impact_type && (
                  <Badge variant="muted" size="sm">
                    {item.impact_type}
                  </Badge>
                )}
                <span className="ml-auto" style={{ color: 'var(--text-muted)' }}>
                  {(item.confidence * 100).toFixed(0)}% conf
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
