import type { AnalysisResult } from '../../types/stocks'
import Badge from '../common/Badge'
import { Cpu } from 'lucide-react'

interface DecisionCardProps {
  result: AnalysisResult
}

const actionColor: Record<string, string> = {
  BUY: 'var(--buy)',
  SELL: 'var(--sell)',
  HOLD: 'var(--hold)',
}

const actionBg: Record<string, string> = {
  BUY: 'rgba(34,197,94,0.08)',
  SELL: 'rgba(239,68,68,0.08)',
  HOLD: 'rgba(245,158,11,0.08)',
}

export default function DecisionCard({ result }: DecisionCardProps) {
  const color = actionColor[result.action] ?? 'var(--text-primary)'
  const bg = actionBg[result.action] ?? 'transparent'
  const confidence = Math.round(result.confidence * 100)

  // Score: -5 to +5, map to 0-100%
  const scoreNormalized = ((result.score + 5) / 10) * 100
  const scoreColor = result.score >= 0 ? 'var(--buy)' : 'var(--sell)'

  return (
    <div
      className="rounded-xl border p-6"
      style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border)' }}
    >
      <div className="flex flex-col md:flex-row md:items-start gap-6">
        {/* Action badge + confidence */}
        <div
          className="flex flex-col items-center justify-center rounded-xl p-6 min-w-[140px]"
          style={{ backgroundColor: bg, border: `2px solid ${color}` }}
        >
          <span className="text-4xl font-black tracking-tight" style={{ color }}>
            {result.action}
          </span>
          <div className="mt-2 text-center">
            <div className="text-2xl font-bold" style={{ color }}>
              {confidence}%
            </div>
            <div className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
              Confidence
            </div>
          </div>
        </div>

        {/* Details */}
        <div className="flex-1 flex flex-col gap-4">
          <div className="flex items-start justify-between flex-wrap gap-2">
            <div>
              <h2 className="text-xl font-bold" style={{ color: 'var(--text-primary)' }}>
                {result.name}
              </h2>
              <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
                {result.symbol} · {result.timeframe} analysis
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Cpu size={14} color="var(--text-muted)" />
              <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                {result.model_source === 'trained'
                  ? `Stock-Specific Model`
                  : `Generic Model (fallback)`}
              </span>
            </div>
          </div>

          {/* Score bar */}
          <div>
            <div className="flex justify-between text-xs mb-1" style={{ color: 'var(--text-muted)' }}>
              <span>Score: {result.score.toFixed(2)}</span>
              <span>Range: -5 to +5</span>
            </div>
            <div className="relative h-3 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--bg-elevated)' }}>
              <div
                className="absolute left-0 top-0 h-full rounded-full transition-all duration-700"
                style={{ width: `${scoreNormalized}%`, backgroundColor: scoreColor }}
              />
              {/* Midpoint marker */}
              <div
                className="absolute top-0 h-full w-0.5"
                style={{ left: '50%', backgroundColor: 'var(--border)' }}
              />
            </div>
          </div>

          {/* Explanation */}
          <p className="text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
            {result.explanation}
          </p>

          {/* Meta badges */}
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge variant={result.action === 'BUY' ? 'buy' : result.action === 'SELL' ? 'sell' : 'hold'}>
              {result.action}
            </Badge>
            <Badge variant="muted">
              Timeframe: {result.timeframe}
            </Badge>
            <Badge variant="muted">
              Sentiment: {result.news_sentiment > 0 ? '+' : ''}{result.news_sentiment.toFixed(2)}
            </Badge>
            <Badge variant={result.model_source === 'trained' ? 'accent' : 'muted'}>
              {result.model_source === 'trained' ? 'Trained Model' : 'Generic Fallback'}
            </Badge>
          </div>
        </div>
      </div>
    </div>
  )
}
