import type { IPOListItem } from '../../types/ipo'
import Badge from '../common/Badge'
import { Calendar, Users, TrendingUp, ArrowRight } from 'lucide-react'

interface IPOCardProps {
  ipo: IPOListItem
  onViewDetails: (id: string) => void
}

function formatDate(dateStr: string): string {
  try {
    const d = new Date(dateStr)
    return d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })
  } catch {
    return dateStr
  }
}

function formatCr(val: number): string {
  if (val >= 1000) return `₹${(val / 1000).toFixed(2)}K Cr`
  return `₹${val.toFixed(0)} Cr`
}

export default function IPOCard({ ipo, onViewDetails }: IPOCardProps) {
  const priceRange = `₹${ipo.issue_price_min}–₹${ipo.issue_price_max}`
  const gmpPositive = ipo.gmp >= 0

  return (
    <div
      className="rounded-xl border p-5 flex flex-col gap-4 hover:border-accent transition-colors cursor-default"
      style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border)' }}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-col gap-1 flex-1 min-w-0">
          <h3 className="font-semibold text-base truncate" style={{ color: 'var(--text-primary)' }}>
            {ipo.company_name}
          </h3>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              {ipo.sector}
            </span>
            {ipo.symbol && (
              <Badge variant="muted" size="sm">
                {ipo.symbol}
              </Badge>
            )}
          </div>
        </div>
        <Badge
          variant={
            ipo.status === 'active' ? 'buy' : ipo.status === 'upcoming' ? 'accent' : 'muted'
          }
          size="md"
        >
          {ipo.status.toUpperCase()}
        </Badge>
      </div>

      {/* Dates */}
      <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
        <Calendar size={12} />
        <span>
          {ipo.status === 'listed' && ipo.listing_date
            ? `Listed: ${formatDate(ipo.listing_date)}`
            : `${formatDate(ipo.open_date)} – ${formatDate(ipo.close_date)}`}
        </span>
      </div>

      {/* Key metrics row */}
      <div className="grid grid-cols-2 gap-2">
        <div className="flex flex-col gap-0.5 px-3 py-2 rounded-lg" style={{ backgroundColor: 'var(--bg-elevated)' }}>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Issue Price</span>
          <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{priceRange}</span>
        </div>
        <div className="flex flex-col gap-0.5 px-3 py-2 rounded-lg" style={{ backgroundColor: 'var(--bg-elevated)' }}>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Issue Size</span>
          <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{formatCr(ipo.issue_size_cr)}</span>
        </div>
        <div className="flex flex-col gap-0.5 px-3 py-2 rounded-lg" style={{ backgroundColor: 'var(--bg-elevated)' }}>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Lot Size</span>
          <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{ipo.lot_size} shares</span>
        </div>
        <div className="flex flex-col gap-0.5 px-3 py-2 rounded-lg" style={{ backgroundColor: 'var(--bg-elevated)' }}>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>GMP</span>
          <span
            className="text-sm font-semibold"
            style={{ color: gmpPositive ? 'var(--buy)' : 'var(--sell)' }}
          >
            {gmpPositive ? '+' : ''}₹{ipo.gmp} ({gmpPositive ? '+' : ''}{ipo.gmp_pct.toFixed(1)}%)
          </span>
        </div>
      </div>

      {/* Subscription */}
      {ipo.subscription_total > 0 && (
        <div className="flex items-center gap-2">
          <Users size={12} color="var(--text-muted)" />
          <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
            Subscription:
          </span>
          <Badge variant="buy" size="sm">
            {ipo.subscription_total.toFixed(2)}x
          </Badge>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            QIB {ipo.subscription_qib.toFixed(1)}x · NII {ipo.subscription_nii.toFixed(1)}x · RII {ipo.subscription_rii.toFixed(1)}x
          </span>
        </div>
      )}

      {/* Recommendation + Risk */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <Badge
            variant={
              ipo.recommendation === 'INVEST' ? 'invest' : ipo.recommendation === 'AVOID' ? 'avoid' : 'neutral'
            }
            size="md"
          >
            {ipo.recommendation}
          </Badge>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Score: {ipo.recommendation_score}/10
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Badge
            variant={
              ipo.risk_level === 'HIGH' ? 'high' : ipo.risk_level === 'MEDIUM' ? 'medium' : 'low'
            }
            size="sm"
          >
            {ipo.risk_level} Risk
          </Badge>
          {ipo.expected_listing_gain_pct != null && (
            <span className="text-xs font-semibold" style={{ color: ipo.expected_listing_gain_pct >= 0 ? 'var(--buy)' : 'var(--sell)' }}>
              <TrendingUp size={12} className="inline mr-0.5" />
              {ipo.expected_listing_gain_pct >= 0 ? '+' : ''}{ipo.expected_listing_gain_pct.toFixed(1)}% est. gain
            </span>
          )}
        </div>
      </div>

      {/* View Details button */}
      <button
        onClick={() => onViewDetails(ipo.id)}
        className="flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors w-full"
        style={{ backgroundColor: 'var(--bg-elevated)', color: 'var(--text-secondary)' }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLButtonElement).style.backgroundColor = 'var(--accent)'
          ;(e.currentTarget as HTMLButtonElement).style.color = '#fff'
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLButtonElement).style.backgroundColor = 'var(--bg-elevated)'
          ;(e.currentTarget as HTMLButtonElement).style.color = 'var(--text-secondary)'
        }}
      >
        View Details <ArrowRight size={14} />
      </button>
    </div>
  )
}
