import type { Fundamentals } from '../../types/stocks'

interface MetricsGridProps {
  fundamentals: Fundamentals
}

function formatIndianCurrency(val?: number): string {
  if (val == null) return 'N/A'
  if (val >= 1e12) return `₹${(val / 1e12).toFixed(2)}T`
  if (val >= 1e7) return `₹${(val / 1e7).toFixed(2)} Cr`
  if (val >= 1e5) return `₹${(val / 1e5).toFixed(2)} L`
  return `₹${val.toFixed(2)}`
}

function formatPrice(val?: number): string {
  if (val == null) return 'N/A'
  return `₹${val.toLocaleString('en-IN', { maximumFractionDigits: 2 })}`
}

function formatRatio(val?: number, suffix?: string): string {
  if (val == null) return 'N/A'
  return `${val.toFixed(2)}${suffix || ''}`
}

interface MetricCardProps {
  label: string
  value: string
  subtext?: string
  valueColor?: string
}

function MetricCard({ label, value, subtext, valueColor }: MetricCardProps) {
  return (
    <div
      className="rounded-lg p-4 flex flex-col gap-1"
      style={{ backgroundColor: 'var(--bg-elevated)' }}
    >
      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
        {label}
      </span>
      <span className="text-lg font-bold" style={{ color: valueColor || 'var(--text-primary)' }}>
        {value}
      </span>
      {subtext && (
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
          {subtext}
        </span>
      )}
    </div>
  )
}

export default function MetricsGrid({ fundamentals }: MetricsGridProps) {
  return (
    <div
      className="rounded-xl border p-5 flex flex-col gap-4"
      style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border)' }}
    >
      <h3 className="font-semibold text-sm uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
        Fundamentals
      </h3>
      <div className="grid grid-cols-2 gap-3">
        <MetricCard
          label="Current Price"
          value={formatPrice(fundamentals.current_price)}
          valueColor="var(--text-primary)"
        />
        <MetricCard
          label="Market Cap"
          value={formatIndianCurrency(fundamentals.market_cap)}
        />
        <MetricCard
          label="P/E Ratio"
          value={formatRatio(fundamentals.pe_ratio, 'x')}
          subtext={fundamentals.pe_ratio && fundamentals.pe_ratio > 30 ? 'High' : fundamentals.pe_ratio ? 'Reasonable' : undefined}
        />
        <MetricCard
          label="ROE"
          value={formatRatio(fundamentals.roe, '%')}
          valueColor={
            fundamentals.roe != null
              ? fundamentals.roe >= 15
                ? 'var(--buy)'
                : fundamentals.roe < 0
                ? 'var(--sell)'
                : 'var(--text-primary)'
              : 'var(--text-primary)'
          }
        />
        <MetricCard
          label="52W High"
          value={formatPrice(fundamentals.week_52_high)}
          valueColor="var(--sell)"
        />
        <MetricCard
          label="52W Low"
          value={formatPrice(fundamentals.week_52_low)}
          valueColor="var(--buy)"
        />
      </div>
      {/* Additional row if data available */}
      {(fundamentals.debt_to_equity != null || fundamentals.book_value != null) && (
        <div className="grid grid-cols-2 gap-3">
          {fundamentals.debt_to_equity != null && (
            <MetricCard
              label="Debt/Equity"
              value={formatRatio(fundamentals.debt_to_equity, 'x')}
              valueColor={
                fundamentals.debt_to_equity > 1
                  ? 'var(--sell)'
                  : fundamentals.debt_to_equity < 0.5
                  ? 'var(--buy)'
                  : 'var(--hold)'
              }
            />
          )}
          {fundamentals.book_value != null && (
            <MetricCard
              label="Book Value"
              value={formatPrice(fundamentals.book_value)}
            />
          )}
        </div>
      )}
    </div>
  )
}
