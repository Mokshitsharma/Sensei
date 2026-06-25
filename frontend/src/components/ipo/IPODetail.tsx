import { X, CheckCircle, XCircle } from 'lucide-react'
import type { IPODetail as IPODetailType } from '../../types/ipo'
import Badge from '../common/Badge'
import LoadingSpinner from '../common/LoadingSpinner'

interface IPODetailProps {
  ipo: IPODetailType | null | undefined
  loading?: boolean
  onClose: () => void
}

function formatDate(dateStr: string): string {
  try {
    return new Date(dateStr).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })
  } catch {
    return dateStr
  }
}

function formatCr(val?: number): string {
  if (val == null) return 'N/A'
  if (val >= 1000) return `₹${(val / 1000).toFixed(2)}K Cr`
  return `₹${val.toFixed(2)} Cr`
}

export default function IPODetail({ ipo, loading, onClose }: IPODetailProps) {
  return (
    <div className="fixed inset-0 z-50 flex justify-end" style={{ backgroundColor: 'rgba(0,0,0,0.6)' }}>
      <div
        className="relative flex flex-col h-full overflow-y-auto w-full max-w-2xl shadow-2xl"
        style={{ backgroundColor: 'var(--bg-primary)', borderLeft: '1px solid var(--border)' }}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between px-6 py-5 border-b sticky top-0 z-10"
          style={{ backgroundColor: 'var(--bg-primary)', borderColor: 'var(--border)' }}
        >
          <div>
            <h2 className="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>
              {ipo?.company_name ?? 'IPO Details'}
            </h2>
            {ipo && (
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                {ipo.sector}
              </p>
            )}
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg transition-colors"
            style={{ color: 'var(--text-muted)' }}
          >
            <X size={20} />
          </button>
        </div>

        {loading ? (
          <div className="flex-1 flex items-center justify-center">
            <LoadingSpinner size="lg" label="Loading IPO details..." />
          </div>
        ) : !ipo ? (
          <div className="flex-1 flex items-center justify-center" style={{ color: 'var(--text-muted)' }}>
            No data available
          </div>
        ) : (
          <div className="flex flex-col gap-6 p-6">
            {/* Status + Recommendation */}
            <div className="flex flex-wrap gap-2">
              <Badge variant={ipo.status === 'active' ? 'buy' : ipo.status === 'upcoming' ? 'accent' : 'muted'}>
                {ipo.status.toUpperCase()}
              </Badge>
              <Badge variant={ipo.recommendation === 'INVEST' ? 'invest' : ipo.recommendation === 'AVOID' ? 'avoid' : 'neutral'}>
                {ipo.recommendation}
              </Badge>
              <Badge variant={ipo.risk_level === 'HIGH' ? 'high' : ipo.risk_level === 'MEDIUM' ? 'medium' : 'low'}>
                {ipo.risk_level} Risk
              </Badge>
            </div>

            {/* Description */}
            {ipo.company_description && (
              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider mb-2" style={{ color: 'var(--text-muted)' }}>
                  About
                </h4>
                <p className="text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                  {ipo.company_description}
                </p>
              </div>
            )}

            {/* Key Info */}
            <div>
              <h4 className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-muted)' }}>
                IPO Details
              </h4>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { label: 'Issue Price', value: `₹${ipo.issue_price_min}–₹${ipo.issue_price_max}` },
                  { label: 'Issue Size', value: formatCr(ipo.issue_size_cr) },
                  { label: 'Lot Size', value: `${ipo.lot_size} shares` },
                  { label: 'Open Date', value: formatDate(ipo.open_date) },
                  { label: 'Close Date', value: formatDate(ipo.close_date) },
                  ...(ipo.listing_date ? [{ label: 'Listing Date', value: formatDate(ipo.listing_date) }] : []),
                  { label: 'Registrar', value: ipo.registrar },
                  { label: 'Rec. Score', value: `${ipo.recommendation_score}/10` },
                ].map((item) => (
                  <div key={item.label} className="px-3 py-2 rounded-lg" style={{ backgroundColor: 'var(--bg-elevated)' }}>
                    <div className="text-xs mb-0.5" style={{ color: 'var(--text-muted)' }}>{item.label}</div>
                    <div className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{item.value}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Subscription breakdown */}
            {ipo.subscription_total > 0 && (
              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-muted)' }}>
                  Subscription ({ipo.subscription_total.toFixed(2)}x overall)
                </h4>
                <div className="flex flex-col gap-2">
                  {[
                    { label: 'QIB', value: ipo.subscription_qib, color: 'var(--accent)' },
                    { label: 'NII', value: ipo.subscription_nii, color: 'var(--hold)' },
                    { label: 'RII', value: ipo.subscription_rii, color: 'var(--buy)' },
                  ].map(({ label, value, color }) => {
                    const max = Math.max(ipo.subscription_qib, ipo.subscription_nii, ipo.subscription_rii, 1)
                    const pct = (value / max) * 100
                    return (
                      <div key={label} className="flex items-center gap-3">
                        <span className="text-xs w-8" style={{ color: 'var(--text-secondary)' }}>{label}</span>
                        <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--bg-elevated)' }}>
                          <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
                        </div>
                        <span className="text-xs w-12 text-right font-semibold" style={{ color }}>{value.toFixed(1)}x</span>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Financials */}
            {ipo.financials && Object.keys(ipo.financials).length > 0 && (
              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-muted)' }}>
                  Financials
                </h4>
                <div className="rounded-lg overflow-hidden border" style={{ borderColor: 'var(--border)' }}>
                  <table className="w-full text-sm">
                    <tbody>
                      {[
                        { label: 'Revenue', value: formatCr(ipo.financials.revenue_cr) },
                        { label: 'EBITDA', value: formatCr(ipo.financials.ebitda_cr) },
                        { label: 'PAT', value: formatCr(ipo.financials.pat_cr) },
                        { label: 'EPS', value: ipo.financials.eps != null ? `₹${ipo.financials.eps.toFixed(2)}` : 'N/A' },
                        { label: 'P/E Ratio', value: ipo.financials.pe_ratio != null ? `${ipo.financials.pe_ratio.toFixed(2)}x` : 'N/A' },
                        { label: 'ROE', value: ipo.financials.roe != null ? `${ipo.financials.roe.toFixed(2)}%` : 'N/A' },
                        { label: 'Debt/Equity', value: ipo.financials.debt_equity != null ? `${ipo.financials.debt_equity.toFixed(2)}x` : 'N/A' },
                      ].map((row, i) => (
                        <tr
                          key={row.label}
                          style={{ backgroundColor: i % 2 === 0 ? 'var(--bg-elevated)' : 'var(--bg-surface)' }}
                        >
                          <td className="px-4 py-2.5" style={{ color: 'var(--text-muted)' }}>{row.label}</td>
                          <td className="px-4 py-2.5 text-right font-medium" style={{ color: 'var(--text-primary)' }}>{row.value}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Promoter holding */}
            {(ipo.promoter_holding_pre != null || ipo.promoter_holding_post != null) && (
              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-muted)' }}>
                  Promoter Holding
                </h4>
                <div className="flex gap-3">
                  {ipo.promoter_holding_pre != null && (
                    <div className="flex-1 px-3 py-2 rounded-lg" style={{ backgroundColor: 'var(--bg-elevated)' }}>
                      <div className="text-xs" style={{ color: 'var(--text-muted)' }}>Pre-IPO</div>
                      <div className="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>{ipo.promoter_holding_pre.toFixed(1)}%</div>
                    </div>
                  )}
                  {ipo.promoter_holding_post != null && (
                    <div className="flex-1 px-3 py-2 rounded-lg" style={{ backgroundColor: 'var(--bg-elevated)' }}>
                      <div className="text-xs" style={{ color: 'var(--text-muted)' }}>Post-IPO</div>
                      <div className="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>{ipo.promoter_holding_post.toFixed(1)}%</div>
                    </div>
                  )}
                  {ipo.industry_pe != null && (
                    <div className="flex-1 px-3 py-2 rounded-lg" style={{ backgroundColor: 'var(--bg-elevated)' }}>
                      <div className="text-xs" style={{ color: 'var(--text-muted)' }}>Industry P/E</div>
                      <div className="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>{ipo.industry_pe.toFixed(1)}x</div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Dimension scores */}
            {ipo.dimension_scores && Object.keys(ipo.dimension_scores).length > 0 && (
              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-muted)' }}>
                  Analysis Dimensions
                </h4>
                <div className="flex flex-col gap-2">
                  {Object.entries(ipo.dimension_scores).map(([key, score]) => (
                    <div key={key} className="flex items-center gap-3">
                      <span className="text-xs w-32 capitalize" style={{ color: 'var(--text-secondary)' }}>
                        {key.replace(/_/g, ' ')}
                      </span>
                      <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--bg-elevated)' }}>
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${(score / 10) * 100}%`,
                            backgroundColor: score >= 7 ? 'var(--buy)' : score >= 4 ? 'var(--hold)' : 'var(--sell)',
                          }}
                        />
                      </div>
                      <span className="text-xs w-8 text-right font-semibold" style={{ color: 'var(--text-primary)' }}>
                        {score}/10
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Objects of issue */}
            {ipo.objects_of_issue?.length > 0 && (
              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider mb-2" style={{ color: 'var(--text-muted)' }}>
                  Objects of Issue
                </h4>
                <ul className="flex flex-col gap-1.5">
                  {ipo.objects_of_issue.map((obj, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm" style={{ color: 'var(--text-secondary)' }}>
                      <span className="mt-1 w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: 'var(--accent)' }} />
                      {obj}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Pros & Cons */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {ipo.pros?.length > 0 && (
                <div>
                  <h4 className="text-xs font-semibold uppercase tracking-wider mb-2" style={{ color: 'var(--buy)' }}>
                    Pros
                  </h4>
                  <ul className="flex flex-col gap-1.5">
                    {ipo.pros.map((pro, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm" style={{ color: 'var(--text-secondary)' }}>
                        <CheckCircle size={13} color="var(--buy)" className="flex-shrink-0 mt-0.5" />
                        {pro}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {ipo.cons?.length > 0 && (
                <div>
                  <h4 className="text-xs font-semibold uppercase tracking-wider mb-2" style={{ color: 'var(--sell)' }}>
                    Cons
                  </h4>
                  <ul className="flex flex-col gap-1.5">
                    {ipo.cons.map((con, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm" style={{ color: 'var(--text-secondary)' }}>
                        <XCircle size={13} color="var(--sell)" className="flex-shrink-0 mt-0.5" />
                        {con}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            {/* Analysis summary */}
            {ipo.analysis_summary && (
              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider mb-2" style={{ color: 'var(--text-muted)' }}>
                  Analysis Summary
                </h4>
                <p className="text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                  {ipo.analysis_summary}
                </p>
              </div>
            )}

            {/* Lead managers */}
            {ipo.lead_managers?.length > 0 && (
              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider mb-2" style={{ color: 'var(--text-muted)' }}>
                  Lead Managers
                </h4>
                <div className="flex flex-wrap gap-2">
                  {ipo.lead_managers.map((lm, i) => (
                    <Badge key={i} variant="muted" size="sm">{lm}</Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
