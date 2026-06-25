import type { BacktestMetrics } from '../../types/stocks'
import { BarChart2 } from 'lucide-react'

interface BacktestCardProps {
  backtest: BacktestMetrics
}

interface MetricProps {
  label: string
  value: string
  valueColor: string
  subtext: string
}

function MetricItem({ label, value, valueColor, subtext }: MetricProps) {
  return (
    <div
      className="flex flex-col gap-1 p-3 rounded-lg"
      style={{ backgroundColor: 'var(--bg-elevated)' }}
    >
      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
        {label}
      </span>
      <span className="text-xl font-bold" style={{ color: valueColor }}>
        {value}
      </span>
      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
        {subtext}
      </span>
    </div>
  )
}

export default function BacktestCard({ backtest }: BacktestCardProps) {
  const returnColor = backtest.total_return >= 0 ? 'var(--buy)' : 'var(--sell)'
  const sharpeColor = backtest.sharpe_ratio >= 1 ? 'var(--buy)' : backtest.sharpe_ratio >= 0.5 ? 'var(--hold)' : 'var(--sell)'

  return (
    <div
      className="rounded-xl border p-5 flex flex-col gap-4"
      style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border)' }}
    >
      <div className="flex items-center gap-2">
        <BarChart2 size={16} color="var(--text-muted)" />
        <h3 className="font-semibold text-sm uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
          Backtest Results
        </h3>
      </div>

      <div className="flex flex-col gap-2">
        <MetricItem
          label="Total Return"
          value={`${backtest.total_return >= 0 ? '+' : ''}${(backtest.total_return * 100).toFixed(2)}%`}
          valueColor={returnColor}
          subtext="Historical strategy return"
        />
        <MetricItem
          label="Sharpe Ratio"
          value={backtest.sharpe_ratio.toFixed(2)}
          valueColor={sharpeColor}
          subtext={backtest.sharpe_ratio >= 1 ? 'Good risk-adjusted return' : backtest.sharpe_ratio >= 0.5 ? 'Moderate' : 'Poor risk-adjusted return'}
        />
        <MetricItem
          label="Max Drawdown"
          value={`-${(Math.abs(backtest.max_drawdown) * 100).toFixed(2)}%`}
          valueColor="var(--sell)"
          subtext="Peak-to-trough decline"
        />
      </div>

      {/* Return bar indicator */}
      <div>
        <div className="text-xs mb-1.5" style={{ color: 'var(--text-muted)' }}>
          Return vs Drawdown
        </div>
        <div className="relative h-4 rounded-full overflow-hidden flex" style={{ backgroundColor: 'var(--bg-elevated)' }}>
          {/* Return portion (green) */}
          <div
            className="h-full transition-all duration-700"
            style={{
              width: `${Math.min(Math.abs(backtest.total_return) * 100 * 2, 60)}%`,
              backgroundColor: 'var(--buy)',
            }}
          />
          {/* Drawdown portion (red) */}
          <div
            className="h-full transition-all duration-700"
            style={{
              width: `${Math.min(Math.abs(backtest.max_drawdown) * 100 * 2, 40)}%`,
              backgroundColor: 'var(--sell)',
            }}
          />
        </div>
        <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
          <span style={{ color: 'var(--buy)' }}>Return</span>
          <span style={{ color: 'var(--sell)' }}>Drawdown</span>
        </div>
      </div>
    </div>
  )
}
