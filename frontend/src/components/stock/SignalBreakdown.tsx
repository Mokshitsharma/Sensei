import type { SignalBreakdown as SignalBreakdownType, ShapFeature } from '../../types/stocks'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

interface SignalBreakdownProps {
  signals: SignalBreakdownType
  shap_ranked: ShapFeature[]
}

function DirectionIcon({ direction }: { direction: string }) {
  if (direction === 'up' || direction === 'positive') return <TrendingUp size={14} color="var(--buy)" />
  if (direction === 'down' || direction === 'negative') return <TrendingDown size={14} color="var(--sell)" />
  return <Minus size={14} color="var(--hold)" />
}

interface SignalRowProps {
  label: string
  value: string
  color?: string
  icon?: React.ReactNode
}

function SignalRow({ label, value, color, icon }: SignalRowProps) {
  return (
    <div
      className="flex items-center justify-between px-3 py-2.5 rounded-lg"
      style={{ backgroundColor: 'var(--bg-elevated)' }}
    >
      <div className="flex items-center gap-2">
        {icon}
        <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
          {label}
        </span>
      </div>
      <span className="text-sm font-semibold" style={{ color: color || 'var(--text-primary)' }}>
        {value}
      </span>
    </div>
  )
}

function mlColor(prob: number): string {
  if (prob >= 0.6) return 'var(--buy)'
  if (prob <= 0.4) return 'var(--sell)'
  return 'var(--hold)'
}

function returnColor(ret: number): string {
  if (ret > 0) return 'var(--buy)'
  if (ret < 0) return 'var(--sell)'
  return 'var(--hold)'
}

function regimeColor(regime: string): string {
  const r = regime.toUpperCase()
  if (r.includes('BULL')) return 'var(--buy)'
  if (r.includes('BEAR')) return 'var(--sell)'
  return 'var(--hold)'
}

function ppoColor(action: string): string {
  const a = action.toUpperCase()
  if (a === 'BUY') return 'var(--buy)'
  if (a === 'SELL') return 'var(--sell)'
  return 'var(--hold)'
}

export default function SignalBreakdown({ signals, shap_ranked }: SignalBreakdownProps) {
  const maxShapValue = Math.max(...shap_ranked.map((f) => Math.abs(f.value)), 0.001)

  return (
    <div
      className="rounded-xl border p-5 flex flex-col gap-5"
      style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border)' }}
    >
      <h3 className="font-semibold text-sm uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
        Signal Breakdown
      </h3>

      {/* Model signals */}
      <div className="flex flex-col gap-2">
        <SignalRow
          label="Rule Signal"
          value={signals.rule_signal}
          color={signals.rule_signal?.toUpperCase() === 'BUY' ? 'var(--buy)' : signals.rule_signal?.toUpperCase() === 'SELL' ? 'var(--sell)' : 'var(--hold)'}
          icon={<TrendingUp size={14} color="var(--text-muted)" />}
        />
        <SignalRow
          label="ML Probability Up"
          value={`${(signals.ml_prob_up * 100).toFixed(1)}%`}
          color={mlColor(signals.ml_prob_up)}
          icon={<span className="text-xs font-bold" style={{ color: 'var(--accent)' }}>ML</span>}
        />
        <SignalRow
          label="LSTM Return"
          value={`${signals.lstm_return >= 0 ? '+' : ''}${(signals.lstm_return * 100).toFixed(2)}%`}
          color={returnColor(signals.lstm_return)}
          icon={<span className="text-xs font-bold" style={{ color: '#818cf8' }}>LSTM</span>}
        />
        <SignalRow
          label="TCN Return"
          value={`${signals.tcn_return >= 0 ? '+' : ''}${(signals.tcn_return * 100).toFixed(2)}%`}
          color={returnColor(signals.tcn_return)}
          icon={<span className="text-xs font-bold" style={{ color: '#60a5fa' }}>TCN</span>}
        />
        <SignalRow
          label="HMM Regime"
          value={signals.regime}
          color={regimeColor(signals.regime)}
          icon={<span className="text-xs font-bold" style={{ color: '#a78bfa' }}>HMM</span>}
        />
        <SignalRow
          label="PPO Action"
          value={signals.ppo_action}
          color={ppoColor(signals.ppo_action)}
          icon={<span className="text-xs font-bold" style={{ color: '#34d399' }}>PPO</span>}
        />
      </div>

      {/* SHAP feature importance */}
      {shap_ranked.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-muted)' }}>
            SHAP Feature Importance
          </h4>
          <div className="flex flex-col gap-2">
            {shap_ranked.slice(0, 8).map((feature, i) => {
              const pct = (Math.abs(feature.value) / maxShapValue) * 100
              const color = feature.direction === 'up' || feature.direction === 'positive'
                ? 'var(--buy)'
                : feature.direction === 'down' || feature.direction === 'negative'
                ? 'var(--sell)'
                : 'var(--hold)'

              return (
                <div key={i} className="flex flex-col gap-1">
                  <div className="flex justify-between text-xs">
                    <div className="flex items-center gap-1">
                      <DirectionIcon direction={feature.direction} />
                      <span style={{ color: 'var(--text-secondary)' }}>{feature.feature}</span>
                    </div>
                    <span style={{ color }}>{feature.value.toFixed(4)}</span>
                  </div>
                  <div className="h-1.5 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--bg-elevated)' }}>
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${pct}%`, backgroundColor: color }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
