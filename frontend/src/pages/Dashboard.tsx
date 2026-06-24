import { useNavigate } from 'react-router-dom'
import { Brain, TrendingUp, Star, Activity, ChevronRight } from 'lucide-react'
import { useAnalyzeStock } from '../hooks/useStocks'
import { useIPOList } from '../hooks/useIPO'
import StockSelector from '../components/stock/StockSelector'
import Badge from '../components/common/Badge'
import LoadingSpinner from '../components/common/LoadingSpinner'

const STATS = [
  { label: '500 Stocks', sub: 'NSE Coverage', icon: TrendingUp, color: 'var(--accent)' },
  { label: '5 AI Models', sub: 'ML + DL + RL', icon: Brain, color: 'var(--buy)' },
  { label: 'Real-time', sub: 'News Sentiment', icon: Activity, color: 'var(--hold)' },
  { label: 'Backtested', sub: 'Strategy Verified', icon: Star, color: '#a78bfa' },
]

function formatDate(dateStr: string): string {
  try {
    return new Date(dateStr).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' })
  } catch {
    return dateStr
  }
}

export default function Dashboard() {
  const navigate = useNavigate()
  const analyzeMutation = useAnalyzeStock()
  const { data: ipos, isLoading: iposLoading } = useIPOList('active')

  function handleAnalyze(symbol: string, timeframe: string) {
    navigate(`/stocks/${symbol}?timeframe=${timeframe}&autorun=1`)
  }

  const activeIPOs = ipos?.slice(0, 3) ?? []

  return (
    <div className="flex flex-col gap-8 max-w-5xl">
      {/* Welcome header */}
      <div>
        <h1 className="text-2xl font-bold mb-1" style={{ color: 'var(--text-primary)' }}>
          Sensei AI
        </h1>
        <p className="text-base" style={{ color: 'var(--text-secondary)' }}>
          Intelligent Stock Analysis — Powered by 5 AI Model Families
        </p>
      </div>

      {/* Quick stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {STATS.map((stat) => (
          <div
            key={stat.label}
            className="rounded-xl border p-4 flex flex-col gap-3"
            style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border)' }}
          >
            <div
              className="w-9 h-9 rounded-lg flex items-center justify-center"
              style={{ backgroundColor: `${stat.color}20` }}
            >
              <stat.icon size={18} color={stat.color} />
            </div>
            <div>
              <div className="font-bold text-lg" style={{ color: 'var(--text-primary)' }}>
                {stat.label}
              </div>
              <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {stat.sub}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Quick analysis card */}
      <div
        className="rounded-xl border p-6"
        style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border)' }}
      >
        <div className="flex items-start justify-between mb-5">
          <div>
            <h2 className="font-semibold text-base" style={{ color: 'var(--text-primary)' }}>
              Quick Analysis
            </h2>
            <p className="text-sm mt-0.5" style={{ color: 'var(--text-muted)' }}>
              Search for any of the 500 NSE stocks and run a full AI analysis
            </p>
          </div>
        </div>
        <StockSelector onAnalyze={handleAnalyze} isAnalyzing={analyzeMutation.isPending} />
      </div>

      {/* AI Models overview */}
      <div
        className="rounded-xl border p-6"
        style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border)' }}
      >
        <h2 className="font-semibold text-base mb-4" style={{ color: 'var(--text-primary)' }}>
          AI Model Pipeline
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
          {[
            { name: 'Rule Engine', tag: 'Technical', color: 'var(--text-secondary)', desc: 'RSI, MACD, Bollinger Bands' },
            { name: 'Random Forest', tag: 'ML', color: 'var(--accent)', desc: 'Probability-based signal' },
            { name: 'LSTM', tag: 'Deep Learning', color: '#818cf8', desc: 'Sequence return prediction' },
            { name: 'TCN', tag: 'Deep Learning', color: '#60a5fa', desc: 'Temporal convolutions' },
            { name: 'PPO Agent', tag: 'RL', color: '#34d399', desc: 'Reinforcement learning action' },
          ].map((model) => (
            <div
              key={model.name}
              className="rounded-lg p-3 flex flex-col gap-1.5"
              style={{ backgroundColor: 'var(--bg-elevated)' }}
            >
              <span className="text-xs font-semibold" style={{ color: model.color }}>
                {model.tag}
              </span>
              <span className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>
                {model.name}
              </span>
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {model.desc}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Active IPOs preview */}
      <div
        className="rounded-xl border p-6"
        style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border)' }}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-semibold text-base" style={{ color: 'var(--text-primary)' }}>
            Active IPOs
          </h2>
          <button
            onClick={() => navigate('/ipo')}
            className="flex items-center gap-1 text-sm font-medium transition-colors"
            style={{ color: 'var(--accent)' }}
          >
            View All <ChevronRight size={14} />
          </button>
        </div>

        {iposLoading ? (
          <div className="flex justify-center py-8">
            <LoadingSpinner size="md" label="Loading IPOs..." />
          </div>
        ) : activeIPOs.length === 0 ? (
          <div className="text-center py-8" style={{ color: 'var(--text-muted)' }}>
            No active IPOs at the moment.{' '}
            <button onClick={() => navigate('/ipo')} style={{ color: 'var(--accent)' }}>
              View upcoming
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {activeIPOs.map((ipo) => (
              <div
                key={ipo.id}
                className="rounded-lg border p-4 flex flex-col gap-3 cursor-pointer"
                style={{ backgroundColor: 'var(--bg-elevated)', borderColor: 'var(--border)' }}
                onClick={() => navigate(`/ipo/${ipo.id}`)}
              >
                <div className="flex items-start justify-between gap-2">
                  <span className="font-semibold text-sm truncate" style={{ color: 'var(--text-primary)' }}>
                    {ipo.company_name}
                  </span>
                  <Badge
                    variant={ipo.recommendation === 'INVEST' ? 'invest' : ipo.recommendation === 'AVOID' ? 'avoid' : 'neutral'}
                    size="sm"
                  >
                    {ipo.recommendation}
                  </Badge>
                </div>
                <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {ipo.sector}
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span style={{ color: 'var(--text-secondary)' }}>
                    ₹{ipo.issue_price_min}–₹{ipo.issue_price_max}
                  </span>
                  <span
                    style={{ color: ipo.gmp >= 0 ? 'var(--buy)' : 'var(--sell)' }}
                  >
                    GMP: {ipo.gmp >= 0 ? '+' : ''}₹{ipo.gmp}
                  </span>
                </div>
                <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  Closes: {formatDate(ipo.close_date)}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
