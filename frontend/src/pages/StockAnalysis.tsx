import { useState, useEffect } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'
import { TrendingUp } from 'lucide-react'
import { useAnalyzeStock } from '../hooks/useStocks'
import type { AnalysisResult } from '../types/stocks'
import StockSelector from '../components/stock/StockSelector'
import DecisionCard from '../components/stock/DecisionCard'
import SignalBreakdown from '../components/stock/SignalBreakdown'
import MetricsGrid from '../components/stock/MetricsGrid'
import CandlestickChart from '../components/stock/CandlestickChart'
import NewsPanel from '../components/stock/NewsPanel'
import BacktestCard from '../components/stock/BacktestCard'
import NarrativeReport from '../components/stock/NarrativeReport'
import LoadingSpinner from '../components/common/LoadingSpinner'
import ErrorCard from '../components/common/ErrorCard'

export default function StockAnalysis() {
  const { symbol } = useParams<{ symbol?: string }>()
  const [searchParams] = useSearchParams()
  const analyzeMutation = useAnalyzeStock()
  const [result, setResult] = useState<AnalysisResult | null>(null)

  // Auto-run if navigated from Dashboard with autorun param
  useEffect(() => {
    if (symbol && searchParams.get('autorun') === '1') {
      const tf = searchParams.get('timeframe') || '1Y'
      handleAnalyze(symbol, tf)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol])

  async function handleAnalyze(sym: string, timeframe: string) {
    try {
      const data = await analyzeMutation.mutateAsync({ symbol: sym, timeframe })
      setResult(data)
    } catch {
      // error handled via analyzeMutation.error
    }
  }

  const isLoading = analyzeMutation.isPending
  const error = analyzeMutation.error

  return (
    <div className="flex flex-col gap-6 max-w-7xl">
      {/* Selector */}
      <div
        className="rounded-xl border p-5"
        style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border)' }}
      >
        <StockSelector onAnalyze={handleAnalyze} isAnalyzing={isLoading} />
      </div>

      {/* Loading overlay */}
      {isLoading && (
        <LoadingSpinner fullPage size="lg" label="Running AI Models... This may take 10–30 seconds" />
      )}

      {/* Error */}
      {error && !isLoading && (
        <ErrorCard
          message={
            error instanceof Error
              ? error.message
              : 'Analysis failed. Please check the stock symbol and try again.'
          }
          onRetry={() => result && handleAnalyze(result.symbol, result.timeframe)}
        />
      )}

      {/* Empty state */}
      {!result && !isLoading && !error && (
        <div
          className="rounded-xl border p-16 flex flex-col items-center gap-6"
          style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border)' }}
        >
          <div
            className="w-16 h-16 rounded-full flex items-center justify-center"
            style={{ backgroundColor: 'var(--bg-elevated)' }}
          >
            <TrendingUp size={28} color="var(--text-muted)" />
          </div>
          <div className="text-center">
            <h3 className="text-lg font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
              No Analysis Yet
            </h3>
            <p className="text-sm max-w-sm" style={{ color: 'var(--text-secondary)' }}>
              Search for a stock above and click "Analyze" to run the full AI pipeline — including ML,
              LSTM, TCN, HMM regime detection, and PPO reinforcement learning.
            </p>
          </div>
          <div className="grid grid-cols-3 gap-3 text-center max-w-md w-full">
            {['Technical Signals', 'Fundamental Data', 'News Sentiment'].map((label) => (
              <div
                key={label}
                className="px-3 py-3 rounded-lg text-xs"
                style={{ backgroundColor: 'var(--bg-elevated)', color: 'var(--text-muted)' }}
              >
                {label}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Analysis results */}
      {result && !isLoading && (
        <div className="flex flex-col gap-5">
          {/* Row 1: Decision card full width */}
          <DecisionCard result={result} />

          {/* Row 2: Chart (2/3) + Signal Breakdown (1/3) */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
            <div className="lg:col-span-2">
              <CandlestickChart
                data={result.chart_data}
                supports={result.support_resistance.supports}
                resistances={result.support_resistance.resistances}
              />
            </div>
            <div className="lg:col-span-1">
              <SignalBreakdown
                signals={result.signals}
                shap_ranked={result.shap_ranked}
              />
            </div>
          </div>

          {/* Row 3: Metrics (1/3) + Backtest (1/3) + News sentiment summary (1/3) */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            <MetricsGrid fundamentals={result.fundamentals} />
            <BacktestCard backtest={result.backtest} />
            {/* News sentiment mini panel */}
            <div
              className="rounded-xl border p-5 flex flex-col gap-4"
              style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border)' }}
            >
              <h3 className="font-semibold text-sm uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
                News Summary
              </h3>
              <div className="flex flex-col gap-3">
                <div
                  className="flex flex-col items-center justify-center rounded-lg p-4"
                  style={{ backgroundColor: 'var(--bg-elevated)' }}
                >
                  <span
                    className="text-3xl font-black"
                    style={{
                      color: result.news_sentiment > 0.1 ? 'var(--buy)'
                        : result.news_sentiment < -0.1 ? 'var(--sell)'
                        : 'var(--hold)',
                    }}
                  >
                    {result.news_sentiment > 0.1 ? 'POSITIVE'
                      : result.news_sentiment < -0.1 ? 'NEGATIVE'
                      : 'NEUTRAL'}
                  </span>
                  <span className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
                    Score: {result.news_sentiment >= 0 ? '+' : ''}{result.news_sentiment.toFixed(3)}
                  </span>
                </div>
                <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                  <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>
                    {result.news.length}
                  </span>{' '}
                  news items analysed
                </div>
                <div className="flex justify-between text-xs" style={{ color: 'var(--text-muted)' }}>
                  <span style={{ color: 'var(--buy)' }}>
                    {result.news.filter((n) => n.label.toUpperCase() === 'POSITIVE').length} Positive
                  </span>
                  <span style={{ color: 'var(--text-muted)' }}>
                    {result.news.filter((n) => n.label.toUpperCase() === 'NEUTRAL').length} Neutral
                  </span>
                  <span style={{ color: 'var(--sell)' }}>
                    {result.news.filter((n) => n.label.toUpperCase() === 'NEGATIVE').length} Negative
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Row 4: Narrative report full width */}
          <NarrativeReport narrative={result.narrative} />

          {/* Row 5: News panel full width */}
          <NewsPanel news={result.news} sentiment_score={result.news_sentiment} />
        </div>
      )}
    </div>
  )
}
