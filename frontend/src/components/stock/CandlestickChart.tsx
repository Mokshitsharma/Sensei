import { useEffect, useRef } from 'react'
import { createChart, ColorType } from 'lightweight-charts'
import type { CandleBar } from '../../types/stocks'

interface CandlestickChartProps {
  data: CandleBar[]
  supports: number[]
  resistances: number[]
}

export default function CandlestickChart({ data, supports, resistances }: CandlestickChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<ReturnType<typeof createChart> | null>(null)

  useEffect(() => {
    if (!containerRef.current || data.length === 0) return

    const container = containerRef.current
    const chart = createChart(container, {
      layout: {
        background: { type: ColorType.Solid, color: '#0f172a' },
        textColor: '#94a3b8',
      },
      grid: {
        vertLines: { color: '#1e293b' },
        horzLines: { color: '#1e293b' },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: '#334155',
      },
      timeScale: {
        borderColor: '#334155',
        timeVisible: true,
        secondsVisible: false,
      },
      width: container.clientWidth,
      height: 340,
    })

    chartRef.current = chart

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderUpColor: '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    })

    // Sort data by time
    const sorted = [...data].sort((a, b) => a.time - b.time)
    candleSeries.setData(
      sorted.map((bar) => ({
        time: bar.time as unknown as import('lightweight-charts').Time,
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
      }))
    )

    // Support lines (green dashed)
    supports.forEach((level) => {
      const line = chart.addLineSeries({
        color: 'rgba(34, 197, 94, 0.6)',
        lineWidth: 1,
        lineStyle: 2, // dashed
        priceLineVisible: false,
        lastValueVisible: false,
      })
      if (sorted.length >= 2) {
        line.setData([
          { time: sorted[0].time as unknown as import('lightweight-charts').Time, value: level },
          { time: sorted[sorted.length - 1].time as unknown as import('lightweight-charts').Time, value: level },
        ])
      }
    })

    // Resistance lines (red dashed)
    resistances.forEach((level) => {
      const line = chart.addLineSeries({
        color: 'rgba(239, 68, 68, 0.6)',
        lineWidth: 1,
        lineStyle: 2, // dashed
        priceLineVisible: false,
        lastValueVisible: false,
      })
      if (sorted.length >= 2) {
        line.setData([
          { time: sorted[0].time as unknown as import('lightweight-charts').Time, value: level },
          { time: sorted[sorted.length - 1].time as unknown as import('lightweight-charts').Time, value: level },
        ])
      }
    })

    chart.timeScale().fitContent()

    // Handle resize
    const observer = new ResizeObserver(() => {
      if (container) {
        chart.applyOptions({ width: container.clientWidth })
      }
    })
    observer.observe(container)

    return () => {
      observer.disconnect()
      chart.remove()
      chartRef.current = null
    }
  }, [data, supports, resistances])

  if (data.length === 0) {
    return (
      <div
        className="rounded-xl border p-5"
        style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border)' }}
      >
        <h3 className="font-semibold text-sm uppercase tracking-wider mb-4" style={{ color: 'var(--text-muted)' }}>
          Price Chart
        </h3>
        <div
          className="flex items-center justify-center h-64 rounded-lg"
          style={{ backgroundColor: 'var(--bg-elevated)', color: 'var(--text-muted)' }}
        >
          No chart data available
        </div>
      </div>
    )
  }

  return (
    <div
      className="rounded-xl border p-5"
      style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border)' }}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-sm uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
          Price Chart
        </h3>
        <div className="flex items-center gap-4 text-xs" style={{ color: 'var(--text-muted)' }}>
          <span className="flex items-center gap-1">
            <span className="w-4 h-0.5 inline-block" style={{ backgroundColor: 'rgba(34,197,94,0.6)' }} />
            Support
          </span>
          <span className="flex items-center gap-1">
            <span className="w-4 h-0.5 inline-block" style={{ backgroundColor: 'rgba(239,68,68,0.6)' }} />
            Resistance
          </span>
        </div>
      </div>
      <div ref={containerRef} className="w-full" />
    </div>
  )
}
