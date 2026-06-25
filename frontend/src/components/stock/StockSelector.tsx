import { useState, useEffect, useRef } from 'react'
import { Search, X, ChevronDown } from 'lucide-react'
import clsx from 'clsx'
import { useStockSearch } from '../../hooks/useStocks'
import type { StockListItem } from '../../types/stocks'
import Badge from '../common/Badge'
import LoadingSpinner from '../common/LoadingSpinner'

const TIMEFRAMES = ['1Y', '2Y', '5Y']

interface StockSelectorProps {
  onAnalyze: (symbol: string, timeframe: string) => void
  isAnalyzing: boolean
}

export default function StockSelector({ onAnalyze, isAnalyzing }: StockSelectorProps) {
  const [query, setQuery] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const [selectedStock, setSelectedStock] = useState<StockListItem | null>(null)
  const [timeframe, setTimeframe] = useState('1Y')
  const [showDropdown, setShowDropdown] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  // Debounce
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedQuery(query), 300)
    return () => clearTimeout(timer)
  }, [query])

  const { data: results, isLoading: searching } = useStockSearch(debouncedQuery)

  // Close dropdown on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setShowDropdown(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  function handleSelect(stock: StockListItem) {
    setSelectedStock(stock)
    setQuery(stock.name)
    setShowDropdown(false)
  }

  function handleClear() {
    setSelectedStock(null)
    setQuery('')
    setDebouncedQuery('')
  }

  function handleAnalyze() {
    if (selectedStock) {
      onAnalyze(selectedStock.symbol, timeframe)
    }
  }

  const showResults = showDropdown && debouncedQuery.length >= 2

  return (
    <div className="flex flex-col gap-4">
      {/* Search input */}
      <div className="relative" ref={dropdownRef}>
        <div
          className="flex items-center gap-2 px-3 py-2.5 rounded-lg border"
          style={{
            backgroundColor: 'var(--bg-elevated)',
            borderColor: showDropdown ? 'var(--accent)' : 'var(--border)',
          }}
        >
          <Search size={16} color="var(--text-muted)" />
          <input
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value)
              setShowDropdown(true)
              if (!e.target.value) setSelectedStock(null)
            }}
            onFocus={() => setShowDropdown(true)}
            placeholder="Search stocks (e.g. HDFC, Infosys, Reliance)..."
            className="flex-1 bg-transparent outline-none text-sm"
            style={{ color: 'var(--text-primary)' }}
          />
          {searching && <LoadingSpinner size="sm" />}
          {query && !searching && (
            <button onClick={handleClear}>
              <X size={14} color="var(--text-muted)" />
            </button>
          )}
          <ChevronDown size={14} color="var(--text-muted)" />
        </div>

        {/* Dropdown */}
        {showResults && (
          <div
            className="absolute top-full left-0 right-0 mt-1 rounded-lg border shadow-xl z-50 overflow-hidden"
            style={{
              backgroundColor: 'var(--bg-surface)',
              borderColor: 'var(--border)',
            }}
          >
            {results && results.length > 0 ? (
              <ul className="max-h-64 overflow-y-auto">
                {results.map((stock) => (
                  <li key={stock.symbol}>
                    <button
                      onClick={() => handleSelect(stock)}
                      className="w-full flex items-center justify-between px-4 py-3 hover:bg-[var(--bg-elevated)] transition-colors text-left"
                    >
                      <div className="flex flex-col gap-0.5">
                        <span className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>
                          {stock.symbol}
                        </span>
                        <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                          {stock.name}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                          {stock.sector}
                        </span>
                        <Badge variant={stock.tier} size="sm">
                          {stock.tier}
                        </Badge>
                      </div>
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="px-4 py-6 text-center text-sm" style={{ color: 'var(--text-muted)' }}>
                {searching ? 'Searching...' : 'No stocks found'}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Timeframe + Analyze row */}
      <div className="flex items-center gap-3">
        <span className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
          Timeframe:
        </span>
        <div className="flex gap-1">
          {TIMEFRAMES.map((tf) => (
            <button
              key={tf}
              onClick={() => setTimeframe(tf)}
              className={clsx(
                'px-3 py-1.5 rounded text-xs font-semibold transition-colors',
                timeframe === tf ? 'text-white' : 'hover:text-white'
              )}
              style={{
                backgroundColor: timeframe === tf ? 'var(--accent)' : 'var(--bg-elevated)',
                color: timeframe === tf ? '#fff' : 'var(--text-secondary)',
              }}
            >
              {tf}
            </button>
          ))}
        </div>

        <button
          onClick={handleAnalyze}
          disabled={!selectedStock || isAnalyzing}
          className={clsx(
            'ml-auto flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-semibold transition-colors',
            (!selectedStock || isAnalyzing) && 'opacity-50 cursor-not-allowed'
          )}
          style={{
            backgroundColor: 'var(--buy)',
            color: '#fff',
          }}
        >
          {isAnalyzing ? (
            <>
              <LoadingSpinner size="sm" />
              Analyzing...
            </>
          ) : (
            'Analyze'
          )}
        </button>
      </div>

      {selectedStock && (
        <div
          className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm"
          style={{ backgroundColor: 'var(--bg-elevated)', color: 'var(--text-secondary)' }}
        >
          <span>Selected:</span>
          <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>
            {selectedStock.symbol}
          </span>
          <span>{selectedStock.name}</span>
          <Badge variant={selectedStock.tier} size="sm">
            {selectedStock.tier}
          </Badge>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {selectedStock.sector}
          </span>
        </div>
      )}
    </div>
  )
}
