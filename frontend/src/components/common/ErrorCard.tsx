import { AlertCircle, RefreshCw } from 'lucide-react'

interface ErrorCardProps {
  message?: string
  onRetry?: () => void
}

export default function ErrorCard({ message = 'Something went wrong.', onRetry }: ErrorCardProps) {
  return (
    <div
      className="flex flex-col items-center gap-4 p-8 rounded-xl border text-center"
      style={{
        backgroundColor: 'var(--bg-surface)',
        borderColor: 'var(--sell)',
      }}
    >
      <AlertCircle size={36} color="var(--sell)" />
      <div>
        <p className="font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>
          Error
        </p>
        <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
          {message}
        </p>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          style={{ backgroundColor: 'var(--accent)', color: '#fff' }}
        >
          <RefreshCw size={14} />
          Retry
        </button>
      )}
    </div>
  )
}
