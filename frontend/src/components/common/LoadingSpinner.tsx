interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg'
  label?: string
  fullPage?: boolean
}

const sizeMap = {
  sm: 'w-4 h-4 border-2',
  md: 'w-8 h-8 border-2',
  lg: 'w-12 h-12 border-3',
}

export default function LoadingSpinner({ size = 'md', label, fullPage }: LoadingSpinnerProps) {
  const spinner = (
    <div className="flex flex-col items-center gap-3">
      <div
        className={`${sizeMap[size]} rounded-full animate-spin`}
        style={{
          borderColor: 'var(--border)',
          borderTopColor: 'var(--accent)',
        }}
      />
      {label && (
        <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
          {label}
        </p>
      )}
    </div>
  )

  if (fullPage) {
    return (
      <div
        className="fixed inset-0 flex items-center justify-center z-50"
        style={{ backgroundColor: 'rgba(15, 23, 42, 0.85)' }}
      >
        {spinner}
      </div>
    )
  }

  return spinner
}
