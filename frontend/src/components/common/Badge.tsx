import clsx from 'clsx'

type BadgeVariant = 'buy' | 'sell' | 'hold' | 'invest' | 'avoid' | 'neutral' | 'accent' | 'muted' | 'large' | 'mid' | 'small' | 'high' | 'medium' | 'low'

interface BadgeProps {
  variant?: BadgeVariant
  children: React.ReactNode
  className?: string
  size?: 'sm' | 'md' | 'lg'
}

const variantStyles: Record<BadgeVariant, { bg: string; color: string }> = {
  buy:     { bg: 'rgba(34,197,94,0.15)',   color: '#22c55e' },
  sell:    { bg: 'rgba(239,68,68,0.15)',   color: '#ef4444' },
  hold:    { bg: 'rgba(245,158,11,0.15)',  color: '#f59e0b' },
  invest:  { bg: 'rgba(34,197,94,0.15)',   color: '#22c55e' },
  avoid:   { bg: 'rgba(239,68,68,0.15)',   color: '#ef4444' },
  neutral: { bg: 'rgba(245,158,11,0.15)',  color: '#f59e0b' },
  accent:  { bg: 'rgba(99,102,241,0.15)',  color: '#6366f1' },
  muted:   { bg: 'rgba(100,116,139,0.2)',  color: '#94a3b8' },
  large:   { bg: 'rgba(99,102,241,0.15)',  color: '#818cf8' },
  mid:     { bg: 'rgba(59,130,246,0.15)',  color: '#60a5fa' },
  small:   { bg: 'rgba(100,116,139,0.2)',  color: '#94a3b8' },
  high:    { bg: 'rgba(239,68,68,0.15)',   color: '#ef4444' },
  medium:  { bg: 'rgba(245,158,11,0.15)',  color: '#f59e0b' },
  low:     { bg: 'rgba(34,197,94,0.15)',   color: '#22c55e' },
}

const sizeStyles = {
  sm: 'text-xs px-1.5 py-0.5',
  md: 'text-xs px-2 py-1',
  lg: 'text-sm px-3 py-1',
}

export default function Badge({ variant = 'muted', children, className, size = 'md' }: BadgeProps) {
  const style = variantStyles[variant]
  return (
    <span
      className={clsx('inline-flex items-center rounded font-semibold', sizeStyles[size], className)}
      style={{ backgroundColor: style.bg, color: style.color }}
    >
      {children}
    </span>
  )
}
