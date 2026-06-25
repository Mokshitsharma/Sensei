import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Star, TrendingUp, Clock, CheckCircle } from 'lucide-react'
import { useIPOList, useIPODetail } from '../hooks/useIPO'
import IPOList from '../components/ipo/IPOList'
import IPODetail from '../components/ipo/IPODetail'
import LoadingSpinner from '../components/common/LoadingSpinner'
import ErrorCard from '../components/common/ErrorCard'

export default function IPOTracker() {
  const { id } = useParams<{ id?: string }>()
  const navigate = useNavigate()
  const [selectedId, setSelectedId] = useState<string | null>(id ?? null)

  const { data: ipos, isLoading, error, refetch } = useIPOList()
  const { data: ipoDetail, isLoading: detailLoading } = useIPODetail(selectedId)

  // Sync URL param to selectedId
  useEffect(() => {
    if (id) setSelectedId(id)
  }, [id])

  function handleViewDetails(ipoId: string) {
    setSelectedId(ipoId)
    navigate(`/ipo/${ipoId}`, { replace: true })
  }

  function handleCloseDetail() {
    setSelectedId(null)
    navigate('/ipo', { replace: true })
  }

  const totalActive = ipos?.filter((i) => i.status === 'active').length ?? 0
  const totalUpcoming = ipos?.filter((i) => i.status === 'upcoming').length ?? 0
  const totalListed = ipos?.filter((i) => i.status === 'listed').length ?? 0

  return (
    <div className="flex flex-col gap-6 max-w-7xl">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold mb-1" style={{ color: 'var(--text-primary)' }}>
          IPO Intelligence
        </h1>
        <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
          Track active, upcoming and recently listed IPOs with AI-powered recommendations
        </p>
      </div>

      {/* Stats */}
      {!isLoading && ipos && (
        <div className="grid grid-cols-3 gap-4">
          {[
            { label: 'Active', value: totalActive, icon: Star, color: 'var(--buy)' },
            { label: 'Upcoming', value: totalUpcoming, icon: Clock, color: 'var(--hold)' },
            { label: 'Listed', value: totalListed, icon: CheckCircle, color: 'var(--accent)' },
          ].map((stat) => (
            <div
              key={stat.label}
              className="rounded-xl border p-4 flex items-center gap-4"
              style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border)' }}
            >
              <div
                className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0"
                style={{ backgroundColor: `${stat.color}20` }}
              >
                <stat.icon size={20} color={stat.color} />
              </div>
              <div>
                <div className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>
                  {stat.value}
                </div>
                <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {stat.label} IPOs
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Content */}
      {isLoading ? (
        <div className="flex justify-center py-16">
          <LoadingSpinner size="lg" label="Loading IPOs..." />
        </div>
      ) : error ? (
        <ErrorCard
          message="Failed to load IPO data. Please try again."
          onRetry={() => refetch()}
        />
      ) : ipos ? (
        <IPOList ipos={ipos} onViewDetails={handleViewDetails} />
      ) : null}

      {/* Detail panel/modal */}
      {selectedId && (
        <IPODetail
          ipo={ipoDetail}
          loading={detailLoading}
          onClose={handleCloseDetail}
        />
      )}
    </div>
  )
}
