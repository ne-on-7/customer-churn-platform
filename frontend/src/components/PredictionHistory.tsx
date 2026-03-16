import { useState, useEffect } from 'react'
import { History, Loader2, ChevronDown, ChevronUp } from 'lucide-react'
import { getPredictionHistory } from '../lib/api'
import type { PredictionHistoryItem } from '../types'

export default function PredictionHistory() {
  const [items, setItems] = useState<PredictionHistoryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [expanded, setExpanded] = useState<string | null>(null)

  useEffect(() => {
    getPredictionHistory()
      .then(setItems)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
          <History size={20} className="text-blue-600 dark:text-blue-400" />
        </div>
        <div>
          <h1 className="text-xl font-semibold">Prediction History</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400">Audit log of all predictions</p>
        </div>
      </div>

      {loading && <div className="flex items-center justify-center py-12"><Loader2 className="animate-spin text-indigo-500" size={32} /></div>}
      {error && <div className="p-4 rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 text-sm">{error}</div>}

      {!loading && !error && items.length === 0 && (
        <div className="text-center py-12 text-gray-500 dark:text-gray-400">
          <History size={48} className="mx-auto mb-3 opacity-30" />
          <p>No predictions yet. Make a prediction to see it here.</p>
        </div>
      )}

      {!loading && !error && items.length > 0 && (
        <div className="space-y-3">
          {items.map((item) => (
            <div key={item.id} className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
              <button
                onClick={() => setExpanded(expanded === item.id ? null : item.id)}
                className="w-full flex items-center justify-between px-5 py-4 hover:bg-gray-50 dark:hover:bg-gray-700/30 transition-colors"
              >
                <div className="flex items-center gap-4">
                  <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                    item.risk_tier === 'High' ? 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400' :
                    item.risk_tier === 'Medium' ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400' :
                    'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400'
                  }`}>{item.risk_tier}</span>
                  <span className="text-sm font-medium tabular-nums">{(item.churn_probability * 100).toFixed(1)}% churn risk</span>
                  <span className="text-xs text-gray-400">{new Date(item.timestamp).toLocaleString()}</span>
                </div>
                {expanded === item.id ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>
              {expanded === item.id && (
                <div className="px-5 pb-4 border-t border-gray-100 dark:border-gray-700/50 pt-3 animate-fade-in">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <p className="text-xs font-medium text-gray-500 mb-2">Top Reasons</p>
                      {item.top_reasons.map((r, i) => (
                        <p key={i} className="text-sm text-gray-600 dark:text-gray-400">{i + 1}. {r}</p>
                      ))}
                      <p className="text-xs text-gray-400 mt-2">Model: {item.model_used}</p>
                    </div>
                    <div>
                      <p className="text-xs font-medium text-gray-500 mb-2">Input Features</p>
                      <div className="text-xs text-gray-500 dark:text-gray-400 space-y-0.5 max-h-40 overflow-y-auto">
                        {Object.entries(item.inputs).map(([k, v]) => (
                          <div key={k} className="flex justify-between">
                            <span>{k}</span>
                            <span className="font-medium text-gray-700 dark:text-gray-300">{String(v)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
