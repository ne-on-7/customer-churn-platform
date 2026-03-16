import { useState, useEffect } from 'react'
import { ShieldAlert, Loader2 } from 'lucide-react'
import { getHighRiskCustomers } from '../lib/api'
import type { HighRiskCustomer } from '../types'

export default function RiskWatchlist() {
  const [customers, setCustomers] = useState<HighRiskCustomer[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [limit, setLimit] = useState(20)

  useEffect(() => {
    setLoading(true)
    getHighRiskCustomers(limit)
      .then(setCustomers)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [limit])

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
            <ShieldAlert size={20} className="text-red-600 dark:text-red-400" />
          </div>
          <div>
            <h1 className="text-xl font-semibold">Risk Watchlist</h1>
            <p className="text-sm text-gray-500 dark:text-gray-400">Top customers most likely to churn</p>
          </div>
        </div>
        <select value={limit} onChange={(e) => setLimit(Number(e.target.value))}
          className="px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm">
          {[10, 20, 50, 100].map((n) => <option key={n} value={n}>Top {n}</option>)}
        </select>
      </div>

      {loading && <div className="flex items-center justify-center py-12"><Loader2 className="animate-spin text-indigo-500" size={32} /></div>}
      {error && <div className="p-4 rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 text-sm">{error}</div>}

      {!loading && !error && (
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  {['#', 'Customer ID', 'Churn Probability', 'Risk Tier', 'Top Reason', 'Monthly Charges', 'Tenure', 'Contract'].map((h) => (
                    <th key={h} className="text-left px-4 py-3 font-medium text-gray-500 dark:text-gray-400 text-xs uppercase">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {customers.map((c, i) => (
                  <tr key={c.customer_id} className="border-b border-gray-100 dark:border-gray-700/50 hover:bg-gray-50 dark:hover:bg-gray-700/30 transition-colors">
                    <td className="px-4 py-3 text-gray-400">{i + 1}</td>
                    <td className="px-4 py-3 font-medium">{c.customer_id}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 rounded-full bg-gray-200 dark:bg-gray-700 overflow-hidden">
                          <div className="h-full rounded-full" style={{
                            width: `${c.churn_probability * 100}%`,
                            backgroundColor: c.risk_tier === 'High' ? '#ef4444' : c.risk_tier === 'Medium' ? '#f59e0b' : '#10b981',
                          }} />
                        </div>
                        <span className="tabular-nums">{(c.churn_probability * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                        c.risk_tier === 'High' ? 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400' :
                        c.risk_tier === 'Medium' ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400' :
                        'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400'
                      }`}>{c.risk_tier}</span>
                    </td>
                    <td className="px-4 py-3 text-gray-500 max-w-xs truncate">{c.top_reason}</td>
                    <td className="px-4 py-3 tabular-nums">${c.monthly_charges.toFixed(2)}</td>
                    <td className="px-4 py-3 tabular-nums">{c.tenure} mo</td>
                    <td className="px-4 py-3 text-gray-500">{c.contract}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
