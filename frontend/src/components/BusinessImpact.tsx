import { useState } from 'react'
import { DollarSign, Loader2 } from 'lucide-react'
import { computeBusinessImpact } from '../lib/api'
import type { BusinessImpactResult } from '../types'

export default function BusinessImpact() {
  const [avgRevenue, setAvgRevenue] = useState(65)
  const [retentionCost, setRetentionCost] = useState(20)
  const [monthsSaved, setMonthsSaved] = useState(6)
  const [result, setResult] = useState<BusinessImpactResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const calculate = async () => {
    setLoading(true)
    setError('')
    try {
      const res = await computeBusinessImpact(avgRevenue, retentionCost, monthsSaved)
      setResult(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Calculation failed')
    }
    setLoading(false)
  }

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl bg-emerald-100 dark:bg-emerald-900/30 flex items-center justify-center">
          <DollarSign size={20} className="text-emerald-600 dark:text-emerald-400" />
        </div>
        <div>
          <h1 className="text-xl font-semibold">Business Impact Calculator</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400">Estimate financial impact of proactive retention</p>
        </div>
      </div>

      {/* Input Form */}
      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Avg Monthly Revenue ($)</label>
            <input type="number" value={avgRevenue} onChange={(e) => setAvgRevenue(Number(e.target.value))}
              className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm focus:ring-2 focus:ring-indigo-500" />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Retention Offer Cost ($)</label>
            <input type="number" value={retentionCost} onChange={(e) => setRetentionCost(Number(e.target.value))}
              className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm focus:ring-2 focus:ring-indigo-500" />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Months Saved: {monthsSaved}</label>
            <input type="range" min={1} max={24} value={monthsSaved} onChange={(e) => setMonthsSaved(Number(e.target.value))}
              className="w-full accent-indigo-500 mt-2" />
          </div>
        </div>
        <button onClick={calculate} disabled={loading}
          className="mt-4 w-full py-2.5 rounded-xl bg-emerald-500 hover:bg-emerald-600 disabled:opacity-50 text-white font-medium transition-colors flex items-center justify-center gap-2">
          {loading ? <><Loader2 size={16} className="animate-spin" /> Calculating...</> : 'Calculate Impact'}
        </button>
      </div>

      {error && (
        <div className="mb-4 p-4 rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="animate-fade-in">
          {/* KPI Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            {[
              { label: 'Revenue Saved', value: `$${result.revenue_saved.toLocaleString()}`, color: 'emerald', border: 'border-l-emerald-500' },
              { label: 'Campaign Cost', value: `$${result.retention_spend.toLocaleString()}`, color: 'red', border: 'border-l-red-500' },
              { label: 'Net Benefit', value: `$${result.net_benefit.toLocaleString()}`, color: 'blue', border: 'border-l-blue-500' },
              { label: 'ROI', value: `${result.roi_percent}%`, color: 'indigo', border: 'border-l-indigo-500' },
            ].map(({ label, value, border }) => (
              <div key={label} className={`bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 border-l-4 ${border} p-4`}>
                <p className="text-xs text-gray-500 dark:text-gray-400">{label}</p>
                <p className="text-2xl font-bold mt-1">{value}</p>
              </div>
            ))}
          </div>

          {/* Confusion Matrix */}
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5">
            <h3 className="font-medium text-sm mb-4">Confusion Matrix Breakdown</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr>
                      <th className="p-2"></th>
                      <th className="p-2 text-center text-xs text-gray-500">Predicted Retained</th>
                      <th className="p-2 text-center text-xs text-gray-500">Predicted Churned</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="p-2 text-xs font-medium text-gray-500">Actually Retained</td>
                      <td className="p-2 text-center bg-emerald-50 dark:bg-emerald-900/20 rounded">{result.true_negatives}</td>
                      <td className="p-2 text-center bg-amber-50 dark:bg-amber-900/20 rounded">{result.false_positives}</td>
                    </tr>
                    <tr>
                      <td className="p-2 text-xs font-medium text-gray-500">Actually Churned</td>
                      <td className="p-2 text-center bg-red-50 dark:bg-red-900/20 rounded">{result.false_negatives}</td>
                      <td className="p-2 text-center bg-emerald-50 dark:bg-emerald-900/20 rounded font-semibold">{result.true_positives}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div className="text-sm space-y-2">
                <p><span className="font-medium text-emerald-600">{result.true_positives}</span> churning customers caught — saved <span className="font-medium">${result.revenue_saved.toLocaleString()}</span></p>
                <p><span className="font-medium text-red-600">{result.false_negatives}</span> churning customers missed — lost <span className="font-medium">${result.missed_revenue.toLocaleString()}</span></p>
                <p><span className="font-medium text-amber-600">{result.false_positives}</span> unnecessary offers — wasted <span className="font-medium">${Math.round(result.retention_spend * result.false_positives / (result.true_positives + result.false_positives)).toLocaleString()}</span></p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
