import { useState, useEffect } from 'react'
import { BarChart3, Loader2 } from 'lucide-react'
import Plot from '../lib/Plot'
import { getModels } from '../lib/api'
import type { ModelResults } from '../types'

const COLORS = ['#6366f1', '#ec4899', '#14b8a6', '#f59e0b', '#8b5cf6']

export default function ModelComparison() {
  const [data, setData] = useState<ModelResults | null>(null)
  const [metric, setMetric] = useState('roc_auc')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getModels().then(setData).catch(console.error).finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="flex items-center justify-center h-full"><Loader2 className="animate-spin text-indigo-500" size={32} /></div>
  if (!data) return <div className="p-6 text-red-500">Failed to load model data</div>

  const modelNames = Object.keys(data.results)
  const metrics = ['roc_auc', 'f1', 'precision', 'recall', 'accuracy']

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center">
          <BarChart3 size={20} className="text-indigo-600 dark:text-indigo-400" />
        </div>
        <div>
          <h1 className="text-xl font-semibold">Model Performance</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400">Best model: <span className="text-indigo-500 font-medium">{data.best_model}</span></p>
        </div>
      </div>

      {/* Metrics Table */}
      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden mb-6">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <th className="text-left px-4 py-3 font-medium text-gray-500 dark:text-gray-400">Model</th>
              {metrics.map((m) => (
                <th key={m} className="text-right px-4 py-3 font-medium text-gray-500 dark:text-gray-400 uppercase text-xs">{m.replace('_', ' ')}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {modelNames.map((name) => {
              const r = data.results[name]
              return (
                <tr key={name} className="border-b border-gray-100 dark:border-gray-700/50 hover:bg-gray-50 dark:hover:bg-gray-700/30 transition-colors">
                  <td className="px-4 py-3 font-medium">
                    {name}
                    {name === data.best_model && <span className="ml-2 text-xs text-indigo-500">Best</span>}
                  </td>
                  {metrics.map((m) => {
                    const val = r[m as keyof typeof r]
                    const isMax = Math.max(...modelNames.map((n) => data.results[n][m as keyof typeof r] as number)) === val
                    return (
                      <td key={m} className={`text-right px-4 py-3 tabular-nums ${isMax ? 'text-emerald-600 dark:text-emerald-400 font-semibold' : ''}`}>
                        {(val as number).toFixed(4)}
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Metric Selector + Chart */}
      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-medium text-sm">Visual Comparison</h3>
          <select value={metric} onChange={(e) => setMetric(e.target.value)}
            className="px-3 py-1.5 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm">
            {metrics.map((m) => <option key={m} value={m}>{m.replace('_', ' ').toUpperCase()}</option>)}
          </select>
        </div>
        <Plot
          data={[{
            type: 'bar',
            x: modelNames,
            y: modelNames.map((n) => data.results[n][metric as keyof typeof data.results[typeof n]]),
            marker: { color: COLORS },
            text: modelNames.map((n) => (data.results[n][metric as keyof typeof data.results[typeof n]] as number).toFixed(4)),
            textposition: 'outside' as const,
          }]}
          layout={{
            height: 400,
            margin: { t: 20, b: 80, l: 60, r: 20 },
            yaxis: { title: metric.replace('_', ' ').toUpperCase() },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#94a3b8' },
          }}
          config={{ displayModeBar: false, responsive: true }}
          className="w-full"
        />
      </div>

      {/* ROC & PR Curves */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
        {[
          { src: '/api/plots/roc_curves.png', title: 'ROC Curves' },
          { src: '/api/plots/pr_curves.png', title: 'Precision-Recall Curves' },
        ].map(({ src, title }) => (
          <div key={title} className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5">
            <h3 className="font-medium text-sm mb-3">{title}</h3>
            <img src={src} alt={title} className="w-full rounded-lg" onError={(e) => (e.currentTarget.style.display = 'none')} />
          </div>
        ))}
      </div>
    </div>
  )
}
