import { useState, useEffect } from 'react'
import { Database, Loader2 } from 'lucide-react'
import Plot from '../lib/Plot'
import { getDataOverview, getChurnRates, getDistribution } from '../lib/api'
import type { DataOverview, ChurnRateData } from '../types'

const COLORS = ['#6366f1', '#ec4899', '#14b8a6', '#f59e0b', '#8b5cf6']

export default function DataExplorer() {
  const [overview, setOverview] = useState<DataOverview | null>(null)
  const [churnData, setChurnData] = useState<ChurnRateData[]>([])
  const [distData, setDistData] = useState<{ bins: number[]; churned: number[]; retained: number[] } | null>(null)
  const [groupBy, setGroupBy] = useState('Contract')
  const [feature, setFeature] = useState('MonthlyCharges')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getDataOverview().then(setOverview).catch(console.error).finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    getChurnRates(groupBy).then(setChurnData).catch(console.error)
  }, [groupBy])

  useEffect(() => {
    getDistribution(feature).then(setDistData).catch(console.error)
  }, [feature])

  if (loading) return <div className="flex items-center justify-center h-full"><Loader2 className="animate-spin text-indigo-500" size={32} /></div>

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center">
          <Database size={20} className="text-indigo-600 dark:text-indigo-400" />
        </div>
        <div>
          <h1 className="text-xl font-semibold">Data Explorer</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400">Explore the telco customer dataset</p>
        </div>
      </div>

      {/* Overview Cards */}
      {overview && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          {[
            { label: 'Total Customers', value: overview.rows.toLocaleString() },
            { label: 'Features', value: overview.columns },
            { label: 'Churn Rate', value: `${(overview.churn_rate * 100).toFixed(1)}%` },
            { label: 'Dataset', value: 'IBM Telco' },
          ].map(({ label, value }) => (
            <div key={label} className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
              <p className="text-xs text-gray-500 dark:text-gray-400">{label}</p>
              <p className="text-xl font-semibold mt-1">{value}</p>
            </div>
          ))}
        </div>
      )}

      {/* Churn Rate Analysis */}
      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 mb-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-medium text-sm">Churn Rate Analysis</h3>
          <select value={groupBy} onChange={(e) => setGroupBy(e.target.value)}
            className="px-3 py-1.5 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm">
            {['Contract', 'InternetService', 'gender', 'PaymentMethod', 'SeniorCitizen'].map((g) => (
              <option key={g} value={g}>{g}</option>
            ))}
          </select>
        </div>
        {churnData.length > 0 && (
          <Plot
            data={[{
              type: 'bar',
              x: churnData.map((d) => d.group),
              y: churnData.map((d) => d.churn_rate),
              marker: { color: COLORS.slice(0, churnData.length) },
              text: churnData.map((d) => `${(d.churn_rate * 100).toFixed(1)}%`),
              textposition: 'outside' as const,
            }]}
            layout={{
              height: 350,
              margin: { t: 20, b: 80, l: 60, r: 20 },
              yaxis: { title: 'Churn Rate', tickformat: '.0%' },
              xaxis: { title: groupBy },
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              font: { color: '#94a3b8' },
            }}
            config={{ displayModeBar: false, responsive: true }}
            className="w-full"
          />
        )}
      </div>

      {/* Feature Distribution */}
      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-medium text-sm">Feature Distribution</h3>
          <select value={feature} onChange={(e) => setFeature(e.target.value)}
            className="px-3 py-1.5 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm">
            {['MonthlyCharges', 'TotalCharges', 'tenure'].map((f) => (
              <option key={f} value={f}>{f}</option>
            ))}
          </select>
        </div>
        {distData && (
          <Plot
            data={[
              { type: 'bar', x: distData.bins, y: distData.retained, name: 'Retained', marker: { color: '#6366f1', opacity: 0.7 } },
              { type: 'bar', x: distData.bins, y: distData.churned, name: 'Churned', marker: { color: '#ef4444', opacity: 0.7 } },
            ]}
            layout={{
              height: 350,
              barmode: 'overlay',
              margin: { t: 20, b: 60, l: 60, r: 20 },
              xaxis: { title: feature },
              yaxis: { title: 'Count' },
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              font: { color: '#94a3b8' },
              legend: { x: 0.8, y: 0.95 },
            }}
            config={{ displayModeBar: false, responsive: true }}
            className="w-full"
          />
        )}
      </div>
    </div>
  )
}
