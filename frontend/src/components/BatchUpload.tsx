import { useState, useRef } from 'react'
import { Upload, Loader2, Download, FileSpreadsheet } from 'lucide-react'
import { predictBatch } from '../lib/api'

export default function BatchUpload() {
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<Record<string, unknown>[] | null>(null)
  const [error, setError] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = (f: File) => {
    if (f.name.endsWith('.csv')) {
      setFile(f)
      setResults(null)
      setError('')
    } else {
      setError('Please upload a CSV file')
    }
  }

  const handleSubmit = async () => {
    if (!file) return
    setLoading(true)
    setError('')
    try {
      const res = await predictBatch(file)
      setResults(res.predictions)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed')
    }
    setLoading(false)
  }

  const downloadCSV = () => {
    if (!results) return
    const headers = Object.keys(results[0])
    const csv = [
      headers.join(','),
      ...results.map((r) => headers.map((h) => JSON.stringify(r[h] ?? '')).join(',')),
    ].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'churn_predictions.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center">
          <Upload size={20} className="text-amber-600 dark:text-amber-400" />
        </div>
        <div>
          <h1 className="text-xl font-semibold">Batch Prediction</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400">Upload a CSV of customers for bulk churn predictions</p>
        </div>
      </div>

      {/* Drop Zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]) }}
        onClick={() => inputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors ${
          dragOver
            ? 'border-indigo-500 bg-indigo-50 dark:bg-indigo-900/20'
            : 'border-gray-300 dark:border-gray-700 hover:border-indigo-400 hover:bg-gray-50 dark:hover:bg-gray-800'
        }`}
      >
        <input ref={inputRef} type="file" accept=".csv" onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} className="hidden" />
        <FileSpreadsheet size={40} className="mx-auto mb-3 text-gray-400" />
        {file ? (
          <p className="text-sm font-medium">{file.name} <span className="text-gray-400">({(file.size / 1024).toFixed(1)} KB)</span></p>
        ) : (
          <>
            <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Drop a CSV file here or click to browse</p>
            <p className="text-xs text-gray-400 mt-1">File should contain the same features as the training data</p>
          </>
        )}
      </div>

      {file && (
        <button onClick={handleSubmit} disabled={loading}
          className="mt-4 w-full py-2.5 rounded-xl bg-indigo-500 hover:bg-indigo-600 disabled:opacity-50 text-white font-medium transition-colors flex items-center justify-center gap-2">
          {loading ? <><Loader2 size={16} className="animate-spin" /> Processing...</> : 'Run Predictions'}
        </button>
      )}

      {error && (
        <div className="mt-4 p-4 rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 text-sm">{error}</div>
      )}

      {results && (
        <div className="mt-6 animate-fade-in">
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm font-medium">
              {results.length > 100 ? `Showing first 100 of ${results.length} predictions` : `${results.length} predictions`}
            </p>
            <button onClick={downloadCSV}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-sm transition-colors">
              <Download size={14} /> Download CSV
            </button>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
            <div className="overflow-x-auto max-h-96">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-white dark:bg-gray-800">
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    {Object.keys(results[0]).map((h) => (
                      <th key={h} className="text-left px-3 py-2 font-medium text-gray-500 text-xs uppercase whitespace-nowrap">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {results.slice(0, 100).map((r, i) => (
                    <tr key={i} className="border-b border-gray-100 dark:border-gray-700/50 hover:bg-gray-50 dark:hover:bg-gray-700/30">
                      {Object.values(r).map((v, j) => (
                        <td key={j} className="px-3 py-2 whitespace-nowrap">{String(v)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
