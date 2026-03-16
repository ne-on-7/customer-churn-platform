import { useState, useEffect } from 'react'
import { FlaskConical, Loader2 } from 'lucide-react'
import Plot from '../lib/Plot'
import { createExperiment, listExperiments, getExperiment, powerAnalysis } from '../lib/api'
import type { ExperimentSummary, PowerAnalysisResult } from '../types'

type SubTab = 'create' | 'power' | 'results' | 'history'

export default function ABTesting() {
  const [subTab, setSubTab] = useState<SubTab>('create')

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center">
          <FlaskConical size={20} className="text-purple-600 dark:text-purple-400" />
        </div>
        <div>
          <h1 className="text-xl font-semibold">A/B Testing</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400">Design and analyze retention experiments</p>
        </div>
      </div>

      {/* Sub-tabs */}
      <div className="flex gap-1 bg-gray-100 dark:bg-gray-800 rounded-xl p-1 mb-6">
        {([['create', 'Create'], ['power', 'Power Analysis'], ['results', 'Results'], ['history', 'History']] as const).map(([key, label]) => (
          <button key={key} onClick={() => setSubTab(key)}
            className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
              subTab === key ? 'bg-white dark:bg-gray-700 shadow-sm' : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            }`}>
            {label}
          </button>
        ))}
      </div>

      {subTab === 'create' && <CreateExperiment />}
      {subTab === 'power' && <PowerAnalysis />}
      {subTab === 'results' && <ExperimentResults />}
      {subTab === 'history' && <ExperimentHistory />}
    </div>
  )
}

function CreateExperiment() {
  const [name, setName] = useState('Discount for High-Risk Customers')
  const [type, setType] = useState('discount')
  const [desc, setDesc] = useState('20% monthly discount for 3 months')
  const [effectSize, setEffectSize] = useState(0.15)
  const [cost, setCost] = useState(20)
  const [riskTiers, setRiskTiers] = useState(['High'])
  const [splitRatio, setSplitRatio] = useState(0.5)
  const [alpha, setAlpha] = useState(0.05)
  const [power, setPower] = useState(0.8)
  const [seed] = useState(42)
  const [avgRev, setAvgRev] = useState(65)
  const [months, setMonths] = useState(6)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<Record<string, unknown> | null>(null)
  const [error, setError] = useState('')

  const handleCreate = async () => {
    setLoading(true); setError(''); setResult(null)
    try {
      const res = await createExperiment({
        name, intervention_type: type, intervention_description: desc,
        expected_effect_size: effectSize, cost_per_customer: cost,
        risk_tiers: riskTiers, split_ratio: splitRatio,
        significance_level: alpha, power, random_seed: seed,
        avg_monthly_revenue: avgRev, months_saved: months,
      })
      setResult(res as Record<string, unknown>)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed')
    }
    setLoading(false)
  }

  const toggleTier = (tier: string) => {
    setRiskTiers((prev) => prev.includes(tier) ? prev.filter((t) => t !== tier) : [...prev, tier])
  }

  const Select = ({ label, value, onChange, options }: { label: string; value: string | number; onChange: (v: string) => void; options: (string | number)[] }) => (
    <div>
      <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{label}</label>
      <select value={value} onChange={(e) => onChange(e.target.value)}
        className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm">
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  )

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-3">
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Experiment Name</label>
            <input value={name} onChange={(e) => setName(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm" />
          </div>
          <Select label="Intervention Type" value={type} onChange={setType} options={['discount', 'personalized_email', 'service_upgrade', 'loyalty_program']} />
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Description</label>
            <input value={desc} onChange={(e) => setDesc(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm" />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Effect Size: {effectSize.toFixed(2)}</label>
            <input type="range" min={0.01} max={0.5} step={0.01} value={effectSize} onChange={(e) => setEffectSize(Number(e.target.value))}
              className="w-full accent-indigo-500" />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Cost per Customer ($)</label>
            <input type="number" value={cost} onChange={(e) => setCost(Number(e.target.value))}
              className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm" />
          </div>
        </div>
        <div className="space-y-3">
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Target Risk Tiers</label>
            <div className="flex gap-2">
              {['High', 'Medium', 'Low'].map((tier) => (
                <button key={tier} onClick={() => toggleTier(tier)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                    riskTiers.includes(tier) ? 'bg-indigo-500 text-white' : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
                  }`}>{tier}</button>
              ))}
            </div>
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Split Ratio: {splitRatio.toFixed(2)}</label>
            <input type="range" min={0.1} max={0.9} step={0.05} value={splitRatio} onChange={(e) => setSplitRatio(Number(e.target.value))}
              className="w-full accent-indigo-500" />
          </div>
          <Select label="Significance Level" value={String(alpha)} onChange={(v) => setAlpha(Number(v))} options={[0.01, 0.05, 0.10]} />
          <Select label="Statistical Power" value={String(power)} onChange={(v) => setPower(Number(v))} options={[0.70, 0.75, 0.80, 0.85, 0.90, 0.95]} />
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Avg Monthly Revenue ($)</label>
            <input type="number" value={avgRev} onChange={(e) => setAvgRev(Number(e.target.value))}
              className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm" />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Months Saved: {months}</label>
            <input type="range" min={1} max={24} value={months} onChange={(e) => setMonths(Number(e.target.value))}
              className="w-full accent-indigo-500" />
          </div>
        </div>
      </div>
      <button onClick={handleCreate} disabled={loading}
        className="mt-4 w-full py-2.5 rounded-xl bg-purple-500 hover:bg-purple-600 disabled:opacity-50 text-white font-medium transition-colors flex items-center justify-center gap-2">
        {loading ? <><Loader2 size={16} className="animate-spin" /> Running...</> : 'Create & Run Experiment'}
      </button>
      {error && <p className="mt-3 text-sm text-red-500">{error}</p>}
      {result && (
        <div className="mt-4 p-4 rounded-xl bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 text-sm animate-fade-in">
          <p className="font-medium text-emerald-700 dark:text-emerald-400">Experiment created successfully!</p>
          <pre className="mt-2 text-xs text-gray-600 dark:text-gray-400 overflow-x-auto max-h-60">{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  )
}

function PowerAnalysis() {
  const [baseline, setBaseline] = useState(0.42)
  const [mde, setMde] = useState(0.10)
  const [alpha, setAlpha] = useState(0.05)
  const [power, setPower] = useState(0.80)
  const [result, setResult] = useState<PowerAnalysisResult | null>(null)
  const [curveData, setCurveData] = useState<{ x: number[]; y: number[] } | null>(null)

  useEffect(() => {
    powerAnalysis(baseline, mde, alpha, power).then(setResult).catch(console.error)
  }, [baseline, mde, alpha, power])

  useEffect(() => {
    const fetchCurve = async () => {
      const xs: number[] = []
      const ys: number[] = []
      for (let es = 0.02; es <= 0.4; es += 0.02) {
        const r = await powerAnalysis(baseline, es, alpha, power)
        xs.push(es)
        ys.push(r.required_sample_size_per_group)
      }
      setCurveData({ x: xs, y: ys })
    }
    fetchCurve().catch(console.error)
  }, [baseline, alpha, power])

  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Baseline Churn: {(baseline * 100).toFixed(0)}%</label>
            <input type="range" min={0.05} max={0.95} step={0.01} value={baseline} onChange={(e) => setBaseline(Number(e.target.value))} className="w-full accent-indigo-500" />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Min Detectable Effect: {(mde * 100).toFixed(0)}%</label>
            <input type="range" min={0.01} max={0.4} step={0.01} value={mde} onChange={(e) => setMde(Number(e.target.value))} className="w-full accent-indigo-500" />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Alpha</label>
            <select value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm">
              {[0.01, 0.05, 0.10].map((a) => <option key={a} value={a}>{a}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Power</label>
            <select value={power} onChange={(e) => setPower(Number(e.target.value))} className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm">
              {[0.70, 0.75, 0.80, 0.85, 0.90, 0.95].map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>
        </div>
      </div>

      {result && (
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
            <p className="text-xs text-gray-500 dark:text-gray-400">Per Group</p>
            <p className="text-2xl font-bold mt-1">{result.required_sample_size_per_group.toLocaleString()}</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
            <p className="text-xs text-gray-500 dark:text-gray-400">Total Required</p>
            <p className="text-2xl font-bold mt-1">{result.total_required.toLocaleString()}</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-4">
            <p className="text-xs text-gray-500 dark:text-gray-400">Achieved Power</p>
            <p className="text-2xl font-bold mt-1">{(result.achieved_power * 100).toFixed(0)}%</p>
          </div>
        </div>
      )}

      {curveData && (
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5">
          <h3 className="font-medium text-sm mb-3">Sample Size vs. Effect Size</h3>
          <Plot
            data={[
              { type: 'scatter', mode: 'lines', x: curveData.x, y: curveData.y, line: { color: '#6366f1', width: 3 } },
            ]}
            layout={{
              height: 350,
              margin: { t: 20, b: 60, l: 80, r: 20 },
              xaxis: { title: 'Minimum Detectable Effect' },
              yaxis: { title: 'Required Sample Size per Group' },
              shapes: [{ type: 'line', x0: mde, x1: mde, y0: 0, y1: 1, yref: 'paper', line: { color: '#ef4444', dash: 'dash' } }],
              annotations: [{ x: mde, y: 1, yref: 'paper', text: `MDE = ${(mde * 100).toFixed(0)}%`, showarrow: false, yanchor: 'bottom', font: { color: '#ef4444' } }],
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              font: { color: '#94a3b8' },
            }}
            config={{ displayModeBar: false, responsive: true }}
            className="w-full"
          />
        </div>
      )}
    </div>
  )
}

function ExperimentResults() {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([])
  const [selectedId, setSelectedId] = useState('')
  const [expData, setExpData] = useState<Record<string, unknown> | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    listExperiments().then((exps) => {
      setExperiments(exps)
      if (exps.length > 0) setSelectedId(exps[0].experiment_id)
    }).catch(console.error).finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (selectedId) {
      getExperiment(selectedId).then(setExpData).catch(console.error)
    }
  }, [selectedId])

  if (loading) return <div className="flex items-center justify-center py-12"><Loader2 className="animate-spin text-indigo-500" /></div>
  if (experiments.length === 0) return <p className="text-sm text-gray-500 py-8 text-center">No experiments yet. Create one first.</p>

  const results = expData?.results as Record<string, unknown> | undefined
  const sa = results?.statistical_analysis as Record<string, unknown> | undefined
  const bi = results?.business_impact as Record<string, unknown> | undefined
  const outcomes = results?.outcomes as Record<string, Record<string, unknown>> | undefined
  const segments = results?.segment_breakdown as { segment: string; control_churn_rate: number; treatment_churn_rate: number }[] | undefined

  return (
    <div className="space-y-6">
      <select value={selectedId} onChange={(e) => setSelectedId(e.target.value)}
        className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm">
        {experiments.map((e) => <option key={e.experiment_id} value={e.experiment_id}>{e.name} ({e.experiment_id})</option>)}
      </select>

      {sa && (
        <>
          <div className={`p-4 rounded-xl text-sm ${
            sa.is_significant ? 'bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 text-emerald-700 dark:text-emerald-400'
              : 'bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 text-amber-700 dark:text-amber-400'
          }`}>
            {sa.is_significant ? 'Statistically Significant' : 'Not Statistically Significant'} (p = {(sa.p_value as number).toFixed(4)})
          </div>

          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {[
              { label: 'Absolute Lift', value: `${((sa.absolute_lift as number) * 100).toFixed(1)}%` },
              { label: 'P-Value', value: (sa.p_value as number).toFixed(4) },
              { label: '95% CI', value: `[${((sa.confidence_interval_95 as number[])[0] * 100).toFixed(1)}%, ${((sa.confidence_interval_95 as number[])[1] * 100).toFixed(1)}%]` },
              { label: 'NNT', value: (sa.number_needed_to_treat as number).toFixed(1) },
              { label: "Cohen's h", value: (sa.effect_size_cohens_h as number).toFixed(3) },
            ].map(({ label, value }) => (
              <div key={label} className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-3">
                <p className="text-xs text-gray-500 dark:text-gray-400">{label}</p>
                <p className="text-lg font-semibold mt-0.5">{value}</p>
              </div>
            ))}
          </div>

          {outcomes && (
            <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5">
              <h3 className="font-medium text-sm mb-3">Churn Rate: Control vs Treatment</h3>
              <Plot
                data={[{
                  type: 'bar',
                  x: ['Control', 'Treatment'],
                  y: [outcomes.control.churn_rate as number, outcomes.treatment.churn_rate as number],
                  marker: { color: ['#ef4444', '#10b981'] },
                  text: [`${((outcomes.control.churn_rate as number) * 100).toFixed(1)}%`, `${((outcomes.treatment.churn_rate as number) * 100).toFixed(1)}%`],
                  textposition: 'outside' as const,
                }]}
                layout={{
                  height: 350,
                  margin: { t: 20, b: 60, l: 60, r: 20 },
                  yaxis: { title: 'Churn Rate', tickformat: '.0%' },
                  paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                  font: { color: '#94a3b8' }, showlegend: false,
                }}
                config={{ displayModeBar: false, responsive: true }}
                className="w-full"
              />
            </div>
          )}

          {segments && segments.length > 0 && (
            <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5">
              <h3 className="font-medium text-sm mb-3">Segment Breakdown</h3>
              <Plot
                data={[
                  { type: 'bar', name: 'Control', x: segments.map((s) => s.segment), y: segments.map((s) => s.control_churn_rate), marker: { color: '#ef4444' } },
                  { type: 'bar', name: 'Treatment', x: segments.map((s) => s.segment), y: segments.map((s) => s.treatment_churn_rate), marker: { color: '#10b981' } },
                ]}
                layout={{
                  height: 350,
                  barmode: 'group',
                  margin: { t: 20, b: 80, l: 60, r: 20 },
                  yaxis: { title: 'Churn Rate', tickformat: '.0%' },
                  paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                  font: { color: '#94a3b8' },
                }}
                config={{ displayModeBar: false, responsive: true }}
                className="w-full"
              />
            </div>
          )}

          {bi && (
            <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5">
              <h3 className="font-medium text-sm mb-3">ROI Waterfall</h3>
              <Plot
                data={[{
                  type: 'waterfall',
                  x: ['Revenue Saved', 'Intervention Cost', 'Net ROI'],
                  y: [bi.revenue_saved as number, -(bi.intervention_cost as number), bi.net_roi as number],
                  measure: ['relative', 'relative', 'total'],
                  text: [`$${(bi.revenue_saved as number).toLocaleString()}`, `-$${(bi.intervention_cost as number).toLocaleString()}`, `$${(bi.net_roi as number).toLocaleString()}`],
                  textposition: 'outside',
                  decreasing: { marker: { color: '#ef4444' } },
                  increasing: { marker: { color: '#10b981' } },
                  totals: { marker: { color: '#6366f1' } },
                }]}
                layout={{
                  height: 350,
                  margin: { t: 20, b: 60, l: 80, r: 20 },
                  yaxis: { title: 'Amount ($)' },
                  paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                  font: { color: '#94a3b8' }, showlegend: false,
                }}
                config={{ displayModeBar: false, responsive: true }}
                className="w-full"
              />
            </div>
          )}
        </>
      )}
    </div>
  )
}

function ExperimentHistory() {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    listExperiments().then(setExperiments).catch(console.error).finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="flex items-center justify-center py-12"><Loader2 className="animate-spin text-indigo-500" /></div>
  if (experiments.length === 0) return <p className="text-sm text-gray-500 py-8 text-center">No experiments yet.</p>

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              {['Name', 'Status', 'Intervention', 'Risk Tiers', 'Lift', 'P-Value', 'Significant', 'ROI'].map((h) => (
                <th key={h} className="text-left px-4 py-3 font-medium text-gray-500 dark:text-gray-400 text-xs uppercase">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {experiments.map((e) => (
              <tr key={e.experiment_id} className="border-b border-gray-100 dark:border-gray-700/50 hover:bg-gray-50 dark:hover:bg-gray-700/30">
                <td className="px-4 py-3 font-medium">{e.name}</td>
                <td className="px-4 py-3"><span className="px-2 py-0.5 rounded-full bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400 text-xs">{e.status}</span></td>
                <td className="px-4 py-3 text-gray-500">{e.intervention_type?.replace(/_/g, ' ')}</td>
                <td className="px-4 py-3 text-gray-500">{e.risk_tiers?.join(', ')}</td>
                <td className="px-4 py-3 tabular-nums">{e.absolute_lift != null ? `${(e.absolute_lift * 100).toFixed(1)}%` : '—'}</td>
                <td className="px-4 py-3 tabular-nums">{e.p_value != null ? e.p_value.toFixed(4) : '—'}</td>
                <td className="px-4 py-3">{e.is_significant ? <span className="text-emerald-500">Yes</span> : <span className="text-gray-400">No</span>}</td>
                <td className="px-4 py-3 tabular-nums">{e.roi_percent != null ? `${e.roi_percent}%` : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
