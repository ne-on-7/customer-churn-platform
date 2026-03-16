import { useState } from 'react'
import { Target, Loader2 } from 'lucide-react'
import { predict } from '../lib/api'
import type { PredictionResult, CustomerInput } from '../types'

const yn = (v: string) => (v === 'Yes' ? 1 : 0)

const COST_MULTIPLIER = 1.15 // ~15% for taxes and fees

function ChurnGauge({ probability }: { probability: number }) {
  const pct = probability * 100
  const circumference = 2 * Math.PI * 45
  const offset = circumference - (probability * circumference)
  const color = pct >= 50 ? '#ef4444' : pct >= 20 ? '#f59e0b' : '#10b981'

  return (
    <div className="flex flex-col items-center">
      <svg width="120" height="120" className="-rotate-90">
        <circle cx="60" cy="60" r="45" fill="none" stroke="currentColor" strokeWidth="8" className="text-gray-200 dark:text-gray-700" />
        <circle cx="60" cy="60" r="45" fill="none" stroke={color} strokeWidth="8"
          strokeDasharray={circumference} strokeDashoffset={offset}
          strokeLinecap="round" className="gauge-ring" />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <div className="text-2xl font-bold" style={{ color }}>{pct.toFixed(1)}%</div>
        <div className="text-xs text-gray-500 dark:text-gray-400">Churn Risk</div>
      </div>
    </div>
  )
}

function Select({ label, value, onChange, options }: { label: string; value: string; onChange: (v: string) => void; options: string[] }) {
  return (
    <div>
      <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{label}</label>
      <select value={value} onChange={(e) => onChange(e.target.value)}
        className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent">
        {options.map((o) => <option key={o}>{o}</option>)}
      </select>
    </div>
  )
}

function NumberInput({ label, value, onChange, min = 0, max = 10000, step = 1 }: { label: string; value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number }) {
  return (
    <div>
      <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{label}</label>
      <input type="number" value={value} min={min} max={max} step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        onBlur={(e) => {
          const raw = Number(e.target.value)
          if (isNaN(raw) || raw < min) onChange(min)
          else if (raw > max) onChange(max)
        }}
        className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent" />
    </div>
  )
}

export default function PredictForm() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [error, setError] = useState('')

  // Form state
  const [gender, setGender] = useState('Female')
  const [senior, setSenior] = useState('No')
  const [partner, setPartner] = useState('No')
  const [dependents, setDependents] = useState('No')
  const [tenure, setTenure] = useState(24)
  const [phone, setPhone] = useState('Yes')
  const [multipleLines, setMultipleLines] = useState('No')
  const [internet, setInternet] = useState('DSL')
  const [security, setSecurity] = useState('Yes')
  const [backup, setBackup] = useState('No')
  const [protection, setProtection] = useState('No')
  const [tech, setTech] = useState('No')
  const [tv, setTv] = useState('No')
  const [movies, setMovies] = useState('No')
  const [contract, setContract] = useState('One year')
  const [paperless, setPaperless] = useState('No')
  const [payment, setPayment] = useState('Bank transfer (automatic)')
  const [monthly, setMonthly] = useState(55)

  const totalCharges = Math.round(tenure * monthly * COST_MULTIPLIER * 100) / 100

  const handleSubmit = async () => {
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const input: CustomerInput = {
        gender: gender === 'Male' ? 1 : 0,
        SeniorCitizen: yn(senior),
        Partner: yn(partner),
        Dependents: yn(dependents),
        tenure,
        PhoneService: yn(phone),
        MultipleLines: yn(multipleLines),
        OnlineSecurity: yn(security),
        OnlineBackup: yn(backup),
        DeviceProtection: yn(protection),
        TechSupport: yn(tech),
        StreamingTV: yn(tv),
        StreamingMovies: yn(movies),
        PaperlessBilling: yn(paperless),
        MonthlyCharges: monthly,
        TotalCharges: totalCharges,
        'InternetService_Fiber optic': internet === 'Fiber optic' ? 1 : 0,
        InternetService_No: internet === 'No' ? 1 : 0,
        'Contract_One year': contract === 'One year' ? 1 : 0,
        'Contract_Two year': contract === 'Two year' ? 1 : 0,
        'PaymentMethod_Credit card (automatic)': payment === 'Credit card (automatic)' ? 1 : 0,
        'PaymentMethod_Electronic check': payment === 'Electronic check' ? 1 : 0,
        'PaymentMethod_Mailed check': payment === 'Mailed check' ? 1 : 0,
      }
      const res = await predict(input)
      setResult(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Prediction failed')
    }
    setLoading(false)
  }

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center">
          <Target size={20} className="text-indigo-600 dark:text-indigo-400" />
        </div>
        <div>
          <h1 className="text-xl font-semibold">Churn Risk Prediction</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400">Enter customer details to predict churn probability</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Demographics */}
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5">
          <h3 className="font-medium mb-4 text-sm text-gray-700 dark:text-gray-300">Demographics</h3>
          <div className="space-y-3">
            <Select label="Gender" value={gender} onChange={setGender} options={['Female', 'Male']} />
            <Select label="Senior Citizen" value={senior} onChange={setSenior} options={['No', 'Yes']} />
            <Select label="Partner" value={partner} onChange={setPartner} options={['No', 'Yes']} />
            <Select label="Dependents" value={dependents} onChange={setDependents} options={['No', 'Yes']} />
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Tenure: {tenure} months</label>
              <input type="range" min={0} max={72} value={tenure} onChange={(e) => setTenure(Number(e.target.value))}
                className="w-full accent-indigo-500" />
            </div>
          </div>
        </div>

        {/* Services */}
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5">
          <h3 className="font-medium mb-4 text-sm text-gray-700 dark:text-gray-300">Services</h3>
          <div className="space-y-3">
            <Select label="Phone Service" value={phone} onChange={setPhone} options={['No', 'Yes']} />
            <Select label="Multiple Lines" value={multipleLines} onChange={setMultipleLines} options={['No', 'Yes']} />
            <Select label="Internet Service" value={internet} onChange={setInternet} options={['DSL', 'Fiber optic', 'No']} />
            <Select label="Online Security" value={security} onChange={setSecurity} options={['No', 'Yes']} />
            <Select label="Online Backup" value={backup} onChange={setBackup} options={['No', 'Yes']} />
            <Select label="Device Protection" value={protection} onChange={setProtection} options={['No', 'Yes']} />
            <Select label="Tech Support" value={tech} onChange={setTech} options={['No', 'Yes']} />
            <Select label="Streaming TV" value={tv} onChange={setTv} options={['No', 'Yes']} />
            <Select label="Streaming Movies" value={movies} onChange={setMovies} options={['No', 'Yes']} />
          </div>
        </div>

        {/* Billing */}
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5">
          <h3 className="font-medium mb-4 text-sm text-gray-700 dark:text-gray-300">Billing</h3>
          <div className="space-y-3">
            <Select label="Contract" value={contract} onChange={setContract} options={['Month-to-month', 'One year', 'Two year']} />
            <Select label="Paperless Billing" value={paperless} onChange={setPaperless} options={['No', 'Yes']} />
            <Select label="Payment Method" value={payment} onChange={setPayment}
              options={['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']} />
            <NumberInput label="Monthly Charges ($)" value={monthly} onChange={setMonthly} min={0} max={200} step={0.5} />
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Total Charges ($)</label>
              <div className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 text-sm text-gray-600 dark:text-gray-400">
                ${totalCharges.toFixed(2)}
                <span className="text-xs ml-1">({tenure} mo x ${monthly} + 15% taxes/fees)</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <button onClick={handleSubmit} disabled={loading}
        className="mt-6 w-full py-3 rounded-xl bg-indigo-500 hover:bg-indigo-600 disabled:opacity-50 text-white font-medium transition-colors flex items-center justify-center gap-2">
        {loading ? <><Loader2 size={18} className="animate-spin" /> Predicting...</> : 'Predict Churn Risk'}
      </button>

      {error && (
        <div className="mt-4 p-4 rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="mt-6 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 animate-fade-in">
          <div className="flex flex-col md:flex-row items-center gap-8">
            <div className="relative">
              <ChurnGauge probability={result.churn_probability} />
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-4">
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  result.risk_tier === 'High' ? 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400' :
                  result.risk_tier === 'Medium' ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400' :
                  'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400'
                }`}>
                  {result.risk_tier} Risk
                </span>
                <span className="text-xs text-gray-500 dark:text-gray-400">Model: {result.model_used}</span>
              </div>
              <h3 className="font-medium text-sm text-gray-700 dark:text-gray-300 mb-2">Top Churn Drivers</h3>
              <div className="space-y-2">
                {result.top_reasons.map((reason, i) => (
                  <div key={i} className="flex items-start gap-2">
                    <span className="w-5 h-5 rounded-full bg-indigo-100 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400 text-xs flex items-center justify-center flex-shrink-0 mt-0.5">
                      {i + 1}
                    </span>
                    <span className="text-sm text-gray-600 dark:text-gray-400">{reason}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
