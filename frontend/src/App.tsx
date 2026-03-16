import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Chat from './components/Chat'
import PredictForm from './components/PredictForm'
import ModelComparison from './components/ModelComparison'
import DataExplorer from './components/DataExplorer'
import BusinessImpact from './components/BusinessImpact'
import ABTesting from './components/ABTesting'
import RiskWatchlist from './components/RiskWatchlist'
import PredictionHistory from './components/PredictionHistory'
import BatchUpload from './components/BatchUpload'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Navigate to="/chat" replace />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/predict" element={<PredictForm />} />
        <Route path="/models" element={<ModelComparison />} />
        <Route path="/explorer" element={<DataExplorer />} />
        <Route path="/impact" element={<BusinessImpact />} />
        <Route path="/experiments" element={<ABTesting />} />
        <Route path="/watchlist" element={<RiskWatchlist />} />
        <Route path="/history" element={<PredictionHistory />} />
        <Route path="/batch" element={<BatchUpload />} />
      </Route>
    </Routes>
  )
}
