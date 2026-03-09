import { Routes, Route } from 'react-router-dom'
import AppShell from './components/layout/AppShell'
import Dashboard from './pages/Dashboard'
import IntelligenceBriefing from './pages/IntelligenceBriefing'
import CorrelationExplorer from './pages/CorrelationExplorer'
import EventTimeline from './pages/EventTimeline'
import WorldMapView from './pages/WorldMapView'
import Signals from './pages/Signals'
import AgentChat from './pages/AgentChat'
import PredictionMarkets from './pages/PredictionMarkets'

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/briefing" element={<IntelligenceBriefing />} />
        <Route path="/markets" element={<PredictionMarkets />} />
        <Route path="/correlation" element={<CorrelationExplorer />} />
        <Route path="/timeline" element={<EventTimeline />} />
        <Route path="/map" element={<WorldMapView />} />
        <Route path="/signals" element={<Signals />} />
        <Route path="/agent" element={<AgentChat />} />
      </Route>
    </Routes>
  )
}
