import { Routes, Route } from 'react-router-dom'
import Layout from './components/layout/Layout'
import Dashboard from './pages/Dashboard'
import StockAnalysis from './pages/StockAnalysis'
import IPOTracker from './pages/IPOTracker'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/stocks/:symbol?" element={<StockAnalysis />} />
        <Route path="/ipo" element={<IPOTracker />} />
        <Route path="/ipo/:id" element={<IPOTracker />} />
      </Routes>
    </Layout>
  )
}
