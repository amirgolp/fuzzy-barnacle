import { Box, Tab, Tabs } from '@mui/material'
import {
  BrowserRouter,
  Link,
  Navigate,
  Route,
  Routes,
  useLocation,
} from 'react-router-dom'

import { BacktestLab } from './pages/BacktestLab'
import { DataExplorer } from './pages/DataExplorer'
import { MarketIntelPage } from './pages/MarketIntelPage'
import { PortfolioOptimizer } from './pages/PortfolioOptimizer'
import { Screener } from './pages/Screener'
import { StrategyBuilder } from './pages/StrategyBuilder'

const NAV_ITEMS = [
  { label: 'Charts', path: '/' },
  { label: 'Intel', path: '/intel' },
  { label: 'Screener', path: '/screener' },
  { label: 'Strategies', path: '/strategies' },
  { label: 'Backtest', path: '/backtest' },
  { label: 'Optimize', path: '/optimize' },
]

function NavBar() {
  const location = useLocation()
  const current = NAV_ITEMS.findIndex((item) => item.path === location.pathname)

  return (
    <Box
      sx={{
        borderBottom: 1,
        borderColor: 'divider',
        bgcolor: 'background.paper',
      }}
    >
      <Tabs value={current === -1 ? 0 : current}>
        {NAV_ITEMS.map((item) => (
          <Tab
            key={item.path}
            label={item.label}
            component={Link}
            to={item.path}
          />
        ))}
      </Tabs>
    </Box>
  )
}

function App() {
  return (
    <BrowserRouter>
      <NavBar />
      <Routes>
        <Route path="/" element={<DataExplorer />} />
        <Route path="/intel" element={<MarketIntelPage />} />
        <Route path="/screener" element={<Screener />} />
        <Route path="/strategies" element={<StrategyBuilder />} />
        <Route path="/backtest" element={<BacktestLab />} />
        <Route path="/optimize" element={<PortfolioOptimizer />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
