import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  IconButton,
  Tabs,
  Tab,
  Badge,
  Tooltip,
  Alert,
  Chip,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  ShowChart as ChartIcon,
  AccountBalance as PortfolioIcon,
  Settings as SettingsIcon,
  Notifications as NotificationsIcon,
  Warning as WarningIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  FullscreenIcon,
} from '@mui/icons-material';

// Import custom components
import MarketOverview from './widgets/MarketOverview';
import PortfolioSummary from './widgets/PortfolioSummary';
import PositionsTable from './widgets/PositionsTable';
import TradingChart from './widgets/TradingChart';
import SystemHealth from './widgets/SystemHealth';
import RecentTrades from './widgets/RecentTrades';
import ControlPanel from './widgets/ControlPanel';
import AlertsPanel from './widgets/AlertsPanel';
import MLInsights from './widgets/MLInsights';
import OrderBook from './widgets/OrderBook';
import PerformanceMetrics from './widgets/PerformanceMetrics';
import RiskMonitor from './widgets/RiskMonitor';

import { useWebSocket } from '@/contexts/WebSocketContext';
import { useTrading } from '@/contexts/TradingContext';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 2 }}>{children}</Box>}
    </div>
  );
}

export default function CommandCenter() {
  const [activeTab, setActiveTab] = useState(0);
  const [systemStatus, setSystemStatus] = useState('READY');
  const [alerts, setAlerts] = useState(3);
  const { connected, lastMessage } = useWebSocket();
  const { tradingEnabled, toggleTrading, emergencyStop } = useTrading();

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const getStatusColor = () => {
    switch (systemStatus) {
      case 'RUNNING':
        return '#00ff88';
      case 'PAUSED':
        return '#ffaa00';
      case 'STOPPED':
        return '#ff3366';
      default:
        return '#666';
    }
  };

  return (
    <Box sx={{ 
      flexGrow: 1, 
      height: '100vh', 
      display: 'flex', 
      flexDirection: 'column',
      bgcolor: 'background.default',
      overflow: 'hidden'
    }}>
      {/* Header Bar */}
      <Paper
        elevation={0}
        sx={{
          p: 2,
          borderRadius: 0,
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
          background: 'linear-gradient(90deg, #1a1a1a 0%, #0a0a0a 100%)',
        }}
      >
        <Grid container alignItems="center" spacing={2}>
          <Grid item>
            <Typography variant="h4" sx={{ fontWeight: 700, color: '#00ff88' }}>
              RRR Command Center
            </Typography>
          </Grid>
          <Grid item xs>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Chip
                label={connected ? 'CONNECTED' : 'DISCONNECTED'}
                color={connected ? 'success' : 'error'}
                size="small"
                icon={<Box sx={{ 
                  width: 8, 
                  height: 8, 
                  borderRadius: '50%',
                  bgcolor: connected ? '#00ff88' : '#ff3366',
                  animation: connected ? 'pulse 2s infinite' : 'none'
                }} />}
              />
              <Chip
                label={`System: ${systemStatus}`}
                sx={{ 
                  bgcolor: getStatusColor() + '20',
                  color: getStatusColor(),
                  borderColor: getStatusColor()
                }}
                variant="outlined"
                size="small"
              />
              <Chip
                label={tradingEnabled ? 'TRADING ACTIVE' : 'TRADING PAUSED'}
                color={tradingEnabled ? 'success' : 'warning'}
                size="small"
              />
            </Box>
          </Grid>
          <Grid item>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Tooltip title="Start Trading">
                <IconButton 
                  color="success" 
                  onClick={() => setSystemStatus('RUNNING')}
                  disabled={systemStatus === 'RUNNING'}
                >
                  <PlayIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="Pause Trading">
                <IconButton 
                  color="warning"
                  onClick={() => {
                    toggleTrading();
                    setSystemStatus('PAUSED');
                  }}
                  disabled={systemStatus === 'PAUSED'}
                >
                  <PauseIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="Emergency Stop">
                <IconButton 
                  color="error"
                  onClick={() => {
                    emergencyStop();
                    setSystemStatus('STOPPED');
                  }}
                >
                  <StopIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="Refresh Data">
                <IconButton color="primary">
                  <RefreshIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="Fullscreen">
                <IconButton color="primary">
                  <FullscreenIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="Notifications">
                <IconButton color="primary">
                  <Badge badgeContent={alerts} color="error">
                    <NotificationsIcon />
                  </Badge>
                </IconButton>
              </Tooltip>
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {/* Navigation Tabs */}
      <Paper
        elevation={0}
        sx={{
          borderRadius: 0,
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <Tabs
          value={activeTab}
          onChange={handleTabChange}
          indicatorColor="primary"
          textColor="primary"
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab icon={<DashboardIcon />} label="Dashboard" />
          <Tab icon={<ChartIcon />} label="Trading" />
          <Tab icon={<PortfolioIcon />} label="Portfolio" />
          <Tab icon={<WarningIcon />} label="Risk" />
          <Tab icon={<SettingsIcon />} label="Settings" />
        </Tabs>
      </Paper>

      {/* Main Content Area */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
        <TabPanel value={activeTab} index={0}>
          {/* Dashboard Tab */}
          <Grid container spacing={2}>
            <Grid item xs={12} md={8}>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <MarketOverview />
                </Grid>
                <Grid item xs={12} md={6}>
                  <PortfolioSummary />
                </Grid>
                <Grid item xs={12} md={6}>
                  <SystemHealth />
                </Grid>
                <Grid item xs={12}>
                  <PositionsTable />
                </Grid>
              </Grid>
            </Grid>
            <Grid item xs={12} md={4}>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <ControlPanel />
                </Grid>
                <Grid item xs={12}>
                  <AlertsPanel />
                </Grid>
                <Grid item xs={12}>
                  <RecentTrades />
                </Grid>
              </Grid>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={activeTab} index={1}>
          {/* Trading Tab */}
          <Grid container spacing={2}>
            <Grid item xs={12} lg={8}>
              <TradingChart />
            </Grid>
            <Grid item xs={12} lg={4}>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <OrderBook />
                </Grid>
                <Grid item xs={12}>
                  <MLInsights />
                </Grid>
              </Grid>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={activeTab} index={2}>
          {/* Portfolio Tab */}
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <PerformanceMetrics />
            </Grid>
            <Grid item xs={12} md={8}>
              <PositionsTable detailed={true} />
            </Grid>
            <Grid item xs={12} md={4}>
              <RecentTrades limit={20} />
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={activeTab} index={3}>
          {/* Risk Tab */}
          <RiskMonitor />
        </TabPanel>

        <TabPanel value={activeTab} index={4}>
          {/* Settings Tab */}
          <Alert severity="info">
            Settings configuration coming soon...
          </Alert>
        </TabPanel>
      </Box>

      {/* Status Bar */}
      <Paper
        elevation={0}
        sx={{
          p: 1,
          borderRadius: 0,
          borderTop: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <Grid container spacing={2} alignItems="center">
          <Grid item xs>
            <Typography variant="caption" color="text.secondary">
              Last Update: {new Date().toLocaleTimeString()}
            </Typography>
          </Grid>
          <Grid item>
            <Typography variant="caption" color="text.secondary">
              API: Polygon.io | Latency: 86ms | CPU: 15% | RAM: 2.1GB
            </Typography>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );
}