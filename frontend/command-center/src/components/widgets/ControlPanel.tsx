import React, { useState } from 'react';
import {
  Paper,
  Box,
  Typography,
  Button,
  ButtonGroup,
  Slider,
  Switch,
  FormControlLabel,
  TextField,
  Grid,
  Divider,
  Alert,
  Chip,
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  Stop,
  Security,
  Speed,
  AttachMoney,
  Warning,
  CheckCircle,
} from '@mui/icons-material';
import toast from 'react-hot-toast';

export default function ControlPanel() {
  const [tradingMode, setTradingMode] = useState<'paper' | 'live'>('paper');
  const [riskLevel, setRiskLevel] = useState(30);
  const [maxPositionSize, setMaxPositionSize] = useState(1000);
  const [autoTrading, setAutoTrading] = useState(true);
  const [emergencyStopEnabled, setEmergencyStopEnabled] = useState(true);
  const [strategies, setStrategies] = useState({
    momentum: true,
    meanReversion: false,
    arbitrage: false,
    marketMaking: false,
  });

  const handleStart = () => {
    toast.success('Trading system started');
  };

  const handlePause = () => {
    toast.warning('Trading system paused');
  };

  const handleStop = () => {
    toast.error('Trading system stopped');
  };

  const handleEmergencyStop = () => {
    if (window.confirm('Are you sure? This will close all positions immediately!')) {
      toast.error('EMERGENCY STOP ACTIVATED - All positions closed');
    }
  };

  const getRiskColor = (level: number) => {
    if (level <= 30) return '#00ff88';
    if (level <= 60) return '#ffaa00';
    return '#ff3366';
  };

  return (
    <Paper
      sx={{
        p: 2,
        height: '100%',
        background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        <Security color="primary" />
        <Typography variant="h6" sx={{ fontWeight: 600 }}>
          Control Panel
        </Typography>
      </Box>

      {/* Trading Mode */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
          Trading Mode
        </Typography>
        <ButtonGroup fullWidth variant="outlined" size="small">
          <Button
            onClick={() => setTradingMode('paper')}
            variant={tradingMode === 'paper' ? 'contained' : 'outlined'}
            color={tradingMode === 'paper' ? 'primary' : 'inherit'}
          >
            Paper Trading
          </Button>
          <Button
            onClick={() => setTradingMode('live')}
            variant={tradingMode === 'live' ? 'contained' : 'outlined'}
            color={tradingMode === 'live' ? 'warning' : 'inherit'}
          >
            Live Trading
          </Button>
        </ButtonGroup>
        {tradingMode === 'live' && (
          <Alert severity="warning" sx={{ mt: 1 }}>
            <Typography variant="caption">
              Live trading with real funds enabled
            </Typography>
          </Alert>
        )}
      </Box>

      {/* Main Controls */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
          System Control
        </Typography>
        <Grid container spacing={1}>
          <Grid item xs={4}>
            <Button
              fullWidth
              variant="contained"
              color="success"
              startIcon={<PlayArrow />}
              onClick={handleStart}
              size="small"
            >
              Start
            </Button>
          </Grid>
          <Grid item xs={4}>
            <Button
              fullWidth
              variant="contained"
              color="warning"
              startIcon={<Pause />}
              onClick={handlePause}
              size="small"
            >
              Pause
            </Button>
          </Grid>
          <Grid item xs={4}>
            <Button
              fullWidth
              variant="contained"
              color="error"
              startIcon={<Stop />}
              onClick={handleStop}
              size="small"
            >
              Stop
            </Button>
          </Grid>
        </Grid>
      </Box>

      <Divider sx={{ my: 2 }} />

      {/* Risk Management */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
          Risk Level: {riskLevel}%
        </Typography>
        <Slider
          value={riskLevel}
          onChange={(_, value) => setRiskLevel(value as number)}
          min={0}
          max={100}
          sx={{
            color: getRiskColor(riskLevel),
            '& .MuiSlider-thumb': {
              bgcolor: getRiskColor(riskLevel),
            },
          }}
        />
        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
          <Typography variant="caption" color="text.secondary">
            Conservative
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Moderate
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Aggressive
          </Typography>
        </Box>
      </Box>

      {/* Position Size */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
          Max Position Size
        </Typography>
        <TextField
          fullWidth
          size="small"
          value={maxPositionSize}
          onChange={(e) => setMaxPositionSize(Number(e.target.value))}
          InputProps={{
            startAdornment: <AttachMoney sx={{ fontSize: 18, mr: 0.5 }} />,
          }}
          type="number"
        />
      </Box>

      <Divider sx={{ my: 2 }} />

      {/* Trading Strategies */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
          Active Strategies
        </Typography>
        <Grid container spacing={1}>
          {Object.entries(strategies).map(([key, enabled]) => (
            <Grid item xs={6} key={key}>
              <Chip
                label={key.charAt(0).toUpperCase() + key.slice(1)}
                color={enabled ? 'success' : 'default'}
                variant={enabled ? 'filled' : 'outlined'}
                size="small"
                onClick={() => setStrategies({ ...strategies, [key]: !enabled })}
                icon={enabled ? <CheckCircle /> : undefined}
                sx={{ width: '100%' }}
              />
            </Grid>
          ))}
        </Grid>
      </Box>

      {/* Switches */}
      <Box sx={{ mb: 2 }}>
        <FormControlLabel
          control={
            <Switch
              checked={autoTrading}
              onChange={(e) => setAutoTrading(e.target.checked)}
              color="primary"
            />
          }
          label="Auto Trading"
        />
        <FormControlLabel
          control={
            <Switch
              checked={emergencyStopEnabled}
              onChange={(e) => setEmergencyStopEnabled(e.target.checked)}
              color="error"
            />
          }
          label="Emergency Stop Ready"
        />
      </Box>

      <Divider sx={{ my: 2 }} />

      {/* Emergency Stop Button */}
      <Button
        fullWidth
        variant="contained"
        color="error"
        startIcon={<Warning />}
        onClick={handleEmergencyStop}
        disabled={!emergencyStopEnabled}
        sx={{
          bgcolor: '#ff3366',
          '&:hover': {
            bgcolor: '#ff0033',
          },
        }}
      >
        EMERGENCY STOP
      </Button>
      <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block', textAlign: 'center' }}>
        Closes all positions immediately
      </Typography>
    </Paper>
  );
}