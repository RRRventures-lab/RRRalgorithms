import React, { useEffect, useState } from 'react';
import {
  Paper,
  Box,
  Typography,
  Grid,
  LinearProgress,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import {
  CheckCircle,
  Error,
  Warning,
  WifiTethering,
  Storage,
  Speed,
  Memory,
  Timeline,
} from '@mui/icons-material';
import { motion } from 'framer-motion';

interface HealthStatus {
  service: string;
  status: 'healthy' | 'degraded' | 'error';
  latency?: number;
  message?: string;
}

interface SystemMetrics {
  cpu: number;
  memory: number;
  disk: number;
  network: number;
}

export default function SystemHealth() {
  const [services, setServices] = useState<HealthStatus[]>([
    { service: 'Polygon.io API', status: 'healthy', latency: 86 },
    { service: 'Database', status: 'healthy', latency: 5 },
    { service: 'WebSocket', status: 'healthy', latency: 12 },
    { service: 'ML Model', status: 'healthy', latency: 45 },
    { service: 'Risk Manager', status: 'healthy', latency: 3 },
  ]);

  const [metrics, setMetrics] = useState<SystemMetrics>({
    cpu: 15,
    memory: 42,
    disk: 65,
    network: 30,
  });

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics({
        cpu: Math.min(100, Math.max(5, metrics.cpu + (Math.random() - 0.5) * 10)),
        memory: Math.min(100, Math.max(20, metrics.memory + (Math.random() - 0.5) * 5)),
        disk: metrics.disk,
        network: Math.min(100, Math.max(10, metrics.network + (Math.random() - 0.5) * 15)),
      });
    }, 3000);

    return () => clearInterval(interval);
  }, [metrics]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle sx={{ color: '#00ff88' }} />;
      case 'degraded':
        return <Warning sx={{ color: '#ffaa00' }} />;
      case 'error':
        return <Error sx={{ color: '#ff3366' }} />;
      default:
        return <CheckCircle sx={{ color: '#666' }} />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return '#00ff88';
      case 'degraded':
        return '#ffaa00';
      case 'error':
        return '#ff3366';
      default:
        return '#666';
    }
  };

  const getMetricColor = (value: number) => {
    if (value < 60) return '#00ff88';
    if (value < 80) return '#ffaa00';
    return '#ff3366';
  };

  const MetricBar = ({ 
    label, 
    value, 
    icon 
  }: { 
    label: string; 
    value: number; 
    icon: React.ReactNode;
  }) => (
    <Box sx={{ mb: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {icon}
          <Typography variant="caption" color="text.secondary">
            {label}
          </Typography>
        </Box>
        <Typography 
          variant="caption" 
          sx={{ 
            color: getMetricColor(value),
            fontFamily: 'monospace',
            fontWeight: 600,
          }}
        >
          {value.toFixed(1)}%
        </Typography>
      </Box>
      <LinearProgress
        variant="determinate"
        value={value}
        sx={{
          height: 6,
          borderRadius: 3,
          bgcolor: 'rgba(255, 255, 255, 0.05)',
          '& .MuiLinearProgress-bar': {
            borderRadius: 3,
            bgcolor: getMetricColor(value),
          },
        }}
      />
    </Box>
  );

  const allHealthy = services.every(s => s.status === 'healthy');

  return (
    <Paper
      sx={{
        p: 2,
        height: '100%',
        background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>
          System Health
        </Typography>
        <Chip
          size="small"
          label={allHealthy ? 'All Systems Operational' : 'Issues Detected'}
          color={allHealthy ? 'success' : 'warning'}
          sx={{
            fontWeight: 600,
            animation: !allHealthy ? 'pulse 2s infinite' : 'none',
          }}
        />
      </Box>

      <Grid container spacing={2}>
        {/* Service Status */}
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Service Status
          </Typography>
          <List dense sx={{ p: 0 }}>
            {services.map((service, index) => (
              <motion.div
                key={service.service}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <ListItem
                  sx={{
                    px: 0,
                    py: 0.5,
                    '&:hover': {
                      bgcolor: 'rgba(255, 255, 255, 0.02)',
                    },
                  }}
                >
                  <ListItemIcon sx={{ minWidth: 32 }}>
                    {getStatusIcon(service.status)}
                  </ListItemIcon>
                  <ListItemText
                    primary={service.service}
                    secondary={
                      service.latency 
                        ? `${service.latency}ms`
                        : service.message
                    }
                    primaryTypographyProps={{
                      variant: 'body2',
                      sx: { fontWeight: 500 },
                    }}
                    secondaryTypographyProps={{
                      variant: 'caption',
                      sx: { 
                        color: getStatusColor(service.status),
                        fontFamily: 'monospace',
                      },
                    }}
                  />
                </ListItem>
              </motion.div>
            ))}
          </List>
        </Grid>

        {/* System Metrics */}
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Resource Usage
          </Typography>
          <Box sx={{ mt: 1 }}>
            <MetricBar 
              label="CPU" 
              value={metrics.cpu} 
              icon={<Speed sx={{ fontSize: 16, color: 'text.secondary' }} />}
            />
            <MetricBar 
              label="Memory" 
              value={metrics.memory} 
              icon={<Memory sx={{ fontSize: 16, color: 'text.secondary' }} />}
            />
            <MetricBar 
              label="Disk" 
              value={metrics.disk} 
              icon={<Storage sx={{ fontSize: 16, color: 'text.secondary' }} />}
            />
            <MetricBar 
              label="Network" 
              value={metrics.network} 
              icon={<WifiTethering sx={{ fontSize: 16, color: 'text.secondary' }} />}
            />
          </Box>
        </Grid>

        {/* Overall Health Score */}
        <Grid item xs={12}>
          <Box
            sx={{
              mt: 1,
              p: 1.5,
              borderRadius: 2,
              bgcolor: allHealthy 
                ? 'rgba(0, 255, 136, 0.05)' 
                : 'rgba(255, 170, 0, 0.05)',
              border: `1px solid ${allHealthy ? 'rgba(0, 255, 136, 0.2)' : 'rgba(255, 170, 0, 0.2)'}`,
            }}
          >
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Timeline sx={{ color: allHealthy ? '#00ff88' : '#ffaa00' }} />
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  System Uptime
                </Typography>
              </Box>
              <Typography 
                variant="body2" 
                sx={{ 
                  fontFamily: 'monospace',
                  fontWeight: 600,
                  color: allHealthy ? '#00ff88' : '#ffaa00',
                }}
              >
                99.95% (24h)
              </Typography>
            </Box>
          </Box>
        </Grid>
      </Grid>
    </Paper>
  );
}