import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface SystemMetrics {
  cpu: number;
  memory: number;
  latency: number;
  throughput: number;
  uptime: number;
}

export interface SystemState {
  metrics: SystemMetrics;
  status: 'healthy' | 'degraded' | 'critical';
  alerts: string[];
  lastUpdate: number;
}

const initialState: SystemState = {
  metrics: {
    cpu: 12,
    memory: 2.1,
    latency: 0.8,
    throughput: 145,
    uptime: 3600,
  },
  status: 'healthy',
  alerts: [],
  lastUpdate: Date.now(),
};

const systemSlice = createSlice({
  name: 'system',
  initialState,
  reducers: {
    updateMetrics: (state, action: PayloadAction<SystemMetrics>) => {
      state.metrics = action.payload;
      state.lastUpdate = Date.now();
      
      // Update status based on metrics
      if (state.metrics.cpu > 80 || state.metrics.memory > 8 || state.metrics.latency > 100) {
        state.status = 'critical';
      } else if (state.metrics.cpu > 60 || state.metrics.memory > 6 || state.metrics.latency > 50) {
        state.status = 'degraded';
      } else {
        state.status = 'healthy';
      }
    },
    addAlert: (state, action: PayloadAction<string>) => {
      state.alerts.push(action.payload);
      if (state.alerts.length > 10) {
        state.alerts.shift();
      }
    },
    clearAlerts: (state) => {
      state.alerts = [];
    },
  },
});

export const { updateMetrics, addAlert, clearAlerts } = systemSlice.actions;
export default systemSlice.reducer;
