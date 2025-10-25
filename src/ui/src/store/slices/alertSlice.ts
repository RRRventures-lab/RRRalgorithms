import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';

// Types
export interface Alert {
  id: string;
  name: string;
  type: 'price' | 'indicator' | 'system' | 'trade' | 'portfolio';
  status: 'active' | 'triggered' | 'expired' | 'disabled';
  created_at: string;
  triggered_at?: string;
  symbol?: string;
  condition: {
    type: 'above' | 'below' | 'crosses_above' | 'crosses_below' | 'equals' | 'change_percent';
    value: number;
    indicator?: string;
    timeframe?: string;
  };
  actions: {
    notification: boolean;
    email?: boolean;
    sound?: boolean;
    execute_strategy?: string;
  };
  message: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  repeat: boolean;
  expires_at?: string;
}

export interface AlertNotification {
  id: string;
  alert_id: string;
  timestamp: string;
  message: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  read: boolean;
  dismissed: boolean;
}

export interface AlertState {
  alerts: Alert[];
  notifications: AlertNotification[];
  unreadCount: number;
  loading: boolean;
  error: string | null;
  soundEnabled: boolean;
}

const initialState: AlertState = {
  alerts: [],
  notifications: [],
  unreadCount: 0,
  loading: false,
  error: null,
  soundEnabled: true,
};

// Async thunks
export const fetchAlerts = createAsyncThunk(
  'alert/fetchAlerts',
  async () => {
    const response = await fetch('http://localhost:8000/api/alerts');
    if (!response.ok) throw new Error('Failed to fetch alerts');
    return await response.json();
  }
);

export const createAlert = createAsyncThunk(
  'alert/createAlert',
  async (alert: Omit<Alert, 'id' | 'created_at' | 'status'>) => {
    const response = await fetch('http://localhost:8000/api/alerts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(alert),
    });
    if (!response.ok) throw new Error('Failed to create alert');
    return await response.json();
  }
);

export const updateAlert = createAsyncThunk(
  'alert/updateAlert',
  async (alert: Alert) => {
    const response = await fetch(`http://localhost:8000/api/alerts/${alert.id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(alert),
    });
    if (!response.ok) throw new Error('Failed to update alert');
    return await response.json();
  }
);

export const deleteAlert = createAsyncThunk(
  'alert/deleteAlert',
  async (alertId: string) => {
    const response = await fetch(`http://localhost:8000/api/alerts/${alertId}`, {
      method: 'DELETE',
    });
    if (!response.ok) throw new Error('Failed to delete alert');
    return alertId;
  }
);

export const fetchNotifications = createAsyncThunk(
  'alert/fetchNotifications',
  async (limit: number = 50) => {
    const response = await fetch(`http://localhost:8000/api/alerts/notifications?limit=${limit}`);
    if (!response.ok) throw new Error('Failed to fetch notifications');
    return await response.json();
  }
);

const alertSlice = createSlice({
  name: 'alert',
  initialState,
  reducers: {
    addNotification: (state, action: PayloadAction<AlertNotification>) => {
      state.notifications.unshift(action.payload);
      if (!action.payload.read) {
        state.unreadCount += 1;
      }
      // Play sound if enabled
      if (state.soundEnabled && action.payload.priority !== 'low') {
        // Sound will be played by component
      }
    },
    markAsRead: (state, action: PayloadAction<string>) => {
      const notification = state.notifications.find(n => n.id === action.payload);
      if (notification && !notification.read) {
        notification.read = true;
        state.unreadCount = Math.max(0, state.unreadCount - 1);
      }
    },
    markAllAsRead: (state) => {
      state.notifications.forEach(n => {
        n.read = true;
      });
      state.unreadCount = 0;
    },
    dismissNotification: (state, action: PayloadAction<string>) => {
      const notification = state.notifications.find(n => n.id === action.payload);
      if (notification) {
        notification.dismissed = true;
      }
    },
    clearNotifications: (state) => {
      state.notifications = state.notifications.filter(n => !n.dismissed);
    },
    toggleSound: (state) => {
      state.soundEnabled = !state.soundEnabled;
    },
    updateAlertStatus: (state, action: PayloadAction<{ id: string; status: Alert['status'] }>) => {
      const alert = state.alerts.find(a => a.id === action.payload.id);
      if (alert) {
        alert.status = action.payload.status;
        if (action.payload.status === 'triggered') {
          alert.triggered_at = new Date().toISOString();
        }
      }
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    // Fetch alerts
    builder.addCase(fetchAlerts.pending, (state) => {
      state.loading = true;
      state.error = null;
    });
    builder.addCase(fetchAlerts.fulfilled, (state, action) => {
      state.loading = false;
      state.alerts = action.payload;
    });
    builder.addCase(fetchAlerts.rejected, (state, action) => {
      state.loading = false;
      state.error = action.error.message || 'Failed to fetch alerts';
    });

    // Create alert
    builder.addCase(createAlert.fulfilled, (state, action) => {
      state.alerts.push(action.payload);
    });
    builder.addCase(createAlert.rejected, (state, action) => {
      state.error = action.error.message || 'Failed to create alert';
    });

    // Update alert
    builder.addCase(updateAlert.fulfilled, (state, action) => {
      const index = state.alerts.findIndex(a => a.id === action.payload.id);
      if (index !== -1) {
        state.alerts[index] = action.payload;
      }
    });

    // Delete alert
    builder.addCase(deleteAlert.fulfilled, (state, action) => {
      state.alerts = state.alerts.filter(a => a.id !== action.payload);
    });

    // Fetch notifications
    builder.addCase(fetchNotifications.fulfilled, (state, action) => {
      state.notifications = action.payload;
      state.unreadCount = action.payload.filter((n: AlertNotification) => !n.read).length;
    });
  },
});

export const {
  addNotification,
  markAsRead,
  markAllAsRead,
  dismissNotification,
  clearNotifications,
  toggleSound,
  updateAlertStatus,
  clearError,
} = alertSlice.actions;

export default alertSlice.reducer;
