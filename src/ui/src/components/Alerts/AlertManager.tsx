import React, { useState, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState, AppDispatch } from '../../store/store';
import {
  fetchAlerts,
  createAlert,
  updateAlert,
  deleteAlert,
  toggleSound,
} from '../../store/slices/alertSlice';
import { Alert } from '../../store/slices/alertSlice';
import AlertForm from './AlertForm';
import AlertNotifications from './AlertNotifications';

const AlertManager: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const { alerts, loading, soundEnabled } = useSelector((state: RootState) => state.alert);
  const [showForm, setShowForm] = useState(false);
  const [editingAlert, setEditingAlert] = useState<Alert | null>(null);
  const [filter, setFilter] = useState<'all' | 'active' | 'triggered' | 'expired'>('all');

  useEffect(() => {
    dispatch(fetchAlerts());
  }, [dispatch]);

  const handleCreateAlert = () => {
    setEditingAlert(null);
    setShowForm(true);
  };

  const handleEditAlert = (alert: Alert) => {
    setEditingAlert(alert);
    setShowForm(true);
  };

  const handleDeleteAlert = async (alertId: string) => {
    if (window.confirm('Are you sure you want to delete this alert?')) {
      await dispatch(deleteAlert(alertId));
    }
  };

  const handleToggleAlert = async (alert: Alert) => {
    const updatedAlert = {
      ...alert,
      status: alert.status === 'active' ? 'disabled' : 'active',
    } as Alert;
    await dispatch(updateAlert(updatedAlert));
  };

  const filteredAlerts = alerts.filter(alert => {
    if (filter === 'all') return true;
    return alert.status === filter;
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'text-bloomberg-green';
      case 'triggered':
        return 'text-bloomberg-amber';
      case 'expired':
        return 'text-terminal-accent';
      case 'disabled':
        return 'text-terminal-border';
      default:
        return 'text-terminal-text';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical':
        return 'bg-bloomberg-red';
      case 'high':
        return 'bg-bloomberg-amber';
      case 'medium':
        return 'bg-bloomberg-blue';
      case 'low':
        return 'bg-terminal-accent';
      default:
        return 'bg-terminal-border';
    }
  };

  if (showForm) {
    return (
      <AlertForm
        alert={editingAlert}
        onSave={() => {
          setShowForm(false);
          dispatch(fetchAlerts());
        }}
        onCancel={() => setShowForm(false)}
      />
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="terminal-panel-header flex justify-between items-center">
        <span className="text-terminal-green text-terminal-sm font-bold">ALERT MANAGER</span>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => dispatch(toggleSound())}
            className={`bloomberg-button text-terminal-xs px-2 py-1 ${
              soundEnabled ? 'bg-bloomberg-green' : 'bg-terminal-border'
            }`}
          >
            {soundEnabled ? '🔊 SOUND ON' : '🔇 SOUND OFF'}
          </button>
          <button
            onClick={handleCreateAlert}
            className="bloomberg-button text-terminal-xs px-3 py-1"
          >
            + CREATE ALERT
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center space-x-2 p-2 border-b border-terminal-border">
        <span className="text-terminal-accent text-terminal-xs">FILTER:</span>
        {(['all', 'active', 'triggered', 'expired'] as const).map(status => (
          <button
            key={status}
            onClick={() => setFilter(status)}
            className={`bloomberg-button text-terminal-xs px-3 py-1 ${
              filter === status ? 'bg-bloomberg-green' : ''
            }`}
          >
            {status.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Alert List */}
      <div className="terminal-content flex-1 overflow-y-auto terminal-scrollbar">
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-terminal-accent">Loading alerts...</div>
          </div>
        ) : filteredAlerts.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <div className="text-terminal-accent mb-2">No alerts found</div>
              <button
                onClick={handleCreateAlert}
                className="bloomberg-button text-terminal-xs px-4 py-2"
              >
                CREATE YOUR FIRST ALERT
              </button>
            </div>
          </div>
        ) : (
          <div className="p-2 space-y-2">
            {filteredAlerts.map((alert) => (
              <div
                key={alert.id}
                className="terminal-panel p-3 hover:bg-terminal-border hover:bg-opacity-30 transition-colors"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-start space-x-3 flex-1">
                    <input
                      type="checkbox"
                      checked={alert.status === 'active'}
                      onChange={() => handleToggleAlert(alert)}
                      className="mt-1 w-4 h-4"
                    />
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <span className="text-terminal-text font-bold text-terminal-sm">
                          {alert.name}
                        </span>
                        <span className={`px-2 py-0.5 rounded text-terminal-xs ${getPriorityColor(alert.priority)}`}>
                          {alert.priority.toUpperCase()}
                        </span>
                        <span className={`text-terminal-xs ${getStatusColor(alert.status)}`}>
                          {alert.status.toUpperCase()}
                        </span>
                      </div>
                      <div className="text-terminal-xs text-terminal-accent mb-1">
                        {alert.type.toUpperCase()}: {alert.symbol && `${alert.symbol} - `}
                        {alert.condition.indicator || 'Price'} {alert.condition.type} {alert.condition.value}
                      </div>
                      <div className="text-terminal-xs text-terminal-text">
                        {alert.message}
                      </div>
                      {alert.triggered_at && (
                        <div className="text-terminal-xs text-bloomberg-amber mt-1">
                          Triggered: {new Date(alert.triggered_at).toLocaleString()}
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => handleEditAlert(alert)}
                      className="bloomberg-button text-terminal-xs px-2 py-1"
                    >
                      EDIT
                    </button>
                    <button
                      onClick={() => handleDeleteAlert(alert.id)}
                      className="bloomberg-button text-terminal-xs px-2 py-1 bg-bloomberg-red hover:bg-red-600"
                    >
                      DELETE
                    </button>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex items-center space-x-2 text-terminal-xs text-terminal-accent">
                  {alert.actions.notification && <span>📱 Notification</span>}
                  {alert.actions.email && <span>📧 Email</span>}
                  {alert.actions.sound && <span>🔊 Sound</span>}
                  {alert.actions.execute_strategy && (
                    <span>⚡ Execute: {alert.actions.execute_strategy}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Stats Footer */}
      <div className="flex items-center justify-between p-2 border-t border-terminal-border text-terminal-xs">
        <div className="text-terminal-accent">
          Total: {alerts.length} | Active: {alerts.filter(a => a.status === 'active').length} |
          Triggered: {alerts.filter(a => a.status === 'triggered').length}
        </div>
        <div className="text-terminal-accent">
          Showing {filteredAlerts.length} alerts
        </div>
      </div>
    </div>
  );
};

export default AlertManager;
