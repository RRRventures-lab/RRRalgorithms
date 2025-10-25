import React, { useState } from 'react';
import { useDispatch } from 'react-redux';
import { AppDispatch } from '../../store/store';
import { createAlert, updateAlert } from '../../store/slices/alertSlice';
import { Alert } from '../../store/slices/alertSlice';

interface AlertFormProps {
  alert: Alert | null;
  onSave: () => void;
  onCancel: () => void;
}

const AlertForm: React.FC<AlertFormProps> = ({ alert, onSave, onCancel }) => {
  const dispatch = useDispatch<AppDispatch>();
  const [formData, setFormData] = useState({
    name: alert?.name || '',
    type: alert?.type || 'price' as Alert['type'],
    symbol: alert?.symbol || 'BTC-USD',
    condition: {
      type: alert?.condition.type || 'above' as Alert['condition']['type'],
      value: alert?.condition.value || 0,
      indicator: alert?.condition.indicator || '',
      timeframe: alert?.condition.timeframe || '1h',
    },
    message: alert?.message || '',
    priority: alert?.priority || 'medium' as Alert['priority'],
    actions: {
      notification: alert?.actions.notification ?? true,
      email: alert?.actions.email ?? false,
      sound: alert?.actions.sound ?? true,
      execute_strategy: alert?.actions.execute_strategy || '',
    },
    repeat: alert?.repeat ?? false,
    expires_at: alert?.expires_at || '',
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (alert) {
      await dispatch(updateAlert({
        ...alert,
        ...formData,
      }));
    } else {
      await dispatch(createAlert(formData as any));
    }

    onSave();
  };

  const handleChange = (field: string, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleConditionChange = (field: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      condition: { ...prev.condition, [field]: value },
    }));
  };

  const handleActionsChange = (field: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      actions: { ...prev.actions, [field]: value },
    }));
  };

  return (
    <div className="h-full flex flex-col">
      <div className="terminal-panel-header flex justify-between items-center">
        <span className="text-terminal-green text-terminal-sm font-bold">
          {alert ? 'EDIT ALERT' : 'CREATE ALERT'}
        </span>
        <button
          onClick={onCancel}
          className="text-terminal-accent hover:text-terminal-text"
        >
          ✕
        </button>
      </div>

      <form onSubmit={handleSubmit} className="terminal-content flex-1 overflow-y-auto terminal-scrollbar p-4">
        <div className="space-y-4 max-w-2xl">
          {/* Basic Info */}
          <div>
            <label className="text-terminal-accent text-terminal-xs font-bold block mb-1">
              ALERT NAME
            </label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => handleChange('name', e.target.value)}
              required
              className="w-full bg-terminal-border border border-terminal-border text-terminal-text px-3 py-2 rounded focus:outline-none focus:border-bloomberg-green"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-terminal-accent text-terminal-xs font-bold block mb-1">
                ALERT TYPE
              </label>
              <select
                value={formData.type}
                onChange={(e) => handleChange('type', e.target.value)}
                className="w-full bg-terminal-border border border-terminal-border text-terminal-text px-3 py-2 rounded focus:outline-none focus:border-bloomberg-green"
              >
                <option value="price">Price Alert</option>
                <option value="indicator">Indicator Alert</option>
                <option value="system">System Alert</option>
                <option value="trade">Trade Alert</option>
                <option value="portfolio">Portfolio Alert</option>
              </select>
            </div>

            <div>
              <label className="text-terminal-accent text-terminal-xs font-bold block mb-1">
                PRIORITY
              </label>
              <select
                value={formData.priority}
                onChange={(e) => handleChange('priority', e.target.value)}
                className="w-full bg-terminal-border border border-terminal-border text-terminal-text px-3 py-2 rounded focus:outline-none focus:border-bloomberg-green"
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
                <option value="critical">Critical</option>
              </select>
            </div>
          </div>

          {(formData.type === 'price' || formData.type === 'indicator') && (
            <div>
              <label className="text-terminal-accent text-terminal-xs font-bold block mb-1">
                SYMBOL
              </label>
              <input
                type="text"
                value={formData.symbol}
                onChange={(e) => handleChange('symbol', e.target.value)}
                placeholder="BTC-USD"
                className="w-full bg-terminal-border border border-terminal-border text-terminal-text px-3 py-2 rounded focus:outline-none focus:border-bloomberg-green"
              />
            </div>
          )}

          {/* Condition */}
          <div className="space-y-3">
            <div className="text-terminal-accent text-terminal-xs font-bold">CONDITION</div>

            {formData.type === 'indicator' && (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-terminal-accent text-terminal-xs block mb-1">
                    INDICATOR
                  </label>
                  <select
                    value={formData.condition.indicator}
                    onChange={(e) => handleConditionChange('indicator', e.target.value)}
                    className="w-full bg-terminal-border border border-terminal-border text-terminal-text px-3 py-2 rounded focus:outline-none focus:border-bloomberg-green"
                  >
                    <option value="">Select...</option>
                    <option value="RSI">RSI</option>
                    <option value="MACD">MACD</option>
                    <option value="SMA">SMA</option>
                    <option value="EMA">EMA</option>
                    <option value="Bollinger Bands">Bollinger Bands</option>
                    <option value="ATR">ATR</option>
                  </select>
                </div>

                <div>
                  <label className="text-terminal-accent text-terminal-xs block mb-1">
                    TIMEFRAME
                  </label>
                  <select
                    value={formData.condition.timeframe}
                    onChange={(e) => handleConditionChange('timeframe', e.target.value)}
                    className="w-full bg-terminal-border border border-terminal-border text-terminal-text px-3 py-2 rounded focus:outline-none focus:border-bloomberg-green"
                  >
                    <option value="1m">1 Minute</option>
                    <option value="5m">5 Minutes</option>
                    <option value="15m">15 Minutes</option>
                    <option value="1h">1 Hour</option>
                    <option value="4h">4 Hours</option>
                    <option value="1d">1 Day</option>
                  </select>
                </div>
              </div>
            )}

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-terminal-accent text-terminal-xs block mb-1">
                  CONDITION TYPE
                </label>
                <select
                  value={formData.condition.type}
                  onChange={(e) => handleConditionChange('type', e.target.value)}
                  className="w-full bg-terminal-border border border-terminal-border text-terminal-text px-3 py-2 rounded focus:outline-none focus:border-bloomberg-green"
                >
                  <option value="above">Above</option>
                  <option value="below">Below</option>
                  <option value="crosses_above">Crosses Above</option>
                  <option value="crosses_below">Crosses Below</option>
                  <option value="equals">Equals</option>
                  <option value="change_percent">% Change</option>
                </select>
              </div>

              <div>
                <label className="text-terminal-accent text-terminal-xs block mb-1">
                  VALUE
                </label>
                <input
                  type="number"
                  value={formData.condition.value}
                  onChange={(e) => handleConditionChange('value', parseFloat(e.target.value))}
                  step="0.01"
                  required
                  className="w-full bg-terminal-border border border-terminal-border text-terminal-text px-3 py-2 rounded focus:outline-none focus:border-bloomberg-green"
                />
              </div>
            </div>
          </div>

          {/* Message */}
          <div>
            <label className="text-terminal-accent text-terminal-xs font-bold block mb-1">
              ALERT MESSAGE
            </label>
            <textarea
              value={formData.message}
              onChange={(e) => handleChange('message', e.target.value)}
              rows={3}
              required
              className="w-full bg-terminal-border border border-terminal-border text-terminal-text px-3 py-2 rounded focus:outline-none focus:border-bloomberg-green resize-none"
            />
          </div>

          {/* Actions */}
          <div className="space-y-2">
            <div className="text-terminal-accent text-terminal-xs font-bold">ACTIONS</div>

            <div className="space-y-2">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={formData.actions.notification}
                  onChange={(e) => handleActionsChange('notification', e.target.checked)}
                  className="w-4 h-4"
                />
                <span className="text-terminal-text text-terminal-sm">Show Notification</span>
              </label>

              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={formData.actions.sound}
                  onChange={(e) => handleActionsChange('sound', e.target.checked)}
                  className="w-4 h-4"
                />
                <span className="text-terminal-text text-terminal-sm">Play Sound</span>
              </label>

              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={formData.actions.email}
                  onChange={(e) => handleActionsChange('email', e.target.checked)}
                  className="w-4 h-4"
                />
                <span className="text-terminal-text text-terminal-sm">Send Email</span>
              </label>
            </div>

            <div>
              <label className="text-terminal-accent text-terminal-xs block mb-1">
                EXECUTE STRATEGY (Optional)
              </label>
              <input
                type="text"
                value={formData.actions.execute_strategy}
                onChange={(e) => handleActionsChange('execute_strategy', e.target.value)}
                placeholder="Strategy ID"
                className="w-full bg-terminal-border border border-terminal-border text-terminal-text px-3 py-2 rounded focus:outline-none focus:border-bloomberg-green"
              />
            </div>
          </div>

          {/* Options */}
          <div className="space-y-2">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={formData.repeat}
                onChange={(e) => handleChange('repeat', e.target.checked)}
                className="w-4 h-4"
              />
              <span className="text-terminal-text text-terminal-sm">Repeat Alert (trigger multiple times)</span>
            </label>

            <div>
              <label className="text-terminal-accent text-terminal-xs block mb-1">
                EXPIRES AT (Optional)
              </label>
              <input
                type="datetime-local"
                value={formData.expires_at}
                onChange={(e) => handleChange('expires_at', e.target.value)}
                className="w-full bg-terminal-border border border-terminal-border text-terminal-text px-3 py-2 rounded focus:outline-none focus:border-bloomberg-green"
              />
            </div>
          </div>

          {/* Buttons */}
          <div className="flex justify-end space-x-2 pt-4">
            <button
              type="button"
              onClick={onCancel}
              className="bloomberg-button px-4 py-2 bg-terminal-border hover:bg-opacity-80"
            >
              CANCEL
            </button>
            <button
              type="submit"
              className="bloomberg-button px-4 py-2"
            >
              {alert ? 'UPDATE ALERT' : 'CREATE ALERT'}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
};

export default AlertForm;
