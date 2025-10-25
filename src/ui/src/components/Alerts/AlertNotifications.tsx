import React, { useEffect, useRef } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState, AppDispatch } from '../../store/store';
import {
  markAsRead,
  markAllAsRead,
  dismissNotification,
  clearNotifications,
} from '../../store/slices/alertSlice';

const AlertNotifications: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const { notifications, unreadCount, soundEnabled } = useSelector(
    (state: RootState) => state.alert
  );
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    // Play notification sound for new alerts
    if (soundEnabled && notifications.length > 0 && !notifications[0].read) {
      playNotificationSound(notifications[0].priority);
    }
  }, [notifications, soundEnabled]);

  const playNotificationSound = (priority: string) => {
    // Create a simple beep sound based on priority
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    // Different frequencies for different priorities
    const frequencies: Record<string, number> = {
      low: 400,
      medium: 600,
      high: 800,
      critical: 1000,
    };

    oscillator.frequency.value = frequencies[priority] || 600;
    oscillator.type = 'sine';

    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);

    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.5);
  };

  const handleDismiss = (notificationId: string) => {
    dispatch(dismissNotification(notificationId));
    setTimeout(() => {
      dispatch(clearNotifications());
    }, 300);
  };

  const handleMarkAsRead = (notificationId: string) => {
    dispatch(markAsRead(notificationId));
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical':
        return 'border-l-4 border-bloomberg-red bg-bloomberg-red bg-opacity-10';
      case 'high':
        return 'border-l-4 border-bloomberg-amber bg-bloomberg-amber bg-opacity-10';
      case 'medium':
        return 'border-l-4 border-bloomberg-blue bg-bloomberg-blue bg-opacity-10';
      case 'low':
        return 'border-l-4 border-terminal-accent bg-terminal-accent bg-opacity-10';
      default:
        return 'border-l-4 border-terminal-border';
    }
  };

  const visibleNotifications = notifications.filter(n => !n.dismissed).slice(0, 5);

  if (visibleNotifications.length === 0) {
    return null;
  }

  return (
    <div className="fixed top-16 right-4 z-50 w-96 space-y-2">
      {/* Unread Counter */}
      {unreadCount > 0 && (
        <div className="flex justify-between items-center mb-2">
          <span className="text-terminal-accent text-terminal-xs">
            {unreadCount} unread notification{unreadCount !== 1 ? 's' : ''}
          </span>
          <button
            onClick={() => dispatch(markAllAsRead())}
            className="bloomberg-button text-terminal-xs px-2 py-1"
          >
            MARK ALL READ
          </button>
        </div>
      )}

      {/* Notifications */}
      {visibleNotifications.map((notification) => (
        <div
          key={notification.id}
          className={`terminal-panel p-3 ${getPriorityColor(notification.priority)} ${
            !notification.read ? 'shadow-lg' : 'opacity-75'
          } transition-all duration-300`}
        >
          <div className="flex items-start justify-between mb-2">
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${
                notification.priority === 'critical' ? 'bg-bloomberg-red' :
                notification.priority === 'high' ? 'bg-bloomberg-amber' :
                notification.priority === 'medium' ? 'bg-bloomberg-blue' :
                'bg-terminal-accent'
              } ${!notification.read ? 'animate-pulse' : ''}`} />
              <span className="text-terminal-text text-terminal-xs font-bold">
                {notification.priority.toUpperCase()} ALERT
              </span>
            </div>
            <button
              onClick={() => handleDismiss(notification.id)}
              className="text-terminal-accent hover:text-bloomberg-red text-terminal-xs"
            >
              ✕
            </button>
          </div>

          <div className="text-terminal-text text-terminal-sm mb-2">
            {notification.message}
          </div>

          <div className="flex items-center justify-between">
            <span className="text-terminal-accent text-terminal-xs">
              {new Date(notification.timestamp).toLocaleTimeString()}
            </span>
            {!notification.read && (
              <button
                onClick={() => handleMarkAsRead(notification.id)}
                className="text-bloomberg-green hover:text-green-400 text-terminal-xs"
              >
                Mark as read
              </button>
            )}
          </div>
        </div>
      ))}

      {notifications.filter(n => !n.dismissed).length > 5 && (
        <div className="text-center text-terminal-accent text-terminal-xs">
          +{notifications.filter(n => !n.dismissed).length - 5} more notifications
        </div>
      )}
    </div>
  );
};

export default AlertNotifications;
