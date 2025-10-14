import React, { useState, useEffect } from 'react';
import { FixedSizeList as List } from 'react-window';

interface ActivityEntry {
  id: string;
  timestamp: string;
  type: 'ORDER' | 'FILL' | 'SYSTEM' | 'ALERT';
  message: string;
  symbol?: string;
  price?: number;
  quantity?: number;
}

const ActivityLog: React.FC = () => {
  const [activities, setActivities] = useState<ActivityEntry[]>([]);

  // Generate sample activities
  useEffect(() => {
    const generateActivity = () => {
      const types: ActivityEntry['type'][] = ['ORDER', 'FILL', 'SYSTEM', 'ALERT'];
      const symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD'];
      const messages = [
        'ORDER PLACED: BUY 0.1 BTC @ 45,230',
        'FILL RECEIVED: 0.1 BTC @ 45,230',
        'SYSTEM: Database backup completed',
        'ALERT: High volatility detected',
        'ORDER PLACED: SELL 2.0 ETH @ 2,812',
        'FILL RECEIVED: 2.0 ETH @ 2,812',
        'SYSTEM: Risk check passed',
        'ALERT: Position limit reached',
      ];

      const newActivity: ActivityEntry = {
        id: Date.now().toString(),
        timestamp: new Date().toLocaleTimeString('en-US', { 
          hour12: false,
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit'
        }),
        type: types[Math.floor(Math.random() * types.length)],
        message: messages[Math.floor(Math.random() * messages.length)],
        symbol: Math.random() > 0.5 ? symbols[Math.floor(Math.random() * symbols.length)] : undefined,
        price: Math.random() > 0.5 ? Math.floor(Math.random() * 50000) : undefined,
        quantity: Math.random() > 0.5 ? Math.random() * 10 : undefined,
      };

      setActivities(prev => [newActivity, ...prev.slice(0, 99)]); // Keep last 100
    };

    // Generate initial activities
    for (let i = 0; i < 20; i++) {
      setTimeout(() => generateActivity(), i * 100);
    }

    // Generate new activity every 2-5 seconds
    const interval = setInterval(() => {
      generateActivity();
    }, Math.random() * 3000 + 2000);

    return () => clearInterval(interval);
  }, []);

  const getTypeColor = (type: ActivityEntry['type']) => {
    switch (type) {
      case 'ORDER':
        return 'text-bloomberg-amber';
      case 'FILL':
        return 'text-bloomberg-green';
      case 'SYSTEM':
        return 'text-bloomberg-blue';
      case 'ALERT':
        return 'text-bloomberg-red';
      default:
        return 'text-terminal-text';
    }
  };

  const ActivityRow: React.FC<{ index: number; style: React.CSSProperties }> = ({ index, style }) => {
    const activity = activities[index];
    if (!activity) return null;

    return (
      <div style={style} className="flex items-center space-x-2 text-terminal-xs border-b border-terminal-border px-2 py-1">
        <span className="text-terminal-accent font-mono w-16">{activity.timestamp}</span>
        <span className={`font-bold w-8 ${getTypeColor(activity.type)}`}>
          {activity.type}
        </span>
        <span className="flex-1">{activity.message}</span>
        {activity.symbol && (
          <span className="text-terminal-accent font-mono w-16">{activity.symbol}</span>
        )}
        {activity.price && (
          <span className="text-terminal-text font-mono w-16">
            ${activity.price.toLocaleString()}
          </span>
        )}
      </div>
    );
  };

  return (
    <div className="h-full flex flex-col">
      <div className="terminal-header">
        <span className="text-terminal-accent font-bold">ACTIVITY LOG</span>
        <div className="flex space-x-2">
          <button className="bloomberg-button text-terminal-xs">CLEAR</button>
          <button className="bloomberg-button text-terminal-xs">EXPORT</button>
        </div>
      </div>
      <div className="terminal-content flex-1">
        {activities.length > 0 ? (
          <List
            height={200}
            width={400}
            itemCount={activities.length}
            itemSize={20}
            className="terminal-scrollbar"
          >
            {ActivityRow}
          </List>
        ) : (
          <div className="text-terminal-text text-terminal-xs text-center py-4">
            No activity
          </div>
        )}
      </div>
    </div>
  );
};

export default ActivityLog;
