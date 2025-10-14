import React from 'react';
import GridLayout from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';
import MarketData from '../MarketData/MarketData';
import Portfolio from '../Portfolio/Portfolio';
import Charts from '../Charts/Charts';
import SystemMetrics from '../System/SystemMetrics';
import ActivityLog from '../System/ActivityLog';
import VoiceInput from '../Jarvis/VoiceInput';
import ChatInterface from '../Jarvis/ChatInterface';

const Terminal: React.FC = () => {
  const layout = [
    { i: 'market', x: 0, y: 0, w: 4, h: 3 },
    { i: 'portfolio', x: 4, y: 0, w: 3, h: 3 },
    { i: 'charts', x: 7, y: 0, w: 5, h: 6 },
    { i: 'jarvis-voice', x: 0, y: 3, w: 3, h: 4 },
    { i: 'jarvis-chat', x: 3, y: 3, w: 4, h: 4 },
    { i: 'system', x: 7, y: 6, w: 3, h: 1 },
    { i: 'activity', x: 0, y: 7, w: 10, h: 2 },
  ];

  return (
    <div className="h-full w-full bg-terminal-bg">
      {/* Header */}
      <div className="terminal-header">
        <div className="flex items-center space-x-4">
          <span className="text-terminal-accent font-bold">RRR TRADING TERMINAL</span>
          <span className="text-terminal-text">v1.0.0</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-terminal-text">
            {new Date().toLocaleTimeString('en-US', { 
              timeZone: 'UTC',
              hour12: false 
            })} UTC
          </span>
          <div className="flex space-x-1">
            <div className="w-3 h-3 bg-terminal-border rounded-full"></div>
            <div className="w-3 h-3 bg-terminal-border rounded-full"></div>
            <div className="w-3 h-3 bg-terminal-border rounded-full"></div>
          </div>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="h-[calc(100vh-40px)] p-2">
        <GridLayout
          className="layout"
          layout={layout}
          cols={12}
          rowHeight={60}
          width={window.innerWidth - 16}
          isDraggable={true}
          isResizable={true}
          margin={[4, 4]}
          containerPadding={[0, 0]}
        >
          <div key="market" className="terminal-panel">
            <MarketData />
          </div>
          <div key="portfolio" className="terminal-panel">
            <Portfolio />
          </div>
          <div key="charts" className="terminal-panel">
            <Charts />
          </div>
          <div key="jarvis-voice" className="terminal-panel">
            <VoiceInput />
          </div>
          <div key="jarvis-chat" className="terminal-panel">
            <ChatInterface />
          </div>
          <div key="system" className="terminal-panel">
            <SystemMetrics />
          </div>
          <div key="activity" className="terminal-panel">
            <ActivityLog />
          </div>
        </GridLayout>
      </div>
    </div>
  );
};

export default Terminal;
