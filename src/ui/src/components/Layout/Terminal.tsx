import React, { Suspense, useState, useEffect } from 'react';
import GridLayout from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';
import { lazyWithRetry } from '../../utils/lazyLoad';

// Eager-loaded core components
import MarketData from '../MarketData/MarketData';
import Portfolio from '../Portfolio/Portfolio';
import Charts from '../Charts/Charts';
import SystemMetrics from '../System/SystemMetrics';

// Lazy-loaded components with code splitting
const ActivityLog = lazyWithRetry(() => import('../System/ActivityLog'));
const VoiceInput = lazyWithRetry(() => import('../Jarvis/VoiceInput'));
const ChatInterface = lazyWithRetry(() => import('../Jarvis/ChatInterface'));
const BacktestResultsViewer = lazyWithRetry(() => import('../Backtest/BacktestResultsViewer'));
const StrategyBuilder = lazyWithRetry(() => import('../Strategy/StrategyBuilder'));
const AlertManager = lazyWithRetry(() => import('../Alerts/AlertManager'));
const AlertNotifications = lazyWithRetry(() => import('../Alerts/AlertNotifications'));
const NeuralNetworkPredictions = lazyWithRetry(() => import('../Visualizations/NeuralNetworkPredictions'));
const MarketInefficiencyHeatmap = lazyWithRetry(() => import('../Visualizations/MarketInefficiencyHeatmap'));

// Loading component
const LoadingPanel: React.FC = () => (
  <div className="h-full w-full flex items-center justify-center">
    <div className="loading-skeleton w-full h-full rounded" />
  </div>
);

type ViewMode = 'trading' | 'backtest' | 'strategy' | 'alerts' | 'ai';

const Terminal: React.FC = () => {
  const [viewMode, setViewMode] = useState<ViewMode>('trading');
  const [isMobile, setIsMobile] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    // Detect mobile
    const checkMobile = () => {
      setIsMobile(window.innerWidth <= 768);
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);

    // Update time every second
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => {
      window.removeEventListener('resize', checkMobile);
      clearInterval(timer);
    };
  }, []);

  const tradingLayout = [
    { i: 'market', x: 0, y: 0, w: 4, h: 3 },
    { i: 'portfolio', x: 4, y: 0, w: 3, h: 3 },
    { i: 'charts', x: 7, y: 0, w: 5, h: 6 },
    { i: 'nn-predictions', x: 0, y: 3, w: 4, h: 4 },
    { i: 'jarvis-chat', x: 4, y: 3, w: 3, h: 4 },
    { i: 'system', x: 7, y: 6, w: 3, h: 2 },
    { i: 'activity', x: 10, y: 6, w: 2, h: 2 },
    { i: 'inefficiencies', x: 0, y: 7, w: 7, h: 3 },
  ];

  const mobileLayout = [
    { i: 'market', x: 0, y: 0, w: 12, h: 4 },
    { i: 'charts', x: 0, y: 4, w: 12, h: 6 },
    { i: 'portfolio', x: 0, y: 10, w: 12, h: 4 },
    { i: 'jarvis-chat', x: 0, y: 14, w: 12, h: 5 },
    { i: 'system', x: 0, y: 19, w: 12, h: 2 },
    { i: 'activity', x: 0, y: 21, w: 12, h: 3 },
  ];

  const layout = isMobile ? mobileLayout : tradingLayout;

  return (
    <div className="h-full w-full bg-terminal-bg">
      {/* Header */}
      <div className="terminal-header">
        <div className="flex items-center space-x-2 md:space-x-4">
          <span className="text-terminal-accent font-bold text-xs md:text-sm">RRR TRADING TERMINAL</span>
          <span className="text-terminal-text text-xs hide-on-mobile">v2.0.0</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-terminal-text text-xs">
            {currentTime.toLocaleTimeString('en-US', {
              timeZone: 'UTC',
              hour12: false
            })} UTC
          </span>
          <div className="flex space-x-1 hide-on-mobile">
            <div className="w-2 h-2 bg-bloomberg-green rounded-full animate-pulse"></div>
            <div className="w-2 h-2 bg-bloomberg-green rounded-full"></div>
            <div className="w-2 h-2 bg-bloomberg-green rounded-full"></div>
          </div>
        </div>
      </div>

      {/* View Mode Selector */}
      <div className="flex items-center space-x-1 p-2 border-b border-terminal-border overflow-x-auto">
        {(['trading', 'backtest', 'strategy', 'alerts', 'ai'] as ViewMode[]).map(mode => (
          <button
            key={mode}
            onClick={() => setViewMode(mode)}
            className={`bloomberg-button text-terminal-xs px-3 py-1 whitespace-nowrap touch-optimized ${
              viewMode === mode ? 'bg-bloomberg-green text-terminal-bg' : ''
            }`}
          >
            {mode.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Alert Notifications */}
      <Suspense fallback={null}>
        <AlertNotifications />
      </Suspense>

      {/* Main Content */}
      <div className="h-[calc(100vh-80px)] md:h-[calc(100vh-88px)] p-2">
        {viewMode === 'trading' && (
          <GridLayout
            className="layout"
            layout={layout}
            cols={12}
            rowHeight={isMobile ? 50 : 60}
            width={isMobile ? window.innerWidth - 16 : window.innerWidth - 16}
            isDraggable={!isMobile}
            isResizable={!isMobile}
            margin={[4, 4]}
            containerPadding={[0, 0]}
          >
            <div key="market" className="terminal-panel gpu-accelerated">
              <MarketData />
            </div>
            <div key="portfolio" className="terminal-panel gpu-accelerated">
              <Portfolio />
            </div>
            <div key="charts" className="terminal-panel gpu-accelerated">
              <Charts />
            </div>
            <div key="nn-predictions" className="terminal-panel gpu-accelerated">
              <Suspense fallback={<LoadingPanel />}>
                <NeuralNetworkPredictions />
              </Suspense>
            </div>
            <div key="jarvis-chat" className="terminal-panel gpu-accelerated">
              <Suspense fallback={<LoadingPanel />}>
                <ChatInterface />
              </Suspense>
            </div>
            <div key="system" className="terminal-panel gpu-accelerated">
              <SystemMetrics />
            </div>
            <div key="activity" className="terminal-panel gpu-accelerated">
              <Suspense fallback={<LoadingPanel />}>
                <ActivityLog />
              </Suspense>
            </div>
            <div key="inefficiencies" className="terminal-panel gpu-accelerated">
              <Suspense fallback={<LoadingPanel />}>
                <MarketInefficiencyHeatmap />
              </Suspense>
            </div>
          </GridLayout>
        )}

        {viewMode === 'backtest' && (
          <div className="h-full terminal-panel">
            <Suspense fallback={<LoadingPanel />}>
              <BacktestResultsViewer />
            </Suspense>
          </div>
        )}

        {viewMode === 'strategy' && (
          <div className="h-full terminal-panel">
            <Suspense fallback={<LoadingPanel />}>
              <StrategyBuilder />
            </Suspense>
          </div>
        )}

        {viewMode === 'alerts' && (
          <div className="h-full terminal-panel">
            <Suspense fallback={<LoadingPanel />}>
              <AlertManager />
            </Suspense>
          </div>
        )}

        {viewMode === 'ai' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full">
            <div className="terminal-panel">
              <Suspense fallback={<LoadingPanel />}>
                <NeuralNetworkPredictions />
              </Suspense>
            </div>
            <div className="terminal-panel">
              <Suspense fallback={<LoadingPanel />}>
                <MarketInefficiencyHeatmap />
              </Suspense>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Terminal;
