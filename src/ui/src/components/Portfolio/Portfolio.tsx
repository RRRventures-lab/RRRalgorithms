import React from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../../store/store';
import Holdings from './Holdings';
import Performance from './Performance';

const Portfolio: React.FC = () => {
  const { totalValue, cash, totalPnl, totalPnlPercent, positions } = useSelector(
    (state: RootState) => state.portfolio
  );

  return (
    <div className="h-full flex flex-col">
      <div className="terminal-header">
        <span className="text-terminal-accent font-bold">PORTFOLIO</span>
      </div>
      <div className="terminal-content flex-1 flex flex-col space-y-2">
        <div className="flex-1">
          <Performance 
            totalValue={totalValue}
            cash={cash}
            totalPnl={totalPnl}
            totalPnlPercent={totalPnlPercent}
          />
        </div>
        <div className="flex-1">
          <Holdings positions={positions} />
        </div>
      </div>
    </div>
  );
};

export default Portfolio;
