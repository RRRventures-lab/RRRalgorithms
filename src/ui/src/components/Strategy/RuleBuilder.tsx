import React from 'react';
import { StrategyRule, StrategyCondition, StrategyAction } from '../../store/slices/strategySlice';

interface RuleBuilderProps {
  rule: StrategyRule;
  ruleType: 'entry' | 'exit';
  availableIndicators: string[];
  onUpdate: (rule: StrategyRule) => void;
  onRemove: () => void;
}

const RuleBuilder: React.FC<RuleBuilderProps> = ({
  rule,
  ruleType,
  availableIndicators,
  onUpdate,
  onRemove,
}) => {
  const handleAddCondition = () => {
    const newCondition: StrategyCondition = {
      id: `condition-${Date.now()}`,
      type: 'indicator',
      indicator: 'RSI',
      operator: '>',
      value: 70,
      timeframe: '1h',
    };
    onUpdate({
      ...rule,
      conditions: [...rule.conditions, newCondition],
    });
  };

  const handleUpdateCondition = (index: number, updates: Partial<StrategyCondition>) => {
    const updatedConditions = [...rule.conditions];
    updatedConditions[index] = { ...updatedConditions[index], ...updates };
    onUpdate({ ...rule, conditions: updatedConditions });
  };

  const handleRemoveCondition = (index: number) => {
    onUpdate({
      ...rule,
      conditions: rule.conditions.filter((_, i) => i !== index),
    });
  };

  return (
    <div className="terminal-panel p-4">
      {/* Rule Header */}
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center space-x-3">
          <input
            type="checkbox"
            checked={rule.enabled}
            onChange={(e) => onUpdate({ ...rule, enabled: e.target.checked })}
            className="w-4 h-4"
          />
          <input
            type="text"
            value={rule.name}
            onChange={(e) => onUpdate({ ...rule, name: e.target.value })}
            className="bg-terminal-bg border border-terminal-border text-terminal-text px-2 py-1 text-terminal-sm rounded focus:outline-none focus:border-bloomberg-green"
          />
          <select
            value={rule.logic}
            onChange={(e) => onUpdate({ ...rule, logic: e.target.value as 'and' | 'or' })}
            className="bg-terminal-bg border border-terminal-border text-terminal-text px-2 py-1 text-terminal-xs rounded focus:outline-none focus:border-bloomberg-green"
          >
            <option value="and">ALL conditions (AND)</option>
            <option value="or">ANY condition (OR)</option>
          </select>
        </div>
        <button
          onClick={onRemove}
          className="bloomberg-button text-terminal-xs px-2 py-1 bg-bloomberg-red hover:bg-red-600"
        >
          DELETE RULE
        </button>
      </div>

      {/* Conditions */}
      <div className="space-y-2 mb-4">
        <div className="flex justify-between items-center">
          <div className="text-terminal-accent text-terminal-xs font-bold">
            CONDITIONS ({rule.conditions.length})
          </div>
          <button
            onClick={handleAddCondition}
            className="bloomberg-button text-terminal-xs px-2 py-1"
          >
            + ADD CONDITION
          </button>
        </div>

        {rule.conditions.map((condition, index) => (
          <div key={condition.id} className="flex items-center space-x-2 bg-terminal-bg p-2 rounded">
            <select
              value={condition.type}
              onChange={(e) => handleUpdateCondition(index, { type: e.target.value as any })}
              className="bg-terminal-border border-terminal-border text-terminal-text px-2 py-1 text-terminal-xs rounded focus:outline-none"
            >
              <option value="indicator">Indicator</option>
              <option value="price">Price</option>
              <option value="volume">Volume</option>
              <option value="time">Time</option>
            </select>

            {condition.type === 'indicator' && (
              <>
                <select
                  value={condition.indicator}
                  onChange={(e) => handleUpdateCondition(index, { indicator: e.target.value })}
                  className="bg-terminal-border border-terminal-border text-terminal-text px-2 py-1 text-terminal-xs rounded focus:outline-none"
                >
                  {availableIndicators.map(ind => (
                    <option key={ind} value={ind}>{ind}</option>
                  ))}
                </select>

                <select
                  value={condition.timeframe}
                  onChange={(e) => handleUpdateCondition(index, { timeframe: e.target.value })}
                  className="bg-terminal-border border-terminal-border text-terminal-text px-2 py-1 text-terminal-xs rounded focus:outline-none"
                >
                  <option value="1m">1m</option>
                  <option value="5m">5m</option>
                  <option value="15m">15m</option>
                  <option value="1h">1h</option>
                  <option value="4h">4h</option>
                  <option value="1d">1d</option>
                </select>
              </>
            )}

            <select
              value={condition.operator}
              onChange={(e) => handleUpdateCondition(index, { operator: e.target.value as any })}
              className="bg-terminal-border border-terminal-border text-terminal-text px-2 py-1 text-terminal-xs rounded focus:outline-none"
            >
              <option value=">">Greater than</option>
              <option value="<">Less than</option>
              <option value=">=">Greater or equal</option>
              <option value="<=">Less or equal</option>
              <option value="=">Equals</option>
              <option value="crosses_above">Crosses above</option>
              <option value="crosses_below">Crosses below</option>
            </select>

            <input
              type="number"
              value={condition.value as number}
              onChange={(e) => handleUpdateCondition(index, { value: parseFloat(e.target.value) })}
              className="bg-terminal-border border-terminal-border text-terminal-text px-2 py-1 text-terminal-xs rounded focus:outline-none w-24"
            />

            <button
              onClick={() => handleRemoveCondition(index)}
              className="text-bloomberg-red hover:text-red-400 text-terminal-xs font-bold px-2"
            >
              ✕
            </button>
          </div>
        ))}

        {rule.conditions.length === 0 && (
          <div className="text-center py-4 text-terminal-accent text-terminal-xs">
            No conditions defined
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="space-y-2">
        <div className="text-terminal-accent text-terminal-xs font-bold">
          ACTIONS
        </div>

        {rule.actions.map((action, index) => (
          <div key={action.id} className="flex items-center space-x-2 bg-terminal-bg p-2 rounded">
            <select
              value={action.type}
              onChange={(e) => {
                const updatedActions = [...rule.actions];
                updatedActions[index] = { ...updatedActions[index], type: e.target.value as any };
                onUpdate({ ...rule, actions: updatedActions });
              }}
              className="bg-terminal-border border-terminal-border text-terminal-text px-2 py-1 text-terminal-xs rounded focus:outline-none"
            >
              <option value="buy">Buy</option>
              <option value="sell">Sell</option>
              <option value="close_long">Close Long</option>
              <option value="close_short">Close Short</option>
            </select>

            <select
              value={action.size_type}
              onChange={(e) => {
                const updatedActions = [...rule.actions];
                updatedActions[index] = { ...updatedActions[index], size_type: e.target.value as any };
                onUpdate({ ...rule, actions: updatedActions });
              }}
              className="bg-terminal-border border-terminal-border text-terminal-text px-2 py-1 text-terminal-xs rounded focus:outline-none"
            >
              <option value="fixed">Fixed Amount</option>
              <option value="percent_equity">% of Equity</option>
              <option value="percent_position">% of Position</option>
            </select>

            <input
              type="number"
              value={action.size_value}
              onChange={(e) => {
                const updatedActions = [...rule.actions];
                updatedActions[index] = { ...updatedActions[index], size_value: parseFloat(e.target.value) };
                onUpdate({ ...rule, actions: updatedActions });
              }}
              min="0"
              step="1"
              className="bg-terminal-border border-terminal-border text-terminal-text px-2 py-1 text-terminal-xs rounded focus:outline-none w-24"
            />

            <span className="text-terminal-text text-terminal-xs">
              {action.size_type === 'fixed' ? '$' : '%'}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default RuleBuilder;
