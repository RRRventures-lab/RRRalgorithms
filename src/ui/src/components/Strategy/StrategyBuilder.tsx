import React, { useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState, AppDispatch } from '../../store/store';
import {
  createNewStrategy,
  updateBuilderStrategy,
  addEntryRule,
  addExitRule,
  updateRule,
  removeRule,
  clearBuilder,
  saveStrategy,
} from '../../store/slices/strategySlice';
import { Strategy, StrategyRule, StrategyCondition, StrategyAction } from '../../store/slices/strategySlice';
import RuleBuilder from './RuleBuilder';

const StrategyBuilder: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const { builderStrategy, availableIndicators, loading } = useSelector(
    (state: RootState) => state.strategy
  );
  const [activeSection, setActiveSection] = useState<'info' | 'entry' | 'exit' | 'risk'>('info');

  const handleCreateNew = () => {
    dispatch(createNewStrategy());
  };

  const handleSave = () => {
    if (builderStrategy) {
      dispatch(saveStrategy(builderStrategy));
    }
  };

  const handleCancel = () => {
    if (window.confirm('Are you sure you want to discard this strategy?')) {
      dispatch(clearBuilder());
    }
  };

  const handleAddEntryRule = () => {
    const newRule: StrategyRule = {
      id: `rule-${Date.now()}`,
      name: 'New Entry Rule',
      conditions: [],
      logic: 'and',
      actions: [{
        id: `action-${Date.now()}`,
        type: 'buy',
        size_type: 'percent_equity',
        size_value: 10,
      }],
      enabled: true,
    };
    dispatch(addEntryRule(newRule));
  };

  const handleAddExitRule = () => {
    const newRule: StrategyRule = {
      id: `rule-${Date.now()}`,
      name: 'New Exit Rule',
      conditions: [],
      logic: 'and',
      actions: [{
        id: `action-${Date.now()}`,
        type: 'sell',
        size_type: 'percent_position',
        size_value: 100,
      }],
      enabled: true,
    };
    dispatch(addExitRule(newRule));
  };

  if (!builderStrategy) {
    return (
      <div className="h-full flex flex-col">
        <div className="terminal-panel-header">
          <span className="text-terminal-green text-terminal-sm font-bold">STRATEGY BUILDER</span>
        </div>
        <div className="terminal-content flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="text-terminal-accent mb-4">No strategy is being built</div>
            <button
              onClick={handleCreateNew}
              className="bloomberg-button px-6 py-3"
            >
              <span className="text-terminal-sm font-bold">CREATE NEW STRATEGY</span>
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="terminal-panel-header flex justify-between items-center">
        <div className="flex items-center space-x-3">
          <span className="text-terminal-green text-terminal-sm font-bold">STRATEGY BUILDER</span>
          <input
            type="text"
            value={builderStrategy.name}
            onChange={(e) => dispatch(updateBuilderStrategy({ name: e.target.value }))}
            className="bg-terminal-border border border-terminal-border text-terminal-text px-2 py-1 text-terminal-sm rounded focus:outline-none focus:border-bloomberg-green"
          />
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={handleCancel}
            className="bloomberg-button text-terminal-xs px-3 py-1 bg-bloomberg-red hover:bg-red-600"
          >
            CANCEL
          </button>
          <button
            onClick={handleSave}
            disabled={loading}
            className="bloomberg-button text-terminal-xs px-3 py-1 disabled:opacity-50"
          >
            {loading ? 'SAVING...' : 'SAVE STRATEGY'}
          </button>
        </div>
      </div>

      {/* Section Tabs */}
      <div className="flex border-b border-terminal-border">
        {['info', 'entry', 'exit', 'risk'].map((section) => (
          <button
            key={section}
            onClick={() => setActiveSection(section as any)}
            className={`px-4 py-2 text-terminal-xs font-bold ${
              activeSection === section
                ? 'text-bloomberg-green border-b-2 border-bloomberg-green'
                : 'text-terminal-accent hover:text-terminal-text'
            }`}
          >
            {section.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="terminal-content flex-1 overflow-y-auto terminal-scrollbar p-4">
        {activeSection === 'info' && (
          <div className="space-y-4 max-w-2xl">
            <div>
              <label className="text-terminal-accent text-terminal-xs font-bold block mb-1">
                STRATEGY NAME
              </label>
              <input
                type="text"
                value={builderStrategy.name}
                onChange={(e) => dispatch(updateBuilderStrategy({ name: e.target.value }))}
                className="w-full bg-terminal-border border border-terminal-border text-terminal-text px-3 py-2 rounded focus:outline-none focus:border-bloomberg-green"
              />
            </div>

            <div>
              <label className="text-terminal-accent text-terminal-xs font-bold block mb-1">
                DESCRIPTION
              </label>
              <textarea
                value={builderStrategy.description}
                onChange={(e) => dispatch(updateBuilderStrategy({ description: e.target.value }))}
                rows={4}
                className="w-full bg-terminal-border border border-terminal-border text-terminal-text px-3 py-2 rounded focus:outline-none focus:border-bloomberg-green resize-none"
              />
            </div>

            <div>
              <label className="text-terminal-accent text-terminal-xs font-bold block mb-1">
                STRATEGY TYPE
              </label>
              <select
                value={builderStrategy.type}
                onChange={(e) => dispatch(updateBuilderStrategy({ type: e.target.value as any }))}
                className="w-full bg-terminal-border border border-terminal-border text-terminal-text px-3 py-2 rounded focus:outline-none focus:border-bloomberg-green"
              >
                <option value="momentum">Momentum</option>
                <option value="mean_reversion">Mean Reversion</option>
                <option value="breakout">Breakout</option>
                <option value="arbitrage">Arbitrage</option>
                <option value="custom">Custom</option>
              </select>
            </div>
          </div>
        )}

        {activeSection === 'entry' && (
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <div className="text-terminal-accent text-terminal-sm font-bold">
                ENTRY RULES ({builderStrategy.entry_rules.length})
              </div>
              <button
                onClick={handleAddEntryRule}
                className="bloomberg-button text-terminal-xs px-3 py-1"
              >
                + ADD ENTRY RULE
              </button>
            </div>

            {builderStrategy.entry_rules.length === 0 ? (
              <div className="text-center py-8 text-terminal-accent">
                No entry rules defined. Click "Add Entry Rule" to get started.
              </div>
            ) : (
              builderStrategy.entry_rules.map((rule) => (
                <RuleBuilder
                  key={rule.id}
                  rule={rule}
                  ruleType="entry"
                  availableIndicators={availableIndicators}
                  onUpdate={(updatedRule) => dispatch(updateRule({ type: 'entry', rule: updatedRule }))}
                  onRemove={() => dispatch(removeRule({ type: 'entry', ruleId: rule.id }))}
                />
              ))
            )}
          </div>
        )}

        {activeSection === 'exit' && (
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <div className="text-terminal-accent text-terminal-sm font-bold">
                EXIT RULES ({builderStrategy.exit_rules.length})
              </div>
              <button
                onClick={handleAddExitRule}
                className="bloomberg-button text-terminal-xs px-3 py-1"
              >
                + ADD EXIT RULE
              </button>
            </div>

            {builderStrategy.exit_rules.length === 0 ? (
              <div className="text-center py-8 text-terminal-accent">
                No exit rules defined. Click "Add Exit Rule" to get started.
              </div>
            ) : (
              builderStrategy.exit_rules.map((rule) => (
                <RuleBuilder
                  key={rule.id}
                  rule={rule}
                  ruleType="exit"
                  availableIndicators={availableIndicators}
                  onUpdate={(updatedRule) => dispatch(updateRule({ type: 'exit', rule: updatedRule }))}
                  onRemove={() => dispatch(removeRule({ type: 'exit', ruleId: rule.id }))}
                />
              ))
            )}
          </div>
        )}

        {activeSection === 'risk' && (
          <div className="space-y-4 max-w-2xl">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-terminal-accent text-terminal-xs font-bold block mb-1">
                  MAX POSITION SIZE (%)
                </label>
                <input
                  type="number"
                  value={builderStrategy.risk_management.max_position_size}
                  onChange={(e) => dispatch(updateBuilderStrategy({
                    risk_management: {
                      ...builderStrategy.risk_management,
                      max_position_size: parseFloat(e.target.value)
                    }
                  }))}
                  min="0"
                  max="100"
                  step="1"
                  className="w-full bg-terminal-border border border-terminal-border text-terminal-text px-3 py-2 rounded focus:outline-none focus:border-bloomberg-green"
                />
              </div>

              <div>
                <label className="text-terminal-accent text-terminal-xs font-bold block mb-1">
                  MAX PORTFOLIO RISK (%)
                </label>
                <input
                  type="number"
                  value={builderStrategy.risk_management.max_portfolio_risk}
                  onChange={(e) => dispatch(updateBuilderStrategy({
                    risk_management: {
                      ...builderStrategy.risk_management,
                      max_portfolio_risk: parseFloat(e.target.value)
                    }
                  }))}
                  min="0"
                  max="100"
                  step="0.1"
                  className="w-full bg-terminal-border border border-terminal-border text-terminal-text px-3 py-2 rounded focus:outline-none focus:border-bloomberg-green"
                />
              </div>

              <div>
                <label className="text-terminal-accent text-terminal-xs font-bold block mb-1">
                  STOP LOSS (%)
                </label>
                <input
                  type="number"
                  value={builderStrategy.risk_management.stop_loss_percent || ''}
                  onChange={(e) => dispatch(updateBuilderStrategy({
                    risk_management: {
                      ...builderStrategy.risk_management,
                      stop_loss_percent: e.target.value ? parseFloat(e.target.value) : undefined
                    }
                  }))}
                  min="0"
                  step="0.1"
                  placeholder="Optional"
                  className="w-full bg-terminal-border border border-terminal-border text-terminal-text px-3 py-2 rounded focus:outline-none focus:border-bloomberg-green"
                />
              </div>

              <div>
                <label className="text-terminal-accent text-terminal-xs font-bold block mb-1">
                  TAKE PROFIT (%)
                </label>
                <input
                  type="number"
                  value={builderStrategy.risk_management.take_profit_percent || ''}
                  onChange={(e) => dispatch(updateBuilderStrategy({
                    risk_management: {
                      ...builderStrategy.risk_management,
                      take_profit_percent: e.target.value ? parseFloat(e.target.value) : undefined
                    }
                  }))}
                  min="0"
                  step="0.1"
                  placeholder="Optional"
                  className="w-full bg-terminal-border border border-terminal-border text-terminal-text px-3 py-2 rounded focus:outline-none focus:border-bloomberg-green"
                />
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="trailing-stop"
                checked={builderStrategy.risk_management.trailing_stop || false}
                onChange={(e) => dispatch(updateBuilderStrategy({
                  risk_management: {
                    ...builderStrategy.risk_management,
                    trailing_stop: e.target.checked
                  }
                }))}
                className="w-4 h-4"
              />
              <label htmlFor="trailing-stop" className="text-terminal-text text-terminal-sm">
                Enable Trailing Stop
              </label>
            </div>

            <div>
              <label className="text-terminal-accent text-terminal-xs font-bold block mb-1">
                MAX DRAWDOWN STOP (%)
              </label>
              <input
                type="number"
                value={builderStrategy.risk_management.max_drawdown_stop || ''}
                onChange={(e) => dispatch(updateBuilderStrategy({
                  risk_management: {
                    ...builderStrategy.risk_management,
                    max_drawdown_stop: e.target.value ? parseFloat(e.target.value) : undefined
                  }
                }))}
                min="0"
                step="0.1"
                placeholder="Optional - Stop trading if drawdown exceeds this value"
                className="w-full bg-terminal-border border border-terminal-border text-terminal-text px-3 py-2 rounded focus:outline-none focus:border-bloomberg-green"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default StrategyBuilder;
