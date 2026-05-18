/**
 * PlannerSection — Task planner configuration
 *
 * Mirrors PromptsSection.tsx. Settings live under `planner.*` in
 * data/config.json and are consumed by TaskPlanner / Agent.run().
 */

import { useState, useEffect } from 'react';
import {
  ListTree,
  RefreshCw,
  Save,
  Check,
  AlertTriangle,
} from 'lucide-react';
import { api } from '../../../lib/api';
import { ModelSelector } from '../../common/ModelSelector';

type ComplexityThreshold = 'simple' | 'moderate' | 'complex';

interface PlannerSettings {
  enabled: boolean;
  model: string;
  temperature: number;
  max_tokens: number;
  prompt_override: string;
  complexity_threshold: ComplexityThreshold;
}

const DEFAULT_SETTINGS: PlannerSettings = {
  enabled: true,
  model: '',
  temperature: 0.3,
  max_tokens: 1000,
  prompt_override: '',
  complexity_threshold: 'moderate',
};

export default function PlannerSection() {
  const [settings, setSettings] = useState<PlannerSettings>(DEFAULT_SETTINGS);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  useEffect(() => {
    void fetchSettings();
  }, []);

  const fetchSettings = async () => {
    setLoading(true);
    try {
      const config = await api.getConfig();
      const p = (config.planner || {}) as Partial<PlannerSettings> & { model?: string | null };
      setSettings({
        enabled: p.enabled ?? true,
        model: p.model || '',
        temperature: p.temperature ?? 0.3,
        max_tokens: p.max_tokens ?? 1000,
        prompt_override: p.prompt_override || '',
        complexity_threshold: (p.complexity_threshold as ComplexityThreshold) || 'moderate',
      });
    } catch (error) {
      console.error('Failed to fetch planner settings:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setMessage(null);
    try {
      await api.updateConfig({
        planner: {
          enabled: settings.enabled,
          // Empty string → null so backend falls back to agent default model.
          model: settings.model.trim() ? settings.model : null,
          temperature: settings.temperature,
          max_tokens: settings.max_tokens,
          prompt_override: settings.prompt_override,
          complexity_threshold: settings.complexity_threshold,
        },
      });
      setMessage({ type: 'success', text: 'Planner settings saved' });
      setTimeout(() => setMessage(null), 3000);
    } catch (error) {
      console.error('Failed to save planner settings:', error);
      setMessage({ type: 'error', text: 'Failed to save planner settings' });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="settings-section fade-in">
      <div className="section-header">
        <div>
          <h2 className="section-title">
            <ListTree size={20} className="section-title-icon" />
            Task Planner
          </h2>
          <p className="section-description">
            Configure how the agent decomposes complex tasks into subtasks before execution.
          </p>
        </div>
      </div>

      {loading ? (
        <div className="loading-state">
          <RefreshCw size={24} className="spin" />
          <span>Loading settings...</span>
        </div>
      ) : (
        <div className="settings-content">
          <div className="setting-row">
            <label className="setting-label">
              <span>Enable Planner</span>
              <span className="setting-hint">
                When off, every task takes the single-pass path regardless of complexity.
              </span>
            </label>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={settings.enabled}
                onChange={(e) => setSettings(prev => ({ ...prev, enabled: e.target.checked }))}
              />
              <span className="toggle-slider"></span>
            </label>
          </div>

          <div className="setting-row">
            <ModelSelector
              label="Planner Model"
              value={settings.model}
              onChange={(modelId) => setSettings(prev => ({ ...prev, model: modelId }))}
              showDefault={true}
              compact
            />
          </div>

          <div className="setting-row">
            <label className="setting-label">
              <span>Complexity Threshold</span>
              <span className="setting-hint">
                Minimum complexity that triggers decomposition.
              </span>
            </label>
            <select
              className="form-input"
              value={settings.complexity_threshold}
              onChange={(e) => setSettings(prev => ({
                ...prev,
                complexity_threshold: e.target.value as ComplexityThreshold,
              }))}
            >
              <option value="simple">Simple — always decompose</option>
              <option value="moderate">Moderate — decompose moderate or complex tasks</option>
              <option value="complex">Complex — only decompose complex tasks</option>
            </select>
          </div>

          <div className="setting-row">
            <label className="setting-label">
              <span>Temperature</span>
              <span className="setting-hint">Lower values produce more deterministic plans.</span>
            </label>
            <div className="input-with-hint">
              <input
                type="number"
                className="form-input"
                value={settings.temperature}
                onChange={(e) => setSettings(prev => ({
                  ...prev,
                  temperature: parseFloat(e.target.value) || 0.3,
                }))}
                min={0}
                max={1}
                step={0.1}
              />
            </div>
          </div>

          <div className="setting-row">
            <label className="setting-label">
              <span>Max Tokens</span>
              <span className="setting-hint">Maximum length of the plan response.</span>
            </label>
            <div className="input-with-hint">
              <input
                type="number"
                className="form-input"
                value={settings.max_tokens}
                onChange={(e) => setSettings(prev => ({
                  ...prev,
                  max_tokens: parseInt(e.target.value) || 1000,
                }))}
                min={100}
                max={4000}
                step={100}
              />
            </div>
          </div>

          <div className="setting-row vertical">
            <label className="setting-label">
              <span>Custom Planner Prompt</span>
              <span className="setting-hint">
                Leave empty to use the default decomposition prompt.
              </span>
            </label>
            <textarea
              className="form-textarea"
              value={settings.prompt_override}
              onChange={(e) => setSettings(prev => ({ ...prev, prompt_override: e.target.value }))}
              placeholder="Override the planner's decomposition system prompt..."
              rows={6}
            />
          </div>

          <div className="setting-actions">
            <button
              className="button-primary"
              onClick={handleSave}
              disabled={saving}
            >
              {saving ? (
                <>
                  <RefreshCw size={16} className="spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Save size={16} />
                  Save Settings
                </>
              )}
            </button>
            {message && (
              <span className={`save-message ${message.type}`}>
                {message.type === 'success' ? <Check size={14} /> : <AlertTriangle size={14} />}
                {message.text}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
