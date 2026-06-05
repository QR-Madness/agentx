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
} from 'lucide-react';
import { api } from '../../../lib/api';
import { useNotify } from '../../../contexts/NotificationContext';
import { Button, SectionHeader } from '../../ui';
import { ModelPickerField } from '../../common/ModelPickerField';

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
  complexity_threshold: 'complex',
};

export default function PlannerSection() {
  const { notifyError, notifySuccess } = useNotify();
  const [settings, setSettings] = useState<PlannerSettings>(DEFAULT_SETTINGS);
  // The built-in decomposition prompt (read-only) used to seed the editor so the
  // user can customize from the real default instead of a blank box.
  const [defaultPrompt, setDefaultPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    void fetchSettings();
  }, []);

  const fetchSettings = async () => {
    setLoading(true);
    try {
      const config = await api.getConfig();
      const p = (config.planner || {}) as Partial<PlannerSettings> & {
        model?: string | null;
        decompose_default?: string;
      };
      const builtinPrompt = p.decompose_default || '';
      setDefaultPrompt(builtinPrompt);
      setSettings({
        enabled: p.enabled ?? true,
        model: p.model || '',
        temperature: p.temperature ?? 0.3,
        max_tokens: p.max_tokens ?? 1000,
        // Seed the editor with the built-in prompt when no override is saved.
        prompt_override: p.prompt_override || builtinPrompt,
        complexity_threshold: (p.complexity_threshold as ComplexityThreshold) || 'complex',
      });
    } catch (error) {
      notifyError(error, 'Failed to load planner settings');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await api.updateConfig({
        planner: {
          enabled: settings.enabled,
          // Empty string → null so backend falls back to agent default model.
          model: settings.model.trim() ? settings.model : null,
          temperature: settings.temperature,
          max_tokens: settings.max_tokens,
          // If the editor still matches the built-in prompt, persist empty so the
          // planner keeps tracking the live default instead of pinning a copy.
          prompt_override:
            settings.prompt_override.trim() === defaultPrompt.trim()
              ? ''
              : settings.prompt_override,
          complexity_threshold: settings.complexity_threshold,
        },
      });
      notifySuccess('Planner settings saved', 'Task Planner');
    } catch (error) {
      notifyError(error, 'Failed to save planner settings');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<ListTree size={20} />}
        title="Task Planner"
        description="Configure how the agent decomposes complex tasks into subtasks before execution."
      />

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
            <ModelPickerField
              label="Planner Model"
              value={settings.model}
              onChange={(modelId) => setSettings(prev => ({ ...prev, model: modelId }))}
              showDefault={true}
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
                Pre-filled with the built-in decomposition prompt — edit it to customize.
                Leaving it unchanged keeps tracking the default automatically.
              </span>
            </label>
            <textarea
              className="form-textarea"
              value={settings.prompt_override}
              onChange={(e) => setSettings(prev => ({ ...prev, prompt_override: e.target.value }))}
              placeholder="Override the planner's decomposition system prompt..."
              rows={10}
            />
            <Button
              variant="ghost"
              onClick={() => setSettings(prev => ({ ...prev, prompt_override: defaultPrompt }))}
              disabled={!defaultPrompt || settings.prompt_override.trim() === defaultPrompt.trim()}
            >
              <RefreshCw size={14} />
              Reset to default
            </Button>
          </div>

          <div className="setting-actions">
            <Button variant="primary" onClick={handleSave} loading={saving}>
              <Save size={16} />
              {saving ? 'Saving...' : 'Save Settings'}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
