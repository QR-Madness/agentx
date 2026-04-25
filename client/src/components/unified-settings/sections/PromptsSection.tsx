/**
 * PromptsSection — Prompt enhancement configuration
 * Extracted from SettingsPanel.tsx lines 656-805
 */

import { useState, useEffect } from 'react';
import {
  Sparkles,
  RefreshCw,
  Save,
  Check,
  AlertTriangle,
} from 'lucide-react';
import { api } from '../../../lib/api';
import { ModelSelector } from '../../common/ModelSelector';

export default function PromptsSection() {
  const [promptEnhanceSettings, setPromptEnhanceSettings] = useState<{
    enabled: boolean;
    model: string;
    temperature: number;
    max_tokens: number;
    system_prompt: string;
  }>({
    enabled: true,
    model: 'anthropic:claude-3-5-haiku-latest',
    temperature: 0.7,
    max_tokens: 1000,
    system_prompt: '',
  });

  const [loadingPromptSettings, setLoadingPromptSettings] = useState(false);
  const [savingPromptSettings, setSavingPromptSettings] = useState(false);
  const [promptSettingsMessage, setPromptSettingsMessage] = useState<{
    type: 'success' | 'error';
    text: string;
  } | null>(null);

  // Fetch prompt enhancement settings on mount
  useEffect(() => {
    fetchPromptSettings();
  }, []);

  const fetchPromptSettings = async () => {
    setLoadingPromptSettings(true);
    try {
      const config = await api.getConfig();
      const promptConfig = (config.prompt_enhancement || {}) as {
        enabled?: boolean;
        model?: string;
        temperature?: number;
        max_tokens?: number;
        system_prompt?: string;
      };
      setPromptEnhanceSettings({
        enabled: promptConfig.enabled ?? true,
        model: promptConfig.model || 'anthropic:claude-3-5-haiku-latest',
        temperature: promptConfig.temperature ?? 0.7,
        max_tokens: promptConfig.max_tokens ?? 1000,
        system_prompt: promptConfig.system_prompt || '',
      });
    } catch (error) {
      console.error('Failed to fetch prompt settings:', error);
    } finally {
      setLoadingPromptSettings(false);
    }
  };

  const handleSavePromptSettings = async () => {
    setSavingPromptSettings(true);
    setPromptSettingsMessage(null);

    try {
      // Save all prompt enhancement settings at once
      await api.updateConfig({
        prompt_enhancement: {
          enabled: promptEnhanceSettings.enabled,
          model: promptEnhanceSettings.model,
          temperature: promptEnhanceSettings.temperature,
          max_tokens: promptEnhanceSettings.max_tokens,
          system_prompt: promptEnhanceSettings.system_prompt,
        },
      });

      setPromptSettingsMessage({ type: 'success', text: 'Prompt settings saved' });
      setTimeout(() => setPromptSettingsMessage(null), 3000);
    } catch (error) {
      console.error('Failed to save prompt settings:', error);
      setPromptSettingsMessage({ type: 'error', text: 'Failed to save prompt settings' });
    } finally {
      setSavingPromptSettings(false);
    }
  };

  return (
    <div className="settings-section fade-in">
      <div className="section-header">
        <div>
          <h2 className="section-title">
            <Sparkles size={20} className="section-title-icon" />
            Prompt Enhancement
          </h2>
          <p className="section-description">
            Configure AI-powered prompt enhancement for better results
          </p>
        </div>
      </div>

      {loadingPromptSettings ? (
        <div className="loading-state">
          <RefreshCw size={24} className="spin" />
          <span>Loading settings...</span>
        </div>
      ) : (
        <div className="settings-content">
          {/* Enable toggle */}
          <div className="setting-row">
            <label className="setting-label">
              <span>Enable Prompt Enhancement</span>
              <span className="setting-hint">Allow AI to improve your prompts before sending</span>
            </label>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={promptEnhanceSettings.enabled}
                onChange={(e) => setPromptEnhanceSettings(prev => ({
                  ...prev,
                  enabled: e.target.checked
                }))}
              />
              <span className="toggle-slider"></span>
            </label>
          </div>

          {/* Model selection - now using ModelSelector component */}
          <div className="setting-row">
            <ModelSelector
              label="Enhancement Model"
              value={promptEnhanceSettings.model}
              onChange={(modelId) => setPromptEnhanceSettings(prev => ({
                ...prev,
                model: modelId
              }))}
              onProviderChange={(provider) => {
                console.log('Provider changed to:', provider);
              }}
              showDefault={false}
              compact
            />
          </div>

          {/* Temperature */}
          <div className="setting-row">
            <label className="setting-label">
              <span>Temperature</span>
              <span className="setting-hint">Creativity level (0.0 - 1.0)</span>
            </label>
            <div className="input-with-hint">
              <input
                type="number"
                className="form-input"
                value={promptEnhanceSettings.temperature}
                onChange={(e) => setPromptEnhanceSettings(prev => ({
                  ...prev,
                  temperature: parseFloat(e.target.value) || 0.7
                }))}
                min={0}
                max={1}
                step={0.1}
              />
            </div>
          </div>

          {/* Max tokens */}
          <div className="setting-row">
            <label className="setting-label">
              <span>Max Tokens</span>
              <span className="setting-hint">Maximum output length for enhanced prompt</span>
            </label>
            <div className="input-with-hint">
              <input
                type="number"
                className="form-input"
                value={promptEnhanceSettings.max_tokens}
                onChange={(e) => setPromptEnhanceSettings(prev => ({
                  ...prev,
                  max_tokens: parseInt(e.target.value) || 1000
                }))}
                min={100}
                max={4000}
                step={100}
              />
            </div>
          </div>

          {/* Custom system prompt */}
          <div className="setting-row vertical">
            <label className="setting-label">
              <span>Custom System Prompt</span>
              <span className="setting-hint">Leave empty to use the default prompt</span>
            </label>
            <textarea
              className="form-textarea"
              value={promptEnhanceSettings.system_prompt}
              onChange={(e) => setPromptEnhanceSettings(prev => ({
                ...prev,
                system_prompt: e.target.value
              }))}
              placeholder="Enter custom instructions for the enhancement model..."
              rows={4}
            />
          </div>

          {/* Save button */}
          <div className="setting-actions">
            <button
              className="button-primary"
              onClick={handleSavePromptSettings}
              disabled={savingPromptSettings}
            >
              {savingPromptSettings ? (
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
            {promptSettingsMessage && (
              <span className={`save-message ${promptSettingsMessage.type}`}>
                {promptSettingsMessage.type === 'success' ? <Check size={14} /> : <AlertTriangle size={14} />}
                {promptSettingsMessage.text}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
