/**
 * ImagesSection — image-generation settings (avatars first; multi-modal pipelines later).
 *
 * OpenRouter-only today: a model picker (default flux.2-klein-4b), the app-level avatar
 * STYLE prompt (the per-profile SUBJECT prompt is appended at generation time), and an
 * enable toggle. Persists under `images.*`. Leads with the OpenRouter-required notice.
 */

import { useEffect, useState } from 'react';
import { ImageIcon, RefreshCw, Save, Info } from 'lucide-react';
import { api } from '../../../lib/api';
import { useNotify } from '../../../contexts/NotificationContext';
import { Button, SectionHeader } from '../../ui';
import { ModelPickerField } from '../../common/ModelPickerField';

interface ImageSettings {
  enabled: boolean;
  default_model: string;
  avatar_style_prompt: string;
}

const FALLBACK: ImageSettings = {
  enabled: true,
  default_model: 'openrouter:black-forest-labs/flux.2-klein-4b',
  avatar_style_prompt:
    'A photorealistic headshot portrait, centered, clean studio lighting, subtle depth of field, with a softly rounded border.',
};

export default function ImagesSection() {
  const { notifyError, notifySuccess } = useNotify();
  const [settings, setSettings] = useState<ImageSettings>(FALLBACK);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    void load();
  }, []);

  const load = async () => {
    setLoading(true);
    try {
      const config = await api.getConfig();
      const im = (config.images || {}) as Partial<ImageSettings>;
      setSettings({
        enabled: im.enabled ?? FALLBACK.enabled,
        default_model: im.default_model || FALLBACK.default_model,
        avatar_style_prompt: im.avatar_style_prompt || FALLBACK.avatar_style_prompt,
      });
    } catch (error) {
      notifyError(error, 'Failed to load image settings');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await api.updateConfig({
        images: {
          enabled: settings.enabled,
          default_model: settings.default_model,
          avatar_style_prompt: settings.avatar_style_prompt,
        },
      });
      notifySuccess('Image settings saved', 'Images');
      await load();
    } catch (error) {
      notifyError(error, 'Failed to save image settings');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<ImageIcon size={20} />}
        title="Images"
        description="Generate agent avatars (and, soon, images in conversations) — via OpenRouter."
      />

      <div
        className="setting-row"
        style={{ display: 'flex', gap: 8, alignItems: 'flex-start', color: 'var(--color-fg-secondary)' }}
      >
        <Info size={15} style={{ flexShrink: 0, marginTop: 2 }} />
        <span className="setting-hint">
          OpenRouter currently powers all AI functionality — image generation, voice, and as
          the model backbone. Other providers are additive; fully-distributed provider support
          is coming.
        </span>
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
              <span>Enable image generation</span>
              <span className="setting-hint">When off, the avatar "Generate" tab is unavailable.</span>
            </label>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={settings.enabled}
                onChange={(e) => setSettings((prev) => ({ ...prev, enabled: e.target.checked }))}
              />
              <span className="toggle-slider"></span>
            </label>
          </div>

          <div className="setting-row">
            <ModelPickerField
              label="Image model"
              value={settings.default_model}
              onChange={(modelId) => setSettings((prev) => ({ ...prev, default_model: modelId }))}
            />
          </div>

          <div className="setting-row">
            <label className="setting-label">
              <span>Avatar style prompt</span>
              <span className="setting-hint">
                The app-wide look. The per-agent subject ("a gray-haired strategist…") is added
                when you generate.
              </span>
            </label>
            <textarea
              className="form-input"
              rows={3}
              value={settings.avatar_style_prompt}
              onChange={(e) => setSettings((prev) => ({ ...prev, avatar_style_prompt: e.target.value }))}
              placeholder={FALLBACK.avatar_style_prompt}
            />
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
