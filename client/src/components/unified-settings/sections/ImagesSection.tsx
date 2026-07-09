/**
 * ImagesSection — image-generation settings (avatars + in-conversation image generation).
 *
 * OpenRouter-only today: a model picker (default flux.2-klein-4b), the app-level avatar
 * STYLE prompt (the per-profile SUBJECT prompt is appended at generation time), and an
 * enable toggle. Persists under `images.*` (vision input under `vision.*`), autosaved
 * via useSettingsAutosave + the settings field kit. Leads with the OpenRouter notice.
 */

import { ImageIcon, RefreshCw, Info } from 'lucide-react';
import { api } from '../../../lib/api';
import { useSettingsAutosave } from '../../../lib/hooks';
import { useNotify } from '../../../contexts/NotificationContext';
import { SectionHeader } from '../../ui';
import { ModelPickerField } from '../../common/ModelPickerField';
import { PromptField, SaveStatusChip, ToggleField } from '../../settings/fields';

interface ImageSettings extends Record<string, unknown> {
  enabled: boolean;
  default_model: string;
  avatar_style_prompt: string;
  // Vision input (image *input* — the user attaches a picture a model can see).
  visionEnabled: boolean;
}

const FALLBACK: ImageSettings = {
  enabled: true,
  default_model: 'openrouter:black-forest-labs/flux.2-klein-4b',
  avatar_style_prompt:
    'A photorealistic headshot portrait, centered, clean studio lighting, subtle depth of field, with a softly rounded border.',
  visionEnabled: true,
};

export default function ImagesSection() {
  const { notifyError } = useNotify();

  const { settings, loading, status, update } = useSettingsAutosave<ImageSettings>({
    load: async () => {
      const config = await api.getConfig();
      const im = (config.images || {}) as Partial<ImageSettings>;
      const vi = (config.vision || {}) as { enabled?: boolean };
      return {
        enabled: im.enabled ?? FALLBACK.enabled,
        default_model: im.default_model || FALLBACK.default_model,
        avatar_style_prompt: im.avatar_style_prompt || FALLBACK.avatar_style_prompt,
        visionEnabled: vi.enabled ?? FALLBACK.visionEnabled,
      };
    },
    save: async changed => {
      // `visionEnabled` persists under vision.*; everything else under images.*.
      const { visionEnabled, ...images } = changed;
      const payload: Parameters<typeof api.updateConfig>[0] = {};
      if (Object.keys(images).length > 0) payload.images = images;
      if (visionEnabled !== undefined) payload.vision = { enabled: visionEnabled };
      await api.updateConfig(payload);
    },
    onError: err => notifyError(err, 'Image settings'),
  });

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<ImageIcon size={20} />}
        title="Images"
        description="Generate agent avatars and images in conversations — via OpenRouter."
        actions={<SaveStatusChip status={status} />}
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

      {loading || !settings ? (
        <div className="loading-state">
          <RefreshCw size={24} className="spin" />
          <span>Loading settings...</span>
        </div>
      ) : (
        <div className="settings-content">
          <ToggleField
            checked={settings.enabled}
            onChange={enabled => update({ enabled })}
            label="Enable image generation"
            hint={'Lets agents generate images in a conversation (the `generate_image` tool and image-output models) and powers the avatar "Generate" tab. When off, both are unavailable.'}
          />

          <ToggleField
            checked={settings.visionEnabled}
            onChange={visionEnabled => update({ visionEnabled })}
            label="Enable vision input"
            hint="Let you attach images to a message so a vision-capable model can see them. When off, the composer's attach button is hidden."
          />

          <div className="setting-row">
            <ModelPickerField
              label="Image model"
              value={settings.default_model}
              onChange={default_model => update({ default_model })}
            />
          </div>

          <PromptField
            label="Avatar style prompt"
            value={settings.avatar_style_prompt}
            onChange={avatar_style_prompt => update({ avatar_style_prompt })}
            onReset={() => update({ avatar_style_prompt: FALLBACK.avatar_style_prompt })}
            placeholder={FALLBACK.avatar_style_prompt}
            rows={3}
            defaultText={FALLBACK.avatar_style_prompt}
          />
          <span className="setting-hint">
            The app-wide look. The per-agent subject ("a gray-haired strategist…") is added
            when you generate.
          </span>
        </div>
      )}
    </div>
  );
}
