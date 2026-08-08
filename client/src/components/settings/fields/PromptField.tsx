/**
 * PromptField — labelled prompt editor (Textarea) with a reset-to-default
 * button, a "leave empty to use default" hint, and (when a default is known) a
 * Diff view comparing the shipped default against the current override.
 *
 * The last field-kit primitive to take the manifest contract, and the one that
 * needed the least of it: it already owns a reset (which knows the shipped
 * *text*, not just the config default) and its own default-vs-yours diff. So a
 * `binding` here buys the two things it lacked — the `data-setting` anchor
 * settings search lands on, and the help popover — while its existing reset
 * stays in charge.
 */

import { useState } from 'react';
import { RotateCcw, GitCompare } from 'lucide-react';
import { Label, Textarea, Button } from '../../ui';
import { LayerDiffModal } from '../../unified-settings/sections/prompt-stack/LayerDiffModal';
import { SettingHelp } from '../SettingHelp';
import type { SettingBinding } from '../../unified-settings/SettingsManifestContext';

interface PromptFieldProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  onReset: () => void;
  placeholder?: string;
  rows?: number;
  /** The shipped default text — enables a "Diff" view (default vs your override). */
  defaultText?: string;
  /** Manifest entry — supplies the anchor and the authored help. */
  binding?: SettingBinding | null;
}

export function PromptField({
  label, value, onChange, onReset, placeholder, rows = 6, defaultText, binding,
}: PromptFieldProps) {
  const [diffOpen, setDiffOpen] = useState(false);
  const canDiff = !!defaultText && defaultText.trim().length > 0;
  const anchor = binding ? `${binding.entry.store}:${binding.entry.key}` : undefined;

  return (
    <div className="setting-textarea" data-setting={anchor}>
      <div className="textarea-header">
        <Label>{label}</Label>
        {binding?.isModified && (
          <span
            className="setting-modified-dot"
            title="Changed from the default"
            aria-label="Changed from the default"
          />
        )}
        <SettingHelp help={binding?.help} label={label} />
        <div style={{ display: 'flex', gap: 4, marginLeft: 'auto' }}>
          {canDiff && (
            <Button variant="ghost" size="sm" onClick={() => setDiffOpen(true)} title="Compare with the default">
              <GitCompare size={14} /> Diff
            </Button>
          )}
          <Button variant="ghost" size="icon" onClick={onReset} title="Reset to default">
            <RotateCcw size={14} />
          </Button>
        </div>
      </div>
      <Textarea
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        rows={rows}
      />
      {!value && <p className="prompt-hint">Leave empty to use default prompt</p>}

      {canDiff && (
        <LayerDiffModal
          open={diffOpen}
          onOpenChange={setDiffOpen}
          title={`${label} — default vs yours`}
          leftLabel="Default"
          leftText={defaultText as string}
          rightLabel="Yours"
          rightText={value.trim() ? value : (defaultText as string)}
          onAdopt={() => {
            onReset();
            setDiffOpen(false);
          }}
        />
      )}
    </div>
  );
}
