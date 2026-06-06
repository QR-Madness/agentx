/**
 * PromptField — labelled prompt editor (Textarea) with a reset-to-default
 * button, a "leave empty to use default" hint, and (when a default is known) a
 * Diff view comparing the shipped default against the current override.
 */

import { useState } from 'react';
import { RotateCcw, GitCompare } from 'lucide-react';
import { Label, Textarea, Button } from '../../ui';
import { LayerDiffModal } from '../../unified-settings/sections/prompt-stack/LayerDiffModal';

interface PromptFieldProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  onReset: () => void;
  placeholder?: string;
  rows?: number;
  /** The shipped default text — enables a "Diff" view (default vs your override). */
  defaultText?: string;
}

export function PromptField({ label, value, onChange, onReset, placeholder, rows = 6, defaultText }: PromptFieldProps) {
  const [diffOpen, setDiffOpen] = useState(false);
  const canDiff = !!defaultText && defaultText.trim().length > 0;

  return (
    <div className="setting-textarea">
      <div className="textarea-header">
        <Label>{label}</Label>
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
