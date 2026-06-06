/**
 * OverridablePromptField — a prompt with a shipped *default* (owned by the app)
 * plus an optional user *override*. Empty override = ride the default. Offers
 * Reset (drop the override), "Load default into editor" (seed for hand-editing),
 * and a Diff (Default vs Yours) via the shared LayerDiffModal.
 *
 * Used for an ambassador's functional voices (briefing / Q&A / draft), where the
 * default lives in backend code and the override lives on the profile.
 */

import { useState } from 'react';
import { RotateCcw, GitCompare, FileDown } from 'lucide-react';
import { Button, Badge } from '../ui';
import { Textarea } from '../ui/Field';
import { LayerDiffModal } from '../unified-settings/sections/prompt-stack/LayerDiffModal';
import './OverridablePromptField.css';

interface OverridablePromptFieldProps {
  title: string;
  description?: string;
  defaultText: string;
  /** The user's override, or null/undefined to ride the default. */
  override: string | null | undefined;
  /** Called with the new override text (empty string stays an override; use onReset to clear). */
  onChange: (value: string) => void;
  /** Clear the override entirely (back to the shipped default). */
  onReset: () => void;
}

export function OverridablePromptField({
  title,
  description,
  defaultText,
  override,
  onChange,
  onReset,
}: OverridablePromptFieldProps) {
  const [diffOpen, setDiffOpen] = useState(false);
  const hasOverride = override !== null && override !== undefined;
  const modified = hasOverride && override !== defaultText;

  return (
    <div className="overridable-field">
      <div className="overridable-field__head">
        <span className="overridable-field__title">{title}</span>
        {modified ? (
          <Badge variant="accent" size="sm">overridden</Badge>
        ) : (
          <Badge variant="neutral" size="sm">default</Badge>
        )}
        <div className="overridable-field__actions">
          <Button type="button" variant="ghost" size="sm" onClick={() => setDiffOpen(true)}>
            <GitCompare size={13} /> Diff
          </Button>
          {!hasOverride ? (
            <Button type="button" variant="ghost" size="sm" onClick={() => onChange(defaultText)}>
              <FileDown size={13} /> Load default
            </Button>
          ) : (
            <Button type="button" variant="ghost" size="sm" onClick={onReset}>
              <RotateCcw size={13} /> Reset
            </Button>
          )}
        </div>
      </div>
      {description && <p className="overridable-field__desc">{description}</p>}
      <Textarea
        className="overridable-field__editor"
        value={hasOverride ? (override as string) : ''}
        rows={5}
        spellCheck={false}
        placeholder="Using the shipped default — type to override it, or “Load default” to start from it."
        onChange={(e) => onChange(e.target.value)}
      />

      <LayerDiffModal
        open={diffOpen}
        onOpenChange={setDiffOpen}
        title={`${title} — default vs yours`}
        leftLabel="Default"
        leftText={defaultText}
        rightLabel="Yours"
        rightText={hasOverride ? (override as string) : defaultText}
        onAdopt={() => {
          onReset();
          setDiffOpen(false);
        }}
      />
    </div>
  );
}
