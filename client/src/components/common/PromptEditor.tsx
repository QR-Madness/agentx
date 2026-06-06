/**
 * PromptEditor — a polished, reusable single-prompt editor.
 *
 * A controlled textarea with an approximate token/char count, an in-place
 * **Enhance** action (rewrites the content via the prompt enhancer, with one-click
 * undo), and an optional **Insert from library** button (the host owns the library
 * surface and inserts by REPLACING the value). Shared by the agent-profile editor
 * and the ambassador "Communications" prompt.
 */

import { useState, forwardRef } from 'react';
import { Sparkles, Undo2, Library } from 'lucide-react';
import { api } from '../../lib/api';
import { useNotify } from '../../contexts/NotificationContext';
import { estimateTokens } from '../../lib/promptStack';
import { Button } from '../ui';
import { Textarea } from '../ui/Field';
import './PromptEditor.css';

interface PromptEditorProps {
  value: string;
  onChange: (value: string) => void;
  label?: string;
  hint?: string;
  placeholder?: string;
  rows?: number;
  /** Show an "Insert from library" button that calls this (host opens its library). */
  onInsertFromLibrary?: () => void;
}

export const PromptEditor = forwardRef<HTMLTextAreaElement, PromptEditorProps>(function PromptEditor(
  { value, onChange, label, hint, placeholder, rows = 6, onInsertFromLibrary },
  ref
) {
  const { notifyError } = useNotify();
  const [enhancing, setEnhancing] = useState(false);
  const [preEnhance, setPreEnhance] = useState<string | null>(null);

  const handleEnhance = async () => {
    if (!value.trim() || enhancing) return;
    setEnhancing(true);
    try {
      const { enhanced_prompt } = await api.enhancePrompt(value);
      if (enhanced_prompt && enhanced_prompt !== value) {
        setPreEnhance(value);
        onChange(enhanced_prompt);
      }
    } catch (err) {
      notifyError(err, 'Failed to enhance prompt');
    } finally {
      setEnhancing(false);
    }
  };

  const handleUndo = () => {
    if (preEnhance === null) return;
    onChange(preEnhance);
    setPreEnhance(null);
  };

  return (
    <div className="prompt-editor">
      <div className="prompt-editor__toolbar">
        {label && <label className="prompt-editor__label">{label}</label>}
        <div className="prompt-editor__actions">
          {preEnhance !== null ? (
            <Button type="button" variant="ghost" size="sm" onClick={handleUndo}>
              <Undo2 size={14} /> Undo enhance
            </Button>
          ) : (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              loading={enhancing}
              disabled={!value.trim()}
              onClick={() => void handleEnhance()}
              title="Rewrite this prompt with the enhancer"
            >
              <Sparkles size={14} /> Enhance
            </Button>
          )}
          {onInsertFromLibrary && (
            <Button type="button" variant="ghost" size="sm" onClick={onInsertFromLibrary}>
              <Library size={14} /> Insert from library
            </Button>
          )}
        </div>
      </div>
      <Textarea
        ref={ref}
        value={value}
        rows={rows}
        spellCheck={false}
        placeholder={placeholder}
        onChange={(e) => {
          if (preEnhance !== null) setPreEnhance(null);
          onChange(e.target.value);
        }}
      />
      <div className="prompt-editor__meta">
        {hint && <span className="prompt-editor__hint">{hint}</span>}
        <span className="prompt-editor__count" title="Approximate">
          ~{estimateTokens(value)} tokens · {value.length} chars
        </span>
      </div>
    </div>
  );
});
